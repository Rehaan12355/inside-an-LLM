"""Inside an LLM: a Jacobian-lens "brain scan" web app.

Ask a local open-weights model a question, then look at which words light up
in its residual stream, layer by layer and token by token. The readout is the
Jacobian lens (J-lens) from Anthropic's "Verbalizable Representations Form a
Global Workspace in Language Models" (2026), via its reference implementation
(github.com/anthropics/jacobian-lens) and a pre-fitted lens from Neuronpedia:

    lens_l(h) = softmax( W_U . norm( J_l @ h ) ),   J_l = E[dh_final / dh_l]

The "workspace" view decomposes a residual into a sparse nonnegative
combination of J-lens vectors (rows of W_U J_l) with gradient pursuit, as the
paper does (k <= 25).

Run:  python server.py   then open http://127.0.0.1:8000
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import torch
import transformers
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import jlens
from jlens.hooks import ActivationRecorder
from jlens.vis import _meaningful_token_mask

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

MODEL_NAME = os.environ.get("JLENS_MODEL", "Qwen/Qwen3.5-4B")
LENS_REPO = os.environ.get("JLENS_LENS_REPO", "neuronpedia/jacobian-lens")
LENS_REVISION = os.environ.get("JLENS_LENS_REVISION", "qwen-n1000")
LENS_FILE = os.environ.get(
    "JLENS_LENS_FILE",
    {
        "Qwen/Qwen3.5-4B": "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
        "Qwen/Qwen3.6-27B": "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    }.get(MODEL_NAME, "lens.pt"),
)
TOP_N = 8  # words kept per (position, layer) cell
MAX_TOKENS = 1024  # prompt + reply cap for one analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("inside-an-llm")


class Brain:
    """Model, lens, and the derived tensors the readouts need."""

    def __init__(self) -> None:
        log.info("loading %s", MODEL_NAME)
        self.hf = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16
        ).cuda()
        self.tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = jlens.from_hf(self.hf, self.tok)
        self.n_layers = self.model.n_layers
        self.final_layer = self.n_layers - 1

        log.info("loading lens %s:%s", LENS_REPO, LENS_FILE)
        lens = jlens.JacobianLens.from_pretrained(
            LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
        )
        self.source_layers: list[int] = lens.source_layers
        # [n_source, d, d] fp32 on the GPU, indexed like source_layers.
        self.J = torch.stack([lens.jacobians[l] for l in self.source_layers]).cuda()
        self.layer_index = {l: i for i, l in enumerate(self.source_layers)}

        self.W_U: torch.Tensor = self.model._lm_head.weight  # [V, d] bf16
        self.vocab_size = self.W_U.shape[0]
        self.word_mask = self._load_word_mask()
        self.gloss = self._load_gloss()
        self._atom_norms: torch.Tensor | None = None
        self._W_eff: torch.Tensor | None = None
        self.lock = threading.Lock()
        self.runs: OrderedDict[str, dict] = OrderedDict()
        log.info(
            "ready: %d layers, d_model=%d, vocab=%d, lens layers %d..%d",
            self.n_layers, self.model.d_model, self.vocab_size,
            self.source_layers[0], self.source_layers[-1],
        )

    # ----------------------------------------------------------------- setup

    def _cache_path(self, name: str) -> Path:
        return CACHE / f"{MODEL_NAME.replace('/', '__')}.{name}.pt"

    def _load_word_mask(self) -> torch.Tensor:
        """Tokens that decode to word characters; display is restricted to
        these (as in the reference slice viewer), ranks stay full-vocab."""
        path = self._cache_path("wordmask")
        if path.exists():
            return torch.load(path).cuda()
        log.info("building word-like token mask (one-time)...")
        mask = _meaningful_token_mask(self.tok, self.vocab_size, torch.device("cpu"))
        torch.save(mask, path)
        return mask.cuda()

    def _load_gloss(self) -> dict[int, str]:
        path = ROOT / "assets" / "qwen_gloss.json.gz"
        if "qwen" not in MODEL_NAME.lower() or not path.exists():
            return {}
        return {int(k): v for k, v in json.load(gzip.open(path)).items()}

    def effective_unembed(self) -> torch.Tensor:
        """W_U with the final RMSNorm gain folded in: W_U diag(gamma). The
        lens logits are a positive rescaling of W_eff @ (J h), so its rows
        times J_l are exactly the J-lens vectors the readout ranks with."""
        if self._W_eff is None:
            x = torch.randn(4, self.model.d_model, device="cuda", dtype=torch.float32)
            normed = self.model._final_norm(x.to(self.W_U.dtype)).float()
            rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            gamma = (normed * rms / x).median(0).values
            self._W_eff = (self.W_U.float() * gamma).to(torch.bfloat16)
        return self._W_eff

    def atom_norms(self) -> torch.Tensor:
        """||row v of W_eff J_l|| for every source layer and token: [n_src, V]."""
        if self._atom_norms is not None:
            return self._atom_norms
        path = self._cache_path("atomnorms")
        if path.exists():
            self._atom_norms = torch.load(path).cuda()
            return self._atom_norms
        log.info("computing J-lens vector norms (one-time, ~a minute)...")
        W = self.effective_unembed()
        norms = torch.empty(len(self.source_layers), self.vocab_size, device="cuda")
        chunk = 16384
        for i in range(len(self.source_layers)):
            Jb = self.J[i].to(torch.bfloat16)
            for s in range(0, self.vocab_size, chunk):
                norms[i, s : s + chunk] = (W[s : s + chunk] @ Jb).float().norm(dim=-1)
        torch.save(norms.cpu(), path)
        self._atom_norms = norms
        return norms

    # ------------------------------------------------------------ inference

    def decode(self, ids: list[int]) -> list[str]:
        return [self.tok.decode([t], clean_up_tokenization_spaces=False) for t in ids]

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int, system: str | None) -> tuple[list[int], int]:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        try:
            text = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        # Stop at the end of the assistant turn, not just the base-model EOS.
        stop = {self.tok.eos_token_id}
        for t in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>"):
            tid = self.tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid != self.tok.unk_token_id:
                stop.add(tid)
        stop.discard(None)
        out = self.hf.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=sorted(stop),
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        return out[0].tolist(), ids.shape[1]

    def reply_text(self, prompt: str, max_new_tokens: int, system: str | None = None) -> str:
        """Generate a reply and return just its text (no lens analysis)."""
        ids, prompt_len = self.generate(prompt, max_new_tokens, system)
        return self.tok.decode(ids[prompt_len:], skip_special_tokens=True).strip()

    @torch.no_grad()
    def analyze(self, ids: list[int], prompt_len: int) -> dict:
        """Forward pass, then J-lens and logit-lens readouts at every
        (position, layer)."""
        input_ids = torch.tensor([ids], device="cuda")
        layers = sorted(set(self.source_layers) | {self.final_layer})
        with ActivationRecorder(self.model.layers, at=layers) as rec:
            self.model.forward(input_ids)
            acts = {l: rec.activations[l][0].detach() for l in layers}

        neg_inf = torch.tensor(float("-inf"), device="cuda")
        out: dict[str, list] = {"jlens": [], "logit": []}
        seen: set[int] = set()
        for layer in layers:
            resid = acts[layer].float()  # [P, d]
            variants = {"logit": resid}
            if layer in self.layer_index:
                variants["jlens"] = resid @ self.J[self.layer_index[layer]].T
            else:  # final layer: J = I, this row is the model's actual output
                variants["jlens"] = resid
            for name, r in variants.items():
                logits = self.model.unembed(r).float()  # [P, V]
                logp = torch.log_softmax(logits, dim=-1)
                masked = torch.where(self.word_mask, logp, neg_inf)
                top = masked.topk(TOP_N, dim=-1)
                ids_ = top.indices.tolist()
                probs = top.values.exp().tolist()
                seen.update(t for row in ids_ for t in row)
                out[name].append(
                    [[[t, round(p, 5)] for t, p in zip(ri, rp)] for ri, rp in zip(ids_, probs)]
                )
                del logits, logp, masked

        run_id = uuid.uuid4().hex[:12]
        seen.update(ids)
        vocab = {t: s for t, s in zip(sorted(seen), self.decode(sorted(seen)))}
        self.runs[run_id] = {
            "acts": {l: acts[l] for l in self.source_layers},
            "ids": ids,
            "prompt_len": prompt_len,
            "layers": layers,
            "readout": out,
            "vocab": vocab,
        }
        while len(self.runs) > 4:
            self.runs.popitem(last=False)

        return {
            "run_id": run_id,
            "model": MODEL_NAME,
            "tokens": ids,
            "token_strs": self.decode(ids),
            "prompt_len": prompt_len,
            "layers": layers,
            "final_layer": self.final_layer,
            # readout[lens][layer_idx][position] = [[token_id, prob], ...]
            "readout": out,
            "vocab": vocab,
            "gloss": {t: self.gloss[t] for t in seen if t in self.gloss},
        }

    @torch.no_grad()
    def workspace(self, run_id: str, position: int, k: int = 25, word_only: bool = True) -> dict:
        """Sparse nonnegative decomposition of the residual at ``position`` at
        every lens layer into k J-lens vectors, by greedy selection plus
        projected gradient-pursuit steps on the support."""
        run = self.runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        n_src = len(self.source_layers)
        h = torch.stack([run["acts"][l][position].float() for l in self.source_layers])  # [L, d]
        W = self.effective_unembed()  # [V, d]
        norms = self.atom_norms()  # [L, V]
        allowed = self.word_mask if word_only else torch.ones_like(self.word_mask)
        banned = (~allowed).expand(n_src, -1).clone()

        support = torch.empty(n_src, 0, dtype=torch.long, device="cuda")
        atoms = torch.empty(n_src, 0, h.shape[1], device="cuda")  # unit-norm J-lens vectors
        coef = torch.empty(n_src, 0, device="cuda")
        resid = h.clone()
        for _ in range(k):
            Jr = torch.bmm(self.J, resid[:, :, None]).squeeze(-1)  # [L, d]
            scores = (Jr.to(torch.bfloat16) @ W.T).float() / norms.clamp_min(1e-6)
            scores[banned] = float("-inf")
            best = scores.argmax(dim=-1)  # [L]
            banned[torch.arange(n_src), best] = True
            new_atom = torch.bmm(W[best].float()[:, None, :], self.J).squeeze(1)
            new_atom = new_atom / new_atom.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            support = torch.cat([support, best[:, None]], 1)
            atoms = torch.cat([atoms, new_atom[:, None, :]], 1)
            coef = torch.cat([coef, torch.zeros(n_src, 1, device="cuda")], 1)
            for _ in range(4):  # projected gradient pursuit on the support
                resid = h - torch.bmm(coef[:, None, :], atoms).squeeze(1)
                g = torch.bmm(atoms, resid[:, :, None]).squeeze(-1)  # [L, s]
                g = torch.where((coef <= 0) & (g < 0), torch.zeros_like(g), g)
                Dg = torch.bmm(g[:, None, :], atoms).squeeze(1)
                step = (g * g).sum(-1) / (Dg * Dg).sum(-1).clamp_min(1e-12)
                coef = (coef + step[:, None] * g).clamp_min(0)
            resid = h - torch.bmm(coef[:, None, :], atoms).squeeze(1)

        explained = 1 - resid.pow(2).sum(-1) / h.pow(2).sum(-1).clamp_min(1e-12)
        result = []
        seen: set[int] = set()
        for i, layer in enumerate(self.source_layers):
            items = [
                (int(t), float(c))
                for t, c in zip(support[i].tolist(), coef[i].tolist())
                if c > 0
            ]
            items.sort(key=lambda x: -x[1])
            seen.update(t for t, _ in items)
            result.append(
                {"layer": layer, "explained": round(float(explained[i]), 4),
                 "norm": round(float(h[i].norm()), 3),
                 "atoms": [[t, round(c, 4)] for t, c in items]}
            )
        vocab = {t: s for t, s in zip(sorted(seen), self.decode(sorted(seen)))}
        return {
            "position": position,
            "layers": result,
            "vocab": vocab,
            "gloss": {t: self.gloss[t] for t in seen if t in self.gloss},
        }


    # -------------------------------------------------------------- summary

    def findings(self, run_id: str) -> dict:
        """Deterministic highlights pulled from a run's J-lens readout."""
        run = self.runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        ids, prompt_len, layers = run["ids"], run["prompt_len"], run["layers"]
        vocab, readout = run["vocab"], run["readout"]
        strs = self.decode(ids)
        n = len(ids)
        has_reply = prompt_len < n
        lo, hi = round(self.n_layers * 0.25), round(self.n_layers * 0.75)
        mid = [li for li, l in enumerate(layers) if lo <= l <= hi]
        jl = readout["jlens"]

        def word(t: int) -> str:
            return vocab[t].strip()

        def key(t: int) -> str:
            return vocab[t].strip().lower()

        def special(s: str) -> bool:
            return s.startswith("<|") or s.strip() in ("<think>", "</think>") or not s.strip()

        # Positions whose readouts count as "the model thinking while writing":
        # the reply plus the last prompt token (where it plans the reply), or
        # the whole text in raw mode.
        start = prompt_len - 1 if has_reply else 0
        scan = [p for p in range(start, n) if not special(strs[p]) or p == prompt_len - 1]

        full_text = "".join(strs).lower()
        text_words = set(re.findall(r"[\w'’-]+", full_text))
        token_keys = [s.strip().lower() for s in strs]

        def meaningful(k: str) -> bool:
            return len(k) >= 3 and k not in STOPWORDS

        # 1. Unspoken: lit up in the middle layers but never written anywhere.
        unspoken: dict[str, dict] = {}
        for li in mid:
            for p in scan:
                for t, prob in jl[li][p]:
                    k = key(t)
                    if not meaningful(k) or k in text_words or k.rstrip("s") in text_words or k + "s" in text_words:
                        continue
                    u = unspoken.setdefault(k, {"word": word(t), "score": 0.0, "gloss": self.gloss.get(t), "pos": p, "best": 0.0})
                    u["score"] += prob
                    if prob > u["best"]:
                        u["best"], u["pos"] = prob, p
        unspoken_top = sorted(unspoken.values(), key=lambda u: -u["score"])[:8]

        # 2. Foresight: a word visible in the middle layers at position p that
        # the model only writes later, at q >= p + 2.
        foresight: dict[str, dict] = {}
        for p in scan:
            for li in mid:
                for t, prob in jl[li][p][:5]:
                    k = key(t)
                    if prob < 0.02 or not meaningful(k) or k in foresight:
                        continue
                    later = [q for q in range(p + 2, n) if token_keys[q] == k]
                    if later and (not has_reply or later[0] >= prompt_len):
                        foresight[k] = {"word": word(t), "seen_at": p, "said_at": later[0],
                                        "lead": later[0] - p, "layer": layers[li], "prob": round(prob, 3)}
        foresight_top = sorted(foresight.values(), key=lambda f: (-f["lead"], -f["prob"]))[:6]

        # 3. Decision depth: earliest layer where the token actually written
        # next is in the lens's top 3, J-lens vs logit lens.
        def first_layer(lens: str, p: int, k: str) -> int | None:
            for li, layer in enumerate(layers):
                if any(key(t) == k for t, _ in readout[lens][li][p][:3]):
                    return layer
            return None

        depth = {"jlens": [], "logit": []}
        for p in range(start, n - 1):
            k = token_keys[p + 1]
            if not meaningful(k) or special(strs[p + 1]):
                continue
            for lens in depth:
                fl = first_layer(lens, p, k)
                if fl is not None:
                    depth[lens].append(fl)

        def median(xs: list[int]) -> float | None:
            xs = sorted(xs)
            return float(xs[len(xs) // 2]) if xs else None

        # 4. What it was thinking just before answering (or at the last token).
        anchor = prompt_len - 1 if has_reply else n - 1
        before: dict[str, float] = {}
        for li in mid:
            for t, prob in jl[li][anchor]:
                if meaningful(key(t)):
                    before[word(t)] = before.get(word(t), 0.0) + prob
        before_top = [w for w, _ in sorted(before.items(), key=lambda x: -x[1])[:6]]

        # 5. Hottest tokens: strongest top-1 activation in the middle layers.
        heat = []
        for p in scan:
            li = max(mid, key=lambda li: jl[li][p][0][1])
            t, prob = jl[li][p][0]
            heat.append({"pos": p, "token": strs[p], "word": word(t), "layer": layers[li], "prob": round(prob, 3)})
        heat.sort(key=lambda h: -h["prob"])

        return {
            "mode": "chat" if has_reply else "raw",
            "n_tokens": n,
            "mid_layers": [lo, hi],
            "final_layer": self.final_layer,
            "prompt": self.tok.decode(ids[:prompt_len], skip_special_tokens=True).strip(),
            "reply": self.tok.decode(ids[prompt_len:], skip_special_tokens=True).strip(),
            "anchor": {"pos": anchor, "token": strs[anchor], "words": before_top},
            "unspoken": unspoken_top,
            "foresight": foresight_top,
            "decision_layer": {
                "jlens": median(depth["jlens"]), "logit": median(depth["logit"]),
                "n_tokens": len(depth["jlens"]),
            },
            "hottest": heat[:4],
        }

    @torch.no_grad()
    def summarize(self, run_id: str) -> dict:
        f = self.findings(run_id)
        lines = [
            f"Prompt: {f['prompt'][:400]!r}",
            f"Model's reply: {f['reply'][:400]!r}" if f["mode"] == "chat"
            else "The model did NOT write anything; it only read this text, and we scanned it at the last word.",
            f"Words active in its middle layers right before it {'answered' if f['mode'] == 'chat' else 'continued'}: "
            + (", ".join(f["anchor"]["words"]) or "none clear"),
        ]
        if f["unspoken"]:
            lines.append("Words it thought about but never wrote: " + ", ".join(
                u["word"] + (f" ({u['gloss']})" if u.get("gloss") else "") for u in f["unspoken"][:6]))
        if f["foresight"]:
            lines.append("Words it was thinking of before writing them: " + "; ".join(
                f"'{x['word']}' {x['lead']} tokens early" for x in f["foresight"][:4]))
        d = f["decision_layer"]
        if d["jlens"] is not None:
            lines.append(
                f"It typically settled on its next word by layer {d['jlens']:.0f} of {f['final_layer']}"
                + (f" (a plain logit lens only sees it at layer {d['logit']:.0f})" if d["logit"] is not None else "")
            )
        facts = "\n".join(f"- {line}" for line in lines)
        instruction = (
            "You are explaining an interpretability scan of a small language model to a curious non-expert. "
            "A 'Jacobian lens' read which words were lighting up inside the model's middle layers. "
            "Using ONLY the facts below, write a 3-sentence plain-English summary of what the model "
            "seemed to be thinking. Strict rules: mention only words that appear in the facts; never name "
            "a word, place, or number that is not listed; do not claim the model answered unless a reply is "
            "given; if a word list looks like noise, say so briefly.\n\n"
            + facts
        )
        narrative = self.reply_text(instruction, max_new_tokens=200)
        return {"findings": f, "narrative": narrative, "facts": lines}


STOPWORDS = set("""
the and for that this with from have has had was were are you your but not they them their its it's
our out who what when where which why how all any can will would could should into about than then
there here been being also just some such very more most other only over under again further once
one two his her she him hers he's i'm we're let's did does doing done get got make made may might
must shall each few both own same too off upon onto per via yet nor
""".split())


# ------------------------------------------------------------------ web app

app = FastAPI(title="Inside an LLM")
brain: Brain | None = None


@app.on_event("startup")
def _load() -> None:
    global brain
    brain = Brain()
    threading.Thread(target=lambda: brain.atom_norms(), daemon=True).start()


class ThinkRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    mode: str = "chat"  # "chat": model replies; "raw": analyse the text as-is
    system: str | None = Field(default=None, max_length=8000)
    max_new_tokens: int = Field(default=96, ge=1, le=400)


class WorkspaceRequest(BaseModel):
    run_id: str
    position: int = Field(ge=0)
    k: int = Field(default=25, ge=1, le=64)
    word_only: bool = True


@app.post("/api/think")
def think(req: ThinkRequest) -> dict:
    assert brain is not None
    with brain.lock:
        if req.mode == "raw":
            ids = brain.tok(req.prompt, add_special_tokens=True).input_ids
            prompt_len = len(ids)
        else:
            ids, prompt_len = brain.generate(req.prompt, req.max_new_tokens, req.system)
        if len(ids) > MAX_TOKENS:
            raise HTTPException(400, f"sequence is {len(ids)} tokens; limit is {MAX_TOKENS}")
        result = brain.analyze(ids, prompt_len)
        torch.cuda.empty_cache()
        return result


@app.post("/api/workspace")
def workspace(req: WorkspaceRequest) -> dict:
    assert brain is not None
    with brain.lock:
        try:
            return brain.workspace(req.run_id, req.position, req.k, req.word_only)
        except KeyError:
            raise HTTPException(404, "run expired; ask again") from None
        except IndexError:
            raise HTTPException(400, "position out of range") from None


class SummaryRequest(BaseModel):
    run_id: str


@app.post("/api/summary")
def summary(req: SummaryRequest) -> dict:
    assert brain is not None
    with brain.lock:
        try:
            return brain.summarize(req.run_id)
        except KeyError:
            raise HTTPException(404, "run expired; ask again") from None


@app.get("/api/info")
def info() -> dict:
    return {"model": MODEL_NAME, "ready": brain is not None,
            "n_layers": brain.n_layers if brain else None}


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
