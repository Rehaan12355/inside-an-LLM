# Inside an LLM

Ask a local language model a question, then see which words light up in its
residual stream, at every layer and every token. It uses the **Jacobian lens**
from Anthropic's *Verbalizable Representations Form a Global Workspace in
Language Models* (2026).

```
lens_l(h) = softmax( W_U · norm( J_l · h ) ),   J_l = E[ ∂h_final / ∂h_l ]
```

- Model: `Qwen/Qwen3.5-4B` (bf16, ~13 GB of VRAM with the lens loaded).
- Lens: Neuronpedia's pre-fitted lens (1000 WikiText prompts), loaded with
  Anthropic's reference code [`jlens`](https://github.com/anthropics/jacobian-lens).

## Run

```powershell
.venv\Scripts\python server.py      # then open http://127.0.0.1:8000
```

First-time setup:

```powershell
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements.txt
```

The first launch downloads the model (~9 GB) and the lens (~400 MB). It also
caches a word-token mask and the J-lens vector norms in `cache/`.

## What you see

| Panel | What it shows |
|---|---|
| **Summary** | Findings computed from the lens: words active right before answering, words thought of before being said, words thought of but never said, the layer where the next word is decided, and the hottest tokens. A short paragraph written by the same local model from those findings sits on top; it can overreach, the findings can't. |
| **Tokens** | Your prompt plus the model's reply. Glow shows how strongly the middle layers light up a word at that token. Click a token to select it. |
| **Thoughts at token** | The lens's top words at each layer, with the output layer on top. Toggle **Logit lens** to compare against the plain baseline. |
| **Workspace** | The residual stream split into a sparse nonnegative mix of ≤25 J-lens vectors per layer, using gradient pursuit as in the paper. The cloud sums the middle layers; the % column is variance explained. |
| **Brain scan** | A layers × tokens grid of each cell's top word, like the paper's slice viewer. Hover a cell for the top 8. |
| **What lit up** | Words summed over the middle layers across the reply. You can limit it to words that never appear in the text: the model's "unspoken" thoughts. |

**Raw text** mode scans text exactly as written, with no reply. Use it for
paper-style prompts like `Fact: The currency used in the country shaped like a boot is`.
In that example, "currency" appears around layer 21 and "euro" around layer 27.

## Notes and choices that go beyond the paper

- Display is limited to word-like tokens, as in the reference viewer's
  `mask_display`. The workspace dictionary uses only those tokens unless
  **words only** is unchecked.
- The final RMSNorm gain is folded into `W_U` when building J-lens vectors, so
  the vectors match what the readout ranks.
- "Middle layers" means 25–75% depth. The paper finds workspace properties
  only in the middle block.
- Qwen's vocabulary has many Chinese tokens. Hover them for an English gloss
  (machine-generated, from the jlens repo).
- Other models: set `JLENS_MODEL` and `JLENS_LENS_FILE` (or fit a lens with
  `jlens.fit`; ~100 prompts is usable).

## License

MIT. See [LICENSE](LICENSE). `assets/qwen_gloss.json.gz` comes from
[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) and
remains under its Apache-2.0 license. Model weights and the pre-fitted lens are
downloaded at run time under their own licenses.
