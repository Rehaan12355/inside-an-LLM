"use strict";

const $ = (s) => document.querySelector(s);
const state = { mode: "chat", lens: "jlens", data: null, sel: null, wsSeq: 0, sumSeq: 0 };

// ---------------------------------------------------------------- helpers

const isSpecial = (s) => /^<\|.*\|>$/.test(s) || /^<\/?think>$/.test(s);
const showTok = (s) => s.replace(/\n/g, "↵").replace(/\t/g, "⇥") || "∅";
const wordOf = (id, vocab) => (vocab[id] ?? `#${id}`);
const norm = (s) => s.trim().toLowerCase();
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function midBand(layer) {
  const n = state.data.final_layer + 1;
  return layer >= Math.round(n * 0.25) && layer <= Math.round(n * 0.75);
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function setStatus(text, cls = "") {
  const s = $("#status");
  s.textContent = text;
  s.className = "status " + cls;
}

// Brightness: sqrt-scaled probability relative to a high percentile of the
// top-1 probabilities in the current lens (final layer excluded), so weak
// lenses still show contrast and strong cells saturate.
function computeScale() {
  const d = state.data, r = d.readout[state.lens], vals = [];
  d.layers.forEach((layer, li) => {
    if (layer === d.final_layer) return;
    for (const cell of r[li]) vals.push(cell[0][1]);
  });
  vals.sort((a, b) => a - b);
  state.scale = Math.max(vals[Math.floor(vals.length * 0.97)] || 1e-3, 1e-4);
}
const alphaOf = (p) => Math.min(1, Math.sqrt(p / state.scale));

// ---------------------------------------------------------------- tooltip

const tip = $("#tip");
function showTip(ev, title, cell) {
  const d = state.data;
  tip.innerHTML = `<div class="g">${esc(title)}</div>` + cell.map(([id, p]) => {
    const g = d.gloss[id] ? ` <span class="g">(${esc(d.gloss[id])})</span>` : "";
    return `<div class="t"><b>${esc(showTok(wordOf(id, d.vocab)))}${g}</b><i>${(p * 100).toFixed(1)}%</i></div>`;
  }).join("");
  tip.hidden = false;
  moveTip(ev);
}
function moveTip(ev) {
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + h > innerHeight - 8) y = ev.clientY - h - pad;
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
const hideTip = () => { tip.hidden = true; };

// ---------------------------------------------------------------- render

function renderAll() {
  computeScale();
  renderTokens();
  renderScan();
  renderLit();
  if (state.sel != null) select(state.sel, false);
}

function tokenGlow(pos) {
  const d = state.data, r = d.readout.jlens;
  let best = 0;
  d.layers.forEach((layer, li) => { if (midBand(layer)) best = Math.max(best, r[li][pos][0][1]); });
  return best;
}

function renderTokens() {
  const d = state.data, box = $("#tokens");
  box.innerHTML = "";
  const lensScale = state.scale;
  d.token_strs.forEach((s, pos) => {
    const t = el("span", "tok", showTok(s));
    if (pos >= d.prompt_len) t.classList.add("reply");
    if (isSpecial(s)) t.classList.add("special");
    t.dataset.pos = pos;
    t.style.setProperty("--a", (Math.min(1, Math.sqrt(tokenGlow(pos) / lensScale)) * 0.55).toFixed(3));
    t.onclick = () => select(pos);
    box.appendChild(t);
    if (s.includes("\n")) box.appendChild(el("span", "tok nl"));
  });
}

function select(pos, scroll = true) {
  const d = state.data;
  state.sel = pos;
  document.querySelectorAll(".tok.sel").forEach((e) => e.classList.remove("sel"));
  const t = document.querySelector(`.tok[data-pos="${pos}"]`);
  if (t) { t.classList.add("sel"); if (scroll) t.scrollIntoView({ block: "nearest" }); }
  document.querySelectorAll(".scan .colsel").forEach((e) => e.classList.remove("colsel"));
  document.querySelectorAll(`.scan [data-pos="${pos}"]`).forEach((e) => e.classList.add("colsel"));
  $("#selTok").textContent = `“${showTok(d.token_strs[pos])}” (#${pos})`;
  renderLadder(pos);
  loadWorkspace(pos);
}

function renderLadder(pos) {
  const d = state.data, r = d.readout[state.lens], box = $("#ladder");
  box.innerHTML = "";
  for (let li = d.layers.length - 1; li >= 0; li--) {
    const layer = d.layers[li], cell = r[li][pos];
    const rung = el("div", "rung");
    if (layer === d.final_layer) rung.classList.add("out");
    if (midBand(layer)) rung.classList.add("mid");
    rung.appendChild(el("span", "ln", layer === d.final_layer ? "out" : `L${layer}`));
    const words = el("div", "words");
    cell.slice(0, 6).forEach(([id, p]) => {
      const a = layer === d.final_layer ? Math.min(1, Math.sqrt(p)) : alphaOf(p);
      const w = el("span", "w" + (a > 0.6 ? " lit" : ""), showTok(wordOf(id, d.vocab)).trim());
      w.style.setProperty("--a", a.toFixed(3));
      words.appendChild(w);
    });
    words.onmouseenter = (ev) => showTip(ev, `layer ${layer}`, cell);
    words.onmousemove = moveTip;
    words.onmouseleave = hideTip;
    rung.appendChild(words);
    box.appendChild(rung);
  }
}

async function loadWorkspace(pos) {
  const d = state.data, seq = ++state.wsSeq;
  const cloud = $("#wsCloud"), layersBox = $("#wsLayers");
  cloud.innerHTML = '<span class="empty">decomposing…</span>';
  layersBox.innerHTML = "";
  let ws;
  try {
    const res = await fetch("/api/workspace", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: d.run_id, position: pos, word_only: $("#wordOnly").checked }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    ws = await res.json();
  } catch (e) {
    if (seq === state.wsSeq) cloud.innerHTML = `<span class="empty err">${esc(String(e.message || e))}</span>`;
    return;
  }
  if (seq !== state.wsSeq) return;

  // Cloud: coefficient as a fraction of the residual norm, summed over the
  // middle band of layers, merged by surface form.
  const agg = new Map();
  for (const L of ws.layers) {
    if (!midBand(L.layer)) continue;
    for (const [id, c] of L.atoms) {
      const word = wordOf(id, ws.vocab).trim();
      const key = norm(word);
      const cur = agg.get(key) || { word, v: 0, gloss: ws.gloss[id] };
      cur.v += c / L.norm;
      agg.set(key, cur);
    }
  }
  drawCloud(cloud, [...agg.values()], 40);

  for (let i = ws.layers.length - 1; i >= 0; i--) {
    const L = ws.layers[i];
    const row = el("div", "wsl" + (midBand(L.layer) ? " mid" : ""));
    row.appendChild(el("span", "ln", `L${L.layer}`));
    row.appendChild(el("span", "ev", `${Math.round(L.explained * 100)}%`)).title = "variance of the residual explained by these atoms";
    const words = el("div", "words");
    const top = L.atoms[0] ? L.atoms[0][1] : 1;
    L.atoms.slice(0, 6).forEach(([id, c]) => {
      const a = Math.min(1, c / top);
      const w = el("span", "w" + (a > 0.6 ? " lit" : ""), wordOf(id, ws.vocab).trim());
      w.style.setProperty("--a", (a * 0.9).toFixed(3));
      w.title = (ws.gloss[id] ? ws.gloss[id] + " · " : "") + `coef ${c.toFixed(2)}`;
      words.appendChild(w);
    });
    row.appendChild(words);
    layersBox.appendChild(row);
  }
}

function drawCloud(box, items, limit) {
  box.innerHTML = "";
  items = items.filter((x) => x.word && x.v > 0).sort((a, b) => b.v - a.v).slice(0, limit);
  if (!items.length) { box.appendChild(el("span", "empty", "nothing lit up")); return; }
  const max = items[0].v;
  // shuffle-ish order for a cloud look while keeping the biggest near the start
  items.forEach((x, i) => { x.o = i < 3 ? i : Math.random() * items.length; });
  items.sort((a, b) => a.o - b.o).forEach((x) => {
    const f = Math.sqrt(x.v / max);
    const s = el("span", null, x.word);
    s.style.setProperty("--fs", `${12 + f * 26}px`);
    s.style.setProperty("--a", (0.35 + 0.65 * f).toFixed(3));
    s.title = (x.gloss ? x.gloss + " · " : "") + `score ${x.v.toFixed(3)}`;
    box.appendChild(s);
  });
}

function renderScan() {
  const d = state.data, r = d.readout[state.lens], grid = $("#scan");
  const stride = +$("#stride").value;
  const replyOnly = $("#replyOnly").checked && d.prompt_len < d.tokens.length;
  const positions = d.tokens.map((_, i) => i).filter((i) => !replyOnly || i >= d.prompt_len);
  const rows = [];
  for (let li = d.layers.length - 1; li >= 0; li--) {
    const layer = d.layers[li];
    if (layer === d.final_layer || layer % stride === 0) rows.push(li);
  }
  grid.innerHTML = "";
  grid.style.gridTemplateColumns = `44px repeat(${positions.length}, 78px)`;
  grid.appendChild(el("div", "c lab hdr corner", "layer"));
  for (const pos of positions) {
    const h = el("div", "c hdr" + (pos >= d.prompt_len ? " reply" : ""), showTok(d.token_strs[pos]));
    h.dataset.pos = pos;
    h.title = `token #${pos}`;
    h.onclick = () => select(pos);
    grid.appendChild(h);
  }
  const frag = document.createDocumentFragment();
  for (const li of rows) {
    const layer = d.layers[li], isOut = layer === d.final_layer;
    frag.appendChild(el("div", "c lab" + (midBand(layer) ? " mid" : "") + (isOut ? " outrow" : ""), isOut ? "out" : `L${layer}`));
    for (const pos of positions) {
      const cell = r[li][pos], [id, p] = cell[0];
      const a = isOut ? Math.min(1, Math.sqrt(p)) : alphaOf(p);
      const c = el("div", "c" + (a > 0.6 ? " lit" : "") + (isOut ? " outrow" : ""), showTok(wordOf(id, d.vocab)).trim());
      c.style.setProperty("--a", (a * 0.9).toFixed(3));
      c.dataset.pos = pos;
      c.onmouseenter = (ev) => showTip(ev, `layer ${layer} · “${showTok(d.token_strs[pos])}”`, cell);
      c.onmousemove = moveTip;
      c.onmouseleave = hideTip;
      c.onclick = () => select(pos);
      frag.appendChild(c);
    }
  }
  grid.appendChild(frag);
}

function renderLit() {
  const d = state.data, r = d.readout[state.lens];
  const replyOnly = $("#cloudReply").checked && d.prompt_len < d.tokens.length;
  const unspoken = $("#unspoken").checked;
  const textWords = new Set(
    d.token_strs.join("").toLowerCase().split(/[^\p{L}\p{N}'’-]+/u).filter(Boolean)
  );
  const agg = new Map();
  d.layers.forEach((layer, li) => {
    if (!midBand(layer)) return;
    r[li].forEach((cell, pos) => {
      if (replyOnly && pos < d.prompt_len) return;
      if (isSpecial(d.token_strs[pos])) return;
      for (const [id, p] of cell) {
        const word = wordOf(id, d.vocab).trim(), key = norm(word);
        if (unspoken && textWords.has(key)) continue;
        const cur = agg.get(key) || { word, v: 0, gloss: d.gloss[id] };
        cur.v += p;
        agg.set(key, cur);
      }
    });
  });
  drawCloud($("#litCloud"), [...agg.values()], 70);
}

// ---------------------------------------------------------------- summary

async function loadSummary() {
  const d = state.data, seq = ++state.sumSeq;
  const nar = $("#narrative"), box = $("#findings");
  nar.className = "narrative loading";
  nar.textContent = "Reading the scan and writing a summary…";
  box.innerHTML = "";
  let s;
  try {
    const res = await fetch("/api/summary", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: d.run_id }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    s = await res.json();
  } catch (e) {
    if (seq === state.sumSeq) { nar.className = "narrative err"; nar.textContent = String(e.message || e); }
    return;
  }
  if (seq !== state.sumSeq) return;
  nar.className = "narrative";
  nar.textContent = s.narrative;
  renderFindings(s.findings);
}

function finding(title) {
  const f = el("div", "finding");
  f.appendChild(el("h3", null, title));
  return f;
}

function findingList(f, items, render, empty) {
  if (!items.length) { f.appendChild(el("div", "none", empty)); return; }
  const ul = el("ul");
  for (const it of items) {
    const li = el("li");
    li.innerHTML = render(it);
    if (it.pos != null) li.onclick = () => select(it.pos);
    ul.appendChild(li);
  }
  f.appendChild(ul);
}

function renderFindings(f) {
  const box = $("#findings"), tokLabel = (s) => `“${esc(showTok(s))}”`;
  box.innerHTML = "";

  const before = finding(f.mode === "chat" ? "Right before answering" : "At the last token");
  findingList(before, f.anchor.words.map((w) => ({ w, pos: f.anchor.pos })),
    (x) => `<b>${esc(x.w)}</b>`, "nothing clear in the middle layers");
  before.querySelector("ul")?.setAttribute("style", "display:flex;flex-wrap:wrap;gap:4px 12px");
  before.appendChild(el("div", "note", `middle layers (${f.mid_layers[0]}–${f.mid_layers[1]}) at ${showTok(f.anchor.token)}`));
  box.appendChild(before);

  const fore = finding("Thought of before saying");
  findingList(fore, f.foresight.map((x) => ({ ...x, pos: x.seen_at })),
    (x) => `<b>${esc(x.word)}</b> <span class="m">${x.lead} tokens early · L${x.layer}</span>`,
    "no word showed up ahead of time");
  box.appendChild(fore);

  const uns = finding("Thought of but never said");
  findingList(uns, f.unspoken,
    (x) => `<b>${esc(x.word)}</b>${x.gloss ? ` <span class="m">(${esc(x.gloss)})</span>` : ""}`,
    "nothing notable");
  box.appendChild(uns);

  const dec = finding("When it decides the next word");
  const dl = f.decision_layer;
  if (dl.jlens != null) {
    dec.appendChild(el("div", "big", `layer ${dl.jlens} / ${f.final_layer}`));
    dec.appendChild(el("div", "note",
      `median over ${dl.n_tokens} tokens: first layer where the next word is in the J-lens top 3` +
      (dl.logit != null ? `. The plain logit lens only sees it at layer ${dl.logit}.` : ".")));
  } else {
    dec.appendChild(el("div", "none", "not enough word tokens to measure"));
  }
  box.appendChild(dec);

  const hot = finding("Hottest tokens");
  findingList(hot, f.hottest,
    (x) => `${tokLabel(x.token)} → <b>${esc(x.word)}</b> <span class="m">${(x.prob * 100).toFixed(0)}% · L${x.layer}</span>`,
    "—");
  box.appendChild(hot);
}

// ---------------------------------------------------------------- actions

async function run() {
  const prompt = $("#prompt").value.trim();
  if (!prompt) return;
  const btn = $("#go");
  btn.disabled = true;
  setStatus(state.mode === "chat" ? "model is thinking and replying…" : "scanning…", "busy");
  try {
    const res = await fetch("/api/think", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, mode: state.mode, max_new_tokens: +$("#maxTokens").value || 80 }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    state.data = await res.json();
    $("#results").hidden = false;
    // Default selection: the last prompt token, where the model plans its reply.
    state.sel = Math.max(0, state.data.prompt_len - 1);
    if (state.mode === "raw") state.sel = state.data.tokens.length - 1;
    renderAll();
    setStatus(`${state.data.model} · ${state.data.tokens.length} tokens`, "ok");
    loadSummary();
  } catch (e) {
    setStatus("error: " + (e.message || e), "");
  } finally {
    btn.disabled = false;
  }
}

function bindSeg(seg, key, onChange) {
  seg.querySelectorAll("button").forEach((b) => {
    b.onclick = () => {
      seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      state[key] = b.dataset[key];
      onChange && onChange();
    };
  });
}

bindSeg(document.querySelector(".composer .seg"), "mode", () => {
  $("#maxTokens").disabled = state.mode === "raw";
});
bindSeg($("#lensSeg"), "lens", () => state.data && renderAll());
$("#go").onclick = run;
$("#resummarize").onclick = () => state.data && loadSummary();
$("#prompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(); });
$("#stride").onchange = () => state.data && (renderScan(), state.sel != null && select(state.sel, false));
$("#replyOnly").onchange = $("#stride").onchange;
$("#unspoken").onchange = () => state.data && renderLit();
$("#cloudReply").onchange = $("#unspoken").onchange;
$("#wordOnly").onchange = () => state.data && state.sel != null && loadWorkspace(state.sel);
document.querySelectorAll("[data-ex]").forEach((b) => {
  b.onclick = () => {
    const [mode, text] = b.dataset.ex.split("|");
    document.querySelector(`.composer .seg [data-mode="${mode}"]`).click();
    $("#prompt").value = text;
    run();
  };
});

(async function poll() {
  try {
    const info = await (await fetch("/api/info")).json();
    setStatus(`${info.model} · ${info.n_layers} layers · ready`, "ok");
  } catch { setStatus("server not reachable"); setTimeout(poll, 2000); }
})();
