/* The time field, and the client that feeds it.
 *
 * The whole visual argument is one mapping: a passage's horizontal position is
 * how long its kind of source has actually been watching. A datasheet's
 * durability claim rests on 500 hours in a weathering cabinet and sits at the
 * left edge; a conservation report rests on an object examined after fifteen
 * years and sits far to the right. When an answer draws on both, you see the
 * span before you read a word of it -- and when every mote piles up in one
 * place, you can see that too, which is the more useful failure to notice.
 *
 * Everything is drawn procedurally. No images, no libraries: the grain, the
 * haze, the axis and the motes are code, so the page is a few kilobytes and
 * looks identical at any pixel density. That is also the Processing habit --
 * the drawing *is* the program, and a seeded field redraws the same way twice.
 */

const API = window.ARTMAT_API || "http://127.0.0.1:8021";

const LAYERS = [
  "manufacturer_datasheet",
  "materials_science",
  "conservation_literature",
  "collection_precedent",
];

const LAYER_LABEL = {
  manufacturer_datasheet: "manufacturer",
  materials_science: "materials science",
  conservation_literature: "conservation",
  collection_precedent: "collection precedent",
};

// Cold at hour zero, amber at sixty years. Same four values as the CSS custom
// properties; duplicated rather than read back out of the stylesheet because
// the canvas needs them as numbers on every frame and getComputedStyle in a
// draw loop is a needless reflow.
const LAYER_RGB = {
  manufacturer_datasheet: [143, 166, 173],
  materials_science: [168, 172, 147],
  conservation_literature: [200, 160, 113],
  collection_precedent: [184, 118, 63],
};

/* ---------- deterministic noise ------------------------------------------ */

// A passage's jitter must not change between frames, and it must not change
// between two people looking at the same answer. So position is derived from
// the chunk id rather than from Math.random: the same passage lands in the
// same place every time, which makes the picture a description of the data
// rather than of when it happened to be drawn.
function hash01(str, salt = 0) {
  let h = 2166136261 ^ salt;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 100000) / 100000;
}

/* ---------- the field ---------------------------------------------------- */

const canvas = document.getElementById("field");
const ctx = canvas.getContext("2d", { alpha: false });

// The band is a second canvas stacked above the horizon fade, so the axis and
// its motes are never occluded by a scrolling paragraph. It is transparent --
// the fade beneath it is what supplies the paper.
const band = document.getElementById("band");
const bctx = band.getContext("2d");

const HOUR = 1 / 8766; // in years
const T_MIN = HOUR;    // one hour
const T_MAX = 100;     // a century

let W = 0, H = 0, DPR = 1;
let BW = 0, BH = 0;
let motes = [];
let grain = null;
let t0 = performance.now();
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = window.innerWidth;
  H = window.innerHeight;
  canvas.width = Math.floor(W * DPR);
  canvas.height = Math.floor(H * DPR);
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);

  const r = band.getBoundingClientRect();
  BW = r.width; BH = r.height;
  band.width = Math.floor(BW * DPR);
  band.height = Math.floor(BH * DPR);
  bctx.setTransform(DPR, 0, 0, DPR, 0, 0);

  grain = null;
  layout();
}

// Log scale, because the interesting distances are ratios. On a linear axis
// 500 hours and 6 months would be the same pixel and the entire left half of
// the argument would collapse into the margin.
function xOf(years) {
  const t = Math.max(T_MIN, Math.min(T_MAX, years));
  const f = Math.log(t / T_MIN) / Math.log(T_MAX / T_MIN);
  return 0.08 * BW + f * 0.84 * BW;
}

function layout() {
  for (const m of motes) {
    m.x = xOf(m.years * (0.55 + 1.1 * hash01(m.id, 7)));
    // Motes sit above the axis line inside the band, spread by a stable hash
    // so two passages from the same layer do not stack into one dot.
    m.baseY = BH * (0.18 + 0.44 * hash01(m.id, 13));
  }
}

function setMotes(hits) {
  motes = hits.map((h) => ({
    id: h.chunk_id,
    layer: h.source_type,
    years: h.horizon_years || 1,
    score: h.score,
    born: performance.now(),
    r: 3 + 9 * Math.min(1, h.score * 1.6),
    phase: hash01(h.chunk_id, 29) * Math.PI * 2,
    speed: 0.25 + 0.5 * hash01(h.chunk_id, 31),
    lit: false,
  }));
  layout();
}

// Grain is generated once per resize and stamped, not recomputed per frame.
// Per-frame noise costs a full-canvas putImageData every 16 ms and buys a
// shimmer nobody asked for; a still grain reads as paper, which is the point.
function makeGrain() {
  const g = document.createElement("canvas");
  g.width = 220; g.height = 220;
  const gc = g.getContext("2d");
  const img = gc.createImageData(220, 220);
  for (let i = 0; i < img.data.length; i += 4) {
    const v = 128 + (Math.random() - 0.5) * 42;
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = 16;
  }
  gc.putImageData(img, 0, 0);
  return g;
}

function draw(now) {
  const t = (now - t0) / 1000;

  // Base wash: bleached paper, very slightly warmer toward the right, because
  // that is where the long time scales are and where everything eventually
  // yellows.
  const wash = ctx.createLinearGradient(0, 0, W, H);
  wash.addColorStop(0, "#f4f1e9");
  wash.addColorStop(0.62, "#f2efe6");
  wash.addColorStop(1, "#efe8d9");
  ctx.fillStyle = wash;
  ctx.fillRect(0, 0, W, H);

  // Overexposure: a soft blown-out band, drifting slowly. This is the one
  // gesture that is purely atmospheric, and it earns its place by making the
  // page feel lit from somewhere rather than filled with a colour.
  const bandY = H * (0.42 + 0.06 * Math.sin(t * 0.06));
  const glow = ctx.createRadialGradient(W * 0.5, bandY, 0, W * 0.5, bandY, Math.max(W, H) * 0.75);
  glow.addColorStop(0, "rgba(255,255,255,0.62)");
  glow.addColorStop(0.45, "rgba(255,255,255,0.18)");
  glow.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, W, H);

  if (!grain) grain = makeGrain();
  const pat = ctx.createPattern(grain, "repeat");
  ctx.fillStyle = pat;
  ctx.fillRect(0, 0, W, H);

  drawBand(t, now);
  requestAnimationFrame(draw);
}

function drawBand(t, now) {
  bctx.clearRect(0, 0, BW, BH);
  if (!motes.length) return;

  drawAxis(t);

  for (const m of motes) {
    const age = Math.min(1, (now - m.born) / 1400);
    const ease = 1 - Math.pow(1 - age, 3);
    const drift = reduced ? 0 : Math.sin(t * m.speed + m.phase) * 7;
    const y = m.baseY + drift;
    const rgb = LAYER_RGB[m.layer] || [150, 150, 150];
    const r = m.r * ease * (m.lit ? 1.7 : 1);

    // Bloom without a shadow filter: three passes at falling alpha. Cheaper
    // than shadowBlur, and it keeps the halo tinted rather than grey.
    for (let i = 3; i >= 1; i--) {
      bctx.beginPath();
      bctx.arc(m.x, y, r * i * 1.5, 0, Math.PI * 2);
      bctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(m.lit ? 0.1 : 0.055) / i})`;
      bctx.fill();
    }

    // A thread down to the axis. Without it a mote is a decorative dot; with
    // it, the dot is standing at a position on a scale, which is the claim.
    bctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.28 * ease})`;
    bctx.lineWidth = 1;
    bctx.beginPath();
    bctx.moveTo(m.x, y + r);
    bctx.lineTo(m.x, BH * 0.72);
    bctx.stroke();

    bctx.beginPath();
    bctx.arc(m.x, y, r, 0, Math.PI * 2);
    bctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.5 + 0.4 * ease})`;
    bctx.fill();
  }
}

const TICKS = [
  [500 * HOUR, "500 hours"],
  [1, "1 year"],
  [10, "10 years"],
  [60, "60 years"],
];

function drawAxis(t) {
  const y = BH * 0.72;
  bctx.strokeStyle = "rgba(90,84,74,0.20)";
  bctx.lineWidth = 1;
  bctx.beginPath();
  bctx.moveTo(0.06 * BW, y);
  bctx.lineTo(0.94 * BW, y);
  bctx.stroke();

  bctx.font = "11px ui-monospace, Menlo, monospace";
  bctx.fillStyle = "rgba(90,84,74,0.5)";
  bctx.textAlign = "center";
  for (const [years, label] of TICKS) {
    const x = xOf(years);
    bctx.beginPath();
    bctx.moveTo(x, y - 4);
    bctx.lineTo(x, y + 4);
    bctx.stroke();
    bctx.fillText(label, x, y + 20);
  }
  bctx.textAlign = "left";
  bctx.fillStyle = "rgba(90,84,74,0.38)";
  bctx.fillText("how long this kind of source has been watching", 0.06 * BW, y + 42);
}

window.addEventListener("resize", resize);
resize();
requestAnimationFrame(draw);

/* ---------- the client --------------------------------------------------- */

const form = document.getElementById("ask");
const input = document.getElementById("q");
const go = document.getElementById("go");
const rewriteEl = document.getElementById("rewrite");
const legend = document.getElementById("axis-legend");
const answerEl = document.getElementById("answer");
const metaEl = document.getElementById("meta");
const passagesEl = document.getElementById("passages");
const passageList = document.getElementById("passage-list");
const errorEl = document.getElementById("error");

document.querySelectorAll("#examples button").forEach((b) => {
  b.addEventListener("click", () => {
    input.value = b.textContent.trim();
    input.focus();
  });
});

function show(el, on = true) { el.hidden = !on; }

function reset() {
  show(rewriteEl, false);
  show(legend, false);
  show(answerEl, false);
  show(metaEl, false);
  show(passagesEl, false);
  show(errorEl, false);
  answerEl.innerHTML = "";
  passageList.innerHTML = "";
  motes = [];
}

function renderRewrite(d) {
  if (!d.used || d.rewritten === d.original) {
    rewriteEl.innerHTML = `<span class="now">${escapeHtml(d.original)}</span>`;
  } else {
    rewriteEl.innerHTML =
      `<span class="was">${escapeHtml(d.original)}</span> ` +
      `<span class="now">${escapeHtml(d.rewritten)}</span>`;
  }
  show(rewriteEl);
}

function renderLegend(hits) {
  const present = new Set(hits.map((h) => h.source_type));
  legend.querySelectorAll("span").forEach((s) => {
    s.classList.toggle("dim", !present.has(s.dataset.layer));
  });
  show(legend);
}

function renderPassages(hits) {
  passageList.innerHTML = "";
  for (const h of hits) {
    const el = document.createElement("div");
    el.className = "passage";
    el.dataset.layer = h.source_type;
    el.dataset.cid = h.chunk_id;
    el.innerHTML =
      `<div class="head"><span>${escapeHtml(h.title)}</span>` +
      `<span class="horizon">${horizonLabel(h.horizon_years)}</span></div>` +
      `<div class="body" hidden>${escapeHtml(h.text)}` +
      `<span class="cid">${escapeHtml(h.chunk_id)}</span></div>`;

    const body = el.querySelector(".body");
    el.querySelector(".head").addEventListener("click", () => {
      body.hidden = !body.hidden;
    });
    // Hovering a passage lights its mote. The link between the paragraph you
    // are reading and the point on the axis it came from is the whole reason
    // both are on screen at once; without it they are two lists.
    el.addEventListener("mouseenter", () => setLit(h.chunk_id, true, el));
    el.addEventListener("mouseleave", () => setLit(h.chunk_id, false, el));
    passageList.appendChild(el);
  }
  show(passagesEl);
}

function setLit(cid, on, el) {
  const m = motes.find((x) => x.id === cid);
  if (m) m.lit = on;
  el.classList.toggle("lit", on);
}

function horizonLabel(years) {
  if (years == null) return "";
  if (years < 0.1) return `${Math.round(years * 8766)} hours`;
  if (years < 1) return `${Math.round(years * 12)} months`;
  return `${Math.round(years)} years`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// Markdown is not parsed. The model emits paragraphs and the occasional bold
// run, and a full parser here would be a lot of surface area for two features
// -- and a way to get raw HTML from a model onto the page. Text is escaped
// first, then the two safe patterns are re-introduced.
function renderAnswer(text, streaming) {
  const html = escapeHtml(text)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .split(/\n{2,}/)
    .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
    .join("");
  answerEl.innerHTML = streaming ? html + '<span class="cursor"></span>' : html;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  reset();
  go.disabled = true;
  go.textContent = "…";

  let acc = "";
  let scrolled = false;
  try {
    const res = await fetch(`${API}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k: 8, variant: "arbitrated", rewrite: true }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);

    // Hand-rolled SSE over fetch rather than EventSource, because EventSource
    // cannot POST and the question does not belong in a URL.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const ev = /^event: (.+)$/m.exec(raw);
        const dt = /^data: (.+)$/m.exec(raw);
        if (!ev || !dt) continue;
        const data = JSON.parse(dt[1]);

        if (ev[1] === "rewrite") {
          renderRewrite(data);
        } else if (ev[1] === "hits") {
          setMotes(data.hits);
          renderLegend(data.hits);
          renderPassages(data.hits);
        } else if (ev[1] === "token") {
          acc += data.text;
          show(answerEl);
          renderAnswer(acc, true);
          // Once, on the first token. A short page puts the opening of the
          // answer down inside the horizon fade, where it is dissolving into
          // paper before it has been read -- the effect is right for text you
          // have finished with and wrong for text that is still arriving.
          if (!scrolled) {
            scrolled = true;
            rewriteEl.scrollIntoView({
              behavior: reduced ? "auto" : "smooth",
              block: "start",
            });
          }
        } else if (ev[1] === "done") {
          renderAnswer(acc, false);
          metaEl.textContent =
            `${data.retrieval_ms} ms retrieval · ${data.generate_ms} ms generation · ` +
            `${data.input_tokens.toLocaleString()} in / ${data.output_tokens.toLocaleString()} out · ` +
            `$${data.cost_usd.toFixed(4)}`;
          show(metaEl);
        } else if (ev[1] === "error") {
          throw new Error(data.message);
        }
      }
    }
  } catch (err) {
    errorEl.textContent = `That did not work: ${err.message}`;
    show(errorEl);
  } finally {
    go.disabled = false;
    go.textContent = "ask";
  }
});
