/* The book, and the client that fills it.
 *
 * Three moments carry the whole thing, and each is a real physical event drawn
 * rather than a fade:
 *
 *   opening   the cover swings, once. Afterwards the book is a surface, and a
 *             further question turns a page rather than reopening anything.
 *             Repeating a ceremony turns it into friction.
 *
 *   sinking   what you wrote is absorbed. Not faded out -- pulled down into
 *             the paper, spreading very slightly as it goes, the way ink
 *             actually behaves on a fibrous sheet.
 *
 *   welling   the answer comes back up through the paper, blurred first and
 *             resolving into letters, in the ink of whichever source speaks.
 *
 * This replaced a first version that plotted each source as a coloured dot on
 * a logarithmic time axis. That version was accurate and unreadable: you had
 * to learn a legend, then learn the axis was logarithmic, then discover that
 * hovering a passage lit its dot -- three things to learn before the picture
 * said anything. Ink age needs none of them. Everyone already knows that fresh
 * writing is black and old writing has gone brown.
 *
 * Everything is generated: the room, the grain, the bleed. No images, no
 * libraries, three files.
 */

const API = window.ARTMAT_API || "http://127.0.0.1:8021";

const LAYER_NAME = {
  manufacturer_datasheet: "manufacturer",
  materials_science: "materials science",
  conservation_literature: "conservation",
  collection_precedent: "collection",
};

// How a source's own words give it away. The `arbitrated` prompt already
// requires attribution in the sentence that makes the claim, so this reads the
// model's phrasing rather than guessing at it. It is a heuristic and it is
// allowed to miss: an untinted sentence just looks like ordinary ink, which is
// the right failure. The chunk ids on the slips stay the provenance of record.
const ATTRIBUTION = [
  [/manufactur|data ?sheet|smooth-?on|the maker|vendor/i, "manufacturer_datasheet"],
  [/conservation|conservator|restorer/i, "conservation_literature"],
  [/collection|tate|precedent|holdings/i, "collection_precedent"],
  [/study|studies|research|peer-reviewed|paper|literature|experiment/i, "materials_science"],
];

/* ---------- the room ------------------------------------------------------ */

const air = document.getElementById("air");
const actx = air.getContext("2d", { alpha: false });
let W = 0, H = 0, DPR = 1, grain = null;
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function sizeAir() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = window.innerWidth; H = window.innerHeight;
  air.width = Math.floor(W * DPR); air.height = Math.floor(H * DPR);
  actx.setTransform(DPR, 0, 0, DPR, 0, 0);
  grain = null;
  paintRoom();
}

// Painted once per resize, not per frame. The room does not move, and a
// requestAnimationFrame loop here would burn a core redrawing the same pixels.
function paintRoom() {
  const wash = actx.createLinearGradient(0, 0, W * 0.4, H);
  wash.addColorStop(0, "#f3f0e8");
  wash.addColorStop(1, "#e9e5da");
  actx.fillStyle = wash;
  actx.fillRect(0, 0, W, H);

  const glow = actx.createRadialGradient(
    W * 0.5, H * 0.34, 0, W * 0.5, H * 0.34, Math.max(W, H) * 0.7);
  glow.addColorStop(0, "rgba(255,255,255,0.7)");
  glow.addColorStop(0.5, "rgba(255,255,255,0.16)");
  glow.addColorStop(1, "rgba(255,255,255,0)");
  actx.fillStyle = glow;
  actx.fillRect(0, 0, W, H);

  if (!grain) grain = makeGrain();
  actx.fillStyle = actx.createPattern(grain, "repeat");
  actx.fillRect(0, 0, W, H);
}

function makeGrain() {
  const g = document.createElement("canvas");
  g.width = g.height = 200;
  const gc = g.getContext("2d");
  const img = gc.createImageData(200, 200);
  for (let i = 0; i < img.data.length; i += 4) {
    const v = 128 + (Math.random() - 0.5) * 46;
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = 18;
  }
  gc.putImageData(img, 0, 0);
  return g;
}

window.addEventListener("resize", sizeAir);
sizeAir();

/* ---------- opening ------------------------------------------------------- */

const body = document.body;
const cover = document.getElementById("cover");
const input = document.getElementById("q");

function openBook() {
  if (body.dataset.state !== "closed") return;
  body.dataset.state = "opening";
  document.getElementById("spread").setAttribute("aria-hidden", "false");
  // The cover takes 1.3 s to swing. Focus lands when the page is actually
  // there, not while it is still edge-on: a caret blinking on a surface the
  // reader cannot see yet is what makes a thing feel like a mock-up.
  setTimeout(() => {
    body.dataset.state = "open";
    input.focus();
  }, reduced ? 20 : 1350);
}

cover.addEventListener("click", openBook);
cover.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openBook(); }
});

/* ---------- ink ----------------------------------------------------------- */

const form = document.getElementById("ask");
const go = document.getElementById("go");
const rewriteEl = document.getElementById("rewrite");
const slipsEl = document.getElementById("slips");
const answerEl = document.getElementById("answer");
const metaEl = document.getElementById("meta");
const errorEl = document.getElementById("error");

input.addEventListener("input", () => {
  body.dataset.wet = input.value.trim() ? "1" : "0";
});

document.querySelectorAll("#examples button").forEach((b) => {
  b.addEventListener("click", () => {
    input.value = b.textContent.trim();
    body.dataset.wet = "1";
    input.focus();
  });
});

/* The sink.
 *
 * Paper absorbing a word does two things at once: the word loses contrast and
 * it spreads. Fading alone reads as a dissolve, blurring alone reads as a
 * lens; doing both while the text also drops a few pixels is what makes it
 * read as *into the page* rather than *off the screen*.
 */
function sink(el) {
  return new Promise((resolve) => {
    if (reduced) { resolve(); return; }
    const start = performance.now();
    const D = 900;
    function step(now) {
      const p = Math.min(1, (now - start) / D);
      const e = p * p;
      el.style.filter = `blur(${e * 2.6}px)`;
      el.style.opacity = String(1 - e);
      el.style.transform = `translateY(${e * 7}px)`;
      el.style.letterSpacing = `${e * 0.06}em`;
      if (p < 1) requestAnimationFrame(step);
      else {
        el.style.filter = ""; el.style.opacity = "";
        el.style.transform = ""; el.style.letterSpacing = "";
        resolve();
      }
    }
    requestAnimationFrame(step);
  });
}

/* The welling-up: text arrives blurred and slightly low, and resolves as if
   absorbed from beneath. Applied per paragraph as each completes, so a long
   answer surfaces in waves instead of shimmering as one block. */
function well(el) {
  if (reduced || !el) return;
  el.animate(
    [
      { filter: "blur(3px)", opacity: 0, transform: "translateY(6px)" },
      { filter: "blur(0px)", opacity: 1, transform: "none" },
    ],
    { duration: 850, easing: "cubic-bezier(0.2,0.7,0.3,1)", fill: "both" }
  );
}

/* ---------- rendering ----------------------------------------------------- */

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function whoSpeaks(sentence) {
  for (const [re, layer] of ATTRIBUTION) if (re.test(sentence)) return layer;
  return null;
}

// Markdown is not parsed. The model emits paragraphs and the occasional bold
// run; a full parser would be a lot of surface area for two features, and a
// way to get raw HTML from a model onto the page. Escape first, then put back
// only the two patterns that are safe.
function renderAnswer(text, streaming) {
  const html = text.split(/\n{2,}/).map((para) => {
    const inked = para.split(/(?<=[.!?])\s+/).map((s) => {
      const layer = whoSpeaks(s);
      const safe = esc(s).replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      return layer ? `<span class="src-${layer}">${safe}</span>` : safe;
    }).join(" ");
    return `<p>${inked.replace(/\n/g, "<br>")}</p>`;
  }).join("");
  answerEl.innerHTML = streaming ? html + '<span class="cursor"></span>' : html;
}

function renderSlips(hits) {
  slipsEl.innerHTML = "";
  hits.forEach((h, i) => {
    const el = document.createElement("div");
    el.className = "slip";
    el.dataset.layer = h.source_type;
    el.style.animationDelay = `${i * 90}ms`;
    el.innerHTML =
      `<span class="who">${esc(LAYER_NAME[h.source_type] || h.source_type)}` +
      ` · ${esc(horizon(h.horizon_years))}</span>` +
      `<span class="ttl">${esc(h.title)}</span>` +
      `<div class="full" hidden>${esc(h.text)}` +
      `<span class="cid">${esc(h.chunk_id)}</span></div>`;
    const full = el.querySelector(".full");
    el.querySelector(".ttl").addEventListener("click", () => {
      full.hidden = !full.hidden;
    });
    slipsEl.appendChild(el);
  });
}

// Said in words rather than plotted. "500 hours of testing" next to "15 years
// of watching" is the same comparison the axis was making, and it needs no
// legend to decode.
function horizon(years) {
  if (years == null) return "";
  if (years < 0.1) return `${Math.round(years * 8766)} hours of testing`;
  if (years < 1) return `${Math.round(years * 12)} months of testing`;
  return `${Math.round(years)} years of watching`;
}

/* ---------- ask ----------------------------------------------------------- */

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  go.disabled = true;
  body.dataset.asked = "1";
  errorEl.hidden = true;
  metaEl.hidden = true;
  answerEl.innerHTML = "";
  slipsEl.innerHTML = "";

  // What you wrote goes down into the paper before anything comes back. The
  // pause is not dead time: it is the only moment in the interaction when
  // nothing is being shown, and it is what makes the answer feel drawn out of
  // the book rather than printed onto it.
  await sink(form);
  const asked = document.createElement("p");
  asked.id = "asked-line";
  asked.textContent = question;
  form.parentNode.insertBefore(asked, form);
  input.value = "";
  body.dataset.wet = "0";
  well(asked);

  let acc = "", paras = 0;
  try {
    const res = await fetch(`${API}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k: 8, variant: "arbitrated", rewrite: true }),
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);

    // Hand-rolled SSE over fetch rather than EventSource, which cannot POST --
    // and the question does not belong in a URL.
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) !== -1) {
        const raw = buf.slice(0, i);
        buf = buf.slice(i + 2);
        const ev = /^event: (.+)$/m.exec(raw);
        const dt = /^data: (.+)$/m.exec(raw);
        if (!ev || !dt) continue;
        const d = JSON.parse(dt[1]);

        if (ev[1] === "rewrite") {
          rewriteEl.innerHTML = d.used && d.rewritten !== d.original
            ? `<span class="was">${esc(d.original)}</span> ${esc(d.rewritten)}`
            : esc(d.original);
          rewriteEl.hidden = false;
          well(rewriteEl);
        } else if (ev[1] === "hits") {
          renderSlips(d.hits);
        } else if (ev[1] === "token") {
          acc += d.text;
          renderAnswer(acc, true);
          const n = acc.split(/\n{2,}/).length;
          if (n > paras) {
            paras = n;
            const ps = answerEl.querySelectorAll("p");
            well(ps[ps.length - 2]);
          }
        } else if (ev[1] === "done") {
          renderAnswer(acc, false);
          const ps = answerEl.querySelectorAll("p");
          well(ps[ps.length - 1]);
          metaEl.textContent =
            `${d.retrieval_ms} ms to find · ${d.generate_ms} ms to write · ` +
            `$${d.cost_usd.toFixed(4)}`;
          metaEl.hidden = false;
        } else if (ev[1] === "error") {
          throw new Error(d.message);
        }
      }
    }
  } catch (err) {
    errorEl.textContent = `The page stayed blank: ${err.message}`;
    errorEl.hidden = false;
  } finally {
    go.disabled = false;
    input.focus();
  }
});
