/* The client.
 *
 * Third version of this front end, and the first two failures are why it is
 * this shape:
 *
 *   1. A logarithmic time axis of coloured motes. Accurate, unreadable -- a
 *      legend, then a log scale, then a hover affordance, all to be learned
 *      before the picture said anything. An artwork may be mysterious; a
 *      control may not.
 *   2. A book that opened, absorbed ink and welled up answers. It was a
 *      container, and containers fight variable-length text: two independent
 *      scroll regions, a fixed box that squeezed long answers, and ~3 s of
 *      ceremony charged on every question.
 *
 * So there is no stage, no canvas and no opening here. The page scrolls, the
 * ask bar sticks, and the only visual argument is carried by CSS from a field
 * that actually exists in the data.
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
// model's phrasing rather than guessing. It is a heuristic and it is allowed to
// miss -- an untinted sentence looks like ordinary text, which is the right
// failure. The chunk ids on the specimens stay the provenance of record.
const ATTRIBUTION = [
  [/manufactur|data ?sheet|smooth-?on|the maker|vendor/i, "manufacturer_datasheet"],
  [/conservation|conservator|restorer/i, "conservation_literature"],
  [/collection|tate|precedent|holdings/i, "collection_precedent"],
  [/study|studies|research|peer-reviewed|paper|literature|experiment/i, "materials_science"],
];

const form = document.getElementById("ask");
const input = document.getElementById("q");
const go = document.getElementById("go");
const rewriteEl = document.getElementById("rewrite");
const answerEl = document.getElementById("answer");
const metaEl = document.getElementById("meta");
const errorEl = document.getElementById("error");
const specimensEl = document.getElementById("specimens");
const listEl = document.getElementById("specimen-list");

document.querySelectorAll("#examples button").forEach((b) => {
  b.addEventListener("click", () => {
    input.value = b.textContent.trim();
    input.focus();
  });
});

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function whoSpeaks(sentence) {
  for (const [re, layer] of ATTRIBUTION) if (re.test(sentence)) return layer;
  return null;
}

// Markdown is not parsed. The model emits paragraphs and the occasional bold
// run; a full parser would be a lot of surface area for two features, and a way
// to get raw HTML from a model onto the page. Escape first, then put back only
// the two patterns that are safe.
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

function renderSpecimens(hits) {
  listEl.innerHTML = "";
  for (const h of hits) {
    const el = document.createElement("div");
    el.className = "spec";
    el.dataset.layer = h.source_type;
    el.innerHTML =
      `<span class="who">${esc(LAYER_NAME[h.source_type] || h.source_type)}` +
      ` · ${esc(h.evidence_kind || "")}</span>` +
      `<span class="ttl">${esc(h.title)}</span>` +
      `<div class="full" hidden>${esc(h.text)}` +
      `<span class="cid">${esc(h.chunk_id)}</span></div>`;
    const full = el.querySelector(".full");
    el.querySelector(".ttl").addEventListener("click", () => {
      full.hidden = !full.hidden;
    });
    listEl.appendChild(el);
  }
  specimensEl.hidden = false;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  go.disabled = true;
  document.body.dataset.asked = "1";
  errorEl.hidden = true;
  metaEl.hidden = true;
  specimensEl.hidden = true;
  answerEl.hidden = true;
  answerEl.innerHTML = "";
  listEl.innerHTML = "";

  let acc = "";
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
          // Out before a single answer token: the rewrite is where a wrong
          // answer usually starts, and seeing it while you wait means you can
          // abandon a bad search instead of reading a paragraph built on one.
          rewriteEl.innerHTML = d.used && d.rewritten !== d.original
            ? `<span class="was">${esc(d.original)}</span> ${esc(d.rewritten)}`
            : esc(d.original);
          rewriteEl.hidden = false;
        } else if (ev[1] === "hits") {
          renderSpecimens(d.hits);
        } else if (ev[1] === "token") {
          acc += d.text;
          answerEl.hidden = false;
          renderAnswer(acc, true);
        } else if (ev[1] === "done") {
          renderAnswer(acc, false);
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
    errorEl.textContent = `That did not work: ${err.message}`;
    errorEl.hidden = false;
  } finally {
    go.disabled = false;
  }
});
