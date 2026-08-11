# Handover — `web/` and `api/`

You are taking over the interface. This is what is running, what the contract
is, what is deliberately constrained, and what is broken.

## What is running right now

| | | |
|---|---|---|
| API | `http://127.0.0.1:8021` | `.venv/bin/uvicorn api.main:app --port 8021` |
| Page | `http://localhost:5173` | `python3 -m http.server 5173 -d web` |

Both are already up. The page server is registered as `artmat-web` in
`.claude/launch.json`. `.env` currently points at the **local** Qdrant and
Postgres; the cloud values are in the same file, commented out.

## What is yours and what is not

```
rag/        retrieval, rewriting, prompts     DO NOT TOUCH from the UI side
app/        Streamlit, the graded build       DO NOT TOUCH
api/        transport only                    change carefully, see below
web/        the interface                     yours
```

The rule that matters: **nothing about retrieval or prompting may live in
`api/`.** Both front ends call the same `rag/` functions, so if you change how
retrieval works for the pretty one, the retrieval evaluation and the LLM-judge
results in the README stop describing what ships. That is the one thing that
would actually damage the project.

## The contract

`POST /api/ask` → `{question, k, variant, rewrite}` → SSE stream:

| event | payload |
|---|---|
| `rewrite` | `{original, rewritten, used, terms, retrieval_ms}` |
| `hits` | `{hits: [{rank, chunk_id, source_type, chunk_type, title, text, url, score, evidence_kind}]}` |
| `token` | `{text}` — one fragment, append it |
| `done` | `{query_id, retrieval_ms, generate_ms, input_tokens, output_tokens, cost_usd, truncated, source_counts}` |
| `error` | `{message}` |

`GET /api/health` → includes `evidence_kinds`.
`POST /api/feedback` → `{query_id, rating: -1|1, comment?}` — **exists, unused
by the page.** Wiring it up is the highest-value small job left, because the
Grafana dashboard has a satisfaction panel with nothing feeding it.

`evidence_kind` is one of four strings: `accelerated testing`, `controlled
specimens`, `examined objects`, `held in a collection`. It is categorical on
purpose — see below.

## Three constraints that are not style opinions

**1. Text never goes on glass.** Body copy lives on `--panel` (solid). Gloss,
chrome, translucency and bevels are allowed on the frame and nowhere near a
paragraph. This is the failure mode of the whole Y2K register and the reason
the headline gradient had to be darkened once already — the canonical chrome
stops assume a dark background and the top two thirds of every letter vanished
on this pale page.

**2. SVG filters never go on a streaming element.** `#craze` and `#craze-deep`
re-rasterise on every repaint. They are on `.spec` chips, which are static once
rendered. Do not put them on `#answer` while tokens are arriving.

**3. There are no invented numbers.** `api/main.py` used to export
`EVIDENCE_HORIZON_YEARS` — 0.06, 0.5, 15.0, 60.0, "how long each layer has been
watching". Nothing measured them. The corpus cannot support them either: real
durations are in the text (708 mentions across datasheets, 474 across materials
science) but a datasheet's "4 hours" is demould time, not observation time, so
extraction needs a model, not a regex. If you want a time axis, a slider, or
rings, **the extraction pass comes first** — otherwise the interface is making
a fluent, plausible, uncheckable claim, which is the exact thing this project
spends its evaluation section measuring against.

## Known bug — fix this first

**A conservation paragraph is being coloured as manufacturer.** Reproduce by
asking "will my cast resin sculpture go yellow outdoors" and inspecting:

```
span.src-manufacturer_datasheet, 218 chars:
"What conservation studies of real, aged objects say. Two Studies in
 Conservation papers looked ... worse than a manufacturer's caution suggests."
```

Two sentences merged into one span, and the second one's "manufacturer's" won.

Root cause is in `renderAnswer()` in `app.js`: the split runs on raw text that
still contains markdown, so `say.** Two` does not split — the character before
the whitespace is `*`, not `.`.

```js
// now
para.split(/(?<=[.!?])\s+/)
// fix
para.split(/(?<=[.!?][*"')\]]{0,2})\s+/)
```

This matters more than a colour glitch. The tinting is a **claim about
provenance**, and it is currently asserting a wrong one on screen. If you would
rather not carry that risk at all, dropping sentence tinting entirely is a
legitimate choice — the chunk ids on the specimen chips are the provenance of
record and they are always correct.

## Design decisions you are free to throw away, and the reasons behind them

Keep the reasons even if you drop the design; they cost two failed attempts.

- **A logarithmic axis of coloured dots** was the first version. Accurate,
  unreadable: a legend, then a log scale, then a hover affordance, all to be
  learned before the picture said anything. *An artwork may be mysterious; a
  control may not.*
- **A book that opened** was the second. Two independent scroll regions inside
  a locked body broke the most practiced gesture on the web; a fixed 16:10
  spread is a container, and containers fight variable-length text; and every
  question paid ~3 s of ceremony. *Ceremony is charged once; friction is
  charged every time.*
- **The current version** is one column, `body` scrolls, ask bar is sticky, no
  opening animation. Those three properties are the fixes. Change the skin
  freely; think hard before giving any of those three up.
- **Why Y2K is not a filter here:** the signature material of that moment,
  translucent coloured ABS, is the one everyone has personally watched turn
  yellow and go brittle. Encoding age in parchment asks the reader to know
  something; encoding it in this asks them to remember an old console. It only
  works if the yellowing actually happens on screen.

## Not done

- feedback buttons (`/api/feedback` is live and waiting)
- mobile has not been looked at below 30rem
- the backend is not deployed anywhere. The page can go on Cloudflare Pages as
  static files; the API needs a host that can hold ~600 MB RSS for fastembed —
  Hugging Face Spaces (Docker, free, no card) is the cheapest route, and the
  root `Dockerfile` already bakes both embedding models in
- `ALLOWED_ORIGINS` must include the Pages origin before any of that works; it
  defaults to localhost only, deliberately, because this API spends money per
  request
