# MATTER — material evidence RAG

**A question-answering system over four kinds of source that disagree with each other on purpose.**

A sculptor decides to cast a two-metre translucent piece for a courtyard. That
one decision needs answers from four incompatible places:

| | knows | cannot know |
|---|---|---|
| **Manufacturer datasheet** | Shore hardness, pot life, mixed viscosity — precise, quantitative | what the piece looks like in year five |
| **Materials science** | degradation mechanisms, measured under controlled ageing | how a real object behaves outdoors |
| **Conservation literature** | what happened to objects that already failed | anything about a product released last year |
| **Collection precedent** | what was actually built and kept | why, or how it aged |

No single source spans these, and **they contradict each other in a way that
matters**. A datasheet saying "UV resistant" is reporting 500 hours of
accelerated weathering. A conservation paper saying the same polymer yellows
badly is reporting a real object after ten winters. Both are true. They are
measuring different things.

An artist needs to see that disagreement, not a confident average of it. The
system's central design choice follows: the four corpora are **parallel layers,
not pipeline stages**, every chunk carries its `source_type`, and the default
answer prompt is instructed to surface conflict rather than resolve it.

### Why RAG, specifically

The answers exist in text, are scattered across sources with different
authority, and change over time. That is the shape of problem retrieval
augmentation is for — and it is *not* a table lookup, because "which resin
yellows slowest" has no column to sort.

The alternative — asking an LLM directly — fails in a specific way, so the
system measures it rather than asserting it. `no_context` is a fourth
evaluation condition running the same questions with no retrieval at all. See
[Answer quality](#answer-quality).

---

## Quick start

```bash
make setup            # venv + pinned dependencies
cp .env.example .env  # add ANTHROPIC_API_KEY
make up               # Qdrant + Postgres + Grafana
make index            # embed 4,115 chunks (~7 min, CPU-only, no API key needed)
make ui               # http://localhost:8501
```

`make all` rebuilds everything from scratch including scraping and both
evaluations. `make help` lists every stage.

Services: UI on `:8501`, Qdrant dashboard on `:6333/dashboard`, Grafana on
`:3000` (anonymous viewing enabled, so no login is required to read the
dashboard).

---

## Corpus

**1,860 documents → 4,115 chunks.** Built, not downloaded.

| Layer | Source | Documents | Licence / access |
|---|---|---:|---|
| Manufacturer datasheet | Smooth-On, 14 mould/casting categories | 324 | scraped, `robots.txt` permits |
| Materials science | OpenAlex, open-access + abstract + field filter | 1,451 | CC0 metadata |
| Conservation literature | OpenAlex → Heritage Science, Studies in Conservation, J. Cultural Heritage | 70 | CC0 metadata |
| Collection precedent | Tate collection open data | 15 aggregates | CC0-1.0 |

### Sources ruled out, and why

Recorded because "why isn't X in here" is the first question a reviewer asks.

- **CAMEO (MFA Boston)** — the most authoritative art-materials encyclopedia
  there is. The entire site sits behind a Cloudflare managed challenge; even
  `robots.txt` requires solving it. Excluded: working around bot detection is
  off-limits, and an ingestion step a grader cannot re-run is worthless.
- **AIC Conservation Wiki** — `robots.txt` declares `ai-train=no`,
  `use=reference`, and reserves rights under EU DSM Article 4. `ai-input`,
  which explicitly covers RAG, is left unstated — neither granted nor refused.
  Excluded as a licensing grey zone.
- **Jesmonite** — blocks ClaudeBot, CCBot, and Bytespider by name. Excluded.
  This is a real coverage gap: AC100-vs-AC730 questions cannot be answered
  from primary sources.

### The Tate layer is not what it was meant to be

The plan was case studies — how a piece was made and what happened to it. Tate
publishes exactly that in curatorial summaries, but renders them client-side:
the static HTML holds 675 characters of sidebar. Keyword matching the CSV's
`medium` field instead gave 1,604 hits that were mostly mounts and frames
("photograph on aluminium", "Perspex frame").

Rebuilt as an aggregated **precedent** layer: 15 dense passages ("fibreglass:
30 works, 17 of them in the 1960s") rather than 1,604 sparse ones. The claim it
supports is narrow and stated in every passage — what was actually built and
collected, which no datasheet or conservation paper reports, and which is *not*
a statement about material properties.

---

## Ingestion

`make ingest` runs four stages, each writing a file the next one reads, each
independently re-runnable.

**Three findings that changed the design:**

1. **Smooth-On's datasheet PDFs live inside HTML comments.** The visible link
   is JS-rendered, so a DOM query returns nothing. Regex over raw markup
   recovers them (verified live, `200 application/pdf`). Missing this would
   have silently dropped the entire safety-data layer.

2. **Elsevier does not deposit abstracts to OpenAlex.** 48 of 50 sampled
   records from the obvious journals had no abstract — 4% usable. Switching
   from a journal whitelist to `has_abstract:true, open_access.is_oa:true`
   raised it to ~98% by routing to MDPI, Springer, Frontiers, PLOS.

3. **OpenAlex meters its free tier by daily credit, not burst rate.** When it
   is spent, the 429 carries `Retry-After: 33787` — nine and a half hours.
   Honouring that literally put the collector to sleep for most of a day while
   appearing to run, which is the worst kind of failure: silent and
   indistinguishable from work. Fixed by capping backoff at 120 s, reading
   `x-ratelimit-remaining` and stopping with headroom, and persisting partial
   harvests. A full production harvest is **50 requests, about half a cent**;
   the quota was exhausted by development, not by the pipeline.

### An LLM as corpus curator

Dropping the journal whitelist was the only way to get usable abstracts, but it
admits cross-discipline collisions that keyword filters cannot express:
"resin yellowing" returns dental composites, "adhesive" returns sea-urchin
biology. A `topics.field.id` filter halves the noise and cannot fix it, because
the collision is semantic.

So 1,971 abstracts were screened one at a time by Haiku 4.5 (Batches API,
structured outputs) against a question keyword matching cannot ask: *would this
help someone making or conserving a physical object?* **520 rejected (26%).**

Validation — the rejections are the interesting part:

- dental/medical journals: 31 of 38 rejected
- *"Çanakkale public sculpture survey"* — **rejected**. The title says
  sculpture; the study is GIS cataloguing with no material investigation. Any
  keyword filter keeps this.
- *bio-based polyurethane engineered to degrade* — **rejected**. Right journal,
  right keywords, exactly inverted intent.

Noise is deliberately not eliminated. A corpus that is 100% relevant makes
retrieval evaluation meaningless — hit-rate is trivially high when every
document is a plausible answer.

---

## Chunking

| chunk_type | count | shape |
|---|---:|---|
| narrative | 2,109 | datasheet prose, ~900-char windows on paragraph bounds |
| abstract | 1,727 | whole unless >2,200 chars |
| spec | 264 | spec table, one per product |
| precedent | 15 | Tate |

**Datasheets are split by kind, not just by length.** A spec table and its
narrative answer different questions by different mechanisms: "Shore 30A" is an
exact-match lookup, "will it survive outdoors" is a semantic one. Indexed
together, 30 terse label/value pairs get diluted by 5 kB of prose and BM25
stops finding them.

Every chunk opens with a context line naming its product or paper. A retrieved
chunk reading "cure time is 4 hours" is unusable without it.

### Boilerplate removal: the threshold sits in empty space

Corpus document frequency, not a hardcoded blocklist — a blocklist rots the
first time the site edits its footer.

```
 73%  Technical Bulletin | SDS | Certifications      site chrome
 61%  You may never have heard of us...              marketing
 59%  5600 Lower Macungie Road...                    address
 53%  Because no two applications are quite...       legal disclaimer
──────────────────────── cut at 30%, in the gap ────────────────
 29%  Colorants for Urethane Rubber, Resin and Foam  cross-sell
 24%  Graduated Container Style Mixing Cups          cross-sell
 22%  IMPORTANT: Shelf life of product is reduced... REAL SAFETY TEXT
```

Lowering the threshold to catch the cross-sell teasers would have deleted the
shelf-life warning two points below them. The teasers are not distinguished by
frequency but by **not being sentences** — all ≤51 chars, while the shortest
genuine technical paragraph is 104. They are removed on that axis, and only
when they also repeat.

### Three data-quality findings, only one of which was bad data

OpenAlex stores abstracts as an inverted index, not text, and reconstruction is
lossy. Oversized chunks exposed three separate failures that a single "drop the
weird records" filter would have conflated:

1. **Missing separators, not missing sentences.** 93 of 1,521 abstracts (6%)
   contain `capabilities.The` — a real boundary whose space was lost. Matching
   both forms recovered 7 of the 11 records first flagged as unreadable.
2. **Publisher markup.** A 2.7 kB `<mml:math>` block reads as one unbroken
   sentence — inside the Paraloid B44/B72/Incralac permeability paper, which is
   directly on topic for "which adhesives do conservators avoid". Stripped, not
   dropped.
3. **Genuine corruption: 2 records.** Inverted indexes with gaps reconstructing
   to word salad. Not recoverable by parsing; dropped.

The degeneracy test applies only to abstracts long enough to need splitting.
Testing all of them rejected a readable Arabic conservation paper — Arabic has
no uppercase for the boundary rule to key on. **A splitting heuristic must not
become a language filter by accident.**

---

## Retrieval

One Qdrant collection, dense and sparse vectors as named vectors on the same
point, fused server-side by Reciprocal Rank Fusion. Both models run locally
through fastembed, so re-indexing needs no API key.

- dense: `BAAI/bge-small-en-v1.5` (384-d)
- sparse: `Qdrant/bm25` with the IDF modifier — real BM25, so the evaluation
  has an honest lexical baseline rather than neural-vs-neural
- rerank: `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder over the top 50

RRF is computed by Qdrant rather than in Python: the usual mistake is fusing
*scores*, which cannot be compared across a cosine similarity and a BM25 sum.

### Query rewriting

Ask *"how long can I work with the silicone before it starts to set"* and the
answer sits in a spec chunk under **Pot Life** — a phrase the question does not
contain and the asker does not know. That vocabulary gap is the problem domain,
not an accident:

| asked | corpus calls it |
|---|---|
| "before it starts to set" | pot life, gel time |
| "go yellow after a few years" | photo-oxidative degradation, yellowing index, UV stabiliser |

Haiku translates the question into document vocabulary, and retrieval runs over
the original **and** the rewrite, fused. Substitution would be a regression: the
rewrite is a guess, and when it is wrong the original query is the only thing
between the user and a confidently irrelevant answer.

---

## Retrieval evaluation

**200 question pairs, 5 methods.** The ground-truth design is the part worth
reading.

The standard recipe — show a chunk to an LLM, ask what it answers, check
retrieval finds it — inflates BM25, because the model writes questions using
the passage's own words. Rather than suppress that, it became the experiment:
every chunk gets **two questions under opposite instructions**.

- `literal` — how someone who half-remembers the document searches. Product
  names and numbers allowed.
- `artist` — how a person in a studio asks, with the passage's distinctive
  vocabulary forbidden.

Compliance audited programmatically: **199/200** artist questions borrow no
proper name, **196/200** no code or number.

| | literal hit@5 | artist hit@5 | artist MRR | p50 ms |
|---|---:|---:|---:|---:|
| dense | 0.935 | 0.445 | 0.353 | 17 |
| sparse (BM25) | **0.990** | 0.550 | 0.425 | **2** |
| hybrid (RRF) | **0.990** | 0.560 | **0.459** | 20 |
| hybrid + rerank | 0.980 | 0.545 | 0.438 | 3879 |
| **rewrite + hybrid** | 0.980 | **0.590** | 0.451 | 30 |

### The headline is the gap, not the best row

Literal → artist costs **0.43–0.49 hit@5**. Reporting only the standard-recipe
number would have claimed 0.99 for a system that delivers 0.56 to the people it
was built for. Query rewriting produces the narrowest gap of any method
(+0.390) — that, not the +0.03 on the headline, is the argument for it.

### A prediction written down in advance, and falsified

The design note predicted BM25 would win `literal` and dense would win
`artist`. **BM25 won both.** The audit removed proper nouns and numerals but
deliberately kept domain nouns — you cannot ask about bronze patina without
"patina" — and those stay lexically matchable, while `bge-small` has little
conservation vocabulary in training. On this corpus a 1970s ranking function
beats neural embedding at 1/8th the latency.

### Reranking costs 194× and buys nothing overall

| artist hit@5 by layer | hybrid | +rerank | median chunk |
|---|---:|---:|---:|
| collection_precedent | 0.533 | **0.867** | 183 tok |
| manufacturer_datasheet | 0.284 | 0.284 | 188 tok |
| conservation_literature | 0.933 | 0.900 | 347 tok |
| materials_science | 0.817 | **0.700** | 372 tok |

Gains and losses cancel. The two layers that lost ground are the two longest,
which is suggestive — but they still fit inside the model's 512-token window,
so truncation does not explain it. A likelier account is that `ms-marco-MiniLM`
was trained on short web passages and scientific abstracts are out of
distribution. **Stated as a hypothesis; not tested. Not shipped as default.**

### The manufacturer layer's 0.284 is mostly a metric artefact

Verified, not assumed. For *"When mixing up a batch of platinum silicone, what
should I keep in mind about glove material and storage temperature"*, gold was
`dragon-skin-15`; retrieval returned **Dragon Skin 20** at rank 3 — a sibling
product carrying the identical latex-inhibition warning, which ~100 platinum
silicone datasheets repeat verbatim.

Document-level hit@5 is also low (0.28), so this is not window splitting: the
layer is genuinely near-duplicate. Chunk-level ground truth demands *the*
correct datasheet where a user needs *a* correct one, so **real usability is
higher than 0.59 suggests**. The fix is deduplication by product family, which
is not done — named here rather than hidden.

---

## Answer quality

Three prompt variants plus a no-retrieval control, scored by **Opus 5** while
Sonnet 5 generates — a model grading its own output rates its own habits
highly. The judge never sees which variant produced an answer.

- `plain` — standard RAG prompt. The control: if the elaborate variants do not
  beat it, they are decoration.
- `sourced` — every claim attributed to its source kind.
- `arbitrated` — sourced, plus instructed to surface disagreement rather than
  average it. The project's thesis as a prompt.
- `no_context` — same question, no retrieval.

**Citation validity is checked in Python, not by the model.** Whether a cited
`chunk_id` exists and was actually shown is a string lookup with a right
answer; asking a model would add noise to the one number that has none. It
distinguishes *valid*, *stale* (real but not retrieved), and *fabricated*.

`no_context` answers are judged against the passages they never saw. That is
the measurement, not a handicap: the claim under test is that an unaided model
produces fluent material advice the literature does not support.

**25 questions × 4 conditions, 1–5 scale:**

| variant | grounded | source discrim. | handles conflict | usable | unsupported claims/answer | citations/answer |
|---|---:|---:|---:|---:|---:|---:|
| plain | 4.68 | 3.72 | 4.04 | **4.28** | **0.76** | 0.44 |
| sourced | 4.68 | 4.28 | 4.08 | **4.28** | 0.88 | 0.96 |
| arbitrated | **4.84** | **4.72** | **4.44** | 4.12 | 1.08 | **1.80** |
| `no_context` | 1.75 | 1.12 | 1.50 | 2.83 | **9.21** | 0.00 |

**Fabricated citations: 0 across all 100 answers.** Stale citations: 0.

### The control is the result

An unaided Sonnet 5 averages **9.21 unsupported claims per answer** against
0.76–1.08 with retrieval. The failure is not vagueness — it is confident
specificity. The worst single answer invented **17** claims:

> Pot life: 45 minutes to 3+ hours · Viscosity: usually 20,000–40,000 cps ·
> Shrinkage: <0.1% · *Mold Star 30 — 45 min pot life, 6 hr demould* ·
> *Sorta-Clear 40 — ~30 min pot life* · *OOMOO 30 (20–30 min pot life)*

The judge's note: the passages actually documented **Mold Star 15 SLOW (50 min
pot life, 4 hr cure)** and **Smooth-Sil 960**, and the answer used none of
them. Several product names are real; the numbers attached to them are not.
This is precisely the failure mode the problem statement claimed — now a number
rather than an assertion.

### The variants trade against each other, and the elaborate one is not free

`arbitrated` wins where it was designed to: **source discrimination +1.00 over
plain**, conflict handling +0.40, groundedness +0.16, and four times as many
citations.

It **loses on usability** (4.12 vs 4.28) and carries **more unsupported claims**
(1.08 vs 0.76). Both follow from the same cause: it writes longer answers that
weigh sources against each other, which is more surface area to be wrong on and
more qualification between the reader and an action.

That is a real trade, not a clean win. `arbitrated` ships as the default
because for this domain provenance is the point — but a user who wants an
answer rather than an assessment is better served by `plain`, so all three are
selectable in the UI and logged per-variant to Grafana.

---

## Interface

Streamlit on `:8501`. Built to be checked rather than believed, because hit@5
of 0.59 is useful and not trustworthy:

- the **rewritten query is shown before the answer streams**, so a user who
  sees the system misunderstand them can stop reading
- every retrieved passage is shown in full, grouped by layer, with its
  `chunk_id`
- the sidebar legend states what each layer *cannot* know

**Latency, measured and fixed.** The first working version took 40 seconds
before anything appeared. Streaming brought time-to-first-token to 18.3 s;
disabling extended thinking brought it to **3.3 s**, and shortened answers from
~2,700 to ~1,600 tokens.

That leaves a discrepancy worth stating plainly: **the offline judge scores
were measured with thinking enabled, and the UI ships with it disabled.** It is
a flag, not a silent default, and the difference is untested.

---

## Monitoring

Postgres captures every query with enough context to answer more than "is it
up"; Grafana reads it directly. **7 panels:**

1. Questions asked over time
2. Which layer is answering — passages served by `source_type`
3. Latency, retrieval vs generation split (20 ms vs tens of seconds; one line
   would hide the only tunable component)
4. Feedback — useful %, **with the response rate beside it**, because a
   satisfaction figure without a denominator is not a measurement
5. Cost per day, priced at call time and stored rather than recomputed
6. Answer style: usage and rating per variant
7. Recent questions beside their rewrites — the fastest way to see the system
   misunderstand someone

Feedback is one click, stored `+1`/`-1` against the query id, `LEFT JOIN`ed so
unrated queries stay in the denominator.

### Exact answer cache

An identical question with identical user-visible settings reuses the newest
complete answer from Postgres before query rewriting, retrieval, or generation
runs. A hit therefore has zero model cost and still returns the original
passages and chunk ids. It also writes a new query row with `cache_hit = true`
and `cache_source_query_id` pointing to the paid source row, so its latency and
feedback belong to this request rather than silently changing the old one.
One negative rating on the source or any reuse removes that answer from future
cache lookups, so a bad answer is not amplified merely because it was first.

The key includes the normalised question, answer variant, passage count,
rewrite setting, model names, and hashes of both prompts. Corpus changes cannot
be inferred safely, so `EXACT_CACHE_NAMESPACE` is the explicit invalidation
switch and must be bumped after reindexing. Set `EXACT_CACHE_ENABLED=false` for
evaluation runs or debugging. Pre-cache historical rows are not reused because
they do not record enough configuration to prove that they are equivalent.

---

## Reproducibility

- Every dependency pinned to an exact version in `requirements.txt`; every
  container image pinned to a patch version verified to exist on Docker Hub.
- `make all` rebuilds from empty. Scrapers cache to disk, so re-runs cost the
  source sites nothing.
- The ground-truth sample uses a **fixed seed**: an evaluation whose question
  set silently changes cannot compare two configurations, which is its only
  purpose.
- Batch results are cached (`data/answers.json`, `data/rewrites.json`) so a
  re-run does not re-pay for generation, and so the two rerank conditions see
  identical rewrites.
- Secrets live in `.env` (gitignored). `.env.example` lists what is needed.

**Cost of a full rebuild: roughly $7.** Embedding and indexing are free and
local. The spend is the relevance gate ($1.13, measured), ground-truth
generation, and the Opus judging pass.

---

## A second front end

`app/` is the graded reference implementation and does not change. `api/` and
`web/` are a second front door onto the same `rag/` package — a FastAPI server
streaming over Server-Sent Events, and a hand-written page with no libraries.

It took three attempts, and the two failures are the useful part.

A **logarithmic axis of coloured dots**, one per retrieved passage, positioned
by how long its layer had been watching. Accurate and unreadable: a legend to
learn, a log scale to understand, a hover affordance to discover — three things
before the picture said anything. An artwork may be mysterious; a control may
not.

A **book that opened**, absorbed ink and welled up answers. It looked right and
was uncomfortable, structurally: two independent scroll regions inside a locked
body broke the most practiced gesture on the web, a fixed spread is a container
and containers fight variable-length text, and every question paid ~3 s of
ceremony. Ceremony is charged once; friction is charged every time.

**The interface nearly asserted four numbers nobody measured.** The third idea
was a slider — drag your work's age forward and watch sources fall silent as
their evidence expires. It would have run on `EVIDENCE_HORIZON_YEARS`, four
constants written by hand. Real durations do exist in the corpus (708 mentions
across the datasheets, 474 across materials science) but a datasheet's "4
hours" is demould time, not observation time, so extracting them needs a model
rather than a regex. Shipping the slider would have made the interface fluent,
plausible and impossible to check — the precise failure this project's
evaluation section exists to measure. The constants were deleted and replaced
with `EVIDENCE_KIND`, four categorical strings that are true by definition of
the layer: *accelerated testing*, *controlled specimens*, *examined objects*,
*held in a collection*.

What ships is one scrolling column with a sticky ask bar and no opening
animation, and a single visual encoding that depends only on `source_type` — a
field that is actually in the data. Each layer is the same translucent plastic
at a different age: the manufacturer's is fresh out of the box, conservation's
is visibly yellowed and crazed, the collection layer has gone amber and nearly
opaque. The signature material of Y2K is the one everybody has personally
watched turn yellow, so the encoding needs no legend — which is exactly what
the first version did need.

Handover notes, including a known provenance-tinting bug, are in
[web/HANDOVER.md](web/HANDOVER.md).

```bash
pip install -r requirements.txt -r requirements-api.txt
uvicorn api.main:app --port 8021      # API
python -m http.server 5173 -d web     # page
```

Nothing about retrieval or prompting lives in `api/` — if the two front ends
ever answered the same question differently, the evaluation both rest on would
be worthless.

---

## Deployment

**Live: <https://artmat-rag-ngpg5rtoak6ppesfbrhe68.streamlit.app>**
(Streamlit Community Cloud, Qdrant Cloud `eu-west-2`, Neon Postgres.)

Runs in two places from one codebase: against the `docker-compose.yml` stack
locally, and against managed equivalents in the cloud — Streamlit Community
Cloud for the UI, Qdrant Cloud for the vectors, Neon for the query log. There is
no production branch and no environment flag. The difference is four environment
variables, which means the deployed configuration is one you can run on your own
machine and actually test.

Two design notes, both in [DEPLOY.md](DEPLOY.md) with the full procedure:

- **Grafana is deliberately not hosted.** Publishing an anonymous-viewer
  dashboard over the query log would publish every question anyone typed. It
  runs locally, pointed at the production database by one variable, and reads
  the same rows.
- **Configuration is read at call time, not import time.** `app/db.py` used to
  bind the DSN into function defaults at import; locally that is invisible,
  because `.env` loads first. On Streamlit Cloud, where secrets arrive through
  `st.secrets` rather than the environment, whichever module imported first
  would freeze the localhost default and the deployed app would log nowhere,
  quietly. `app/secrets.py` bridges the two, and one file — not the whole rag
  package — is the one that knows Streamlit exists.

Logging is allowed to fail without taking the answer with it: a managed Postgres
can be asleep, and losing a log row is not a reason to refuse a question. When
it happens the page says monitoring is degraded and disables the feedback
buttons, rather than recording feedback against a row that was never written.

`Dockerfile` builds the same app as a container for anywhere that takes one,
with both embedding models baked in so there is no cold-start download.

---

## Known limitations

Stated because a system that reports only its wins has not been evaluated.

1. **Manufacturer datasheets are near-duplicate.** ~100 platinum silicones
   repeat the same warnings verbatim. Deduplication by product family is not
   implemented; chunk-level retrieval metrics understate real usability.
2. **One vendor.** Smooth-On only. Jesmonite blocks crawlers, so a question the
   project was designed around cannot be answered from primary sources.
3. **Reranking is not shipped**, and the reason it underperforms on long
   scientific abstracts is a hypothesis, not a finding.
4. **The judge scores describe a configuration the UI does not run** (thinking
   enabled vs disabled).
5. **The artist question set is LLM-written.** Compliance was audited
   mechanically, but it is still a model's idea of how a fabricator talks.
   Questions from actual artists would be better evidence.
6. **English only.** The corpus is overwhelmingly English, and the sentence
   splitter is Latin-centric by construction.
