# MATTER — build status

Working notes for a two-day build. Delete before submission; the graded
document is `README.md`.

## The problem (draft for README — review this first)

A sculptor decides to cast a two-metre translucent piece for a courtyard. That
single decision needs answers from four incompatible places:

- **Which resin?** The manufacturer's datasheet gives Shore hardness, pot life
  and mixed viscosity — precise, quantitative, and silent about year five.
- **Will it yellow?** Polymer science has measured this, in journals written
  for chemists, indexed under vocabulary no artist uses.
- **Will it survive outdoors?** Conservation literature knows, because it has
  examined objects that already failed.
- **Can I even buy it?** Supplier reality: pack sizes, shelf life, whether the
  product still exists.

No single source spans these. Worse, they disagree in a way that matters. A
datasheet saying "UV resistant" is reporting 500 hours of accelerated
weathering. A conservation paper saying the same polymer yellows badly is
reporting a real object after ten winters. **Both are true.** The artist needs
to see the disagreement, not a confident average of it.

Asking an LLM directly fails in a specific, demonstrable way: it produces
fluent, plausible, unciteable material advice — invented product codes,
half-remembered cure times, and no distinction between what a vendor claims
and what a conservator observed. That failure mode is reproduced as a control
experiment in `eval/` rather than merely asserted.

### Why RAG, specifically

The answers exist in text, are scattered across sources with different
authority, and change over time (products are discontinued; ageing studies are
published). That is the shape of problem retrieval augmentation is for. It is
*not* a table lookup — "which resin yellows slowest" has no column to sort.

## Scope

Locked to **mould-making and casting**: silicones, urethanes, epoxies,
mineral composites, and the adhesives/coatings around them. Chosen because it
is where the corpus is densest and where the motivating questions live.

Out of scope (stated so the boundary is deliberate, not accidental): fabric,
paint, digital media, metal foundry process.

## Corpus

| Layer | Source | Status | Notes |
|---|---|---|---|
| Manufacturer datasheet | Smooth-On, 14 categories | ✅ 324 products | 264 with spec tables, 314 technical-bulletin PDFs, 313 SDS PDFs, median 14 specs/product |
| Conservation literature | OpenAlex → Heritage Science, Studies in Conservation, J. Cultural Heritage | ✅ 70 works | Journal whitelist, high precision (98% clean when measured). Not screened. |
| Materials science | OpenAlex → OA + has_abstract + field filter, 5 pages/query | ✅ 1,971 works | Broad recall, ~40–60% noise by inspection. Screened by the LLM gate. |
| Precedent | Tate collection open data (CC0) | ✅ 15 aggregates | Reframed — see below |

### Sources ruled out, and why

Recorded because "why isn't X in here" is the first question a reviewer asks.

- **CAMEO (MFA Boston)** — the most authoritative art-materials encyclopedia
  available. Entire site sits behind a Cloudflare managed challenge; even
  `robots.txt` requires solving it. Excluded: working around bot detection is
  off-limits, and an ingestion step a grader cannot re-run is worthless.
- **AIC Conservation Wiki** — `robots.txt` declares `ai-train=no`,
  `use=reference`, and reserves rights under Article 4 of the EU DSM
  Directive. `ai-input` (which explicitly covers RAG) is left unstated, i.e.
  neither granted nor refused. Excluded as a licensing grey zone.
- **Jesmonite** — `robots.txt` blocks CCBot, Bytespider, Applebot-Extended and
  ClaudeBot by name, with `ai-train=no`. Excluded. This is a real coverage
  gap: AC100-vs-AC730 questions cannot be answered from primary sources and
  must fall back to third-party literature.
- **INCCA artist interviews** — HTTP 403. Excluded. The artist-intent layer is
  consequently thin, surviving only where Tate Papers quotes artists directly.

## Findings that shaped the design

1. **Smooth-On's datasheet PDFs are inside HTML comments.** The visible link is
   JS-rendered; a DOM query returns nothing. Regex over raw markup recovers
   them (verified live, `200 application/pdf`). Missing this would have
   silently dropped the entire safety layer.
2. **Elsevier does not deposit abstracts to OpenAlex.** 48 of 50 sampled
   records from Polymer Degradation and Stability et al. had no abstract at
   all — a 4% usable rate. Switching from a journal whitelist to
   `has_abstract:true, open_access.is_oa:true` raised it to ~98% by routing to
   MDPI/Springer/Frontiers/PLOS.
3. **Keyword filtering alone cannot clean this corpus.** Dropping the journal
   whitelist admits cross-discipline collisions: "resin yellowing" returns
   dental composites and dye removal in sugar processing; "adhesive" returns
   sea-urchin biology. A field filter halves it but does not fix it. Hence the
   LLM relevance gate at ingestion — and hence a deliberately imperfect corpus,
   which is what makes the retrieval evaluation and the reranker meaningful
   rather than decorative.
4. **~12% of each product page is boilerplate** (company blurb, address, phone)
   appearing in >50% of documents. Removed by corpus-level document-frequency
   filtering rather than a hardcoded blocklist, so it survives template changes.
5. **OpenAlex now meters the free tier by daily credits**, not just burst rate.
   Anonymous callers get $0.10/day = 1,000 requests at `$0.0001` each. When it
   is spent, the 429 carries `Retry-After: 33787` — nine and a half hours.
   Honouring that literally put the collector to sleep for most of a day while
   appearing to run, which is the worst kind of failure: silent and
   indistinguishable from work.

   Three fixes: cap backoff at 120s and raise `QuotaExhausted` beyond it; read
   `x-ratelimit-remaining` and stop with headroom left; persist partial
   harvests so a resumed run accumulates rather than overwrites.

   Registering a **free** API key lifts the allowance 10× to $1.00/day (10,000
   requests); prepaid balance stacks on top. The key travels in the
   `Authorization` header, not the documented `api_key` query parameter — a key
   in a query string leaks into access logs, proxy records, and this project's
   own on-disk cache filenames.

   Worth stating plainly for the write-up: a full production harvest is **50
   requests, about half a cent**. The quota was exhausted by iterative
   development — probing, diagnostics, and re-runs — not by the pipeline. With
   the key, pagination went from 1 page to 5 per query, taking a single query
   from ~50 usable works to ~220.
6. **The Tate case-study layer does not exist as conceived.** Artwork pages
   carry curatorial summaries, but render them client-side — the static HTML
   holds 675 characters of sidebar. The CSV's `medium` field is all that is
   reachable offline, and a bare material phrase is an inventory entry, not
   knowledge. Keyword matching over it is also badly noisy: of 1,604 raw hits,
   most are mounts and frames ("photograph on aluminium", "Perspex frame").
   Reframed as an aggregated **precedent** layer instead — 15 dense passages
   ("fibreglass: 30 works, 17 of them in the 1960s") rather than 1,604 sparse
   ones. The claim it supports is narrow and honest: what was actually built
   and collected, which no datasheet or conservation paper reports.

## Chunking: 4,115 chunks (`ingestion/chunk.py`)

| chunk_type | count | source |
|---|---:|---|
| narrative | 2,109 | Smooth-On prose, ~900-char windows on paragraph bounds |
| abstract | 1,727 | literature, whole unless >2,200 chars |
| spec | 264 | Smooth-On spec tables, one per product |
| precedent | 15 | Tate |

Median 880 chars, p90 1,932.

**Datasheets are split by chunk type, not just by length.** A spec table and its
narrative answer different questions by different mechanisms: "Shore 30A" is an
exact-match lookup, "will it survive outdoors" is a semantic one. Indexed
together, 30 terse label/value pairs get diluted by 5 kB of prose and BM25 stops
finding them. Every chunk also opens with a context line naming its product or
paper — a retrieved chunk reading "cure time is 4 hours" is unusable without it.

### Boilerplate: the threshold is in empty space, not tuned

Document frequency over the scraped corpus, not a hardcoded blocklist (a
blocklist rots the first time the site edits its footer):

```
 73%  cats=14  Technical Bulletin | SDS | Certifications      site chrome
 61%  cats=14  You may never have heard of us...              marketing
 59%  cats=14  5600 Lower Macungie Road...                    address
 53%  cats=13  Because no two applications are quite...       legal disclaimer
────────────────────────────────────── cut at 30%, in the gap ──────
 29%  cats= 4  Colorants for Urethane Rubber, Resin and Foam  cross-sell
 24%  cats= 4  Graduated Container Style Mixing Cups          cross-sell
 22%  cats= 3  IMPORTANT: Shelf life of product is reduced... REAL SAFETY TEXT
```

Lowering the threshold to catch the cross-sell teasers would have deleted the
shelf-life warning sitting 2 points below them. The teasers are not
distinguished by frequency but by not being sentences — all ≤51 chars, while the
shortest genuine technical paragraph is 104. They are removed on that axis
instead, and only when they also repeat, so a one-line note unique to one
product survives. Drops 10.6% of manufacturer text.

### Three data-quality findings, only one of which was bad data

OpenAlex stores abstracts as an inverted index (token → positions), not as text,
and reconstruction is lossy. Chunk sizes exposed three separate failures that a
single "drop the weird records" filter would have conflated:

1. **Missing separators, not missing sentences.** 93 of 1,521 abstracts (6%,
   364 occurrences, 2.5% of all boundaries) contain `capabilities.The` — a real
   boundary whose space was lost. A splitter requiring whitespace under-splits
   these silently. Both forms are now matched, which recovered 7 of the 11
   records first flagged as unreadable.
2. **Publisher markup.** 3 abstracts carry MathML, 11 inline HTML, 45 entities.
   One 2.7 kB `<mml:math>` block reads as a single unbroken sentence — and it
   sits inside the Paraloid B44/B72/Incralac permeability paper, which is
   directly on topic for "which adhesives do conservators avoid". Stripped
   before splitting.
3. **Genuine corruption: 2 records.** Inverted indexes with gaps, reconstructing
   to tens of kB of word salad ("For the corrosion-resistant was to occur only
   by localized corrosion that was by a pit growth rate that with"). Not
   recoverable by parsing; dropped.

The degeneracy test applies only to abstracts long enough to need splitting.
Testing all of them rejected a readable Arabic conservation paper — Arabic has
no uppercase for the boundary rule to key on, and at 1,551 chars it was never
going to be split anyway. **A splitting heuristic must not become a language
filter by accident.**

## Retrieval design

The five layers are **parallel corpora, not pipeline stages**. Each chunk
carries a `source_type`, retrieval fans out across all of them, and fusion is
source-aware. This is what lets an answer surface the vendor/conservator
disagreement instead of letting one layer overwrite the other.

```
query
  → rewrite (artistic intent → material vocabulary; "透明感" → cast acrylic,
     water-clear urethane, refractive index, light transmission)
  → parallel retrieval: BM25 (exact: "Shore 30A", "PB29", product names)
                      + dense (semantic: "won't crack in winter")
  → reciprocal rank fusion → cross-encoder rerank, source_type weighted
  → LLM synthesis, every claim tagged with its layer
```

## Retrieval evaluation: 200 question pairs, 5 methods

Ground truth is **two questions per chunk under opposite instructions** rather
than one (`eval/ground_truth.py`). The standard recipe — show a chunk to an LLM,
ask what it answers, check retrieval finds it — measures whether the system can
find its own words. Compliance audited: 199/200 `artist` questions borrow no
proper name, 196/200 no code or number.

| | literal hit@5 | artist hit@5 | artist MRR | p50 ms |
|---|---:|---:|---:|---:|
| dense | 0.935 | 0.445 | 0.353 | 17 |
| sparse (BM25) | **0.990** | 0.550 | 0.425 | **2** |
| hybrid (RRF) | **0.990** | 0.560 | **0.459** | 20 |
| hybrid + rerank | 0.980 | 0.545 | 0.438 | 3879 |
| **rewrite + hybrid** | 0.980 | **0.590** | 0.451 | 30 (+~700 LLM) |

### 1. The headline is the gap, not the best row

Literal → artist costs **0.43–0.49 hit@5**. Reporting only the standard-recipe
number would have claimed 0.99 for a system that delivers 0.56 to the users it
was built for. Query rewriting produces the narrowest gap of any method
(+0.390), which is the argument for it — not the +0.03 on the headline.

### 2. A written-down prediction that turned out wrong

The design note in `ground_truth.py` predicted BM25 would win `literal` and
dense would win `artist`. **BM25 won both.** The audit removed proper nouns and
numerals but deliberately kept domain nouns — you cannot ask about bronze patina
without "patina" — and those stay lexically matchable. Meanwhile `bge-small` is
a general-purpose encoder with little conservation or casting vocabulary in
training. On this corpus a 1970s ranking function beats neural embedding, at
1/8th the latency. Kept in the write-up because it was predicted in advance and
falsified.

### 3. Reranking: 194× the cost, no aggregate gain — but strongly two-sided

| layer (artist hit@5) | hybrid | +rerank | median chunk |
|---|---:|---:|---:|
| collection_precedent | 0.533 | **0.867** | 183 tok |
| manufacturer_datasheet | 0.284 | 0.284 | 188 tok |
| conservation_literature | 0.933 | 0.900 | 347 tok |
| materials_science | 0.817 | **0.700** | 372 tok |

The gains and losses cancel to nothing overall. The two layers that lost ground
are the two longest, which is suggestive — but they still fit inside the model's
512-token window, so truncation does not explain it. A likelier account is that
`ms-marco-MiniLM` was trained on short web passages and scientific abstracts are
out of distribution for it. **Stated as a hypothesis: not tested.** Not shipped
as the default.

### 4. The manufacturer layer's 0.284 is mostly a metric artefact

Verified rather than assumed. For *"When mixing up a batch of platinum silicone,
what should I keep in mind about glove material and storage temperature"*, gold
was `dragon-skin-15`; retrieval returned **Dragon Skin 20** at rank 3 — a sibling
product carrying the identical latex-inhibition warning. Roughly 100 platinum
silicone datasheets repeat it verbatim.

Document-level hit@5 is also low (0.28), so this is not window splitting: the
layer is genuinely near-duplicate. Chunk-level ground truth demands *the* correct
datasheet where the user needs *a* correct one, so **real usability is higher
than 0.56**. The fix is not a better retriever; it is deduplicating by product
family, which is left undone and named here rather than hidden.

### 5. Rewriting helps where it was designed to and hurts where it was not

Biggest gain is exactly the vocabulary-gap layer: manufacturer 0.284 → **0.347**.
But `literal` MRR *drops* 0.922 → 0.871 — when a question already speaks document
vocabulary, appending synonyms dilutes the exact match that was already winning.
The rewrite is fused with the original query rather than replacing it, so the
damage is bounded; it is not free.

## Remaining plan

- [x] Finish OpenAlex collection + LLM relevance gate
- [x] Tate collection open data (reframed: precedent layer)
- [x] Clean + chunk (boilerplate DF filter; specs vs narrative split)
- [x] Qdrant index (dense + BM25 sparse, 4,115 points)
- [x] Ground-truth question set + retrieval eval (5 methods)
- [x] Ingestion = automated Python scripts (dlt not needed for full marks)
- [x] RAG chain + 3 prompt variants + LLM-as-judge
- [x] Streamlit UI (streaming) + feedback capture
- [x] Grafana dashboard (7 panels, all SQL verified)
- [x] docker-compose, pinned versions, README, Makefile
- [x] Cloud deploy — live at artmat-rag-ngpg5rtoak6ppesfbrhe68.streamlit.app
      Verified end to end from the public URL: retrieval from Qdrant Cloud
      (eu-west-2, 4,115 points, green), generation, and a row in Neon.

7. **Prompt caching silently did nothing.** The gate's screening brief is
   identical across ~2000 requests, so a `cache_control` marker on it looked
   like free money. It isn't: Haiku 4.5's minimum cacheable prefix is 4096
   tokens and the brief measures 354. Below the threshold the marker is
   ignored with no error — two identical calls returned
   `cache_creation_input_tokens: 0` and `cache_read_input_tokens: 0`.

   The marker was removed rather than left as decoration, because a cost model
   built on a cache that never engages is wrong by ~30% and nothing in the
   system ever says so. Measured cost for the gate: **$1.13** (1.48 MTok in,
   0.16 MTok out, Haiku 4.5 at batch pricing) — established with
   `messages.count_tokens` on a real sample, not a chars/4 rule of thumb.

## Gate scope: screen the noisy layer only

The gate runs on `materials_science` and not on `conservation_literature`.

The whitelist layer measured 98% clean (2 rejects in 89). Screening it costs
real money to change almost nothing. The broad-recall layer is where the noise
actually is — a random sample of 18 turned up prosthetic dentistry, essential
oils in a Siberian forestry journal, Spanish-language megalithic architecture
history, and 1955 anion-exchange-resin analytical chemistry.

Unscreened records still pass through to the output file with
`relevance.screened: false`, so the corpus stays complete and the decision
stays auditable.

## Relevance gate: validated

Batch of 89 abstracts, 89 succeeded / 0 errored, ~4.5 minutes end to end. The
Batches + structured-outputs pipeline is proven and gets reused for both
evaluation stages.

Only 2 of 89 were rejected — expected, since this batch came from the
high-precision journal whitelist. What matters is *which* two, because both are
exactly the failure mode keyword filtering cannot reach: **right journal, right
keywords, wrong subject.**

- *"Bio-based polyurethane with photolabile o-nitrobenzyl molecules for
  degradation"* — Polymer Degradation and Stability. On-topic journal, on-topic
  keywords. Rejected because it is about engineering polymers **to** degrade on
  demand for medical and recycling use. An artist needs the opposite: materials
  that do not degrade unintentionally. Same vocabulary, inverted intent.
- *"TOPO-Loss for continuity-preserving crack detection using deep learning"* —
  Construction and Building Materials. Rejected as a computer-vision paper
  about earthquake damage assessment, with no bearing on material behaviour.

Use these two as the worked example in the README for why the gate exists.
Domain split across the 89: conservation_practice 36, adhesives_coatings 29,
cementitious 8, polymers_resins 5, metals 5, pigments_surfaces 4, off_topic 2.

## Model ladder

Three LLM jobs, three tiers, matched to task difficulty rather than picked once
and reused everywhere.

| Job | Model | Why this tier |
|---|---|---|
| Relevance gate | `claude-haiku-4-5` | ~1000 binary classifications. Cheapest current model is sufficient for a judgement this narrow. |
| Generation | `claude-sonnet-5` | The RAG answer — the thing under evaluation. |
| LLM judge | `claude-opus-5` | Deliberately **stronger than the generator**. A judge at or below the generator's level shares its blind spots, and scoring noise then swamps the prompt-variant differences the evaluation exists to measure. |

Embeddings run locally via fastembed (zero API cost) — the retrieval evaluation
issues hundreds of queries, and paying per embedding would dominate the budget
for no quality gain at this corpus size.

Two API features do real work here rather than being decoration:

- **Batches API** — every evaluation run (3 retrieval strategies × the
  ground-truth set; 3 prompt variants × judge) is offline batch work with no
  latency requirement. Half price.
- **Structured outputs** (`output_config.format`) — the gate's verdict and the
  judge's score come back schema-validated. Regex-parsing model prose fails on
  a few calls in a thousand, and a silent failure inside a corpus filter or a
  scoring loop is invisible until the numbers look inexplicable.

## Needed from you

`ANTHROPIC_API_KEY` in `artmat-rag/.env` (no credentials on this machine yet —
`ANTHROPIC_API_KEY` unset, `ant` CLI not installed). Estimated total for the
whole build including evaluation: **~$7** at batch pricing.

Pinned versions so far: `anthropic==0.121.0`, `httpx==0.28.1`,
`selectolax==0.3.27`, `tenacity==9.1.2`, Python 3.13.
