"""Streamlit interface.

The design follows from what the evaluation found. Retrieval hit@5 on
realistically-phrased questions is 0.59 -- good enough to be useful, nowhere
near good enough to be trusted silently. So the interface is built to be
checked rather than believed:

- the rewritten query is shown, because it is where a wrong answer usually
  starts, and a user who sees "pot life" appear can tell instantly whether the
  system understood them;
- retrieved passages are shown in full, grouped by layer, so a claim can be
  traced to the sentence it came from;
- the layer legend states what each source *cannot* know, not just what it is.

Feedback is a single click, stored against the query id. Asking for a rating
before the answer is read would measure patience; asking with a form would
mean almost no one answers.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Streamlit puts the *script's* directory on sys.path, not the project root, so
# `import rag.…` fails unless the process happens to have been started from
# here. Fixed at the entry point rather than by requiring a particular working
# directory, because "run it from the right folder" is exactly the instruction
# that works for the author and fails for everyone else.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic
import streamlit as st

# Order matters and is load-bearing. `app.secrets` populates os.environ from
# Streamlit Cloud's secrets store; `rag.env` fills the gaps from `.env` locally
# without overriding anything already set. Both must run before any module that
# reads configuration.
import app.secrets  # noqa: F401  -- bridges st.secrets into os.environ
import rag.env  # noqa: F401  -- loads .env on import

from app.cache import cache_fields, hit_record, hit_to_dict, lookup
from app.db import estimate_cost, init_schema, log_feedback, log_query
from rag.generate import LAYER_DESCRIPTIONS, PROMPTS, stream
from rag.index import connect as qdrant_connect
from rag.rewrite import Rewrite, rewrite, search_rewritten
from rag.search import (
    CHUNK_TYPES,
    COLLECTION_MATERIALS,
    FILTER_YEAR_MAX,
    FILTER_YEAR_MIN,
    LITERATURE_DOMAINS,
    MANUFACTURER_CATEGORIES,
    PRODUCTION_METHOD,
    PRODUCTION_REWRITE_METHOD,
    SOURCE_TYPES,
    SearchFilters,
    search,
)
from rag.route import ModelRoute, route_question

st.set_page_config(page_title="MATTER — material evidence", layout="wide")

EXAMPLES = [
    "will my cast resin sculpture go yellow if I put it outdoors",
    "how long can I work with the silicone before it starts to set",
    "can I cast concrete thin enough for a wall panel that won't crack",
    "cast aluminium or stainless steel for a piece by the sea",
    "why do conservators avoid certain adhesives on artworks",
]

LAYER_LABEL = {
    "manufacturer_datasheet": "Manufacturer",
    "materials_science": "Materials science",
    "conservation_literature": "Conservation",
    "collection_precedent": "Collection precedent",
}


@st.cache_resource
def bootstrap():
    """One Qdrant client, one warmed embedder, one schema check per process.

    `cache_resource` rather than `cache_data`: these are connections, not
    values, and Streamlit re-runs this module top to bottom on every widget
    interaction. Without it, every keystroke would open a new client.

    Two things happen here that only matter once this is deployed:

    The embedding model is *warmed*. fastembed downloads and initialises the
    130 MB ONNX model the first time it embeds anything, which on a cold
    container is 20-40 seconds. Paid at boot, that is a slow deploy; paid on
    first query, it is a user who thinks the app is broken.

    The schema check is allowed to fail. A managed Postgres can be asleep, or
    briefly unreachable, and losing the query log is not a reason to refuse to
    answer questions. The failure is returned rather than swallowed, so the
    page can say monitoring is degraded instead of quietly recording nothing.
    """
    db_error = None
    try:
        init_schema()
    except Exception as exc:
        db_error = str(exc)

    qdrant = qdrant_connect()
    warm(qdrant)
    return qdrant, anthropic.Anthropic(), db_error


def warm(qdrant) -> None:
    """Run one throwaway hybrid search to load both embedding models.

    Deliberately a real search rather than a direct fastembed call: qdrant-client
    caches its embedders internally, so warming through the query path loads the
    same instances the first user query will use, instead of a second copy that
    doubles the memory and saves nothing. It also proves the collection exists
    at boot rather than at first question.
    """
    search(qdrant, "warm", method=PRODUCTION_METHOD, limit=1)


def retrieve(question: str, k: int, use_rewrite: bool, filters: SearchFilters):
    qdrant, client, _ = bootstrap()
    started = time.perf_counter()
    if use_rewrite:
        rw = rewrite(question, client)
        hits = search_rewritten(
            qdrant, rw, limit=k, rerank=False, filters=filters
        )
    else:
        rw = Rewrite(original=question, rewritten=question, used=False)
        hits = search(
            qdrant, question, method=PRODUCTION_METHOD, limit=k, filters=filters
        )
    return rw, hits, int((time.perf_counter() - started) * 1000)


def finish_query(
    question: str,
    variant: str,
    k: int,
    use_rewrite: bool,
    rw,
    hits,
    answer,
    retrieval_ms: int,
    generate_ms: int,
    filters: SearchFilters,
    decision: ModelRoute,
) -> dict:
    """Log a completed query. Split out from retrieval so the UI can stream the
    answer first and record it afterwards, without the log losing anything."""

    source_counts: dict[str, int] = {}
    for hit in hits:
        source_counts[hit.source_type] = source_counts.get(hit.source_type, 0) + 1

    record = {
        "question": question,
        "rewritten": rw.rewritten,
        "rewrite_used": rw.used,
        "variant": variant,
        "method": PRODUCTION_REWRITE_METHOD if use_rewrite else PRODUCTION_METHOD,
        "answer": answer.text,
        "source_counts": source_counts,
        "chunk_ids": [h.chunk_id for h in hits],
        "rewrite_terms": rw.terms,
        "hits": [hit_to_dict(h) for h in hits],
        "n_hits": len(hits),
        "retrieval_ms": retrieval_ms,
        "generate_ms": generate_ms,
        "total_ms": retrieval_ms + generate_ms,
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
        "cost_usd": estimate_cost(
            answer.model, answer.input_tokens, answer.output_tokens
        ),
        "truncated": answer.truncated,
        "error": None,
        **cache_fields(question, variant, k, use_rewrite, filters),
        "filters": filters.as_dict(),
        "generate_model": answer.model,
        "route_tier": decision.tier,
        "route_reason": decision.reason,
    }

    # An answer the user can already read must not be thrown away because the
    # log is unavailable. `query_id = None` disables the feedback buttons for
    # this answer -- feedback keyed to a row that was never written would be
    # worse than no feedback.
    try:
        query_id = log_query(record)
    except Exception as exc:
        query_id = None
        st.warning(f"Answer not logged — monitoring is degraded ({exc}).")

    return {
        "query_id": query_id,
        "answer": answer,
        "hits": hits,
        "rewrite": rw,
        "retrieval_ms": retrieval_ms,
        "generate_ms": generate_ms,
        "cache_hit": False,
        "route": decision,
    }


def finish_cached_query(cached, total_ms: int) -> dict:
    """Give a cache hit its own log id so its feedback remains independent."""
    try:
        query_id = log_query(hit_record(cached, total_ms))
    except Exception as exc:
        query_id = None
        st.warning(f"Cache hit not logged — monitoring is degraded ({exc}).")
    return {
        "query_id": query_id,
        "answer": cached.answer,
        "hits": cached.hits,
        "rewrite": cached.rewrite,
        "retrieval_ms": 0,
        "generate_ms": 0,
        "total_ms": total_ms,
        "cache_hit": True,
        "route": cached.route,
    }


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.header("What this is")
    st.markdown(
        "A question-answering system over four kinds of source that **disagree "
        "with each other on purpose**. A datasheet's *UV resistant* is 500 "
        "hours of accelerated testing; a conservation study's *it yellowed* is "
        "a real object after ten winters. Both are true. The disagreement is "
        "usually the answer."
    )

    st.subheader("Settings")
    variant = st.selectbox(
        "Answer style",
        [v for v in PROMPTS if v != "no_context"],
        index=list(PROMPTS).index("arbitrated"),
        help="`arbitrated` surfaces source disagreement; `plain` is a standard "
        "RAG prompt, kept so the difference is visible.",
    )
    k = st.slider("Passages retrieved", 4, 16, 8)
    use_rewrite = st.toggle(
        "Query rewriting",
        value=True,
        help="Translates studio language into document vocabulary "
        "('before it sets' → 'pot life'). Measured +0.03 hit@5 overall, "
        "+0.06 on manufacturer datasheets.",
    )

    with st.expander("Filter evidence"):
        selected_sources = st.multiselect(
            "Source layers",
            SOURCE_TYPES,
            format_func=lambda value: LAYER_LABEL.get(value, value),
            help="Empty means all four layers.",
        )
        selected_chunks = st.multiselect(
            "Passage types",
            CHUNK_TYPES,
            format_func=lambda value: value.replace("_", " ").title(),
            help="Empty means all passage types.",
        )
        selected_categories = st.multiselect(
            "Manufacturer categories",
            MANUFACTURER_CATEGORIES,
            format_func=lambda value: value.replace("-", " ").title(),
        )
        selected_domains = st.multiselect(
            "Literature domains",
            LITERATURE_DOMAINS,
            format_func=lambda value: value.replace("_", " ").title(),
        )
        selected_materials = st.multiselect(
            "Collection materials",
            COLLECTION_MATERIALS,
            format_func=str.title,
        )
        limit_year = st.toggle(
            "Limit publication year",
            help="Only literature carries publication years; enabling this "
            "excludes datasheets and collection precedents.",
        )
        year_range = st.slider(
            "Publication years",
            FILTER_YEAR_MIN,
            FILTER_YEAR_MAX,
            (FILTER_YEAR_MIN, FILTER_YEAR_MAX),
            disabled=not limit_year,
        )

    filters = SearchFilters(
        source_types=tuple(selected_sources),
        chunk_types=tuple(selected_chunks),
        categories=tuple(selected_categories),
        domains=tuple(selected_domains),
        materials=tuple(selected_materials),
        year_from=year_range[0] if limit_year else None,
        year_to=year_range[1] if limit_year else None,
    )

    st.subheader("Source layers")
    for layer, label in LAYER_LABEL.items():
        st.caption(f"**{label}** — {LAYER_DESCRIPTIONS[layer].split('—', 1)[1].strip()}")

    _, _, db_error = bootstrap()
    if db_error:
        st.divider()
        st.warning(
            "Query logging is unavailable, so answers here are not being "
            "recorded and feedback is disabled. Everything else works.",
            icon="⚠️",
        )
        st.caption(f"`{db_error[:180]}`")

# --- main -------------------------------------------------------------------

st.title("Materials for making")
st.caption(
    "Mould-making and casting. 4,115 passages from manufacturer datasheets, "
    "conservation and materials-science literature, and the Tate collection."
)

if "question" not in st.session_state:
    st.session_state.question = ""

cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example[:34] + "…", use_container_width=True, help=example):
        st.session_state.question = example

# `key=`, not `value=`. With `value=st.session_state.question` and no key, the
# widget is rebuilt from that variable on every rerun -- and a rerun happens on
# any interaction at all, including the ⌘+Enter that the text area itself
# advertises. Typed text that had not yet been submitted was silently wiped:
# nudge the slider mid-question and the question is gone.
#
# It survived local testing because clicking Ask sends the widget's current
# value in the same message that triggers the rerun, so the one path that was
# exercised by hand happened to be the one path that worked. It showed up on
# the deployed app, where latency leaves room for a rerun to land in between.
#
# `key="question"` binds the widget to session_state instead: Streamlit owns
# the value, reruns preserve it, and the example buttons above still work
# because they assign to the same key before this line runs.
question = st.text_area(
    "Your question",
    key="question",
    placeholder="Describe what you are trying to make, avoid, or fix.",
    height=88,
)

if st.button("Ask", type="primary") and question.strip():
    st.session_state.result = None
    try:
        cache_started = time.perf_counter()
        try:
            cached = lookup(question.strip(), variant, k, use_rewrite, filters)
        except Exception:
            cached = None

        if cached is not None:
            cache_ms = int((time.perf_counter() - cache_started) * 1000)
            st.session_state.result = finish_cached_query(cached, cache_ms)
            st.rerun()

        with st.spinner("Searching…"):
            rw, hits, retrieval_ms = retrieve(
                question.strip(), k, use_rewrite, filters
            )

        # Shown before the answer streams, not after. The rewrite is the first
        # place a query goes wrong, and a user who sees it while waiting can
        # abandon a bad search instead of reading a paragraph built on it.
        if rw.used and rw.rewritten != rw.original:
            st.info(f"**Searched as:** {rw.rewritten}", icon="🔎")

        _, client, _ = bootstrap()
        decision = route_question(question.strip(), hits, variant)
        gen_started = time.perf_counter()
        answer_holder = {}

        def token_stream():
            for piece in stream(
                question.strip(), hits, variant, client, model=decision.model
            ):
                if isinstance(piece, str):
                    yield piece
                else:
                    answer_holder["answer"] = piece

        st.write_stream(token_stream())
        generate_ms = int((time.perf_counter() - gen_started) * 1000)

        st.session_state.result = finish_query(
            question.strip(), variant, k, use_rewrite, rw,
            question.strip(), variant, k, use_rewrite, rw,
            hits, answer_holder["answer"], retrieval_ms, generate_ms,
            filters, decision,
        )
        # Re-run so the page renders from session state alone. Without this the
        # streamed text is drawn by this branch, and every later interaction --
        # a feedback click, a slider nudge -- re-runs the script, takes the
        # other path, and the answer vanishes.
        st.rerun()
    except Exception as exc:  # surfaced, not swallowed
        st.session_state.result = None
        st.error(f"Query failed: {exc}")

result = st.session_state.get("result")
if result:
    answer, hits, rw = result["answer"], result["hits"], result["rewrite"]

    if rw.used and rw.rewritten != rw.original:
        st.info(f"**Searched as:** {rw.rewritten}", icon="🔎")
    if answer.truncated:
        st.warning("This answer hit the token limit and may be cut off.")

    st.markdown(answer.text)

    st.caption(
        (f"{result['total_ms']} ms exact cache · $0.0000"
         if result.get("cache_hit") else
         f"{result['retrieval_ms']} ms retrieval · {result['generate_ms']} ms "
         f"generation · {answer.input_tokens:,} in / {answer.output_tokens:,} out "
         f"· ${estimate_cost(answer.model, answer.input_tokens, answer.output_tokens):.4f}")
    )
    decision = result.get("route")
    if decision:
        st.caption(
            f"Model route: {decision.tier} · `{answer.model}` · {decision.reason}"
        )

    if result["query_id"] is not None:
        left, right, _ = st.columns([1, 1, 6])
        if left.button("👍 Useful"):
            log_feedback(result["query_id"], 1)
            st.toast("Thanks — recorded.")
        if right.button("👎 Not useful"):
            log_feedback(result["query_id"], -1)
            st.toast("Thanks — recorded.")

    st.divider()
    st.subheader("Passages this answer was built from")

    by_layer: dict[str, list] = {}
    for hit in hits:
        by_layer.setdefault(hit.source_type, []).append(hit)

    for layer, group in by_layer.items():
        st.markdown(f"**{LAYER_LABEL.get(layer, layer)}** ({len(group)})")
        for hit in group:
            with st.expander(f"{hit.title[:88]}  ·  score {hit.score:.3f}"):
                st.write(hit.text)
                st.caption(f"`{hit.chunk_id}`")
                if hit.url:
                    st.caption(hit.url)
