"""HTTP API over the same retrieval and generation code the Streamlit app uses.

Why this exists as a second front door rather than a rewrite:

The Streamlit app is the graded reference implementation and it stays exactly as
it is. But Streamlit owns its own layout, typography, and rerun model, and the
thing worth building next -- an interface where the *kind of evidence* behind a
claim is visible rather than described -- needs control over every pixel.
Fighting a framework for that is how you end up with neither a clean reference
implementation nor a good interface.

So: `rag/` is untouched and shared. This module adds transport, nothing else.
Anything that looks like a decision about retrieval or prompting belongs in
`rag/`, not here -- if the two front ends ever answer the same question
differently, the comparison the whole evaluation rests on is worthless.

Streaming is Server-Sent Events rather than WebSockets. The traffic is one
direction, the payload is text, and SSE survives a proxy that has never heard
of this application. `EventSource` is four lines in the browser.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import rag.env  # noqa: F401  -- loads .env on import

from app.cache import cache_fields, hit_record, hit_to_dict, lookup
from app.db import estimate_cost, init_schema, log_feedback, log_query
from rag.generate import MODEL, PROMPTS, stream
from rag.index import connect as qdrant_connect
from rag.rewrite import Rewrite, rewrite, search_rewritten
from rag.search import search

# What kind of evidence each layer rests on. Categorical, and every value is
# true by the definition of the layer rather than estimated.
#
# This replaced `EVIDENCE_HORIZON_YEARS`, four numbers -- 0.06, 0.5, 15.0, 60.0
# -- that said how long each layer had "been watching". They were mine. Nothing
# measured them, and the corpus cannot support them: real durations do appear in
# the text (708 mentions across the datasheets, 474 across materials science)
# but a datasheet's "4 hours" is demould time, not observation time, so pulling
# them out needs a model that reads the context, not a regular expression.
#
# An interface driven by four invented numbers would have been fluent,
# plausible and impossible to check -- which is the exact failure this project
# spent its evaluation measuring. Better to say the thing that is defensible:
# accelerated testing is not the same evidence as an examined object, and that
# distinction needs no number to be true.
EVIDENCE_KIND = {
    "manufacturer_datasheet": "accelerated testing",
    "materials_science": "controlled specimens",
    "conservation_literature": "examined objects",
    "collection_precedent": "held in a collection",
}

app = FastAPI(title="MATTER", version="0.1")

# The browser front end is served from a different origin (Cloudflare Pages in
# production, a file server in development), so CORS is not optional. Origins
# are read from the environment rather than set to "*": this API spends money
# per request, and an open one is an open invoice.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_state: dict = {}


@app.on_event("startup")
def startup() -> None:
    """Connect, warm the embedder, and tolerate a sleeping query log.

    The warm-up runs one real hybrid search rather than calling fastembed
    directly, for the same reason the Streamlit app does: qdrant-client caches
    its own embedder instances, so warming through the query path loads the
    ones the first request will actually use instead of a second copy.
    """
    _state["qdrant"] = qdrant_connect()
    search(_state["qdrant"], "warm", method="hybrid", limit=1)
    try:
        init_schema()
        _state["db_error"] = None
    except Exception as exc:  # logging is not worth refusing to answer over
        _state["db_error"] = str(exc)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    variant: str = "arbitrated"
    k: int = Field(default=8, ge=1, le=16)
    rewrite: bool = True


class FeedbackRequest(BaseModel):
    query_id: int
    rating: int = Field(ge=-1, le=1)
    comment: str | None = Field(default=None, max_length=2000)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def hit_json(hit, rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id": hit.chunk_id,
        "source_type": hit.source_type,
        "chunk_type": hit.chunk_type,
        "title": hit.title,
        "text": hit.text,
        "url": hit.url,
        "score": hit.score,
        "evidence_kind": EVIDENCE_KIND.get(hit.source_type),
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "model": MODEL,
        "variants": [v for v in PROMPTS if v != "no_context"],
        "db": "down" if _state.get("db_error") else "up",
        "evidence_kinds": EVIDENCE_KIND,
    }


@app.post("/api/ask")
async def ask(req: AskRequest):
    if req.variant not in PROMPTS:
        raise HTTPException(400, f"unknown variant {req.variant!r}")

    async def events():
        loop = asyncio.get_running_loop()
        question = req.question.strip()

        cache_started = time.perf_counter()
        try:
            cached = await loop.run_in_executor(
                None, lambda: lookup(question, req.variant, req.k, req.rewrite)
            )
        except Exception:
            # The cache is an optimisation backed by the optional query log.
            # A sleeping Postgres must not turn a working RAG request into 503.
            cached = None

        if cached is not None:
            cache_ms = int((time.perf_counter() - cache_started) * 1000)
            yield sse("rewrite", {
                "original": cached.rewrite.original,
                "rewritten": cached.rewrite.rewritten,
                "used": cached.rewrite.used,
                "terms": cached.rewrite.terms,
                "retrieval_ms": 0,
            })
            yield sse("hits", {
                "hits": [hit_json(hit, i) for i, hit in enumerate(cached.hits)]
            })
            yield sse("token", {"text": cached.answer.text})

            query_id = None
            try:
                query_id = await loop.run_in_executor(
                    None, log_query, hit_record(cached, cache_ms)
                )
            except Exception:
                query_id = None
            yield sse("done", {
                "query_id": query_id,
                "retrieval_ms": 0,
                "generate_ms": 0,
                "total_ms": cache_ms,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0,
                "truncated": False,
                "source_counts": cached.source_counts,
                "cache_hit": True,
                "cache_source_query_id": cached.source_query_id,
            })
            return

        qdrant = _state.get("qdrant")
        if qdrant is None:
            yield sse("error", {"message": "not ready"})
            return

        started = time.perf_counter()
        if req.rewrite:
            rw = await loop.run_in_executor(None, rewrite, question)
            hits = await loop.run_in_executor(
                None, lambda: search_rewritten(qdrant, rw, limit=req.k, rerank=False)
            )
        else:
            rw = Rewrite(original=question, rewritten=question, used=False)
            hits = await loop.run_in_executor(
                None, lambda: search(qdrant, question, method="hybrid", limit=req.k)
            )
        retrieval_ms = int((time.perf_counter() - started) * 1000)

        # The rewrite goes out before a single answer token. It is where a wrong
        # answer usually starts, and a reader who sees "pot life" appear knows
        # instantly whether they were understood -- while the model is still
        # thinking, not after they have read a paragraph built on a misreading.
        yield sse("rewrite", {
            "original": rw.original, "rewritten": rw.rewritten,
            "used": rw.used, "terms": rw.terms, "retrieval_ms": retrieval_ms,
        })
        yield sse("hits", {"hits": [hit_json(h, i) for i, h in enumerate(hits)]})

        # `stream()` is a blocking generator, so it runs on a worker thread and
        # hands pieces back through a queue. Iterating it directly here would
        # block the event loop and stall every other connection for the length
        # of one answer.
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def pump():
            try:
                for piece in stream(question, hits, req.variant):
                    loop.call_soon_threadsafe(queue.put_nowait, piece)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

        gen_started = time.perf_counter()
        loop.run_in_executor(None, pump)

        answer = None
        while True:
            piece = await queue.get()
            if piece is SENTINEL:
                break
            if isinstance(piece, Exception):
                yield sse("error", {"message": str(piece)})
                return
            if isinstance(piece, str):
                yield sse("token", {"text": piece})
            else:
                answer = piece
        generate_ms = int((time.perf_counter() - gen_started) * 1000)

        if answer is None:
            yield sse("error", {"message": "generation produced no answer"})
            return

        source_counts: dict[str, int] = {}
        for hit in hits:
            source_counts[hit.source_type] = source_counts.get(hit.source_type, 0) + 1

        cost = estimate_cost(MODEL, answer.input_tokens, answer.output_tokens)
        query_id = None
        try:
            query_id = await loop.run_in_executor(None, log_query, {
                "question": question,
                "rewritten": rw.rewritten,
                "rewrite_used": rw.used,
                "variant": req.variant,
                "method": "rewrite_hybrid" if req.rewrite else "hybrid",
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
                "cost_usd": cost,
                "truncated": answer.truncated,
                "error": None,
                **cache_fields(question, req.variant, req.k, req.rewrite),
            })
        except Exception:
            # Same rule as the Streamlit app: an answer the reader can already
            # see must not be discarded because the log is unavailable. A null
            # query_id disables feedback for this answer rather than keying it
            # to a row that was never written.
            query_id = None

        yield sse("done", {
            "query_id": query_id,
            "retrieval_ms": retrieval_ms,
            "generate_ms": generate_ms,
            "input_tokens": answer.input_tokens,
            "output_tokens": answer.output_tokens,
            "cost_usd": round(cost, 6),
            "truncated": answer.truncated,
            "source_counts": source_counts,
            "cache_hit": False,
        })

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # nginx and friends buffer by default, which turns a stream into one
            # long pause followed by everything at once.
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> dict:
    if req.rating == 0:
        raise HTTPException(400, "rating must be -1 or 1")
    try:
        log_feedback(req.query_id, req.rating, req.comment)
    except Exception as exc:
        raise HTTPException(503, f"feedback not recorded: {exc}")
    return {"ok": True}
