"""HTTP API over the same retrieval and generation code the Streamlit app uses.

Why this exists as a second front door rather than a rewrite:

The Streamlit app is the graded reference implementation and it stays exactly as
it is. But Streamlit owns its own layout, typography, and rerun model, and the
thing worth building next -- an interface where the *evidence horizon* of each
source is visible rather than described -- needs a canvas, an animation frame,
and control over every pixel. Fighting a framework for that is how you end up
with neither a clean reference implementation nor a good interface.

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

from app.db import estimate_cost, init_schema, log_feedback, log_query
from rag.generate import MODEL, PROMPTS, stream
from rag.index import connect as qdrant_connect
from rag.rewrite import Rewrite, rewrite, search_rewritten
from rag.search import search

# How long each layer has actually been watching. This is the project's thesis
# reduced to four numbers, and it is an editorial claim rather than a measured
# field -- so it is stated here, in one place, where the interface can render it
# and a reader can disagree with it.
#
# A datasheet's durability claim rests on a few hundred hours in a weathering
# cabinet. A conservation report rests on an object someone examined after it
# had been outside for a decade. Both are honest; they are not the same
# evidence, and an interface that lays them out at the same distance is
# flattening the one distinction that matters most.
EVIDENCE_HORIZON_YEARS = {
    "manufacturer_datasheet": 0.06,   # ~500 h accelerated weathering
    "materials_science": 0.5,         # controlled ageing, weeks to months
    "conservation_literature": 15.0,  # examined objects, a decade or three
    "collection_precedent": 60.0,     # what a collection has actually held
}

app = FastAPI(title="artmat", version="0.1")

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
        "horizon_years": EVIDENCE_HORIZON_YEARS.get(hit.source_type),
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "model": MODEL,
        "variants": [v for v in PROMPTS if v != "no_context"],
        "db": "down" if _state.get("db_error") else "up",
        "horizons": EVIDENCE_HORIZON_YEARS,
    }


@app.post("/api/ask")
async def ask(req: AskRequest):
    if req.variant not in PROMPTS:
        raise HTTPException(400, f"unknown variant {req.variant!r}")

    qdrant = _state.get("qdrant")
    if qdrant is None:
        raise HTTPException(503, "not ready")

    async def events():
        loop = asyncio.get_running_loop()
        question = req.question.strip()

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
                "n_hits": len(hits),
                "retrieval_ms": retrieval_ms,
                "generate_ms": generate_ms,
                "total_ms": retrieval_ms + generate_ms,
                "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens,
                "cost_usd": cost,
                "truncated": answer.truncated,
                "error": None,
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
