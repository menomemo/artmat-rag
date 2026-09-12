"""Postgres: query log and user feedback.

This is the monitoring substrate. Every question asked through the UI is
recorded with enough context to answer the questions a dashboard should be able
to answer -- not just "how many queries" but "are the answers getting worse",
"which layer is actually carrying the corpus", "is the rewrite earning its
latency", "what is this costing".

Two tables rather than one. Feedback arrives seconds to minutes after the
query, from a different user action, and may never arrive at all; folding it
into the query row would mean an UPDATE on a table that is otherwise
append-only, and would lose the distinction between "rated neutral" and "not
rated". `LEFT JOIN` keeps unrated queries visible in the dashboard, which is
the honest denominator for a satisfaction rate.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

LOCAL_DSN = "postgresql://artmat:artmat@localhost:5432/artmat"


def dsn() -> str:
    """Read the DSN at call time, not at import time.

    This was `DSN = os.environ.get(...)` as a module constant, bound into every
    function's default argument. That works locally, where `.env` is loaded
    before the first import, and fails silently on Streamlit Cloud, where the
    secrets bridge populates `os.environ` at startup: whichever module imported
    `app.db` first would freeze the localhost default, and the deployed app
    would keep trying to log to a Postgres that is not there.

    A managed database is also the one connection that can hang rather than
    refuse -- `connect_timeout` makes that a 10-second error instead of a page
    that never renders.
    """
    return os.environ.get("POSTGRES_DSN", LOCAL_DSN)

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id              BIGSERIAL PRIMARY KEY,
    asked_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    question        TEXT         NOT NULL,
    rewritten       TEXT,
    rewrite_used    BOOLEAN      NOT NULL DEFAULT FALSE,
    variant         TEXT         NOT NULL,
    method          TEXT         NOT NULL,
    answer          TEXT         NOT NULL,
    -- Which layers the answer was actually built from. A jsonb map of
    -- source_type -> count, so the dashboard can show corpus balance without
    -- joining back to Qdrant, which has no notion of time.
    source_counts   JSONB        NOT NULL DEFAULT '{}'::jsonb,
    chunk_ids       JSONB        NOT NULL DEFAULT '[]'::jsonb,
    n_hits          INTEGER      NOT NULL DEFAULT 0,
    retrieval_ms    INTEGER      NOT NULL DEFAULT 0,
    generate_ms     INTEGER      NOT NULL DEFAULT 0,
    total_ms        INTEGER      NOT NULL DEFAULT 0,
    input_tokens    INTEGER      NOT NULL DEFAULT 0,
    output_tokens   INTEGER      NOT NULL DEFAULT 0,
    -- Cost is stored, not derived at query time. Prices change; what a query
    -- cost on the day it ran does not.
    cost_usd        NUMERIC(10,6) NOT NULL DEFAULT 0,
    truncated       BOOLEAN      NOT NULL DEFAULT FALSE,
    error           TEXT,
    -- Exact-cache fields are deliberately part of the append-only query log.
    -- A cache hit gets its own row (and therefore its own feedback id), while
    -- cache_source_query_id preserves which paid answer it reused.
    cache_key        TEXT,
    cache_namespace  TEXT,
    cache_hit        BOOLEAN      NOT NULL DEFAULT FALSE,
    cache_source_query_id BIGINT REFERENCES queries(id) ON DELETE SET NULL,
    rewrite_terms    JSONB        NOT NULL DEFAULT '[]'::jsonb,
    hits             JSONB        NOT NULL DEFAULT '[]'::jsonb
);

-- CREATE TABLE IF NOT EXISTS does not add columns to an existing deployment.
ALTER TABLE queries ADD COLUMN IF NOT EXISTS cache_key TEXT;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS cache_namespace TEXT;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS cache_hit BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS cache_source_query_id BIGINT REFERENCES queries(id) ON DELETE SET NULL;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS rewrite_terms JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE queries ADD COLUMN IF NOT EXISTS hits JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE INDEX IF NOT EXISTS queries_asked_at_idx ON queries (asked_at DESC);
CREATE INDEX IF NOT EXISTS queries_variant_idx  ON queries (variant);
CREATE INDEX IF NOT EXISTS queries_exact_cache_idx
    ON queries (cache_key, asked_at DESC)
    WHERE cache_hit = FALSE AND error IS NULL AND truncated = FALSE;

CREATE TABLE IF NOT EXISTS feedback (
    id          BIGSERIAL PRIMARY KEY,
    query_id    BIGINT      NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    given_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- +1 / -1 rather than a boolean, so "no opinion" stays absent rather than
    -- being encoded as false.
    rating      SMALLINT    NOT NULL CHECK (rating IN (-1, 1)),
    comment     TEXT,
    -- One rating per query; a second click revises rather than double-counts.
    UNIQUE (query_id)
);

CREATE INDEX IF NOT EXISTS feedback_given_at_idx ON feedback (given_at DESC);
"""

# Published per-MTok prices for the models this app calls. Kept here so the
# figure in the dashboard is auditable rather than a magic constant buried in
# an f-string.
PRICING_USD_PER_MTOK = {
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-opus-5": {"input": 15.00, "output": 75.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICING_USD_PER_MTOK.get(model)
    if not price:
        return 0.0
    return (
        input_tokens * price["input"] + output_tokens * price["output"]
    ) / 1_000_000


@contextmanager
def connect(url: str | None = None):
    with psycopg.connect(
        url or dsn(), row_factory=dict_row, connect_timeout=10
    ) as conn:
        yield conn


def init_schema(url: str | None = None) -> None:
    with connect(url) as conn:
        conn.execute(SCHEMA)
        conn.commit()


def log_query(record: dict, url: str | None = None) -> int:
    """Insert one query, return its id so feedback can reference it."""
    with connect(url) as conn:
        row = conn.execute(
            """
            INSERT INTO queries (
                question, rewritten, rewrite_used, variant, method, answer,
                source_counts, chunk_ids, n_hits,
                retrieval_ms, generate_ms, total_ms,
                input_tokens, output_tokens, cost_usd, truncated, error,
                cache_key, cache_namespace, cache_hit, cache_source_query_id,
                rewrite_terms, hits
            ) VALUES (
                %(question)s, %(rewritten)s, %(rewrite_used)s, %(variant)s,
                %(method)s, %(answer)s, %(source_counts)s, %(chunk_ids)s,
                %(n_hits)s, %(retrieval_ms)s, %(generate_ms)s, %(total_ms)s,
                %(input_tokens)s, %(output_tokens)s, %(cost_usd)s,
                %(truncated)s, %(error)s, %(cache_key)s, %(cache_namespace)s,
                %(cache_hit)s, %(cache_source_query_id)s, %(rewrite_terms)s,
                %(hits)s
            ) RETURNING id
            """,
            {
                **record,
                "source_counts": json.dumps(record.get("source_counts", {})),
                "chunk_ids": json.dumps(record.get("chunk_ids", [])),
                "cache_key": record.get("cache_key"),
                "cache_namespace": record.get("cache_namespace"),
                "cache_hit": record.get("cache_hit", False),
                "cache_source_query_id": record.get("cache_source_query_id"),
                "rewrite_terms": json.dumps(record.get("rewrite_terms", [])),
                "hits": json.dumps(record.get("hits", [])),
            },
        ).fetchone()
        conn.commit()
        return row["id"]


def find_cached_query(cache_key: str, url: str | None = None) -> dict | None:
    """Return the newest complete paid answer for an exact cache key.

    Cache-hit rows are never used as sources. This keeps the provenance chain
    one hop long and means deleting or inspecting the original paid query is
    sufficient to understand every reuse of it.
    """
    with connect(url) as conn:
        return conn.execute(
            """
            SELECT q.id, q.question, q.rewritten, q.rewrite_used,
                   q.rewrite_terms, q.variant, q.method, q.answer,
                   q.source_counts, q.chunk_ids, q.hits, q.n_hits,
                   q.cache_key, q.cache_namespace
              FROM queries q
             WHERE q.cache_key = %s
               AND q.cache_hit = FALSE
               AND q.error IS NULL
               AND q.truncated = FALSE
               AND q.answer <> ''
               -- One negative rating on the source or any reuse invalidates
               -- the answer. Caching must not amplify an answer a reader has
               -- already identified as unhelpful.
               AND NOT EXISTS (
                   SELECT 1
                     FROM feedback f
                     JOIN queries rated ON rated.id = f.query_id
                    WHERE f.rating = -1
                      AND (rated.id = q.id OR rated.cache_source_query_id = q.id)
               )
             ORDER BY q.asked_at DESC
             LIMIT 1
            """,
            (cache_key,),
        ).fetchone()


def log_feedback(query_id: int, rating: int, comment: str | None = None,
                 url: str | None = None) -> None:
    with connect(url) as conn:
        conn.execute(
            """
            INSERT INTO feedback (query_id, rating, comment)
            VALUES (%s, %s, %s)
            ON CONFLICT (query_id) DO UPDATE
              SET rating = EXCLUDED.rating,
                  comment = EXCLUDED.comment,
                  given_at = now()
            """,
            (query_id, rating, comment),
        )
        conn.commit()


if __name__ == "__main__":
    init_schema()
    with connect() as conn:
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        ).fetchall()
    print("schema ready:", [t["table_name"] for t in tables])
