"""Safe exact-answer cache shared by both user interfaces.

This module decides cache identity; transport modules do not. The identity
includes every user-visible pipeline setting plus hashes of the live prompts.
`EXACT_CACHE_NAMESPACE` is the deliberate invalidation switch for corpus or
index changes, which code cannot infer reliably from Qdrant.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass

from app.db import find_cached_query
from rag.generate import MODEL as GENERATE_MODEL, PROMPTS, Answer
from rag.rewrite import MODEL as REWRITE_MODEL, SYSTEM_PROMPT, Rewrite
from rag.search import Hit


@dataclass(frozen=True)
class ExactCacheIdentity:
    key: str
    namespace: str


@dataclass
class CachedQuery:
    source_query_id: int
    identity: ExactCacheIdentity
    answer: Answer
    rewrite: Rewrite
    hits: list[Hit]
    source_counts: dict[str, int]
    method: str


def enabled() -> bool:
    return os.environ.get("EXACT_CACHE_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off"
    }


def normalize_question(question: str) -> str:
    """Normalise formatting, not meaning: Unicode form and repeated spaces."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", question)).strip()


def identity_for(
    question: str, variant: str, k: int, use_rewrite: bool
) -> ExactCacheIdentity:
    namespace = (
        os.environ.get("EXACT_CACHE_NAMESPACE", "corpus-v1").strip()
        or "corpus-v1"
    )
    material = {
        "namespace": namespace,
        "question": normalize_question(question),
        "variant": variant,
        "k": k,
        "rewrite": use_rewrite,
        "retrieval": "rewrite_hybrid" if use_rewrite else "hybrid",
        "generate_model": GENERATE_MODEL,
        "rewrite_model": REWRITE_MODEL if use_rewrite else None,
        "answer_prompt": PROMPTS[variant],
        "rewrite_prompt": SYSTEM_PROMPT if use_rewrite else None,
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True).encode()
    return ExactCacheIdentity(hashlib.sha256(encoded).hexdigest(), namespace)


def hit_to_dict(hit: Hit) -> dict:
    return asdict(hit)


def lookup(
    question: str, variant: str, k: int, use_rewrite: bool
) -> CachedQuery | None:
    if not enabled():
        return None
    identity = identity_for(question, variant, k, use_rewrite)
    row = find_cached_query(identity.key)
    if row is None:
        return None

    hits = [Hit(**value) for value in (row.get("hits") or [])]
    rewrite = Rewrite(
        original=question,
        rewritten=row.get("rewritten") or question,
        terms=list(row.get("rewrite_terms") or []),
        used=bool(row.get("rewrite_used")),
    )
    answer = Answer(
        question=question,
        variant=variant,
        text=row["answer"],
        hits=hits,
        input_tokens=0,
        output_tokens=0,
        truncated=False,
    )
    return CachedQuery(
        source_query_id=row["id"],
        identity=identity,
        answer=answer,
        rewrite=rewrite,
        hits=hits,
        source_counts=dict(row.get("source_counts") or {}),
        method=row["method"],
    )


def cache_fields(
    question: str, variant: str, k: int, use_rewrite: bool
) -> dict:
    identity = identity_for(question, variant, k, use_rewrite)
    return {
        "cache_key": identity.key,
        "cache_namespace": identity.namespace,
        "cache_hit": False,
        "cache_source_query_id": None,
    }


def hit_record(cached: CachedQuery, total_ms: int) -> dict:
    """Build a zero-cost log row for this request, not the source request."""
    return {
        "question": cached.answer.question,
        "rewritten": cached.rewrite.rewritten,
        "rewrite_used": cached.rewrite.used,
        "rewrite_terms": cached.rewrite.terms,
        "variant": cached.answer.variant,
        "method": cached.method,
        "answer": cached.answer.text,
        "source_counts": cached.source_counts,
        "chunk_ids": [hit.chunk_id for hit in cached.hits],
        "hits": [hit_to_dict(hit) for hit in cached.hits],
        "n_hits": len(cached.hits),
        "retrieval_ms": 0,
        "generate_ms": 0,
        "total_ms": total_ms,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0,
        "truncated": False,
        "error": None,
        "cache_key": cached.identity.key,
        "cache_namespace": cached.identity.namespace,
        "cache_hit": True,
        "cache_source_query_id": cached.source_query_id,
    }
