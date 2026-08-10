"""Four retrievers over one collection, so they can be compared honestly.

`dense`, `sparse`, and `hybrid` differ only in how candidates are scored -- same
index, same chunks, same payloads. `hybrid_rerank` adds a cross-encoder pass on
top of hybrid. Anything that differed between them other than the ranking would
make the evaluation a comparison of two pipelines rather than of four rankers.

Fusion happens inside Qdrant via the Query API rather than in Python. Reciprocal
Rank Fusion only needs ranks, so it is trivial to reimplement badly: the usual
mistake is fusing *scores*, which cannot be compared across a cosine similarity
and a BM25 sum. Letting the engine do it removes that opportunity.

The cross-encoder is the expensive one. Bi-encoders embed the query and the
document separately and compare vectors, so a document's representation cannot
depend on the question; a cross-encoder reads both together and scores the pair
directly. That is far more accurate and far too slow to run over 4,115 chunks,
so it reranks the top `candidates` from hybrid only -- which is exactly the
tradeoff that makes reranking worth having.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient, models

from rag.index import (
    COLLECTION,
    DENSE,
    DENSE_MODEL,
    SPARSE,
    SPARSE_MODEL,
    dense_doc,
    sparse_doc,
)

RERANK_MODEL = os.environ.get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

# How many candidates each arm contributes before fusion, and how many the
# cross-encoder rescores. Larger is more accurate and slower; 50 keeps a
# reranked query near a second on CPU, which the Streamlit UI has to live with.
CANDIDATES = 50

METHODS = ("dense", "sparse", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    source_type: str
    chunk_type: str
    title: str
    text: str
    url: str | None
    score: float

    @classmethod
    def from_point(cls, point) -> "Hit":
        payload = point.payload or {}
        return cls(
            chunk_id=payload.get("chunk_id", ""),
            doc_id=payload.get("doc_id", ""),
            source_type=payload.get("source_type", ""),
            chunk_type=payload.get("chunk_type", ""),
            title=payload.get("title", ""),
            text=payload.get("text", ""),
            url=payload.get("url"),
            score=point.score,
        )


@lru_cache(maxsize=1)
def _reranker():
    # Imported and constructed lazily: loading the ONNX cross-encoder costs a
    # few seconds and ~90 MB, and three of the four retrievers never touch it.
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=RERANK_MODEL)


def _source_filter(source_types: list[str] | None) -> models.Filter | None:
    if not source_types:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="source_type", match=models.MatchAny(any=source_types)
            )
        ]
    )


def search(
    client: QdrantClient,
    query: str,
    method: str = "hybrid_rerank",
    limit: int = 5,
    source_types: list[str] | None = None,
) -> list[Hit]:
    """One entry point for all four methods, so callers cannot diverge."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

    query_filter = _source_filter(source_types)

    if method == "dense":
        points = client.query_points(
            collection_name=COLLECTION,
            query=dense_doc(query),
            using=DENSE,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        ).points
        return [Hit.from_point(p) for p in points]

    if method == "sparse":
        points = client.query_points(
            collection_name=COLLECTION,
            query=sparse_doc(query),
            using=SPARSE,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        ).points
        return [Hit.from_point(p) for p in points]

    # hybrid and hybrid_rerank share their candidate generation exactly. The
    # rerank variant asks for more candidates because the cross-encoder can
    # only promote what fusion already surfaced -- a document missed here is
    # missed for good, no matter how good the reranker is.
    fetch = CANDIDATES if method == "hybrid_rerank" else limit
    points = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_doc(query), using=DENSE, limit=CANDIDATES),
            models.Prefetch(query=sparse_doc(query), using=SPARSE, limit=CANDIDATES),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=fetch,
        query_filter=query_filter,
        with_payload=True,
    ).points

    hits = [Hit.from_point(p) for p in points]
    if method == "hybrid":
        return hits[:limit]

    if not hits:
        return []
    scores = list(_reranker().rerank(query, [h.text for h in hits]))
    for hit, score in zip(hits, scores):
        hit.score = float(score)
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]


if __name__ == "__main__":
    import argparse

    from rag.index import connect

    parser = argparse.ArgumentParser(description="Query the collection")
    parser.add_argument("query")
    parser.add_argument("--method", default="hybrid_rerank", choices=METHODS)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    client = connect()
    for i, hit in enumerate(search(client, args.query, args.method, args.limit), 1):
        print(f"{i}. [{hit.source_type}] {hit.title[:70]}  ({hit.score:.4f})")
        print(f"   {hit.text[:180].strip()}...")
