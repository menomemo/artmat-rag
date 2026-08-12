"""Retrievers over one collection, so they can be compared honestly.

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
import re
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt

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

# Production diversity settings. MMR operates only on the candidate set already
# retrieved by hybrid search; it cannot introduce an irrelevant document that
# retrieval did not find. Two chunks per document preserve useful spec/prose
# pairs while preventing one long datasheet from occupying the whole context.
MMR_CANDIDATES = int(os.environ.get("MMR_CANDIDATES", "32"))
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", "0.95"))
MMR_MAX_PER_DOC = int(os.environ.get("MMR_MAX_PER_DOC", "2"))
MMR_MAX_PER_FAMILY = int(os.environ.get("MMR_MAX_PER_FAMILY", "2"))
MMR_RELEVANCE_HEAD = int(os.environ.get("MMR_RELEVANCE_HEAD", "5"))

METHODS = ("dense", "sparse", "hybrid", "hybrid_mmr", "hybrid_rerank")
PRODUCTION_DIVERSITY_ENABLED = os.environ.get(
    "PRODUCTION_DIVERSITY_ENABLED", "false"
).strip().lower() not in {"0", "false", "no", "off"}
PRODUCTION_METHOD = "hybrid_mmr" if PRODUCTION_DIVERSITY_ENABLED else "hybrid"
PRODUCTION_REWRITE_METHOD = f"rewrite_{PRODUCTION_METHOD}"

SOURCE_TYPES = (
    "manufacturer_datasheet",
    "materials_science",
    "conservation_literature",
    "collection_precedent",
)
CHUNK_TYPES = ("spec", "narrative", "abstract", "precedent")
MANUFACTURER_CATEGORIES = (
    "adhesives", "color-and-fillers", "concrete-gypsum-additives",
    "epoxy-casting-and-laminating-resins", "epoxy-putties",
    "epoxy-urethane-coatings", "platinum-silicone", "polysulfide-rubber",
    "sealers-release-agents", "silicone-expanding-foam-platinum-cure",
    "tin-silicone", "urethane-expanding-foams", "urethane-resin",
    "urethane-rubber",
)
LITERATURE_DOMAINS = (
    "adhesives_coatings", "cementitious", "conservation_practice", "metals",
    "pigments_surfaces", "polymers_resins",
)
COLLECTION_MATERIALS = (
    "aluminium", "bronze", "concrete / cement", "epoxy resin", "fibreglass",
    "latex / rubber", "lead", "plaster", "polyester resin", "polystyrene",
    "polyurethane", "silicone", "steel", "unspecified resin", "wax",
)
FILTER_YEAR_MIN = 1952
FILTER_YEAR_MAX = 2026


@dataclass(frozen=True)
class SearchFilters:
    source_types: tuple[str, ...] = ()
    chunk_types: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    materials: tuple[str, ...] = ()
    year_from: int | None = None
    year_to: int | None = None

    def __post_init__(self) -> None:
        checks = (
            ("source type", self.source_types, SOURCE_TYPES),
            ("chunk type", self.chunk_types, CHUNK_TYPES),
            ("category", self.categories, MANUFACTURER_CATEGORIES),
            ("domain", self.domains, LITERATURE_DOMAINS),
            ("material", self.materials, COLLECTION_MATERIALS),
        )
        for label, values, allowed in checks:
            unknown = set(values) - set(allowed)
            if unknown:
                raise ValueError(f"unknown {label}: {sorted(unknown)}")
        if (
            self.year_from is not None and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must be less than or equal to year_to")
        for label, value in (("year_from", self.year_from), ("year_to", self.year_to)):
            if value is not None and not FILTER_YEAR_MIN <= value <= FILTER_YEAR_MAX:
                raise ValueError(
                    f"{label} must be between {FILTER_YEAR_MIN} and {FILTER_YEAR_MAX}"
                )

    def as_dict(self) -> dict:
        return {
            "source_types": list(self.source_types),
            "chunk_types": list(self.chunk_types),
            "categories": list(self.categories),
            "domains": list(self.domains),
            "materials": list(self.materials),
            "year_from": self.year_from,
            "year_to": self.year_to,
        }


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


@dataclass
class Candidate:
    hit: Hit
    vector: list[float]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


GRADE_WORDS = {
    "fast", "slow", "medium", "std", "standard", "trial", "kit", "starter"
}


def family_key(hit: Hit) -> str:
    """Conservative manufacturer series key; other sources stay per-document.

    Smooth-On titles consistently put a numeric grade and speed after the
    series ("Mold Star 15 SLOW", "Smooth-Sil 950"). Removing only those
    suffixes groups genuine variants without pretending every product in the
    broad `platinum-silicone` category is interchangeable.
    """
    if hit.source_type != "manufacturer_datasheet":
        return hit.doc_id
    words = re.findall(r"[a-z0-9]+", hit.title.casefold())
    while len(words) > 1 and (
        any(character.isdigit() for character in words[-1])
        or words[-1] in GRADE_WORDS
    ):
        words.pop()
    return " ".join(words) or hit.doc_id


def diversify(candidates: list[Candidate], limit: int) -> list[Hit]:
    """Select relevant but non-redundant candidates with deterministic MMR."""
    if not candidates or limit <= 0:
        return []

    scores = [candidate.hit.score for candidate in candidates]
    low, high = min(scores), max(scores)
    relevance = [
        (score - low) / (high - low) if high > low else 1.0
        for score in scores
    ]

    # Preserve the evaluated relevance head exactly. Diversity is for widening
    # an eight-passage context, not for moving a known-good top-five result out
    # of reach. Family/document caps apply only to additional selections.
    head = min(limit, MMR_RELEVANCE_HEAD, len(candidates))
    selected: list[int] = list(range(head))
    remaining = list(range(head, len(candidates)))
    per_doc: dict[str, int] = {}
    per_family: dict[str, int] = {}
    for index in selected:
        hit = candidates[index].hit
        per_doc[hit.doc_id] = per_doc.get(hit.doc_id, 0) + 1
        family = family_key(hit)
        per_family[family] = per_family.get(family, 0) + 1

    while remaining and len(selected) < limit:
        best_index = None
        best_value = float("-inf")
        for index in remaining:
            candidate = candidates[index]
            if per_doc.get(candidate.hit.doc_id, 0) >= MMR_MAX_PER_DOC:
                continue
            family = family_key(candidate.hit)
            if per_family.get(family, 0) >= MMR_MAX_PER_FAMILY:
                continue
            similarity = max(
                (_cosine(candidate.vector, candidates[chosen].vector)
                 for chosen in selected),
                default=0.0,
            )
            value = MMR_LAMBDA * relevance[index] - (1 - MMR_LAMBDA) * max(
                similarity, 0.0
            )
            if value > best_value:
                best_value = value
                best_index = index

        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
        doc_id = candidates[best_index].hit.doc_id
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        family = family_key(candidates[best_index].hit)
        per_family[family] = per_family.get(family, 0) + 1

    return [candidates[index].hit for index in selected]


def candidates_from_points(points) -> list[Candidate]:
    return [
        Candidate(Hit.from_point(point), list((point.vector or {}).get(DENSE, [])))
        for point in points
    ]


@lru_cache(maxsize=1)
def _reranker():
    # Imported and constructed lazily: loading the ONNX cross-encoder costs a
    # few seconds and ~90 MB, and three of the four retrievers never touch it.
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=RERANK_MODEL)


def _metadata_filter(
    filters: SearchFilters | None = None,
    source_types: list[str] | None = None,
) -> models.Filter | None:
    if filters is not None and source_types:
        raise ValueError("pass filters or source_types, not both")
    filters = filters or SearchFilters(
        source_types=tuple(source_types or ())
    )
    conditions: list[models.Condition] = []
    keyword_fields = (
        ("source_type", filters.source_types),
        ("chunk_type", filters.chunk_types),
        ("category", filters.categories),
        ("domain", filters.domains),
        ("material", filters.materials),
    )
    for field, values in keyword_fields:
        if values:
            conditions.append(
                models.FieldCondition(
                    key=field, match=models.MatchAny(any=list(values))
                )
            )
    if filters.year_from is not None or filters.year_to is not None:
        conditions.append(
            models.FieldCondition(
                key="year",
                range=models.Range(gte=filters.year_from, lte=filters.year_to),
            )
        )
    if not conditions:
        return None
    return models.Filter(must=conditions)


# Compatibility name used by older callers and evaluations.
def _source_filter(source_types: list[str] | None) -> models.Filter | None:
    return _metadata_filter(source_types=source_types)


def search(
    client: QdrantClient,
    query: str,
    method: str = "hybrid_rerank",
    limit: int = 5,
    source_types: list[str] | None = None,
    filters: SearchFilters | None = None,
) -> list[Hit]:
    """One entry point for every method, so callers cannot diverge."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

    query_filter = _metadata_filter(filters, source_types)

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
    diversity_active = method == "hybrid_mmr" and limit > MMR_RELEVANCE_HEAD
    if method == "hybrid_rerank":
        fetch = CANDIDATES
    elif method in ("hybrid", "hybrid_mmr"):
        fetch = min(CANDIDATES, max(limit, MMR_CANDIDATES))
    else:
        fetch = limit
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
        with_vectors=[DENSE] if diversity_active else False,
    ).points
    points.sort(
        key=lambda point: (
            -point.score, (point.payload or {}).get("chunk_id", "")
        )
    )

    if diversity_active:
        return diversify(candidates_from_points(points), limit)

    hits = [Hit.from_point(p) for p in points]
    if method in ("hybrid", "hybrid_mmr"):
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
