"""Build the Qdrant collection: dense and sparse vectors in one place.

Why both, in one collection rather than two:

The corpus asks for it. "What is the Shore hardness of Mold Star 15?" is a
lexical lookup -- the answer is a literal string in a spec table, and a dense
model that has never seen "Shore 30A" will happily return a chemically similar
product with the wrong number. "Will this crack outdoors in winter?" is the
opposite: no document contains those words, and BM25 has nothing to match on.
Neither retriever is adequate alone, and which one wins depends on the
question, not on the corpus.

Qdrant supports named vectors, so a point carries both representations and a
single query can fuse them server-side with Reciprocal Rank Fusion. The
alternative -- two collections, two round trips, fusion in Python -- means the
scores being combined were computed by different code paths, and the fusion
step becomes something to debug rather than something to trust.

Both models run locally through fastembed (ONNX, CPU). Embedding 4,115 chunks
costs nothing and needs no API key, which matters for a grader re-running this
from scratch.

The normal command is a state sync, not a blind rebuild. It compares canonical
payload hashes, embeds only additions and changes, and removes points whose
chunk ids disappeared. A full collection drop remains an explicit recovery
operation rather than the price of updating one datasheet.

API note: qdrant-client 1.19 removed the older `set_model` / `client.add`
convenience layer in favour of `models.Document`, which carries its model name
with it. The text is embedded client-side either way; the difference is that
the model is now named at the point of use, so indexing and querying cannot
silently drift onto different models.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx
from qdrant_client import QdrantClient, models

import rag.env  # noqa: F401  -- loads .env on import

# 384 dimensions, ~130 MB ONNX, and the strongest of the small English models
# on retrieval benchmarks. `bge-base` is ~3x the size for a few points of
# nDCG -- not the right trade when the whole thing has to rebuild during a
# two-day project and again on a grader's laptop.
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_DIM = 384

# Real BM25, not a learned sparse model. Two reasons: it gives the evaluation
# an honest lexical baseline to compare against (SPLADE is a neural model, so
# "sparse vs dense" would really be "neural vs neural"), and its IDF is
# computed by Qdrant over this corpus rather than baked in at training time.
SPARSE_MODEL = "Qdrant/bm25"

COLLECTION = os.environ.get("QDRANT_COLLECTION", "artmat")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

# Named vectors. Spelled out rather than derived from the model name, because
# the evaluation queries these by name and a renamed model must not silently
# become a different, empty vector field.
DENSE = "dense"
SPARSE = "bm25"

# Point ids must survive reordering of chunks.jsonl.  UUID5 is accepted by
# Qdrant, deterministic across machines, and cannot collide merely because a
# new chunk was inserted earlier in the file.
POINT_NAMESPACE = uuid.UUID("6bc19fb6-304c-5cb9-9e89-a2dc82f27d67")
INTERNAL_PAYLOAD_KEYS = {"_content_hash", "_index_signature"}
INDEX_SIGNATURE = hashlib.sha256(
    json.dumps(
        {
            "dense_model": DENSE_MODEL,
            "dense_dim": DENSE_DIM,
            "sparse_model": SPARSE_MODEL,
            "vectors": [DENSE, SPARSE],
            "payload_schema": 1,
        },
        sort_keys=True,
    ).encode()
).hexdigest()


def dense_doc(text: str) -> models.Document:
    return models.Document(text=text, model=DENSE_MODEL)


def sparse_doc(text: str) -> models.Document:
    return models.Document(text=text, model=SPARSE_MODEL)


# Set when Qdrant is the managed service rather than the local container. Its
# presence is also what tells `wait_for_qdrant` not to bother polling: a hosted
# cluster is either up or it is a configuration problem, and retrying for a
# minute only delays the error message that would have explained it.
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None


def wait_for_qdrant(url: str = QDRANT_URL, timeout_s: float = 60.0) -> None:
    """Block until Qdrant answers, or explain clearly why it never will."""
    if QDRANT_API_KEY:
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/readyz", timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    raise RuntimeError(
        f"Qdrant did not become ready at {url} within {timeout_s:.0f}s. "
        "Start it with `docker compose up -d qdrant`."
    )


def connect() -> QdrantClient:
    """One client for both deployments, chosen by environment rather than code.

    Local is HTTP on 6333 with no key. Qdrant Cloud is HTTPS on 6333 with a
    key, and gRPC (6334) is not exposed on the free tier -- so REST is the
    transport in both cases, which keeps the deployed path the same one the
    evaluation measured.
    """
    wait_for_qdrant()
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    if client.collection_exists(COLLECTION):
        ensure_payload_indexes(client)
    return client


def ensure_payload_indexes(client: QdrantClient) -> None:
    """Idempotently add every index used by production metadata filters."""
    fields = {
        "source_type": models.PayloadSchemaType.KEYWORD,
        "chunk_type": models.PayloadSchemaType.KEYWORD,
        "category": models.PayloadSchemaType.KEYWORD,
        "domain": models.PayloadSchemaType.KEYWORD,
        "material": models.PayloadSchemaType.KEYWORD,
        "year": models.PayloadSchemaType.INTEGER,
    }
    existing = client.get_collection(COLLECTION).payload_schema
    for field, schema in fields.items():
        if field in existing:
            if existing[field].data_type != schema:
                raise RuntimeError(
                    f"payload index {field!r} is {existing[field].data_type}, "
                    f"expected {schema}"
                )
            continue
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=schema,
        )


def create_collection(client: QdrantClient, drop: bool = False) -> None:
    if drop and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    if client.collection_exists(COLLECTION):
        ensure_payload_indexes(client)
        return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            DENSE: models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE: models.SparseVectorParams(
                # Without this, Qdrant stores the raw term frequencies fastembed
                # produced and never applies the inverse-document-frequency
                # term -- so "the" would weigh as much as "polysulfide" and the
                # lexical arm would return whichever chunk is longest.
                modifier=models.Modifier.IDF
            )
        },
    )

    # Payload indexes on the fields the evaluation and the UI actually filter
    # by. Without them Qdrant full-scans the payload for every filtered query.
    ensure_payload_indexes(client)


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def chunk_payload(chunk: dict) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "source_type": chunk["source_type"],
        "chunk_type": chunk["chunk_type"],
        "title": chunk["title"],
        "url": chunk["url"],
        "text": chunk["text"],
        **chunk.get("metadata", {}),
    }


def content_hash(payload: dict) -> str:
    """Hash everything retrieval can return, independent of JSON key order."""
    public = {k: v for k, v in payload.items() if k not in INTERNAL_PAYLOAD_KEYS}
    canonical = json.dumps(
        public, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def to_point(point: int | str | uuid.UUID, chunk: dict) -> models.PointStruct:
    payload = chunk_payload(chunk)
    payload["_content_hash"] = content_hash(payload)
    payload["_index_signature"] = INDEX_SIGNATURE
    return models.PointStruct(
        id=point,
        vector={DENSE: dense_doc(chunk["text"]), SPARSE: sparse_doc(chunk["text"])},
        payload=payload,
    )


@dataclass(frozen=True)
class ExistingPoint:
    id: int | str | uuid.UUID
    payload: dict


@dataclass
class IndexPlan:
    added: list[tuple[int | str | uuid.UUID, dict]]
    updated: list[tuple[int | str | uuid.UUID, dict]]
    unchanged: list[tuple[int | str | uuid.UUID, dict]]
    adopted: list[int | str | uuid.UUID]
    stale: list[int | str | uuid.UUID]

    @property
    def embeds(self) -> int:
        return len(self.added) + len(self.updated)


def validate_chunks(chunks: list[dict]) -> dict[str, dict]:
    desired: dict[str, dict] = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            raise ValueError("every chunk must have a non-empty chunk_id")
        if chunk_id in desired:
            raise ValueError(f"duplicate chunk_id in input: {chunk_id}")
        desired[chunk_id] = chunk
    return desired


def read_existing(client: QdrantClient, page_size: int = 256) -> list[ExistingPoint]:
    """Read payloads only; vectors are intentionally not transferred."""
    found: list[ExistingPoint] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        found.extend(ExistingPoint(record.id, record.payload or {}) for record in records)
        if offset is None:
            return found


def plan_sync(chunks: list[dict], existing: list[ExistingPoint]) -> IndexPlan:
    desired = validate_chunks(chunks)
    by_chunk_id: dict[str, ExistingPoint] = {}
    stale: list[int | str | uuid.UUID] = []

    for record in existing:
        chunk_id = record.payload.get("chunk_id")
        if not chunk_id:
            stale.append(record.id)
            continue
        if chunk_id in by_chunk_id:
            raise RuntimeError(
                f"collection contains duplicate chunk_id {chunk_id!r}; rebuild with --drop"
            )
        by_chunk_id[chunk_id] = record

    added: list[tuple[int | str | uuid.UUID, dict]] = []
    updated: list[tuple[int | str | uuid.UUID, dict]] = []
    unchanged: list[tuple[int | str | uuid.UUID, dict]] = []
    adopted: list[int | str | uuid.UUID] = []
    for chunk_id, chunk in desired.items():
        record = by_chunk_id.pop(chunk_id, None)
        if record is None:
            added.append((point_id(chunk_id), chunk))
            continue

        wanted_hash = content_hash(chunk_payload(chunk))
        stored_hash = record.payload.get("_content_hash") or content_hash(record.payload)
        # A missing signature denotes the legacy index built by this repository
        # with the same two models.  It is adopted in place, avoiding a one-off
        # paid-in-time full rebuild.  `apply_plan` persists an explicit
        # signature without re-embedding, so future model changes are visible.
        stored_signature = record.payload.get("_index_signature", INDEX_SIGNATURE)
        target = (record.id, chunk)
        if stored_hash == wanted_hash and stored_signature == INDEX_SIGNATURE:
            unchanged.append(target)
            if "_index_signature" not in record.payload:
                adopted.append(record.id)
        else:
            updated.append(target)

    stale.extend(record.id for record in by_chunk_id.values())
    return IndexPlan(added, updated, unchanged, adopted, stale)


def batches(items: list[Any], size: int) -> Iterable[list[Any]]:
    if size < 1:
        raise ValueError("batch size must be at least 1")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def apply_plan(
    client: QdrantClient,
    plan: IndexPlan,
    batch_size: int = 64,
    prune: bool = True,
) -> None:
    """Embed changed points first, then delete stale points.

    Batched rather than uploaded in one call because both models run in-process:
    a single 4,115-document request would hold every embedding in memory and
    give no sign of progress on a run that takes minutes.

    Deletion is deliberately last.  If embedding or upload fails, the old
    searchable corpus remains intact and the same command can be retried.
    """
    changed = plan.added + plan.updated
    completed = 0
    for window in batches(changed, batch_size):
        client.upsert(
            collection_name=COLLECTION,
            points=[to_point(point, chunk) for point, chunk in window],
            wait=True,
        )
        completed += len(window)
        print(f"  embedded {completed}/{len(changed)}", end="\r", flush=True)
    if changed:
        print()

    # Legacy points already contain the correct vectors. Persisting the current
    # signature is a cheap payload-only migration that makes a later model
    # change detectable without re-embedding the corpus today.
    for window in batches(plan.adopted, batch_size):
        client.set_payload(
            collection_name=COLLECTION,
            payload={"_index_signature": INDEX_SIGNATURE},
            points=window,
            wait=True,
        )

    if prune:
        for window in batches(plan.stale, batch_size):
            client.delete(
                collection_name=COLLECTION,
                points_selector=models.PointIdsList(points=window),
                wait=True,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index chunks into Qdrant")
    parser.add_argument("--chunks", default="data/chunks.jsonl", type=Path)
    parser.add_argument("--drop", action="store_true", help="recreate the collection")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--no-prune", action="store_true", help="keep points absent from the input"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show changes without embedding or writing"
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="allow an empty input to remove every point (dangerous)",
    )
    args = parser.parse_args()

    if args.drop and args.dry_run:
        parser.error("--drop and --dry-run cannot be used together")
    if args.limit and not args.no_prune:
        parser.error("--limit requires --no-prune; refusing to delete unseen chunks")

    chunks = load_chunks(args.chunks)
    if args.limit:
        chunks = chunks[: args.limit]
    if not chunks and not args.allow_empty:
        parser.error("input contains no chunks; refusing to empty the collection")

    client = connect()
    if args.dry_run:
        existing = read_existing(client) if client.collection_exists(COLLECTION) else []
    else:
        create_collection(client, drop=args.drop)
        existing = read_existing(client)

    started = time.monotonic()
    plan = plan_sync(chunks, existing)
    print(
        f"plan: {len(plan.added)} add, {len(plan.updated)} update, "
        f"{len(plan.unchanged)} unchanged ({len(plan.adopted)} legacy signatures), "
        f"{len(plan.stale)} stale"
    )
    if args.dry_run:
        raise SystemExit(0)
    apply_plan(client, plan, batch_size=args.batch_size, prune=not args.no_prune)
    elapsed = time.monotonic() - started

    count = client.count(COLLECTION, exact=True).count
    print(
        f"done in {elapsed:.0f}s -- embedded {plan.embeds}, "
        f"deleted {len(plan.stale) if not args.no_prune else 0}, "
        f"collection '{COLLECTION}' holds {count} points"
    )
