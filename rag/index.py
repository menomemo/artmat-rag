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

API note: qdrant-client 1.19 removed the older `set_model` / `client.add`
convenience layer in favour of `models.Document`, which carries its model name
with it. The text is embedded client-side either way; the difference is that
the model is now named at the point of use, so indexing and querying cannot
silently drift onto different models.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

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
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)


def create_collection(client: QdrantClient, drop: bool = False) -> None:
    if drop and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    if client.collection_exists(COLLECTION):
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
    for field in ("source_type", "chunk_type"):
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def to_point(index: int, chunk: dict) -> models.PointStruct:
    return models.PointStruct(
        # Deterministic ids: re-running the indexer updates points in place
        # instead of duplicating the corpus, so a partial run can simply be
        # repeated. `chunk_id` is kept in the payload as the human-readable key.
        id=index,
        vector={DENSE: dense_doc(chunk["text"]), SPARSE: sparse_doc(chunk["text"])},
        payload={
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "source_type": chunk["source_type"],
            "chunk_type": chunk["chunk_type"],
            "title": chunk["title"],
            "url": chunk["url"],
            "text": chunk["text"],
            **chunk.get("metadata", {}),
        },
    )


def upload(client: QdrantClient, chunks: list[dict], batch_size: int = 64) -> None:
    """Embed and upsert in batches.

    Batched rather than uploaded in one call because both models run in-process:
    a single 4,115-document request would hold every embedding in memory and
    give no sign of progress on a run that takes minutes.
    """
    for start in range(0, len(chunks), batch_size):
        window = chunks[start : start + batch_size]
        client.upsert(
            collection_name=COLLECTION,
            points=[to_point(start + i, c) for i, c in enumerate(window)],
        )
        done = min(start + batch_size, len(chunks))
        print(f"  {done}/{len(chunks)}", end="\r", flush=True)
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index chunks into Qdrant")
    parser.add_argument("--chunks", default="data/chunks.jsonl", type=Path)
    parser.add_argument("--drop", action="store_true", help="recreate the collection")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    if args.limit:
        chunks = chunks[: args.limit]

    client = connect()
    create_collection(client, drop=args.drop)

    started = time.monotonic()
    print(f"embedding + indexing {len(chunks)} chunks ...", flush=True)
    upload(client, chunks)
    elapsed = time.monotonic() - started

    count = client.count(COLLECTION, exact=True).count
    print(f"done in {elapsed:.0f}s -- collection '{COLLECTION}' holds {count} points")
