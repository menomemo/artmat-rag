"""Evaluate the four retrievers against the ground-truth set.

Metrics
-------
Each question was written from exactly one chunk, so there is exactly one
relevant document and the usual pair applies:

- **hit-rate@k** -- was the source chunk returned in the top k at all. This is
  what decides whether the generator can possibly be right.
- **MRR@k** -- 1/rank of the source chunk. This is what decides whether the
  right passage is near the top of a context window the model actually reads.

Both are reported at chunk level and at document level. Chunk level is the
strict reading and the standard one. Document level exists because it is often
the fairer one: a datasheet's narrative is split into several windows, and
retrieving window 2 when the question was written from window 3 of the same
product is not really a miss. Reporting only the strict number would understate
every retriever equally, which is fine for ranking them and misleading about
whether the system works.

The two question styles are always kept apart. Averaging them would produce one
number that describes no one -- neither the user who remembers a product name
nor the user who only knows what they are trying to make.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient

from rag.index import connect
from rag.rewrite import Rewrite, rewrite, search_rewritten
from rag.search import METHODS, search

STYLES = ("literal", "artist")
K = 5

# Rewriting methods are named separately rather than added to rag.search's
# METHODS, because they are not pure functions of the index: they call an LLM.
REWRITE_METHODS = ("rewrite_hybrid", "rewrite_hybrid_rerank")
ALL_METHODS = METHODS + REWRITE_METHODS


def load_rewrites(questions: list[dict], path: Path) -> dict[str, Rewrite]:
    """Rewrite every question once, then cache to disk.

    Two things depend on this. The evaluation compares `rewrite_hybrid` against
    `rewrite_hybrid_rerank`, and they must see *identical* rewrites or the
    comparison measures LLM sampling noise instead of the reranker. And a rerun
    of the suite has to reproduce, which a fresh set of generations would not.
    """
    cache: dict[str, dict] = {}
    if path.exists():
        cache = json.loads(path.read_text(encoding="utf-8"))

    texts = [row[style] for row in questions for style in STYLES]
    missing = [t for t in texts if t not in cache]
    if missing:
        print(f"  rewriting {len(missing)} queries ({len(cache)} cached)")
        for i, text in enumerate(missing, 1):
            rw = rewrite(text)
            cache[text] = {
                "rewritten": rw.rewritten,
                "terms": rw.terms,
                "used": rw.used,
            }
            if i % 50 == 0:
                print(f"    {i}/{len(missing)}", flush=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    failed = sum(1 for t in texts if not cache[t]["used"])
    if failed:
        print(f"  note: {failed} rewrites failed and fell back to the original")
    return {
        t: Rewrite(
            original=t,
            rewritten=cache[t]["rewritten"],
            terms=cache[t]["terms"],
            used=cache[t]["used"],
        )
        for t in texts
    }


def reciprocal_rank(hits: list, key: str, target: str) -> float:
    for rank, hit in enumerate(hits, 1):
        if getattr(hit, key) == target:
            return 1.0 / rank
    return 0.0


def evaluate(
    client: QdrantClient,
    questions: list[dict],
    method: str,
    k: int = K,
    rewrites: dict[str, Rewrite] | None = None,
) -> dict:
    """Run one retriever over every question, both styles.

    Rewrite latency is deliberately excluded from the timing: the rewrites are
    pre-computed and cached, so what is measured is retrieval. The LLM call
    adds roughly 700 ms in production and is reported separately rather than
    folded into a number labelled "retrieval latency".
    """
    per_style: dict[str, list[dict]] = defaultdict(list)
    latencies: list[float] = []

    for i, row in enumerate(questions, 1):
        for style in STYLES:
            started = time.perf_counter()
            if method in REWRITE_METHODS:
                hits = search_rewritten(
                    client,
                    rewrites[row[style]],
                    limit=k,
                    rerank=method.endswith("rerank"),
                )
            else:
                hits = search(client, row[style], method=method, limit=k)
            latencies.append(time.perf_counter() - started)
            per_style[style].append(
                {
                    "source_type": row["source_type"],
                    "chunk_rr": reciprocal_rank(hits, "chunk_id", row["chunk_id"]),
                    "doc_rr": reciprocal_rank(hits, "doc_id", row["doc_id"]),
                }
            )
        if i % 25 == 0:
            print(f"    {method}: {i}/{len(questions)}", end="\r", flush=True)

    summary: dict = {
        "method": method,
        "k": k,
        "latency_ms_p50": statistics.median(latencies) * 1000,
        "latency_ms_p95": (
            statistics.quantiles(latencies, n=20)[18] * 1000
            if len(latencies) >= 20
            else max(latencies) * 1000
        ),
        "styles": {},
    }
    for style, rows in per_style.items():
        summary["styles"][style] = {
            "n": len(rows),
            "hit_rate": sum(r["chunk_rr"] > 0 for r in rows) / len(rows),
            "mrr": sum(r["chunk_rr"] for r in rows) / len(rows),
            "hit_rate_doc": sum(r["doc_rr"] > 0 for r in rows) / len(rows),
            "mrr_doc": sum(r["doc_rr"] for r in rows) / len(rows),
            "by_source": {
                source: {
                    "n": len(group),
                    "hit_rate": sum(r["chunk_rr"] > 0 for r in group) / len(group),
                    "mrr": sum(r["chunk_rr"] for r in group) / len(group),
                }
                for source, group in _group(rows).items()
            },
        }
    return summary


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[row["source_type"]].append(row)
    return out


def print_report(results: list[dict]) -> None:
    k = results[0]["k"]
    print(f"\n{'=' * 72}\nRetrieval evaluation (k={k})\n{'=' * 72}")

    for style in STYLES:
        print(f"\n--- {style} questions " + "-" * (52 - len(style)))
        print(
            f"{'method':<16}{'hit@k':>8}{'MRR':>8}"
            f"{'hit@k doc':>12}{'MRR doc':>10}{'p50 ms':>10}"
        )
        for result in results:
            s = result["styles"][style]
            print(
                f"{result['method']:<16}{s['hit_rate']:>8.3f}{s['mrr']:>8.3f}"
                f"{s['hit_rate_doc']:>12.3f}{s['mrr_doc']:>10.3f}"
                f"{result['latency_ms_p50']:>10.0f}"
            )

    print(f"\n--- gap between styles (hit@k literal - artist) " + "-" * 24)
    for result in results:
        gap = (
            result["styles"]["literal"]["hit_rate"]
            - result["styles"]["artist"]["hit_rate"]
        )
        print(f"{result['method']:<16}{gap:>+8.3f}")

    print(f"\n--- hit@k by source layer, artist questions " + "-" * 28)
    sources = sorted(results[0]["styles"]["artist"]["by_source"])
    print(f"{'method':<16}" + "".join(f"{s[:14]:>16}" for s in sources))
    for result in results:
        cells = "".join(
            f"{result['styles']['artist']['by_source'][s]['hit_rate']:>16.3f}"
            for s in sources
        )
        print(f"{result['method']:<16}{cells}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval")
    parser.add_argument("--ground-truth", default="data/ground_truth.jsonl", type=Path)
    parser.add_argument("--out", default="data/retrieval_eval.json", type=Path)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--methods", nargs="*", default=list(ALL_METHODS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--rewrite-cache", default="data/rewrites.json", type=Path
    )
    args = parser.parse_args()

    questions = [
        json.loads(line) for line in args.ground_truth.open(encoding="utf-8")
    ]
    if args.limit:
        questions = questions[: args.limit]
    print(f"{len(questions)} question pairs, {len(args.methods)} methods")

    client = connect()
    rewrites = (
        load_rewrites(questions, args.rewrite_cache)
        if any(m in REWRITE_METHODS for m in args.methods)
        else None
    )

    results = []
    for method in args.methods:
        started = time.monotonic()
        results.append(evaluate(client, questions, method, args.k, rewrites))
        print(f"  {method}: done in {time.monotonic() - started:.0f}s" + " " * 20)

    # Merge with any previous run rather than overwrite. `hybrid_rerank` takes
    # 23 minutes; re-measuring it to add a fast method to the same table would
    # be a waste, and keeping two result files invites comparing numbers from
    # different corpus states.
    merged: dict[str, dict] = {}
    if args.out.exists():
        merged = {
            r["method"]: r for r in json.loads(args.out.read_text(encoding="utf-8"))
        }
    merged.update({r["method"]: r for r in results})
    ordered = [merged[m] for m in ALL_METHODS if m in merged]

    print_report(ordered)
    args.out.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
