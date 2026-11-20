"""Pipeline vs agentic retrieval, on the same questions, scored the same way.

The question this answers is not "does tool calling work" -- it does -- but
"is it worth what it costs *on this corpus*". So both arms are measured on the
things that would decide that:

- **evidence**: did the answer's passages include the chunk the question was
  written from (target hit), how many passages, how many of the four layers;
- **answer quality**: the same blind Opus judge and deterministic citation
  audit as `eval/judge.py`;
- **cost and latency**: every model call counted, including the pipeline's
  Haiku rewrite, because leaving it out would flatter the cheaper arm.

Design decisions that shape the result:

- **Both arms ship-configured**: arbitrated prompt, extended thinking off, the
  same generator model. The pipeline's rewrite + hybrid k=8 is its UI default.
- **One reference per question.** The judge scores both answers against the
  union of the passages either arm retrieved. Judging each against only what
  it saw would let an arm that searched less look more grounded for having
  less to be checked against -- and "did it find the evidence that exists" is
  part of what is being compared.
- **Two question sets, reported separately.** The 25 ground-truth questions
  were each generated from one chunk, so they test whether the target is
  found, but rarely need more than one layer. The 5 UI examples are the
  cross-layer questions the agent exists for; they have no target, only
  judge scores. Averaging the two would hide which kind of question the
  agent helps with.
- **The agent's passages are not capped at 8.** It can see up to
  `MAX_SEARCHES * PER_SEARCH`. That is the honest comparison of the two
  systems as built, and the passage count is reported next to every quality
  number so nobody reads "more passages" as "better retrieval".
"""

from __future__ import annotations

import hashlib
import json
import statistics as stats
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import anthropic

import rag.env  # noqa: F401
from app.db import estimate_cost
from eval.judge import JUDGE_MODEL, DIMENSIONS, build_judge_request, citation_audit, collect, wait
from rag.agent import AGENT_VERSION, MAX_SEARCHES, PER_SEARCH, run_agent
from rag.generate import MODEL as GEN_MODEL
from rag.generate import format_context, generate
from rag.index import COLLECTION, connect
from rag.rewrite import MODEL as REWRITE_MODEL
from rag.rewrite import rewrite, search_rewritten

EXAMPLES = [
    "will my cast resin sculpture go yellow if I put it outdoors",
    "how long can I work with the silicone before it starts to set",
    "can I cast concrete thin enough for a wall panel that won't crack",
    "cast aluminium or stainless steel for a piece by the sea",
    "why do conservators avoid certain adhesives on artworks",
]
ARMS = ("pipeline", "agentic")


def corpus_ids(qdrant) -> set[str]:
    """Every chunk id in the index. Read from Qdrant rather than
    `data/chunks.jsonl`, which is regenerable but not always on disk -- and
    the index is what answers were actually retrieved from."""
    ids, offset = set(), None
    while True:
        points, offset = qdrant.scroll(
            COLLECTION, limit=1000, offset=offset, with_payload=["chunk_id"], with_vectors=False
        )
        ids.update(p.payload["chunk_id"] for p in points)
        if offset is None:
            return ids


def run_pipeline(question: str, qdrant, client) -> dict:
    started = time.perf_counter()
    rw = rewrite(question, client)
    hits16 = search_rewritten(qdrant, rw, limit=16, rerank=False)
    # RRF ranks do not depend on the final limit, so the first 8 of a k=16
    # query are exactly the k=8 result the UI shows. The other 8 cost nothing
    # and give the control the agent's passage count has to be compared with.
    hits = hits16[:8]
    answer = generate(question, hits, "arbitrated", client, thinking=False)
    return {
        "text": answer.text,
        "hits": [asdict(h) for h in hits],
        "k16": [[h.chunk_id, h.source_type] for h in hits16],
        "ms": int((time.perf_counter() - started) * 1000),
        "cost_usd": estimate_cost(GEN_MODEL, answer.input_tokens, answer.output_tokens)
        + estimate_cost(REWRITE_MODEL, rw.input_tokens, rw.output_tokens),
        "input_tokens": answer.input_tokens + rw.input_tokens,
        "output_tokens": answer.output_tokens + rw.output_tokens,
        "truncated": answer.truncated,
        "searches": [],
        "turns": 1,
        "budget_exhausted": False,
    }


def run_agentic(question: str, qdrant, client) -> dict:
    started = time.perf_counter()
    answer = run_agent(question, qdrant, client, thinking=False)
    return {
        "text": answer.text,
        "hits": [asdict(h) for h in answer.hits],
        "ms": int((time.perf_counter() - started) * 1000),
        "cost_usd": estimate_cost(GEN_MODEL, answer.input_tokens, answer.output_tokens),
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
        "truncated": answer.truncated,
        "searches": [asdict(c) for c in answer.searches],
        "turns": answer.turns,
        "budget_exhausted": answer.budget_exhausted,
    }


def k16_control(question: str, qdrant, client) -> list:
    """Retrieval only, for a pipeline run reused from an earlier eval that
    predates the control. A fresh Haiku rewrite, so its top 8 may differ
    slightly from the reused answer's passages; reported as retrieval only."""
    rw = rewrite(question, client)
    return [[h.chunk_id, h.source_type] for h in search_rewritten(qdrant, rw, limit=16, rerank=False)]


def run_both(item: dict, qdrant, client, reused: dict | None = None) -> dict:
    out = dict(item)
    arms = [("pipeline", run_pipeline), ("agentic", run_agentic)]
    if reused and "error" not in reused.get("pipeline", {"error": 1}):
        out["pipeline"] = dict(reused["pipeline"], reused=True)
        if "k16" not in out["pipeline"]:
            out["pipeline"]["k16"] = k16_control(item["question"], qdrant, client)
        arms = arms[1:]
    for arm, fn in arms:
        for attempt in range(3):
            try:
                out[arm] = fn(item["question"], qdrant, client)
                break
            except anthropic.APIError as exc:  # rate limits under concurrency
                if attempt == 2:
                    out[arm] = {"error": str(exc)[:300]}
                time.sleep(10 * (attempt + 1))
    return out


def reference_context(run: dict) -> tuple[str, set[str]]:
    from rag.search import Hit

    union: dict[str, dict] = {}
    for arm in ARMS:
        for h in run[arm].get("hits", []):
            union.setdefault(h["chunk_id"], h)
    return format_context([Hit(**h) for h in union.values()]), set(union)


def arm_metrics(run: dict, arm: dict, corpus: set[str]) -> dict:
    shown = {h["chunk_id"] for h in arm["hits"]}
    return {
        "target_hit": (run["target"] in shown) if run.get("target") else None,
        "n_passages": len(shown),
        "n_layers": len({h["source_type"] for h in arm["hits"]}),
        "n_searches": len(arm["searches"]),
        "filtered_searches": sum(bool(s["source_types"]) for s in arm["searches"]),
        "citations": citation_audit(arm["text"], shown, corpus),
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(stats.mean(xs), 3) if xs else None


def summarise(runs: list[dict], verdicts: dict) -> dict:
    out = {}
    for subset in ("ground_truth", "examples"):
        rows = [r for r in runs if r["set"] == subset]
        block = {}
        k16 = [r for r in rows if r["pipeline"].get("k16") and r.get("target")]
        block["pipeline_k16"] = {
            "target_hit_rate": mean([r["target"] in {c for c, _ in r["pipeline"]["k16"]} for r in k16]),
            "layers": mean([len({l for _, l in r["pipeline"]["k16"]}) for r in rows if r["pipeline"].get("k16")]),
        }
        for arm in ARMS:
            ok = [r for r in rows if "error" not in r[arm] and not r[arm]["truncated"]]
            m = [r[f"{arm}_metrics"] for r in ok]
            v = [verdicts.get(f"{arm}-{r['id']}") for r in ok]
            v = [x for x in v if x]
            block[arm] = {
                "n": len(ok),
                "n_judged": len(v),
                "target_hit_rate": mean([x["target_hit"] for x in m]),
                "passages": mean([x["n_passages"] for x in m]),
                "layers": mean([x["n_layers"] for x in m]),
                "searches": mean([x["n_searches"] for x in m]),
                "searches_using_layer_filter": sum(x["filtered_searches"] for x in m),
                "budget_exhausted": sum(r[arm]["budget_exhausted"] for r in ok),
                **{d: mean([x[d] for x in v]) for d in DIMENSIONS},
                "unsupported_claims": mean([len(x["unsupported_claims"]) for x in v]),
                "cited_per_answer": mean([x["citations"]["n_cited"] for x in m]),
                "fabricated_citations": sum(bool(x["citations"]["fabricated"]) for x in m),
                "stale_citations": sum(bool(x["citations"]["stale"]) for x in m),
                "cost_usd": mean([r[arm]["cost_usd"] for r in ok]),
                "latency_s_median": round(stats.median(r[arm]["ms"] for r in ok) / 1000, 1) if ok else None,
                "input_tokens": mean([r[arm]["input_tokens"] for r in ok]),
                "output_tokens": mean([r[arm]["output_tokens"] for r in ok]),
                "truncated": sum(r[arm].get("truncated", False) for r in rows if "error" not in r[arm]),
                "errors": sum("error" in r[arm] for r in rows),
            }
        # Paired: per question, which arm the judge scored higher overall.
        wins = defaultdict(int)
        for r in rows:
            a, b = verdicts.get(f"pipeline-{r['id']}"), verdicts.get(f"agentic-{r['id']}")
            if not (a and b):
                continue
            sa, sb = sum(a[d] for d in DIMENSIONS), sum(b[d] for d in DIMENSIONS)
            wins["agentic" if sb > sa else "pipeline" if sa > sb else "tie"] += 1
        block["paired_judge_wins"] = dict(wins)
        out[subset] = block
    return out


def print_report(summary: dict) -> None:
    keys = [
        "n_judged", "target_hit_rate", "passages", "layers", "searches",
        *DIMENSIONS, "unsupported_claims", "cited_per_answer",
        "fabricated_citations", "cost_usd", "latency_s_median", "input_tokens",
    ]
    for subset, block in summary.items():
        print(f"\n== {subset} ==")
        print(f"{'':<24}{'pipeline':>12}{'agentic':>12}")
        for k in keys:
            print(f"{k:<24}{str(block['pipeline'][k]):>12}{str(block['agentic'][k]):>12}")
        print(f"{'paired judge wins':<24}{block['paired_judge_wins']}")
        print(f"{'pipeline k=16 control':<24}{block['pipeline_k16']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare pipeline and agentic retrieval")
    parser.add_argument("--n", type=int, default=25, help="ground-truth questions")
    parser.add_argument("--runs", default=f"data/agentic_runs_{AGENT_VERSION.split('-')[0]}.json", type=Path)
    parser.add_argument("--out", default=f"data/agentic_eval_{AGENT_VERSION.split('-')[0]}.json", type=Path)
    parser.add_argument("--reuse-pipeline", type=Path,
                        help="runs file whose pipeline answers to reuse instead of regenerating")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the Opus batch: retrieval, citation, cost and latency only")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    rows = [json.loads(l) for l in open("data/ground_truth.jsonl", encoding="utf-8")][: args.n]
    items = [
        {"set": "ground_truth", "question": r["artist"], "target": r["chunk_id"]} for r in rows
    ] + [{"set": "examples", "question": q, "target": None} for q in EXAMPLES]
    for item in items:
        item["id"] = hashlib.sha1(item["question"].encode()).hexdigest()[:10]

    qdrant = connect()
    client = anthropic.Anthropic(max_retries=5)
    corpus = corpus_ids(qdrant)
    print(f"{len(corpus)} chunks in index; {len(items)} questions; "
          f"agent budget {MAX_SEARCHES} x {PER_SEARCH}")

    runs: dict[str, dict] = {}
    if args.runs.exists():
        runs = {r["id"]: r for r in json.loads(args.runs.read_text(encoding="utf-8"))}
        print(f"  reusing {len(runs)} cached runs")
    reused = {}
    if args.reuse_pipeline:
        reused = {r["id"]: r for r in json.loads(args.reuse_pipeline.read_text(encoding="utf-8"))}
        print(f"  reusing pipeline answers for {len(reused)} questions from {args.reuse_pipeline}")
    todo = [i for i in items if i["id"] not in runs
            or any("error" in runs[i["id"]].get(a, {"error": 1}) for a in ARMS)]
    if todo:
        print(f"running {len(todo)} questions through both arms ({args.workers} workers)")
        with ThreadPoolExecutor(args.workers) as pool:
            for n, run in enumerate(pool.map(lambda i: run_both(i, qdrant, client, reused.get(i["id"])), todo), 1):
                runs[run["id"]] = run
                print(f"  {n}/{len(todo)}", end="\r", flush=True)
        args.runs.write_text(json.dumps(list(runs.values()), ensure_ascii=False, indent=1), encoding="utf-8")
        print()

    runs_list = [runs[i["id"]] for i in items]
    records = []
    for run in runs_list:
        if any("error" in run[a] for a in ARMS):
            continue
        context, _ = reference_context(run)
        for arm in ARMS:
            run[f"{arm}_metrics"] = arm_metrics(run, run[arm], corpus)
            if not run[arm]["truncated"]:
                records.append({
                    "custom_id": f"{arm}-{run['id']}",
                    "question": run["question"],
                    "context": context,
                    "answer": run[arm]["text"],
                })

    verdicts: dict = {}
    if args.no_judge:
        print("--no-judge: skipping the Opus batch")
    else:
        print(f"judging {len(records)} answers")
        batch = client.messages.batches.create(requests=[build_judge_request(r) for r in records])
        print(f"  batch {batch.id}")
        wait(batch.id, client)
        verdicts = collect(batch.id, client)

    summary = summarise([r for r in runs_list if all("error" not in r[a] for a in ARMS)], verdicts)
    print_report(summary)
    per_question = [
        {
            "set": r["set"], "question": r["question"],
            **{f"{a}_{k}": r[f"{a}_metrics"][k] for a in ARMS
               for k in ("target_hit", "n_passages", "n_layers", "n_searches")},
            **{f"{a}_judge": {d: verdicts.get(f"{a}-{r['id']}", {}).get(d) for d in DIMENSIONS}
               for a in ARMS},
            "agentic_queries": [s["query"] for s in r["agentic"]["searches"]],
        }
        for r in runs_list if all(f"{a}_metrics" in r for a in ARMS)
    ]
    args.out.write_text(json.dumps({
        "config": {"agent_version": AGENT_VERSION, "generator": GEN_MODEL,
                   "judge": None if args.no_judge else JUDGE_MODEL,
                   "pipeline_reused_from": str(args.reuse_pipeline) if args.reuse_pipeline else None,
                   "agent_max_searches": MAX_SEARCHES, "agent_per_search": PER_SEARCH,
                   "pipeline_k": 8, "thinking": False},
        "summary": summary,
        "per_question": per_question,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
