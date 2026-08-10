"""LLM-as-judge over the four answer conditions.

Two evaluations run here, deliberately separated:

**Deterministic, in Python.** Whether a cited `chunk_id` exists in the corpus,
and whether it was actually among the passages that answer was shown. This is a
string lookup with a right answer, and asking a model to do it would introduce
noise into the one number that has none. It is also the only check that catches
the specific failure this project claims to prevent -- a citation that looks
authoritative and refers to nothing.

**Model judgement, by Opus.** Everything that is genuinely a judgement call:
whether claims are supported, whether vendor and conservation voices are kept
distinct, whether the answer is usable by someone standing at a bench.

Design decisions that matter for the result:

- **Every answer is judged against the same retrieved context, including
  `no_context`.** The no-retrieval condition is scored on groundedness against
  passages it never saw. That is not unfair, it is the measurement: the claim
  under test is that an unaided model produces fluent material advice that the
  literature does not support, and this is what makes it a number.
- **The judge is a different model from the generator.** Sonnet writes, Opus
  scores. A model grading its own output rates its own habits highly.
- **The judge never sees which variant produced an answer.** Labels would let
  it reward the prompt it finds most impressive rather than the answer.
- **Truncated answers are excluded, not scored.** A cut-off answer measures the
  token budget, not the prompt.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import anthropic

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-5")

# Generous for the same reason as in rag/generate.py: thinking tokens and
# output share this budget, and a truncated verdict is an unusable verdict.
MAX_TOKENS = 2000

CHUNK_ID = re.compile(r"\b((?:smoothon|openalex|tate):[A-Za-z0-9._-]+#[a-z]+(?:-\d+)?)\b")

JUDGE_SYSTEM = """\
You are evaluating answers given to artists, fabricators, and conservators who \
make physical objects. You will see a question, the reference passages \
retrieved for it, and one answer.

The answer may or may not have been written using those passages. Judge it \
against them regardless: the passages are the evidence available on this \
topic, and a claim they do not support is unsupported whether or not the \
writer had them.

Score four dimensions, 1 to 5.

**grounded** — are the answer's factual claims supported by the passages?
 5: every substantive claim traceable to a passage
 3: broadly consistent, but claims material specifics the passages do not contain
 1: confident specifics (products, numbers, timescales) with no basis in the passages

**source_discrimination** — does the answer distinguish *kinds* of evidence?
A manufacturer's accelerated-ageing claim and a conservation study of a \
decade-old object are different sorts of knowledge. Does the reader learn \
which is which?
 5: attributes claims to source type and explains what each is positioned to know
 3: attributes some claims, or attributes without explaining the difference
 1: presents everything in one undifferentiated voice

**handles_conflict** — where the passages disagree, or where evidence is \
missing, does the answer say so?
 5: names disagreements and gaps explicitly, explains why sources differ
 3: hedges vaguely, or notes a gap without explaining it
 1: presents a single confident answer where the evidence is mixed or absent
 If the passages genuinely do not conflict and have no notable gap, score 4 \
for an answer that correctly says the evidence is consistent.

**usable** — could someone act on this at a workbench?
 5: specific, ordered, honest about what it cannot tell them
 3: correct but vague, or buried in qualification
 1: unusable — evasive, or so hedged it gives no purchase

Judge what is there. Do not reward length, confident tone, or formatting.\
"""

# `enum`, not `minimum`/`maximum`. Structured outputs rejects numeric range
# keywords outright -- "For 'integer' type, properties maximum, minimum are not
# supported" -- and it rejects them at request time, so all 100 requests in the
# first judging batch errored identically. An explicit enum both passes
# validation and constrains the range more tightly than a range check would.
SCORE = {"type": "integer", "enum": [1, 2, 3, 4, 5]}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": SCORE,
        "source_discrimination": SCORE,
        "handles_conflict": SCORE,
        "usable": SCORE,
        "unsupported_claims": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific claims in the answer that the passages do "
            "not support. Quote them briefly. Empty if none.",
        },
        "reason": {"type": "string", "description": "Two sentences, no more."},
    },
    "required": [
        "grounded",
        "source_discrimination",
        "handles_conflict",
        "usable",
        "unsupported_claims",
        "reason",
    ],
    "additionalProperties": False,
}

DIMENSIONS = ("grounded", "source_discrimination", "handles_conflict", "usable")


def citation_audit(answer_text: str, shown_ids: set[str], corpus_ids: set[str]) -> dict:
    """Deterministic citation check.

    Three outcomes, and the third is the one worth catching:
      valid     -- cited, and was in this answer's context
      stale     -- exists in the corpus, but was not retrieved for this answer
      fabricated-- matches the id format and does not exist at all
    """
    cited = set(CHUNK_ID.findall(answer_text))
    return {
        "n_cited": len(cited),
        "valid": sorted(cited & shown_ids),
        "stale": sorted((cited & corpus_ids) - shown_ids),
        "fabricated": sorted(cited - corpus_ids),
    }


def build_judge_request(record: dict) -> dict:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    content = (
        f"QUESTION\n{record['question']}\n\n"
        f"REFERENCE PASSAGES\n{record['context']}\n\n"
        f"ANSWER TO JUDGE\n{record['answer']}"
    )
    return Request(
        custom_id=record["custom_id"],
        params=MessageCreateParamsNonStreaming(
            model=JUDGE_MODEL,
            max_tokens=MAX_TOKENS,
            system=JUDGE_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
            messages=[{"role": "user", "content": content}],
        ),
    )


def wait(batch_id: str, client: anthropic.Anthropic, poll_s: int = 20):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            counts = batch.request_counts
            print(
                f"  batch ended: {counts.succeeded} succeeded, "
                f"{counts.errored} errored, {counts.expired} expired"
            )
            return batch
        print(f"    {batch.request_counts.processing} processing ...", flush=True)
        time.sleep(poll_s)


def collect(batch_id: str, client: anthropic.Anthropic) -> dict[str, dict]:
    """Same tolerance as every other collector here: one bad item must not
    destroy the batch, and `succeeded` does not mean `complete`."""
    out: dict[str, dict] = {}
    skipped = 0
    # Reasons, not just a count. The first version reported "skipped 100
    # unusable verdicts" and threw away the API's message, which said exactly
    # what was wrong with the request schema. A failure counter that discards
    # the failure is worse than no counter: it looks like observability.
    reasons: Counter[str] = Counter()
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            skipped += 1
            detail = getattr(getattr(result.result, "error", None), "error", None)
            reasons[
                f"{result.result.type}: {getattr(detail, 'message', 'no detail')}"[:160]
            ] += 1
            continue
        message = result.result.message
        if message.stop_reason == "max_tokens":
            skipped += 1
            reasons["truncated (max_tokens)"] += 1
            continue
        text = next((b.text for b in message.content if b.type == "text"), None)
        if not text:
            skipped += 1
            reasons["no text block"] += 1
            continue
        try:
            out[result.custom_id] = json.loads(text)
        except json.JSONDecodeError:
            skipped += 1
            reasons["unparseable JSON"] += 1
    if skipped:
        print(f"  skipped {skipped} unusable verdicts:")
        for reason, count in reasons.most_common(5):
            print(f"    {count:>4}  {reason}")
    return out


def summarise(records: list[dict], verdicts: dict[str, dict]) -> dict:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        verdict = verdicts.get(record["custom_id"])
        if verdict:
            by_variant[record["variant"]].append({**record, **verdict})

    summary = {}
    for variant, rows in by_variant.items():
        n = len(rows)
        summary[variant] = {
            "n": n,
            **{d: sum(r[d] for r in rows) / n for d in DIMENSIONS},
            "unsupported_claims_per_answer": sum(
                len(r["unsupported_claims"]) for r in rows
            )
            / n,
            "citations_per_answer": sum(r["citations"]["n_cited"] for r in rows) / n,
            "answers_with_fabricated_citation": sum(
                bool(r["citations"]["fabricated"]) for r in rows
            ),
            "answers_with_stale_citation": sum(
                bool(r["citations"]["stale"]) for r in rows
            ),
        }
    return summary


def print_report(summary: dict, order: list[str]) -> None:
    print(f"\n{'=' * 78}\nLLM-as-judge ({JUDGE_MODEL}), 1-5 scale\n{'=' * 78}")
    print(
        f"{'variant':<14}{'n':>4}{'grounded':>10}{'sources':>9}"
        f"{'conflict':>10}{'usable':>8}{'unsup/ans':>11}{'cites/ans':>11}"
    )
    for variant in order:
        s = summary.get(variant)
        if not s:
            continue
        print(
            f"{variant:<14}{s['n']:>4}{s['grounded']:>10.2f}"
            f"{s['source_discrimination']:>9.2f}{s['handles_conflict']:>10.2f}"
            f"{s['usable']:>8.2f}{s['unsupported_claims_per_answer']:>11.2f}"
            f"{s['citations_per_answer']:>11.2f}"
        )
    print(f"\n{'variant':<14}{'fabricated citations':>22}{'stale citations':>18}")
    for variant in order:
        s = summary.get(variant)
        if not s:
            continue
        print(
            f"{variant:<14}{s['answers_with_fabricated_citation']:>22}"
            f"{s['answers_with_stale_citation']:>18}"
        )


# --- generation pass ---------------------------------------------------------


def build_generation_request(question: str, context: str, variant: str, cid: str) -> dict:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    from rag.generate import MAX_TOKENS as GEN_MAX_TOKENS
    from rag.generate import MODEL as GEN_MODEL
    from rag.generate import PROMPTS

    content = (
        question
        if variant == "no_context"
        else f"{context}\n\n---\n\nQuestion: {question}"
    )
    return Request(
        custom_id=cid,
        params=MessageCreateParamsNonStreaming(
            model=GEN_MODEL,
            max_tokens=GEN_MAX_TOKENS,
            system=PROMPTS[variant],
            messages=[{"role": "user", "content": content}],
        ),
    )


def collect_answers(batch_id: str, client: anthropic.Anthropic) -> dict[str, dict]:
    out: dict[str, dict] = {}
    truncated = 0
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            continue
        message = result.result.message
        text = "\n".join(b.text for b in message.content if b.type == "text")
        if message.stop_reason == "max_tokens":
            truncated += 1
            continue
        out[result.custom_id] = {
            "text": text.strip(),
            "output_tokens": message.usage.output_tokens,
        }
    if truncated:
        print(f"  {truncated} answers truncated and excluded from scoring")
    return out


if __name__ == "__main__":
    import argparse
    import hashlib

    from rag.generate import PROMPTS, format_context
    from rag.index import connect
    from rag.rewrite import Rewrite, search_rewritten

    parser = argparse.ArgumentParser(description="Generate and judge answers")
    parser.add_argument("--ground-truth", default="data/ground_truth.jsonl", type=Path)
    parser.add_argument("--rewrite-cache", default="data/rewrites.json", type=Path)
    parser.add_argument("--answers", default="data/answers.json", type=Path)
    parser.add_argument("--out", default="data/judge_eval.json", type=Path)
    parser.add_argument("--n", type=int, default=25, help="questions to evaluate")
    parser.add_argument("--k", type=int, default=8, help="passages per answer")
    args = parser.parse_args()

    variants = list(PROMPTS)
    questions = [json.loads(l) for l in args.ground_truth.open(encoding="utf-8")]
    # The `artist` phrasing only. `literal` questions are a retrieval probe --
    # nobody walks into a studio and asks a question phrased from the document
    # they are trying to find.
    sample = questions[: args.n]
    rewrites = json.loads(args.rewrite_cache.read_text(encoding="utf-8"))

    print(f"retrieving context for {len(sample)} questions (k={args.k})")
    qdrant = connect()
    corpus_ids = {
        json.loads(l)["chunk_id"] for l in open("data/chunks.jsonl", encoding="utf-8")
    }

    records = []
    for row in sample:
        question = row["artist"]
        cached = rewrites.get(question)
        rw = Rewrite(
            original=question,
            rewritten=(cached or {}).get("rewritten", question),
            terms=(cached or {}).get("terms", []),
            used=bool(cached),
        )
        hits = search_rewritten(qdrant, rw, limit=args.k, rerank=False)
        context = format_context(hits)
        shown = {h.chunk_id for h in hits}
        digest = hashlib.sha1(question.encode()).hexdigest()[:10]
        for variant in variants:
            records.append(
                {
                    "custom_id": f"{variant}-{digest}",
                    "question": question,
                    "variant": variant,
                    "context": context,
                    "shown_ids": sorted(shown),
                }
            )

    client = anthropic.Anthropic()

    answers: dict[str, dict] = {}
    if args.answers.exists():
        answers = json.loads(args.answers.read_text(encoding="utf-8"))
        print(f"  reusing {len(answers)} cached answers")
    missing = [r for r in records if r["custom_id"] not in answers]
    if missing:
        print(f"generating {len(missing)} answers ({len(variants)} variants)")
        gen_batch = client.messages.batches.create(
            requests=[
                build_generation_request(
                    r["question"], r["context"], r["variant"], r["custom_id"]
                )
                for r in missing
            ]
        )
        print(f"  batch {gen_batch.id}")
        wait(gen_batch.id, client)
        answers.update(collect_answers(gen_batch.id, client))
        args.answers.write_text(
            json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    records = [r for r in records if r["custom_id"] in answers]
    for record in records:
        record["answer"] = answers[record["custom_id"]]["text"]
        record["citations"] = citation_audit(
            record["answer"], set(record["shown_ids"]), corpus_ids
        )

    print(f"judging {len(records)} answers with {JUDGE_MODEL}")
    judge_batch = client.messages.batches.create(
        requests=[build_judge_request(r) for r in records]
    )
    print(f"  batch {judge_batch.id}")
    wait(judge_batch.id, client)
    verdicts = collect(judge_batch.id, client)

    summary = summarise(records, verdicts)
    print_report(summary, variants)
    args.out.write_text(
        json.dumps(
            {"summary": summary, "n_questions": len(sample), "k": args.k},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")
