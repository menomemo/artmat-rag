"""Generate the retrieval ground-truth set.

The usual recipe -- show a chunk to an LLM, ask for questions it answers, then
check whether retrieval returns that chunk -- has a flaw that inflates the
result. The model is looking at the passage while it writes, so it reuses the
passage's vocabulary, and BM25 then scores well for a reason that will not hold
when a real person asks. Reporting a single hit-rate off such a set overstates
lexical retrieval and understates the case for anything else.

Rather than try to suppress the leakage, this generates **two questions per
chunk under opposite instructions** and reports the two populations separately:

- `literal`  -- how someone who half-remembers the document would ask it.
                Technical terms allowed. This is close to the standard recipe,
                and is the friendly case for BM25.
- `artist`   -- how the target user actually asks: by intent and outcome,
                with distinctive terms from the passage explicitly forbidden.
                BM25 has little to match on here.

The gap between the two hit-rates *is* the measurement. It says how much of a
lexical retriever's apparent quality is an artefact of how the questions were
written, and it is the honest argument for hybrid search -- not "hybrid is best
practice" but "these two question styles are both real, they favour different
retrievers, and one index has to serve both".

A chunk is a valid ground-truth target only if the question is answerable
*specifically* from it. Both prompts insist on that; questions so generic that
any passage would do are useless as relevance judgements.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import anthropic

# Question generation is a writing task, not a classification task -- the
# `artist` style in particular requires knowing how a fabricator talks. Haiku
# produces serviceable `literal` questions and noticeably flatter `artist`
# ones, which would bias the very comparison this set exists to make.
MODEL = "claude-sonnet-5"

# 1200, not 400. Sonnet 5 emits thinking blocks by default and `max_tokens`
# budgets thinking *and* output together, so 400 was enough for the two
# questions but not for the reasoning in front of them. 4 of 200 requests hit
# the ceiling and returned JSON cut off mid-string -- and because the batch
# reported them as `succeeded` (they did complete; they completed by running
# out of room), nothing flagged it until parsing failed.
MAX_TOKENS = 1200

# The evaluation must not be dominated by whichever layer happens to be
# largest. Manufacturer narrative is 51% of the corpus; sampled proportionally
# it would also be 51% of the questions, and the literature layer -- where the
# hard retrieval problems live -- would be a footnote. Sampling is stratified
# with a floor instead, then hit-rates are reported per layer as well as
# overall.
SAMPLE_PER_STRATUM = {
    ("manufacturer_datasheet", "spec"): 40,
    ("manufacturer_datasheet", "narrative"): 55,
    ("materials_science", "abstract"): 60,
    ("conservation_literature", "abstract"): 30,
    ("collection_precedent", "precedent"): 15,
}

SYSTEM_PROMPT = """\
You write evaluation questions for a retrieval system used by artists, \
fabricators, and conservators who make physical objects -- sculpture, \
installation, cast and moulded work.

You will be given one passage. Write exactly two questions, both answerable \
specifically by that passage.

QUESTION 1 -- "literal": how someone who half-remembers this document would \
look for it again. Technical terms, product names, and numbers are fine.

QUESTION 2 -- "artist": how someone working in a studio would actually ask, \
before they know what the answer is made of. Phrase it by intent, outcome, or \
problem -- what they are trying to make, avoid, or fix.

Hard constraint on QUESTION 2: do not reuse the passage's distinctive \
vocabulary. No product names, no trade names, no chemical names, no numeric \
values, no unusual technical terms lifted from the text. Ordinary material \
words (resin, mould, concrete, steel) are allowed. If the passage is about \
"Mold Star 15 pot life 50 minutes", ask how long there is to work before a \
silicone starts setting -- not about Mold Star or 50 minutes.

Both questions must be specific enough that this passage answers them and a \
randomly chosen other passage would not. Reject the temptation to write \
something broad and safe: a question any document could answer is worthless \
here.

Write plainly. No preamble, no numbering inside the strings.\
"""

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "literal": {
            "type": "string",
            "description": "Question reusing the passage's own terminology.",
        },
        "artist": {
            "type": "string",
            "description": "Intent-phrased question avoiding the passage's "
            "distinctive vocabulary.",
        },
    },
    "required": ["literal", "artist"],
    "additionalProperties": False,
}


def stratified_sample(chunks: list[dict], seed: int = 20260810) -> list[dict]:
    """Sample per (source_type, chunk_type), with a fixed seed.

    The seed is pinned so the ground-truth set is regenerable: an evaluation
    whose question set silently changes between runs cannot be used to compare
    two retrieval configurations, which is the only thing it is for.
    """
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for chunk in chunks:
        buckets[(chunk["source_type"], chunk["chunk_type"])].append(chunk)

    rng = random.Random(seed)
    sampled: list[dict] = []
    for stratum, target in SAMPLE_PER_STRATUM.items():
        pool = buckets.get(stratum, [])
        if not pool:
            print(f"  warning: no chunks in stratum {stratum}")
            continue
        take = min(target, len(pool))
        sampled.extend(rng.sample(pool, take))
        if take < target:
            print(f"  stratum {stratum}: only {take} available (wanted {target})")
    rng.shuffle(sampled)
    return sampled


def custom_id(chunk_id: str) -> str:
    """Stable batch key for a chunk_id.

    A digest rather than `hash()`: Python randomises string hashing per
    process, so a batch submitted in one run could not be collected in the
    next. The readable prefix is kept so a failed request can be traced back to
    its chunk without a lookup table.
    """
    digest = hashlib.sha1(chunk_id.encode()).hexdigest()[:12]
    readable = re.sub(r"[^a-zA-Z0-9_-]", "_", chunk_id)[:45]
    return f"{readable}-{digest}"


def build_request(chunk: dict) -> dict:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    return Request(
        custom_id=custom_id(chunk["chunk_id"]),
        params=MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            output_config={
                "format": {"type": "json_schema", "schema": QUESTION_SCHEMA}
            },
            messages=[
                {
                    "role": "user",
                    "content": f"Passage ({chunk['source_type']}):\n\n{chunk['text']}",
                }
            ],
        ),
    )


# --- leakage audit -----------------------------------------------------------

# What the audit is for: catching borrowing that hands BM25 the answer. Two
# earlier attempts got this wrong in opposite ways.
#
# A hand-written stopword list flagged `away, dark, over, painting, restore,
# varnish` -- so the list would have had to grow forever, and every word added
# to it was a judgement call made to get a nicer number.
#
# Corpus document frequency (flag terms in <2% of chunks) looked principled and
# was worse. Over a corpus that is *entirely* technical prose, ordinary English
# is rare: it flagged `actually`, `evidence`, `what`, `old`, `likely`,
# `instead`. Measuring rarity against a specialised corpus does not measure
# distinctiveness, it measures register.
#
# So the audit tests the two things that are unambiguously decisive, and
# nothing else. Domain nouns are deliberately not counted: a question about
# bronze patina has to say "patina", and calling that a violation would mean
# the only compliant question is one nobody would ask.
# "Paraloid", "Incralac", "Jesmonite" -- capitalised away from a sentence start.
PROPER_NOUN = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-Z]{2,})\b", re.M)

# "B72", "AC100", "30A", "2831" -- grade codes, which are pure lexical giveaways.
GRADE_CODE = re.compile(r"\b(?:[A-Za-z]{1,4}[-\s]?\d{1,4}[A-Za-z]?|\d{1,4}[A-Za-z])\b")

NUMBER = re.compile(r"\d+(?:\.\d+)?")


def borrowed(question: str, passage: str) -> dict[str, list[str]]:
    """Retrieval-decisive vocabulary the `artist` question took from its passage.

    Returns the two categories separately. A compliance check, not a score --
    the real measurement is the literal/artist hit-rate gap in
    `eval/retrieval.py`, and this only establishes that the gap is not an
    artefact of the generator quietly copying product names.
    """
    # Case-sensitive on both sides. Matching a capitalised passage token against
    # a lowercased question flagged `Time`, `Hardness`, `Steel`, `The` -- spec
    # labels and title words, not trade names. Someone reusing "Paraloid" writes
    # "Paraloid"; someone asking about cure time writes "time".
    names = set(PROPER_NOUN.findall(passage)) & set(PROPER_NOUN.findall(question))
    codes = set(GRADE_CODE.findall(question)) & set(GRADE_CODE.findall(passage))
    numbers = set(NUMBER.findall(question)) & set(NUMBER.findall(passage))
    return {
        "names": sorted(names),
        "values": sorted(codes | numbers),
    }


def submit(chunks: list[dict], client: anthropic.Anthropic) -> str:
    batch = client.messages.batches.create(
        requests=[build_request(c) for c in chunks]
    )
    print(f"submitted batch {batch.id} with {len(chunks)} requests")
    return batch.id


def wait(batch_id: str, client: anthropic.Anthropic, poll_s: int = 20):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            counts = batch.request_counts
            print(
                f"batch ended: {counts.succeeded} succeeded, "
                f"{counts.errored} errored, {counts.expired} expired"
            )
            return batch
        print(f"  {batch.request_counts.processing} processing ...", flush=True)
        time.sleep(poll_s)


def collect(batch_id: str, client: anthropic.Anthropic) -> dict[str, dict]:
    """Collect what parsed; report what did not.

    Every failure is per-request. An earlier version let `json.loads` raise,
    and four truncated responses destroyed the other 196 -- results that had
    already been paid for and were sitting on the server. A collector for a
    batch of independent items must never let one item abort the batch.

    Note that a truncated response still arrives with `type == "succeeded"`:
    the request did complete, it completed by exhausting `max_tokens`. Only
    `stop_reason` says so, which is why it is checked separately.
    """
    out: dict[str, dict] = {}
    truncated = unparsed = failed = 0

    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            failed += 1
            continue
        message = result.result.message
        if message.stop_reason == "max_tokens":
            truncated += 1
            continue
        text = next((b.text for b in message.content if b.type == "text"), None)
        if not text:
            unparsed += 1
            continue
        try:
            out[result.custom_id] = json.loads(text)
        except json.JSONDecodeError:
            unparsed += 1

    if truncated or unparsed or failed:
        print(
            f"  collected {len(out)}; skipped {truncated} truncated, "
            f"{unparsed} unparseable, {failed} failed"
        )
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate retrieval ground truth")
    parser.add_argument("--chunks", default="data/chunks.jsonl", type=Path)
    parser.add_argument("--out", default="data/ground_truth.jsonl", type=Path)
    parser.add_argument("--batch-id", default=None, help="resume an existing batch")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-topup", action="store_true", help="skip the retry pass")
    args = parser.parse_args()

    chunks = [json.loads(line) for line in args.chunks.open(encoding="utf-8")]
    sampled = stratified_sample(chunks)
    if args.limit:
        sampled = sampled[: args.limit]
    print(f"sampled {len(sampled)} chunks:")
    for stratum, count in Counter(
        (c["source_type"], c["chunk_type"]) for c in sampled
    ).most_common():
        print(f"    {count:>4}  {stratum[0]} / {stratum[1]}")

    client = anthropic.Anthropic()
    batch_id = args.batch_id or submit(sampled, client)
    wait(batch_id, client)
    questions = collect(batch_id, client)

    # Top-up pass. Whatever the first batch dropped -- truncation, a transient
    # error -- is resubmitted rather than left as a hole in the sample, because
    # the sample is stratified and silently losing four spec chunks quietly
    # unbalances the strata the stratification exists to protect.
    # One retry only: a chunk that fails twice is a chunk with a real problem,
    # and looping on it would burn tokens without converging.
    missing = [c for c in sampled if custom_id(c["chunk_id"]) not in questions]
    if missing and not args.no_topup:
        print(f"\ntopping up {len(missing)} missing question pairs")
        topup_id = submit(missing, client)
        wait(topup_id, client)
        questions.update(collect(topup_id, client))
        still = [c for c in sampled if custom_id(c["chunk_id"]) not in questions]
        if still:
            print(f"  {len(still)} still missing after retry; proceeding without them")

    rows, clean_names, clean_values = [], 0, 0
    for chunk in sampled:
        pair = questions.get(custom_id(chunk["chunk_id"]))
        if not pair:
            continue
        took = borrowed(pair["artist"], chunk["text"])
        clean_names += not took["names"]
        clean_values += not took["values"]
        rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "source_type": chunk["source_type"],
                "chunk_type": chunk["chunk_type"],
                "literal": pair["literal"],
                "artist": pair["artist"],
                "artist_borrowed_names": took["names"],
                "artist_borrowed_values": took["values"],
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = max(len(rows), 1)
    print(f"\nwrote {len(rows)} question pairs -> {args.out}")
    print("  artist-question compliance (no retrieval-decisive borrowing):")
    print(f"    free of proper names:  {clean_names}/{len(rows)} ({clean_names / n:.0%})")
    print(f"    free of codes/numbers: {clean_values}/{len(rows)} ({clean_values / n:.0%})")
