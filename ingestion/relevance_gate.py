"""LLM relevance gate over the OpenAlex literature layer.

Why this exists: dropping the journal whitelist was the only way to get usable
abstracts out of OpenAlex (Elsevier deposits none), but keyword search over the
whole open-access corpus admits cross-discipline collisions -- "resin
yellowing" returns dental composites, "adhesive" returns sea-urchin biology.
A `topics.field.id` filter halves the noise and cannot fix it, because the
collision is semantic, not disciplinary.

So the corpus is cleaned by asking a model, one abstract at a time, a question
keyword matching cannot express: *would this help someone making or conserving
a physical art object?*

Design notes:

- **Batch, not sync.** This is offline work with no latency requirement, so it
  goes through the Batches API at half price. ~1000 abstracts is a few cents.
- **Structured outputs.** The verdict comes back as schema-validated JSON.
  Parsing model prose with a regex fails silently on a handful of calls out of
  a thousand, and a silent failure in a corpus filter is invisible until the
  retrieval evaluation looks inexplicably bad.
- **Nothing is deleted.** The verdict is written back as a field. The noisy
  corpus stays reproducible, and the retrieval evaluation can be run with the
  gate on and off to show what it bought.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import anthropic

MODEL = "claude-haiku-4-5"

# The gate is a cheap binary judgement, so it runs on the smallest current
# model. Generation and the LLM judge use larger models -- see eval/.
MAX_TOKENS = 300

SYSTEM_PROMPT = """\
You screen scientific abstracts for a knowledge base used by artists, \
fabricators, and conservators who make and maintain physical art objects: \
sculpture, installation, and architectural artwork.

The knowledge base exists to answer questions like:
- Which casting resin yellows least outdoors?
- Can this concrete be cast in thin sections?
- Cast aluminium or stainless steel for a coastal site?
- What happens to this material after ten years outside?
- Why do museums avoid a particular adhesive?

An abstract is RELEVANT when its findings would inform one of those decisions \
-- typically work on polymers, resins, silicones, elastomers, cementitious \
composites, metals, adhesives, coatings, or pigments, studied for ageing, \
weathering, degradation, mechanical behaviour, conservation, or fabrication.

An abstract is NOT RELEVANT when it merely shares vocabulary with that domain. \
The most common false positives, all of which must be rejected:
- dental and biomedical materials ("resin composite" in a clinical context)
- food, agricultural, or pharmaceutical science
- water treatment and dye removal ("yellow" as a dye name, "adhesive" as a \
biological adhesion mechanism)
- marine biology, tissue adhesion, and cell culture
- pure synthesis or characterisation with no bearing on how a material \
behaves in a made object

Judge the abstract's actual subject, not its keywords. A study of epoxy \
photodegradation is relevant even if it never mentions art; a study of dental \
resin yellowing is not relevant even though it is about resin yellowing.\
"""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {
            "type": "boolean",
            "description": "True if the abstract informs an art-fabrication "
            "or conservation decision.",
        },
        "domain": {
            "type": "string",
            "enum": [
                "polymers_resins",
                "cementitious",
                "metals",
                "adhesives_coatings",
                "pigments_surfaces",
                "conservation_practice",
                "off_topic",
            ],
            "description": "Primary subject area; 'off_topic' when not relevant.",
        },
        "reason": {
            "type": "string",
            "description": "One sentence justifying the verdict.",
        },
    },
    "required": ["relevant", "domain", "reason"],
    "additionalProperties": False,
}


def doc_key(work: dict) -> str:
    """Batch `custom_id` for a work record.

    Prefers the stored `doc_id`, but derives one when absent: records written
    before `doc_id` was serialised explicitly are already on disk, and the
    OpenAlex daily quota makes re-harvesting them expensive.
    """
    doc_id = work.get("doc_id")
    if not doc_id:
        doc_id = "openalex:" + str(work["openalex_id"]).rsplit("/", 1)[-1]
    return doc_id.replace(":", "_")[:64]


def build_request(work: dict) -> dict:
    """One batch request for one abstract."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    abstract = work["abstract"][:4000]
    user_content = (
        f"Journal: {work.get('journal') or 'unknown'}\n"
        f"Year: {work.get('year') or 'unknown'}\n"
        f"Title: {work['title']}\n\n"
        f"Abstract:\n{abstract}"
    )

    return Request(
        custom_id=doc_key(work),
        params=MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            # No `cache_control` here, deliberately. The screening brief is
            # identical across all ~2000 requests, so caching looks like an
            # obvious win -- but Haiku 4.5's minimum cacheable prefix is 4096
            # tokens and this prompt measures 354. Below the threshold the
            # marker is silently ignored: no error, and
            # `cache_creation_input_tokens` / `cache_read_input_tokens` both
            # stay 0 (verified empirically on two identical calls).
            #
            # Leaving a decorative marker in place would be worse than omitting
            # it, because the cost model built on top of it would be wrong by
            # ~30% and nothing would ever say so.
            system=SYSTEM_PROMPT,
            output_config={
                "format": {"type": "json_schema", "schema": VERDICT_SCHEMA}
            },
            messages=[{"role": "user", "content": user_content}],
        ),
    )


def submit(works: list[dict], client: anthropic.Anthropic) -> str:
    requests = [build_request(work) for work in works]
    batch = client.messages.batches.create(requests=requests)
    print(f"submitted batch {batch.id} with {len(requests)} requests")
    return batch.id


def wait(batch_id: str, client: anthropic.Anthropic, poll_s: int = 30):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            counts = batch.request_counts
            print(
                f"batch ended: {counts.succeeded} succeeded, "
                f"{counts.errored} errored, {counts.expired} expired"
            )
            return batch
        print(
            f"  {batch.processing_status}: "
            f"{batch.request_counts.processing} still processing",
            flush=True,
        )
        time.sleep(poll_s)


def collect_verdicts(batch_id: str, client: anthropic.Anthropic) -> dict[str, dict]:
    """Map custom_id -> verdict. Results arrive in arbitrary order."""
    verdicts: dict[str, dict] = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            # A failed screen is recorded rather than silently dropped: an
            # unscreened abstract should stay in the corpus, not vanish.
            verdicts[result.custom_id] = {
                "relevant": True,
                "domain": "off_topic",
                "reason": f"screening failed ({result.result.type}); kept by default",
                "screened": False,
            }
            continue
        text = next(
            (b.text for b in result.result.message.content if b.type == "text"), "{}"
        )
        verdict = json.loads(text)
        verdict["screened"] = True
        verdicts[result.custom_id] = verdict
    return verdicts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Screen OpenAlex abstracts")
    parser.add_argument("--in", dest="infile", default="data/raw/openalex.jsonl", type=Path)
    parser.add_argument("--out", default="data/raw/openalex_screened.jsonl", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="screen only N (smoke test)")
    parser.add_argument(
        "--source-type",
        default=None,
        help="screen only this layer. The conservation-journal whitelist "
        "measured 98%% clean (2 rejects in 89), so screening it buys almost "
        "nothing; the broad materials_science recall is where the noise is.",
    )
    parser.add_argument("--batch-id", default=None, help="resume an existing batch")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    all_works = [json.loads(line) for line in args.infile.open(encoding="utf-8")]

    # Records outside the screened layer pass through unscreened rather than
    # being dropped -- the output file must stay a complete corpus.
    if args.source_type:
        works = [w for w in all_works if w.get("source_type") == args.source_type]
        passthrough = [w for w in all_works if w.get("source_type") != args.source_type]
        print(
            f"screening {len(works)} × {args.source_type}; "
            f"{len(passthrough)} pass through unscreened"
        )
    else:
        works, passthrough = all_works, []

    if args.limit:
        works = works[: args.limit]

    batch_id = args.batch_id or submit(works, client)
    wait(batch_id, client)
    verdicts = collect_verdicts(batch_id, client)

    kept = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for work in passthrough:
            work["relevance"] = {
                "relevant": True,
                "domain": "conservation_practice",
                "reason": "high-precision journal whitelist; not screened",
                "screened": False,
            }
            handle.write(json.dumps(work, ensure_ascii=False) + "\n")

        for work in works:
            key = doc_key(work)
            work["relevance"] = verdicts.get(
                key,
                {"relevant": True, "domain": "off_topic",
                 "reason": "no verdict returned; kept by default", "screened": False},
            )
            kept += bool(work["relevance"]["relevant"])
            handle.write(json.dumps(work, ensure_ascii=False) + "\n")

    print(f"screened {len(works)} works -> {args.out}")
    print(f"  relevant: {kept}  filtered: {len(works) - kept}")
