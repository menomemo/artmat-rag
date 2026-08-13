"""Answer generation, in three prompt variants that exist to be compared.

The variants are not three phrasings of the same instruction. They encode three
different beliefs about what this system is for, and the evaluation is meant to
show that the choice has consequences:

- **plain** -- the default RAG prompt. Answer from the context. This is the
  control: if the elaborate variants do not beat it, they are decoration.
- **sourced** -- every claim must carry the layer it came from. Costs tokens
  and fluency; buys a reader the ability to check.
- **arbitrated** -- sourced, plus an explicit instruction to surface
  disagreement between layers rather than average it. This is the project's
  thesis stated as a prompt: a datasheet's "UV resistant" (500 hours of
  accelerated weathering) and a conservation paper's "yellows badly" (a real
  object after ten winters) are both true, and flattening them into one
  confident sentence is the failure this system exists to prevent.

A fourth condition, `no_context`, runs the same question with no retrieval at
all. It is not a prompt variant -- it is the control that shows what RAG is
buying, and it is what makes the claim in the problem statement ("fluent,
plausible, unciteable material advice") a measurement rather than an assertion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import anthropic

from rag.route import COMPLEX_MODEL
from rag.search import Hit

MODEL = COMPLEX_MODEL

# 4000, and the reason is worth writing down because this project hit it twice.
# Sonnet 5 emits thinking blocks, and `max_tokens` budgets thinking *and*
# visible output together. At 1500 the model spent most of the budget
# reasoning and the answer was cut mid-sentence -- once after two sentences,
# once after 3.8 kB, because thinking length varies per call.
#
# The failure is quiet by construction: the response arrives well-formed, and
# only `stop_reason` distinguishes "finished" from "ran out of room". So the
# budget is generous *and* truncation is recorded on the Answer rather than
# trusted not to happen.
MAX_TOKENS = 4000

# Extended thinking is on by default for this model and costs 15 seconds before
# the first visible character: measured 18.3 s to first token with it, 3.3 s
# without, on the same question and context. Output also drops from ~2,700 to
# ~1,600 tokens, which for a workshop reader is closer to right.
#
# The interactive path therefore disables it and the offline evaluation does
# not. That is a real difference between what was measured and what ships, and
# it is recorded in STATUS.md rather than smoothed over -- the judge scores in
# `eval/judge.py` describe the thinking-enabled configuration.
THINKING_DISABLED = {"type": "disabled"}


def thinking_config(
    enabled: bool, model: str = MODEL
) -> dict | anthropic.NotGiven:
    # The cheap lookup model does not need extended-thinking configuration.
    # Omitting the field also keeps routing compatible with models that do not
    # expose Anthropic's thinking control at all.
    if model != MODEL:
        return anthropic.NOT_GIVEN
    return anthropic.NOT_GIVEN if enabled else THINKING_DISABLED

# How the layers should be described to the model. Naming them by what they can
# and cannot know is more useful than naming them by where they came from: the
# model has to weigh them, not cite them decoratively.
LAYER_DESCRIPTIONS = {
    "manufacturer_datasheet": (
        "MANUFACTURER — the maker's own published data. Precise and "
        "quantitative. Ageing claims rest on accelerated testing, and it has "
        "a commercial interest in the product."
    ),
    "materials_science": (
        "MATERIALS SCIENCE — peer-reviewed studies of how materials behave "
        "and degrade. Rigorous, but usually on test specimens rather than "
        "artworks."
    ),
    "conservation_literature": (
        "CONSERVATION — published examination of real objects, often ones "
        "that already failed. The only layer reporting decades of real time."
    ),
    "collection_precedent": (
        "PRECEDENT — what a national collection actually holds. Evidence that "
        "work was made and kept, not a statement about material properties."
    ),
}

SHARED_RULES = """\
Ground every factual claim in the passages provided. If they do not answer the \
question, say so plainly and say what is missing -- a partial answer that names \
its gap is useful; an invented one is not.

Never invent product names, grade codes, or numeric values. If a passage gives \
a number, quote it as given, with its units.

Write for someone standing in a workshop. Plain sentences, no marketing tone, \
no bulleted restatement of the question.\
"""

PROMPTS = {
    "plain": f"""\
You answer questions about materials for making physical artworks, using the \
passages supplied.

{SHARED_RULES}\
""",
    "sourced": f"""\
You answer questions about materials for making physical artworks, using the \
passages supplied.

The passages come from sources of different kinds, and each is labelled. \
Attribute every claim to the kind of source it came from, in the sentence that \
makes the claim -- "the manufacturer states", "a conservation study found", \
"the collection record shows". A reader must be able to tell, without \
scrolling, whether a statement is a vendor's number or an examined outcome.

{SHARED_RULES}\
""",
    "arbitrated": f"""\
You answer questions about materials for making physical artworks, using the \
passages supplied.

The passages come from sources of different kinds, and each is labelled. \
Attribute every claim to the kind of source it came from, in the sentence that \
makes the claim.

These sources routinely disagree, and the disagreement is usually the most \
useful thing you can tell the reader. A datasheet reporting "UV resistant" is \
reporting a few hundred hours of accelerated weathering; a conservation study \
reporting that the same polymer yellowed is reporting an object after ten \
years outdoors. Both are true and they are not in conflict about facts -- they \
are measuring different things.

So: where the passages disagree, say so explicitly, and explain what each \
source is in a position to know. Do not average them into a single confident \
recommendation. Where they agree, say that too -- agreement across a vendor \
and an independent study is worth more than either alone.

If only one kind of source is present, say which perspective is missing.

{SHARED_RULES}\
""",
    "no_context": """\
You answer questions about materials for making physical artworks.

Answer from your own knowledge. Be specific and practical: name products, \
give numbers where you can.\
""",
}


@dataclass
class Answer:
    question: str
    variant: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    # True when the model ran out of `max_tokens`. A truncated answer must
    # never be scored as if it were a finished one -- the judge would be
    # measuring the token budget and calling it answer quality.
    truncated: bool = False
    model: str = MODEL


def format_context(hits: list[Hit]) -> str:
    """Render retrieved chunks with their provenance.

    Grouped by layer rather than by rank. Rank order is a retrieval artefact;
    what the model needs is to see the manufacturer block and the conservation
    block as blocks, so that a disagreement between them is visible as a
    disagreement rather than as two adjacent numbered passages.
    """
    by_layer: dict[str, list[Hit]] = {}
    for hit in hits:
        by_layer.setdefault(hit.source_type, []).append(hit)

    blocks = []
    for layer, group in by_layer.items():
        header = LAYER_DESCRIPTIONS.get(layer, layer.upper())
        passages = "\n\n".join(
            f"[{h.chunk_id}] {h.title}\n{h.text}" for h in group
        )
        blocks.append(f"### {header}\n\n{passages}")
    return "\n\n".join(blocks)


def build_content(question: str, hits: list[Hit], variant: str) -> str:
    if variant == "no_context":
        return question
    return f"{format_context(hits)}\n\n---\n\nQuestion: {question}"


def stream(
    question: str,
    hits: list[Hit],
    variant: str = "arbitrated",
    client: anthropic.Anthropic | None = None,
    thinking: bool = False,
    model: str | None = None,
):
    """Yield answer text as it arrives, then yield the finished `Answer` last.

    Measured before this existed: 40 seconds from click to first character, for
    a 3,237-token answer. The work was not wasteful -- the answer is long
    because it is comparing four sources -- but a blank screen for 40 seconds
    reads as broken, and a user who cannot see the answer forming cannot start
    reading the part that is already correct.

    The generator's last yield is the `Answer` object rather than a string, so
    the caller still gets token counts and `stop_reason` for logging. A
    streaming UI that silently stopped recording cost and truncation would have
    traded the monitoring story for the latency one.
    """
    if variant not in PROMPTS:
        raise ValueError(f"unknown variant {variant!r}; expected {list(PROMPTS)}")
    client = client or anthropic.Anthropic()
    model = model or MODEL
    hits = [] if variant == "no_context" else hits

    parts: list[str] = []
    with client.messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
        system=PROMPTS[variant],
        thinking=thinking_config(thinking, model),
        messages=[
            {"role": "user", "content": build_content(question, hits, variant)}
        ],
    ) as streamed:
        for text in streamed.text_stream:
            parts.append(text)
            yield text
        message = streamed.get_final_message()

    yield Answer(
        question=question,
        variant=variant,
        text="".join(parts).strip(),
        hits=hits,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        truncated=message.stop_reason == "max_tokens",
        model=model,
    )


def generate(
    question: str,
    hits: list[Hit],
    variant: str = "arbitrated",
    client: anthropic.Anthropic | None = None,
    thinking: bool = True,
    model: str | None = None,
) -> Answer:
    if variant not in PROMPTS:
        raise ValueError(f"unknown variant {variant!r}; expected {list(PROMPTS)}")
    client = client or anthropic.Anthropic()
    model = model or MODEL

    if variant == "no_context":
        hits = []
    content = build_content(question, hits, variant)

    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=PROMPTS[variant],
        thinking=thinking_config(thinking, model),
        messages=[{"role": "user", "content": content}],
    )
    text = "\n".join(b.text for b in message.content if b.type == "text")
    return Answer(
        question=question,
        variant=variant,
        text=text.strip(),
        hits=hits,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        truncated=message.stop_reason == "max_tokens",
        model=model,
    )


def answer_question(
    question: str,
    qdrant,
    variant: str = "arbitrated",
    limit: int = 8,
    use_rewrite: bool = True,
    client: anthropic.Anthropic | None = None,
) -> tuple[Answer, object]:
    """End-to-end: rewrite, retrieve, generate. Returns the answer and rewrite.

    `hybrid` without the cross-encoder is the default retrieval path. The
    evaluation measured reranking at 194x the latency for no aggregate gain on
    this corpus, so it is not on by default -- see STATUS.md.
    """
    from rag.rewrite import Rewrite, rewrite, search_rewritten
    from rag.search import search

    if variant == "no_context":
        return generate(question, [], variant, client), None

    if use_rewrite:
        rw = rewrite(question, client)
        hits = search_rewritten(qdrant, rw, limit=limit, rerank=False)
    else:
        rw = Rewrite(original=question, rewritten=question, used=False)
        hits = search(qdrant, question, method="hybrid", limit=limit)

    return generate(question, hits, variant, client), rw


if __name__ == "__main__":
    import argparse

    from rag.index import connect

    parser = argparse.ArgumentParser(description="Answer one question")
    parser.add_argument("question")
    parser.add_argument("--variant", default="arbitrated", choices=list(PROMPTS))
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    qdrant = connect()
    answer, rw = answer_question(args.question, qdrant, args.variant, args.limit)
    if rw is not None and rw.used:
        print(f"[rewritten] {rw.rewritten}\n")
    print(answer.text)
    if answer.hits:
        print("\n--- sources")
        for hit in answer.hits:
            print(f"  [{hit.source_type}] {hit.title[:64]}")
