"""Agentic retrieval: the model chooses what to search, layer by layer.

The fixed pipeline makes one retrieval decision for the model: rewrite the
question once, fetch the top `k` passages across all four layers, answer. That
is cheap and predictable, and it has one specific blind spot for this corpus.
If the top `k` happen to be six datasheets and two abstracts, the arbitrated
prompt is asked to weigh a conservation perspective it was never shown -- and
"if only one kind of source is present, say which perspective is missing" is
the best it can do.

Here retrieval is a tool instead. The model gets `search_layer(layer, query)`
and decides which layers the question needs, then searches each one in that
layer's own vocabulary -- "pot life" for a datasheet, "photo-oxidative
degradation" for a conservation paper. The tool is a thin wrapper over the
same `hybrid` search the pipeline uses, so what changes between the two paths
is *who decides the queries and the layer mix*, not the retriever.

Whether that is worth it is an empirical question, and `eval/agentic.py`
exists to answer it. The costs are structural, not incidental:

- every tool round re-sends the conversation so far, so input tokens grow with
  each search rather than adding up;
- the first visible character waits for every search the model chooses to do;
- the model can search badly, repeat itself, or stop too early.

So the loop is bounded. `MAX_SEARCHES` caps tool calls -- each requested call
spends budget whether or not its input was valid, which is what guarantees the
loop terminates -- and once the budget is gone the final turn is made with
`tool_choice: none`, which forces an answer from what was found instead of a
request for more.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Iterator

import anthropic

from rag.generate import (
    LAYER_DESCRIPTIONS,
    MAX_TOKENS,
    MODEL,
    PROMPTS,
    thinking_config,
)
from rag.search import Hit, search

# v2. v1 offered an *optional* `source_types` filter and the model used it in
# 2 of 120 searches (STATUS.md), so layer targeting -- the reason this mode
# exists -- never happened. v2 makes it structural: every search names exactly
# one layer. 4 passages per layer-search keeps a four-layer sweep at 16
# passages, the same budget as the pipeline k=16 control in eval/agentic.py;
# 6 searches leaves room for two follow-ups after that sweep.
AGENT_VERSION = "v2-per-layer"
MAX_SEARCHES = int(os.environ.get("AGENT_MAX_SEARCHES", "6"))
PER_SEARCH = int(os.environ.get("AGENT_PER_SEARCH", "4"))
LAYERS = list(LAYER_DESCRIPTIONS)

TOOL = {
    "name": "search_layer",
    "description": (
        "Search ONE source layer of the corpus of passages about materials "
        "for making artworks. Hybrid keyword + semantic search; returns up to "
        f"{PER_SEARCH} passages from that layer, each labelled with its chunk "
        "id. Passages already returned are not repeated. Write the query in "
        "the vocabulary that layer's documents use: a datasheet says 'pot "
        "life', 'Shore A'; a conservation paper says 'photo-oxidative "
        "degradation', 'consolidant'; a collection record names materials as "
        "accessioned."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "layer": {
                "type": "string",
                "enum": LAYERS,
                "description": "The one layer to search.",
            },
            "query": {
                "type": "string",
                "description": "What to search for, in that layer's vocabulary.",
            },
        },
        "required": ["layer", "query"],
        "additionalProperties": False,
    },
}

AGENT_PREAMBLE = f"""\
You start with no passages. Find them with the search_layer tool before \
answering; you may search up to {MAX_SEARCHES} times in total.

The corpus has four layers that know different things:
{chr(10).join(f"- {layer}: {desc}" for layer, desc in LAYER_DESCRIPTIONS.items())}

First decide which layers this question needs -- usually two to four. A \
question about how a material behaves or ages needs the manufacturer's claim \
and at least one independent layer; collection_precedent matters when the \
question is about what has actually been made or kept. Then search every \
layer you chose, all in parallel in your first turn, one call per layer, each \
query phrased in that layer's vocabulary. Afterwards, only if a layer \
returned nothing relevant, you may search it once more with a better query. \
Treat only passages the tool returned as evidence.

Cite each claim with the id of the passage it rests on, in square brackets \
exactly as it appears in the results, e.g. [openalex:W123#abstract-0].

When you answer, begin with the answer itself. Do not describe your searches \
or announce that you have enough evidence -- the reader sees the answer, not \
the search log.

Once you have searched, answer under these rules:

"""

SYSTEM_PROMPT = AGENT_PREAMBLE + PROMPTS["arbitrated"]


@dataclass
class SearchCall:
    query: str
    source_types: list[str] | None
    returned: int
    new: int
    ms: int
    error: str | None = None


@dataclass
class AgentAnswer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    searches: list[SearchCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0
    truncated: bool = False
    # True when the model still wanted to search after the budget ran out.
    # Not a failure -- the forced final turn still answers -- but worth
    # counting: a cap that binds on most questions is set too low.
    budget_exhausted: bool = False
    variant: str = "agentic"


def format_results(hits: list[Hit], repeated: list[str]) -> str:
    """Render one tool result. Same passage format as the pipeline's context,
    so the generator sees evidence the same way on both paths."""
    if not hits and not repeated:
        return "No passages matched this query."
    parts = [
        f"[{h.chunk_id}] ({h.source_type}) {h.title}\n{h.text}" for h in hits
    ]
    if repeated:
        parts.append(
            "Also matched, already returned earlier: " + ", ".join(repeated)
        )
    if not hits:
        parts.insert(0, "Nothing new for this query.")
    return "\n\n".join(parts)


def run_tool(qdrant, tool_input: dict, seen: dict[str, Hit]) -> tuple[str, SearchCall, list[Hit]]:
    """Execute one search request. Invalid input comes back to the model as a
    tool error it can correct, never as an exception that ends the answer."""
    started = time.perf_counter()
    query = (tool_input.get("query") or "").strip()
    layer = tool_input.get("layer")
    source_types = [layer] if isinstance(layer, str) else None

    error = None
    if layer not in LAYERS:
        error = f"layer must be one of {LAYERS}"
    elif not query:
        error = "query must be a non-empty string"
    if error:
        call = SearchCall(query, source_types, 0, 0, 0, error=error)
        return f"Error: {error}", call, []

    found = search(qdrant, query, method="hybrid", limit=PER_SEARCH, source_types=source_types)
    new = [h for h in found if h.chunk_id not in seen]
    repeated = [h.chunk_id for h in found if h.chunk_id in seen]
    call = SearchCall(
        query=query,
        source_types=source_types,
        returned=len(found),
        new=len(new),
        ms=int((time.perf_counter() - started) * 1000),
    )
    return format_results(new, repeated), call, new


def stream_agent(
    question: str,
    qdrant,
    client: anthropic.Anthropic | None = None,
    max_searches: int = MAX_SEARCHES,
    thinking: bool = False,
) -> Iterator[tuple[str, object]]:
    """Run the search-then-answer loop, yielding events as they happen.

    Events: ("text", str) for answer text as it streams; ("reset", None) when
    text already streamed turned out to be preamble to a tool call rather than
    the answer; ("search", SearchCall) after each search; ("done",
    AgentAnswer) last. The UI needs the first three to avoid a silent wait;
    the evaluation only reads the last.
    """
    client = client or anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": question}]
    seen: dict[str, Hit] = {}
    searches: list[SearchCall] = []
    usage_in = usage_out = turns = 0
    budget_exhausted = False
    final = None

    while True:
        budget_left = max_searches - len(searches)
        tool_choice = {"type": "auto"} if budget_left > 0 else {"type": "none"}
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[TOOL],
            tool_choice=tool_choice,
            thinking=thinking_config(thinking),
            messages=messages,
        ) as streamed:
            for text in streamed.text_stream:
                yield "text", text
            message = streamed.get_final_message()

        turns += 1
        usage_in += message.usage.input_tokens
        usage_out += message.usage.output_tokens
        tool_uses = [b for b in message.content if b.type == "tool_use"]

        if message.stop_reason != "tool_use" or not tool_uses:
            final = message
            break

        yield "reset", None
        messages.append({"role": "assistant", "content": message.content})
        results = []
        for block in tool_uses:
            if len(searches) >= max_searches:
                budget_exhausted = True
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Search budget exhausted. Answer now from the passages you have.",
                    "is_error": True,
                })
                continue
            content, call, new = run_tool(qdrant, dict(block.input), seen)
            for hit in new:
                seen[hit.chunk_id] = hit
            searches.append(call)
            yield "search", call
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
                **({"is_error": True} if call.error else {}),
            })
        messages.append({"role": "user", "content": results})

    text = "\n".join(b.text for b in final.content if b.type == "text").strip()
    yield "done", AgentAnswer(
        question=question,
        text=text,
        hits=list(seen.values()),
        searches=searches,
        input_tokens=usage_in,
        output_tokens=usage_out,
        turns=turns,
        truncated=final.stop_reason == "max_tokens",
        budget_exhausted=budget_exhausted,
    )


def run_agent(question: str, qdrant, client: anthropic.Anthropic | None = None,
              max_searches: int = MAX_SEARCHES, thinking: bool = False) -> AgentAnswer:
    for kind, payload in stream_agent(question, qdrant, client, max_searches, thinking):
        if kind == "done":
            return payload
    raise RuntimeError("agent loop ended without an answer")


if __name__ == "__main__":
    import argparse

    import rag.env  # noqa: F401
    from rag.index import connect

    parser = argparse.ArgumentParser(description="Answer one question with agentic search")
    parser.add_argument("question")
    parser.add_argument("--max-searches", type=int, default=MAX_SEARCHES)
    args = parser.parse_args()

    answer = run_agent(args.question, connect(), max_searches=args.max_searches)
    for i, call in enumerate(answer.searches, 1):
        layers = ",".join(call.source_types) if call.source_types else "all"
        print(f"search {i}: {call.query!r} [{layers}] -> {call.new} new / {call.returned}")
    print(f"\n{answer.text}\n")
    print(f"{answer.turns} turns, {answer.input_tokens:,} in / {answer.output_tokens:,} out")
