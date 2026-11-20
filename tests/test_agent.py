"""The agent loop's contract, tested without a network.

What must hold regardless of what the model decides: the loop terminates, the
search budget is enforced even when the model asks for more, bad tool input
becomes a correctable tool error, passages are never shown twice, and token
usage is summed across turns -- the last one being what the cost comparison
rests on.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

import rag.agent as agent
from rag.search import Hit


def hit(cid: str, layer: str = "manufacturer_datasheet") -> Hit:
    return Hit(cid, cid.split("#")[0], layer, "spec", f"title {cid}", f"text {cid}", None, 1.0)


def text_block(t):
    return NS(type="text", text=t)


def tool_block(i, layer="manufacturer_datasheet", **inp):
    return NS(type="tool_use", id=f"tu{i}", name="search_layer", input={"layer": layer, **inp})


class FakeStream:
    def __init__(self, message):
        self.message = message
        self.text_stream = [b.text for b in message.content if b.type == "text"]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class FakeClient:
    """Replays scripted assistant turns and records every request."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []
        self.messages = self

    def stream(self, **kwargs):
        self.requests.append(kwargs)
        content, stop = self.turns.pop(0)
        return FakeStream(NS(content=content, stop_reason=stop,
                             usage=NS(input_tokens=100, output_tokens=10)))


@pytest.fixture
def fake_search(monkeypatch):
    calls = []

    def search(qdrant, query, method, limit, source_types):
        calls.append((query, source_types))
        return [hit("smoothon:a#spec"), hit("openalex:b#abstract-0", "materials_science")]

    monkeypatch.setattr(agent, "search", search)
    return calls


def test_one_search_then_answer(fake_search):
    client = FakeClient([
        ([text_block("Let me look."), tool_block(1, query="pot life")], "tool_use"),
        ([text_block("Pot life is 20 minutes.")], "end_turn"),
    ])
    events = list(agent.stream_agent("how long can I work", None, client))
    kinds = [k for k, _ in events]
    answer = events[-1][1]

    assert kinds.count("search") == 1 and "reset" in kinds
    assert answer.text == "Pot life is 20 minutes."  # preamble text excluded
    assert [h.chunk_id for h in answer.hits] == ["smoothon:a#spec", "openalex:b#abstract-0"]
    assert (answer.turns, answer.input_tokens, answer.output_tokens) == (2, 200, 20)
    assert not answer.budget_exhausted
    # tool result is paired with the request id it answers
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "tu1" and "[smoothon:a#spec]" in result["content"]


def test_budget_forces_an_answer(fake_search):
    searches = [([tool_block(i, query=f"q{i}")], "tool_use") for i in range(2)]
    client = FakeClient(searches + [([text_block("done")], "end_turn")])
    answer = agent.run_agent("q", None, client, max_searches=2)

    assert len(answer.searches) == 2
    assert [r["tool_choice"]["type"] for r in client.requests] == ["auto", "auto", "none"]
    assert answer.text == "done"


def test_parallel_calls_beyond_budget_are_refused(fake_search):
    client = FakeClient([
        ([tool_block(1, query="a"), tool_block(2, query="b"), tool_block(3, query="c")], "tool_use"),
        ([text_block("answer")], "end_turn"),
    ])
    answer = agent.run_agent("q", None, client, max_searches=2)

    assert len(fake_search) == 2
    assert answer.budget_exhausted
    refused = client.requests[1]["messages"][-1]["content"][2]
    assert refused["is_error"] and "budget" in refused["content"].lower()


def test_invalid_layer_is_a_tool_error_not_a_crash(fake_search):
    client = FakeClient([
        ([tool_block(1, query="x", layer="blogs")], "tool_use"),
        ([text_block("ok")], "end_turn"),
    ])
    answer = agent.run_agent("q", None, client, max_searches=3)

    assert fake_search == []  # never reached the retriever
    assert answer.searches[0].error and len(answer.searches) == 1  # still spent budget
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] and "layer must be" in result["content"]


def test_repeated_passages_are_not_shown_twice(fake_search):
    client = FakeClient([
        ([tool_block(1, query="a")], "tool_use"),
        ([tool_block(2, query="b")], "tool_use"),
        ([text_block("ok")], "end_turn"),
    ])
    answer = agent.run_agent("q", None, client, max_searches=4)

    assert len(answer.hits) == 2
    assert [c.new for c in answer.searches] == [2, 0]
    second = client.requests[2]["messages"][-1]["content"][0]["content"]
    assert second.startswith("Nothing new") and "already returned" in second


def test_layer_filter_reaches_the_retriever(fake_search):
    client = FakeClient([
        ([tool_block(1, query="yellowing", layer="conservation_literature")], "tool_use"),
        ([text_block("ok")], "end_turn"),
    ])
    agent.run_agent("q", None, client)
    assert fake_search == [("yellowing", ["conservation_literature"])]


def test_missing_layer_is_rejected(fake_search):
    client = FakeClient([
        ([NS(type="tool_use", id="tu1", name="search_layer", input={"query": "x"})], "tool_use"),
        ([text_block("ok")], "end_turn"),
    ])
    answer = agent.run_agent("q", None, client)
    assert fake_search == [] and answer.searches[0].error


def test_prompt_asks_for_citations_and_per_layer_search():
    # The two behaviours v1 lacked (STATUS.md). Cheap to assert, easy to lose
    # in a later prompt edit.
    assert "Cite each claim" in agent.SYSTEM_PROMPT
    assert "one call per layer" in agent.SYSTEM_PROMPT
    assert agent.TOOL["input_schema"]["required"] == ["layer", "query"]


def test_truncation_is_recorded(fake_search):
    client = FakeClient([([text_block("cut")], "max_tokens")])
    assert agent.run_agent("q", None, client).truncated
