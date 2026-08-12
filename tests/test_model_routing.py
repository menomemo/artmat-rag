from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag.generate import stream
from rag.route import COMPLEX_MODEL, SIMPLE_MODEL, route_question
from rag.search import Hit


def hit(
    source_type: str = "manufacturer_datasheet",
    chunk_type: str = "spec",
) -> Hit:
    return Hit(
        "chunk", "doc", source_type, chunk_type, "Product", "Evidence", None, 1.0
    )


@patch.dict(os.environ, {"MODEL_ROUTING_ENABLED": "true"})
class ModelRoutingTests(unittest.TestCase):
    def test_clear_single_source_spec_lookup_uses_simple_model(self):
        route = route_question(
            "What is the pot life and Shore A hardness?", [hit(), hit()], "arbitrated"
        )
        self.assertEqual(route.tier, "simple")
        self.assertEqual(route.model, SIMPLE_MODEL)

    def test_cross_source_evidence_uses_complex_model(self):
        route = route_question(
            "What is the cure time?",
            [hit(), hit("materials_science", "abstract")],
            "arbitrated",
        )
        self.assertEqual(route.tier, "complex")
        self.assertEqual(route.model, COMPLEX_MODEL)

    def test_comparison_or_ageing_language_uses_complex_model(self):
        for question in (
            "Which resin has the better cure time?",
            "Will the stated hardness change after years outdoors?",
            "Should I choose the faster mix ratio?",
            "What mix ratio and glove type are recommended?",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    route_question(question, [hit()], "arbitrated").tier,
                    "complex",
                )

    def test_narrative_evidence_does_not_take_cheap_route(self):
        route = route_question(
            "What is the pot life?", [hit(chunk_type="narrative")], "plain"
        )
        self.assertEqual(route.tier, "complex")

    def test_no_context_control_always_uses_complex_model(self):
        route = route_question("What is the hardness?", [hit()], "no_context")
        self.assertEqual(route.reason, "no-context control")
        self.assertEqual(route.model, COMPLEX_MODEL)

    def test_routing_can_be_disabled_without_code_change(self):
        with patch.dict(os.environ, {"MODEL_ROUTING_ENABLED": "false"}):
            route = route_question("What is the pot life?", [hit()], "plain")
        self.assertEqual(route.reason, "routing disabled")
        self.assertEqual(route.model, COMPLEX_MODEL)

    def test_stream_uses_routed_model_and_records_it_on_answer(self):
        captured = {}

        class FakeStream:
            text_stream = iter(["answer"])

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def get_final_message(self):
                return SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=10, output_tokens=2),
                    stop_reason="end_turn",
                )

        class Messages:
            def stream(self, **kwargs):
                captured.update(kwargs)
                return FakeStream()

        client = SimpleNamespace(messages=Messages())
        pieces = list(
            stream(
                "What is the pot life?",
                [hit()],
                "plain",
                client=client,
                model=SIMPLE_MODEL,
            )
        )
        self.assertEqual(captured["model"], SIMPLE_MODEL)
        self.assertEqual(pieces[-1].model, SIMPLE_MODEL)


if __name__ == "__main__":
    unittest.main()
