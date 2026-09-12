from __future__ import annotations

import unittest
from unittest.mock import patch

from api.main import AskRequest, _state, ask
from rag.generate import Answer
from rag.route import ModelRoute, SIMPLE_MODEL
from rag.search import Hit


def hit() -> Hit:
    return Hit(
        "chunk", "doc", "manufacturer_datasheet", "spec",
        "Product", "Evidence", None, 1.0,
    )


class ApiModelRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_route_is_streamed_and_logged_with_actual_model(self):
        previous = _state.get("qdrant")
        _state["qdrant"] = object()
        decision = ModelRoute(
            SIMPLE_MODEL, "simple", "single-source specification lookup"
        )
        answer = Answer(
            "What is the pot life?",
            "plain",
            "20 minutes",
            [hit()],
            input_tokens=10,
            output_tokens=2,
            model=SIMPLE_MODEL,
        )
        try:
            with (
                patch("api.main.search", return_value=[hit()]),
                patch("api.main.route_question", return_value=decision),
                patch("api.main.stream", return_value=iter(["20 minutes", answer]))
                as generate,
                patch("api.main.log_query", return_value=91) as log,
            ):
                response = await ask(
                    AskRequest(
                        question="What is the pot life?",
                        variant="plain",
                        rewrite=False,
                    )
                )
                body = ""
                async for piece in response.body_iterator:
                    body += piece.decode() if isinstance(piece, bytes) else piece
        finally:
            if previous is None:
                _state.pop("qdrant", None)
            else:
                _state["qdrant"] = previous

        self.assertIn("event: route", body)
        self.assertIn(f'"model": "{SIMPLE_MODEL}"', body)
        self.assertIn('"route_tier": "simple"', body)
        self.assertEqual(generate.call_args.kwargs["model"], SIMPLE_MODEL)
        record = log.call_args.args[0]
        self.assertEqual(record["generate_model"], SIMPLE_MODEL)
        self.assertEqual(record["route_tier"], "simple")


if __name__ == "__main__":
    unittest.main()
