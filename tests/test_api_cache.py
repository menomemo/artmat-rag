from __future__ import annotations

import unittest
from unittest.mock import patch

from api.main import AskRequest, ask
from app.cache import CachedQuery, ExactCacheIdentity
from rag.generate import Answer
from rag.rewrite import Rewrite
from rag.search import Hit


class ApiExactCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_hit_answers_without_qdrant_or_models(self):
        hit = Hit(
            "chunk-1",
            "doc-1",
            "materials_science",
            "abstract",
            "Study",
            "Evidence",
            None,
            0.8,
        )
        cached = CachedQuery(
            source_query_id=7,
            identity=ExactCacheIdentity("key", "corpus-v1"),
            answer=Answer("question", "arbitrated", "cached answer", [hit]),
            rewrite=Rewrite("question", "technical question", ["term"], True),
            hits=[hit],
            source_counts={"materials_science": 1},
            method="rewrite_hybrid",
        )

        with (
            patch("api.main.lookup", return_value=cached),
            patch("api.main.log_query", return_value=99) as log,
            patch("api.main.rewrite", side_effect=AssertionError("model called")),
            patch("api.main.search", side_effect=AssertionError("search called")),
            patch("api.main.stream", side_effect=AssertionError("model called")),
        ):
            response = await ask(AskRequest(question="question"))
            body = ""
            async for piece in response.body_iterator:
                body += piece.decode() if isinstance(piece, bytes) else piece

        self.assertIn('event: token', body)
        self.assertIn('"text": "cached answer"', body)
        self.assertIn('"cache_hit": true', body)
        self.assertIn('"query_id": 99', body)
        log.assert_called_once()
        self.assertTrue(log.call_args.args[0]["cache_hit"])


if __name__ == "__main__":
    unittest.main()
