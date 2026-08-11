from __future__ import annotations

import unittest
from pathlib import Path

from rag.route import routing_enabled
from rag.search import PRODUCTION_DIVERSITY_ENABLED, PRODUCTION_METHOD


class ProductionContractTests(unittest.TestCase):
    def test_unevaluated_features_default_off(self):
        self.assertFalse(routing_enabled())
        self.assertFalse(PRODUCTION_DIVERSITY_ENABLED)
        self.assertEqual(PRODUCTION_METHOD, "hybrid")

    def test_answer_body_does_not_guess_source_from_words(self):
        source = Path("web/app.js").read_text(encoding="utf-8")
        self.assertNotIn("function whoSpeaks", source)
        self.assertNotIn('class="src-${', source)
        self.assertIn("Source identity must come from retrieved chunk ids", source)


if __name__ == "__main__":
    unittest.main()
