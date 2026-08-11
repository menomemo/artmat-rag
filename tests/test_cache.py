from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.cache import (
    cache_fields,
    hit_record,
    identity_for,
    lookup,
    normalize_question,
)
from rag.search import SearchFilters


class ExactCacheTests(unittest.TestCase):
    def test_normalization_changes_formatting_not_case(self):
        self.assertEqual(normalize_question("  Resin\n\t  cure?  "), "Resin cure?")
        self.assertNotEqual(normalize_question("Resin"), normalize_question("resin"))

    def test_identity_covers_user_visible_pipeline_settings(self):
        base = identity_for("Will it yellow?", "arbitrated", 8, True)
        self.assertNotEqual(
            base.key, identity_for("Will it yellow?", "plain", 8, True).key
        )
        self.assertNotEqual(
            base.key, identity_for("Will it yellow?", "arbitrated", 4, True).key
        )
        self.assertNotEqual(
            base.key, identity_for("Will it yellow?", "arbitrated", 8, False).key
        )

    def test_namespace_invalidates_all_keys(self):
        with patch.dict(os.environ, {"EXACT_CACHE_NAMESPACE": "first"}):
            first = identity_for("question", "arbitrated", 8, True)
        with patch.dict(os.environ, {"EXACT_CACHE_NAMESPACE": "second"}):
            second = identity_for("question", "arbitrated", 8, True)
        self.assertNotEqual(first.key, second.key)
        self.assertEqual(second.namespace, "second")

    def test_identity_includes_metadata_filters(self):
        unfiltered = identity_for("question", "arbitrated", 8, True)
        filtered = identity_for(
            "question",
            "arbitrated",
            8,
            True,
            SearchFilters(source_types=("conservation_literature",)),
        )
        other_filter = identity_for(
            "question",
            "arbitrated",
            8,
            True,
            SearchFilters(source_types=("manufacturer_datasheet",)),
        )
        self.assertNotEqual(unfiltered.key, filtered.key)
        self.assertNotEqual(filtered.key, other_filter.key)

    @patch("app.cache.find_cached_query")
    def test_lookup_restores_answer_rewrite_and_evidence(self, find):
        identity = identity_for("question", "arbitrated", 8, True)
        find.return_value = {
            "id": 41,
            "answer": "cached answer",
            "rewritten": "technical question",
            "rewrite_used": True,
            "rewrite_terms": ["pot life"],
            "method": "rewrite_hybrid",
            "source_counts": {"manufacturer_datasheet": 1},
            "hits": [
                {
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1",
                    "source_type": "manufacturer_datasheet",
                    "chunk_type": "spec",
                    "title": "Product",
                    "text": "Pot life: 20 minutes",
                    "url": None,
                    "score": 0.9,
                }
            ],
            "filters": {"source_types": ["manufacturer_datasheet"]},
            "generate_model": "claude-haiku-4-5",
            "route_tier": "simple",
            "route_reason": "single-source specification lookup",
        }

        cached = lookup("question", "arbitrated", 8, True)

        find.assert_called_once_with(identity.key)
        self.assertEqual(cached.source_query_id, 41)
        self.assertEqual(cached.answer.text, "cached answer")
        self.assertEqual(cached.answer.input_tokens, 0)
        self.assertEqual(cached.rewrite.terms, ["pot life"])
        self.assertEqual(cached.hits[0].chunk_id, "chunk-1")
        self.assertEqual(cached.answer.model, "claude-haiku-4-5")
        self.assertEqual(cached.route.tier, "simple")

        record = hit_record(cached, 12)
        self.assertTrue(record["cache_hit"])
        self.assertEqual(record["cache_source_query_id"], 41)
        self.assertEqual(record["cost_usd"], 0)
        self.assertEqual(record["total_ms"], 12)
        self.assertEqual(record["generate_model"], "claude-haiku-4-5")
        self.assertEqual(record["filters"]["source_types"], ["manufacturer_datasheet"])

    def test_disabled_cache_does_not_touch_database(self):
        with patch.dict(os.environ, {"EXACT_CACHE_ENABLED": "false"}):
            with patch("app.cache.find_cached_query") as find:
                self.assertIsNone(lookup("question", "arbitrated", 8, True))
                find.assert_not_called()

    def test_new_paid_query_is_marked_as_cache_source(self):
        fields = cache_fields("question", "arbitrated", 8, True)
        self.assertFalse(fields["cache_hit"])
        self.assertIsNone(fields["cache_source_query_id"])
        self.assertEqual(len(fields["cache_key"]), 64)


if __name__ == "__main__":
    unittest.main()
