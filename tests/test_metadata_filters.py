from __future__ import annotations

import unittest

from rag.search import SearchFilters, _metadata_filter


class MetadataFilterTests(unittest.TestCase):
    def test_empty_filters_do_not_send_qdrant_filter(self):
        self.assertIsNone(_metadata_filter(SearchFilters()))

    def test_values_are_or_within_fields_and_and_between_fields(self):
        query_filter = _metadata_filter(
            SearchFilters(
                source_types=("materials_science", "conservation_literature"),
                chunk_types=("abstract",),
                domains=("metals", "conservation_practice"),
                year_from=2000,
                year_to=2020,
            )
        )

        conditions = {condition.key: condition for condition in query_filter.must}
        self.assertEqual(
            conditions["source_type"].match.any,
            ["materials_science", "conservation_literature"],
        )
        self.assertEqual(conditions["chunk_type"].match.any, ["abstract"])
        self.assertEqual(
            conditions["domain"].match.any,
            ["metals", "conservation_practice"],
        )
        self.assertEqual(conditions["year"].range.gte, 2000)
        self.assertEqual(conditions["year"].range.lte, 2020)

    def test_unknown_indexed_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown category"):
            SearchFilters(categories=("imaginary-resin",))

    def test_legacy_source_argument_cannot_conflict_with_filters(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            _metadata_filter(
                SearchFilters(domains=("metals",)),
                source_types=["materials_science"],
            )

    def test_reversed_year_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "year_from"):
            SearchFilters(year_from=2020, year_to=2000)

    def test_year_must_exist_in_current_corpus_range(self):
        with self.assertRaisesRegex(ValueError, "between 1952 and 2026"):
            SearchFilters(year_from=1900)

    def test_serialized_filters_are_json_safe(self):
        filters = SearchFilters(
            source_types=("manufacturer_datasheet",),
            categories=("platinum-silicone",),
        )
        self.assertEqual(
            filters.as_dict()["source_types"], ["manufacturer_datasheet"]
        )
        self.assertEqual(filters.as_dict()["categories"], ["platinum-silicone"])


if __name__ == "__main__":
    unittest.main()
