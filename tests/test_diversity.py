from __future__ import annotations

import unittest

from rag.search import Candidate, Hit, diversify, family_key


def candidate(index: int, title: str, vector: list[float]) -> Candidate:
    return Candidate(
        Hit(
            chunk_id=f"chunk-{index}",
            doc_id=f"doc-{index}",
            source_type="manufacturer_datasheet",
            chunk_type="spec",
            title=title,
            text=title,
            url=None,
            score=1 - index / 100,
        ),
        vector,
    )


class DiversityTests(unittest.TestCase):
    def test_manufacturer_family_removes_only_grade_suffixes(self):
        cases = {
            "Mold Star™ 15 SLOW": "mold star",
            "Smooth-Sil™ 950": "smooth sil",
            "Dragon Skin™ 10 MEDIUM": "dragon skin",
            "Smooth-Cast™ ONYX™ FAST": "smooth cast onyx",
            "Forton™ MG": "forton mg",
        }
        for title, expected in cases.items():
            hit = candidate(0, title, [1, 0]).hit
            self.assertEqual(family_key(hit), expected)

    def test_non_manufacturer_sources_are_not_grouped_by_title(self):
        hit = candidate(0, "Repeated title", [1, 0]).hit
        hit.source_type = "materials_science"
        self.assertEqual(family_key(hit), hit.doc_id)

    def test_relevance_head_is_preserved_and_tail_is_diversified(self):
        rows = [
            candidate(0, "Mold Star 10", [1, 0]),
            candidate(1, "Smooth-Sil 930", [0.99, 0.01]),
            candidate(2, "Mold Star 20", [0.98, 0.02]),
            candidate(3, "Smooth-Sil 940", [0.97, 0.03]),
            candidate(4, "Plat-Cat", [0.96, 0.04]),
            candidate(5, "Mold Star 30", [0.95, 0.05]),
            candidate(6, "Smooth-Sil 950", [0.94, 0.06]),
            candidate(7, "Ecoflex 00-35 FAST", [0.8, 0.2]),
            candidate(8, "Solaris", [0.7, 0.3]),
            candidate(9, "SLO-JO", [0.6, 0.4]),
        ]

        selected = diversify(rows, 8)

        self.assertEqual(
            [hit.chunk_id for hit in selected[:5]],
            [f"chunk-{index}" for index in range(5)],
        )
        self.assertNotIn("chunk-5", [hit.chunk_id for hit in selected])
        self.assertNotIn("chunk-6", [hit.chunk_id for hit in selected])
        self.assertEqual(len(selected), 8)


if __name__ == "__main__":
    unittest.main()
