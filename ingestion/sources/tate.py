"""Tate collection open data: the *precedent* layer.

What this layer is, and what it deliberately is not.

The original intent was a case-study layer -- "how did this artist make this
piece, and what happened to it". Tate's artwork pages carry exactly that in
curatorial summaries, but those pages render client-side: the static HTML holds
675 characters of sidebar and no summary text. Harvesting them would mean
driving a headless browser across hundreds of pages, which is not a good trade
inside a two-day build.

What remains is `artwork_data.csv` (CC0, 69,201 works): an accession record per
artwork, whose `medium` field is a short material phrase. A bare phrase like
"Polychromed aluminium and rubber coated steel" is an inventory entry, not
knowledge -- it says what Koons used and nothing about why, how, or how it
aged. Indexed one-per-artwork it would add 1,600 low-signal chunks that answer
none of the questions this system exists for.

So the layer is built by aggregation instead. Counting which materials actually
recur in a national collection, and when, produces a claim neither of the other
two layers can make: a manufacturer datasheet says what a resin does, a
conservation paper says how it fails, and this says whether anyone actually
built lasting work out of it. That is worth ~25 dense chunks, not 1,600 sparse
ones.

Licence: CC0-1.0. Source: github.com/tategallery/collection.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

CSV_URL = (
    "https://raw.githubusercontent.com/tategallery/collection/master/artwork_data.csv"
)

# Materials that are cast, moulded, or fabricated as the substance of an
# object. Ordered most-specific-first so a medium naming several materials is
# attributed to the one that best characterises how it was made.
FABRICATION_MATERIALS = {
    "jesmonite": ["jesmonite"],
    "epoxy resin": ["epoxy"],
    "polyurethane": ["polyurethane"],
    "polyester resin": ["polyester resin", "polyester and", "glass-reinforced polyester"],
    "fibreglass": ["fibreglass", "fiberglass", "glass fibre", "glass-reinforced"],
    "silicone": ["silicone"],
    "unspecified resin": ["resin"],
    "concrete / cement": ["concrete", "cement"],
    "plaster": ["plaster"],
    "wax": ["wax"],
    "latex / rubber": ["latex", "rubber"],
    "polystyrene": ["polystyrene"],
    "bronze": ["bronze"],
    "aluminium": ["aluminium", "aluminum"],
    "steel": ["steel"],
    "lead": ["lead"],
}

# Media where a material name appears but the object was not cast or moulded
# from it -- the material is a mount, a frame, a support, or a print substrate.
# Without these, keyword matching reports "photograph on aluminium" as an
# aluminium artwork and the prevalence counts become meaningless.
NON_FABRICATION_CONTEXTS = [
    r"\bon paper\b",
    r"\bscreenprint\b",
    r"\blithograph\b",
    r"\betching\b",
    r"\bphotograph\w*\b.{0,30}\bon\b",
    r"\bprint\b.{0,20}\bon\b",
    r"\bmounted (?:on|between)\b",
    r"\bframe[ds]?\b",
    r"\bon canvas\b",
    r"\bpolyester film\b",
]


@dataclass
class MaterialPrecedent:
    """Aggregated evidence that a material was actually used in collected work."""

    material: str
    artwork_count: int
    first_year: int | None
    last_year: int | None
    by_decade: dict[str, int] = field(default_factory=dict)
    notable_artists: list[str] = field(default_factory=list)
    example_media: list[str] = field(default_factory=list)
    source_type: str = "collection_precedent"

    @property
    def doc_id(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", self.material.lower()).strip("-")
        return f"tate:{slug}"

    def to_passage(self) -> str:
        """Render as prose. The retrieval unit is text, not a record, and the
        embedding of a bare dict is not something a semantic query can match.
        """
        span = (
            f"{self.first_year}–{self.last_year}"
            if self.first_year and self.last_year and self.first_year != self.last_year
            else str(self.first_year or "date unrecorded")
        )
        decades = ", ".join(
            f"{decade}: {count}"
            for decade, count in sorted(self.by_decade.items())
            if count
        )
        lines = [
            f"{self.material.capitalize()} in the Tate collection: "
            f"{self.artwork_count} catalogued artworks, {span}.",
        ]
        if decades:
            lines.append(f"Distribution by decade of production -- {decades}.")
        if self.notable_artists:
            lines.append(
                "Artists represented include "
                + ", ".join(self.notable_artists[:8])
                + "."
            )
        if self.example_media:
            lines.append(
                "Example catalogue descriptions: "
                + "; ".join(f'"{m}"' for m in self.example_media[:3])
                + "."
            )
        lines.append(
            "This records what was actually made and acquired by a national "
            "collection, which is evidence of practical viability over time -- "
            "it is not a statement about the material's technical properties "
            "or its long-term condition."
        )
        return " ".join(lines)


# Honorifics Tate appends to the catalogue name field. They survive a naive
# comma split and turn "Moore, Henry, OM, CH" into three "artists".
HONORIFICS = {
    "om", "ch", "cbe", "obe", "mbe", "ra", "pra", "rha", "kt", "bt", "dbe",
    "kbe", "sir", "dame", "lord", "baron",
}


def readable_name(catalogue_name: str) -> str:
    """"Moore, Henry, OM, CH" -> "Henry Moore". Retrieved text is read by a
    person, and catalogue inversion reads as noise inside a sentence.
    """
    parts = [p.strip() for p in catalogue_name.split(",") if p.strip()]
    parts = [p for p in parts if p.lower().rstrip(".") not in HONORIFICS]
    if len(parts) >= 2:
        return f"{parts[1]} {parts[0]}"
    return parts[0] if parts else catalogue_name


def is_fabrication(medium: str) -> bool:
    """True when the medium describes an object made *of* the material."""
    lowered = medium.lower()
    return not any(re.search(p, lowered) for p in NON_FABRICATION_CONTEXTS)


def classify(medium: str) -> str | None:
    lowered = medium.lower()
    for material, patterns in FABRICATION_MATERIALS.items():
        if any(p in lowered for p in patterns):
            return material
    return None


def build(csv_path: Path) -> list[MaterialPrecedent]:
    csv.field_size_limit(sys.maxsize)
    grouped: dict[str, list[dict]] = defaultdict(list)

    with csv_path.open(encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            medium = (row.get("medium") or "").strip()
            if not medium or not is_fabrication(medium):
                continue
            material = classify(medium)
            if material:
                grouped[material].append(row)

    precedents = []
    for material, rows in grouped.items():
        years = []
        for row in rows:
            raw = (row.get("year") or "").strip()
            if raw.lstrip("?c. ").isdigit():
                years.append(int(raw.lstrip("?c. ")))

        decades: Counter[str] = Counter()
        for year in years:
            decades[f"{year // 10 * 10}s"] += 1

        artists = [
            readable_name(a)
            for a, _ in Counter(r["artist"] for r in rows).most_common(8)
        ]
        examples = [
            m
            for m in dict.fromkeys((r.get("medium") or "").strip() for r in rows)
            if 30 < len(m) < 110
        ][:3]

        precedents.append(
            MaterialPrecedent(
                material=material,
                artwork_count=len(rows),
                first_year=min(years) if years else None,
                last_year=max(years) if years else None,
                by_decade=dict(decades.most_common(8)),
                notable_artists=artists,
                example_media=examples,
            )
        )

    precedents.sort(key=lambda p: p.artwork_count, reverse=True)
    return precedents


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Tate precedent layer")
    parser.add_argument("--csv", default="data/tate_artwork_data.csv", type=Path)
    parser.add_argument("--out", default="data/raw/tate.jsonl", type=Path)
    args = parser.parse_args()

    precedents = build(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for precedent in precedents:
            record = asdict(precedent)
            record["doc_id"] = precedent.doc_id
            record["passage"] = precedent.to_passage()
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(precedents)} material precedents -> {args.out}")
    for precedent in precedents:
        span = f"{precedent.first_year}-{precedent.last_year}"
        print(f"  {precedent.artwork_count:>5}  {precedent.material:<20} {span}")
