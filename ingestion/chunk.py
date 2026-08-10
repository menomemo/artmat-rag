"""Turn three heterogeneous corpora into one flat stream of retrievable chunks.

The three layers do not want the same treatment, because they do not fail in
the same way:

- A **manufacturer datasheet** is two documents wearing one coat. Its spec table
  is short, numeric, and answers "what is the Shore hardness" by exact match;
  its narrative is prose and answers "will this survive outdoors" by meaning.
  Indexed together, the table's 30 terse label/value pairs get diluted by 5 kB
  of prose and BM25 stops finding them. They are split.
- A **journal abstract** is a single argument. Cutting it in half puts the
  method in one chunk and the finding in the other, and the finding is the part
  worth retrieving. It stays whole unless it is genuinely long.
- A **collection precedent** is already written as one dense paragraph by
  `tate.py`, sized deliberately. It passes through.

Two rules apply to every chunk regardless of layer:

1. **Self-containment.** A chunk reading "cure time is 4 hours at room
   temperature" is worthless once retrieved -- four hours of *what*? Every
   chunk therefore opens with a context line naming its document. This costs a
   few tokens and is the difference between a citable answer and a plausible
   one.
2. **`source_type` travels with the text.** Downstream fusion has to be able to
   tell a manufacturer's claim from a conservator's, because the whole premise
   of this system is that those two sources disagree and the disagreement is
   the answer.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --- boilerplate removal ----------------------------------------------------

# Paragraphs appearing on more than this share of manufacturer pages are site
# chrome, not product information. The measured document-frequency curve has a
# wide gap here (53% -> 29%), so the threshold sits in empty space rather than
# on a slope; it was not tuned to a target output size.
BOILERPLATE_DF = 0.30

# The cross-sell teasers that survive the DF cut ("Colorants for Urethane
# Rubber, Resin and Foam", 29%) cannot be removed by lowering that threshold:
# the next real paragraph below them is a shelf-life safety warning at 22%, and
# a corpus filter that deletes safety text to catch an advert is a bad trade.
#
# They are removed on a different axis instead. They are widget labels, not
# prose -- every one is under 60 characters, while the shortest genuine
# technical paragraph in the corpus is 104. Length only disqualifies a
# paragraph that also *repeats*, so a one-line note unique to one product is
# kept.
TEASER_MAX_CHARS = 60
TEASER_MIN_DF = 0.05


def boilerplate_paragraphs(descriptions: list[str]) -> set[str]:
    """Corpus-level, not a hardcoded blocklist.

    A blocklist would be shorter to write and would rot the first time
    Smooth-On edits its footer. Measuring document frequency over whatever was
    actually scraped keeps the filter honest when the corpus changes.
    """
    n = len(descriptions)
    df: Counter[str] = Counter()
    for description in descriptions:
        for para in set(description.split("\n\n")):
            df[para] += 1

    return {
        para
        for para, count in df.items()
        if count / n > BOILERPLATE_DF
        or (len(para) <= TEASER_MAX_CHARS and count / n > TEASER_MIN_DF)
    }


# --- chunk model ------------------------------------------------------------

# Long abstracts are windowed rather than truncated. 2200 characters is roughly
# the 90th percentile of the literature layer, so the split path runs on ~7% of
# records and the common case stays whole.
ABSTRACT_MAX_CHARS = 2200
ABSTRACT_OVERLAP_SENTENCES = 1

# Narrative windows are smaller: datasheet prose is a sequence of loosely
# related notes (mixing, curing, demoulding, safety), so packing more of it
# into one embedding blurs the vector across topics that a query asks about
# separately.
NARRATIVE_TARGET_CHARS = 900


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_type: str
    chunk_type: str  # spec | narrative | abstract | precedent
    title: str
    text: str
    url: str | None = None
    metadata: dict = field(default_factory=dict)


# Sentence boundaries, in two forms. The first is ordinary prose. The second
# has no whitespace at all, and exists because OpenAlex abstracts are not
# stored as text: they are stored as an inverted index of token -> positions,
# and reconstruction rejoins tokens with single spaces. When the deposit lost a
# separator the result is "self-healing capabilities.The self-healing mechanism
# is driven by", a real boundary with the space missing.
#
# Measured over the kept literature layer: 93 of 1521 abstracts (6%) contain
# one, 364 occurrences, 2.5% of all sentence boundaries in the corpus. Without
# the second pattern those abstracts silently under-split, and the few with
# many occurrences produce single "sentences" thousands of characters long.
SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z(])"  # normal: terminator, space, capital
    r"|(?<=[a-z)\"'][.!?])(?=[A-Z])"  # glued: terminator, no space, capital
)

# A sentence longer than this is not a sentence. It means the record's
# punctuation did not survive deposit at all -- see `is_degenerate`.
MAX_PLAUSIBLE_SENTENCE = 1500


def split_sentences(text: str) -> list[str]:
    """Good enough for abstracts, deliberately not a parser.

    The lookbehind spares the abbreviations that actually occur in this corpus
    (et al., e.g., approx., vs.) and the decimal points in measurements, which
    is where a naive split-on-period does visible damage: "cured for 1." /
    "5 hours". The glued branch requires a lowercase letter, quote, or closing
    bracket before the terminator for the same reason -- "1.5" and "Fig.2A"
    must not split.
    """
    parts = SENTENCE_BOUNDARY.split(text.strip())
    return [p for p in (p.strip() for p in parts) if p]


# Publisher markup that survives into OpenAlex abstracts. Rare -- 3 records
# carry MathML, 11 carry inline HTML, 45 carry entities -- but a single MathML
# block is 900 characters of `<mml:mi>` tags with no sentence boundary in it,
# which is enough to make an otherwise clean paper look unparseable.
MARKUP = re.compile(
    r"<mml:math.*?</mml:math>"  # MathML block
    r"|\$\$.*?\$\$"  # display LaTeX
    r"|</?[a-z]{1,10}(?:\s[^<>]*)?>",  # inline HTML tags
    re.S,
)
ENTITY = re.compile(r"&(?:[a-zA-Z]+|#\d+);")


def clean_abstract(text: str) -> str:
    """Strip publisher markup, then normalise whitespace.

    Done before splitting rather than after, because the markup is what breaks
    the split: the Paraloid B44/B72/Incralac permeability paper -- directly on
    topic for "which adhesives do conservators avoid" -- carries a 2.7 kB
    MathML equation that reads as one unbroken sentence.
    """
    text = MARKUP.sub(" ", text)
    text = ENTITY.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_degenerate(text: str) -> bool:
    """True when reconstruction produced unreadable output.

    Two records in the corpus reconstruct to tens of kB of word salad -- "For
    the corrosion-resistant was to occur only by localized corrosion that was
    by a pit growth rate that with" -- because their inverted index has gaps.
    The connective words are simply absent, so this is not recoverable by
    better parsing.

    The test is structural, not linguistic: after cleaning and splitting, no
    plausible sentence runs past `MAX_PLAUSIBLE_SENTENCE` characters.

    It applies only to abstracts long enough to need splitting. An abstract
    that fits in a single chunk is never split, so its internal sentence
    structure is irrelevant -- and testing it anyway rejected a perfectly
    readable Arabic conservation paper, whose script has no uppercase for the
    boundary rule to key on. A splitting heuristic must not become a language
    filter by accident.
    """
    if len(text) <= ABSTRACT_MAX_CHARS:
        return False
    return any(len(s) > MAX_PLAUSIBLE_SENTENCE for s in split_sentences(text))


def pack(units: list[str], target: int, overlap: int = 0) -> list[str]:
    """Greedily group text units into windows of roughly `target` chars.

    Splitting happens between units wherever possible, so a sentence or a
    paragraph is not cut in half. A unit that is itself larger than the target
    is cut on whitespace as a last resort: letting it through whole was how a
    single record produced a 15 kB chunk that no embedding model would accept.
    """
    windows: list[list[str]] = []
    current: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal current, size
        if current:
            windows.append(current)
        current = current[-overlap:] if overlap else []
        size = sum(len(u) for u in current)

    for unit in units:
        while len(unit) > target:
            cut = unit.rfind(" ", 0, target)
            flush()
            windows.append([unit[: cut if cut > 0 else target]])
            unit = unit[cut + 1 if cut > 0 else target :]
            current, size = [], 0
        if current and size + len(unit) > target:
            flush()
        current.append(unit)
        size += len(unit)
    if current:
        windows.append(current)
    return [" ".join(w) for w in windows]


# --- per-layer chunking -----------------------------------------------------


def chunk_manufacturer(row: dict, boilerplate: set[str]) -> list[Chunk]:
    doc_id = f"smoothon:{row['slug']}"
    name = row["name"]
    category = row["category"].replace("-", " ")
    header = f"{name} ({category}, Smooth-On)"
    chunks: list[Chunk] = []

    if row["specs"]:
        # Rendered as label/value lines rather than a sentence. The retrieval
        # target here is a literal token -- someone types "Shore 30A" or "pot
        # life 50 minutes" -- and prose-ifying it ("the Shore hardness is 30A")
        # inserts filler between the two terms that have to match.
        lines = "\n".join(f"{k}: {v}" for k, v in row["specs"].items())
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}#spec",
                doc_id=doc_id,
                source_type="manufacturer_datasheet",
                chunk_type="spec",
                title=name,
                text=f"Technical specifications for {header}.\n{lines}",
                url=row["url"],
                metadata={
                    "category": row["category"],
                    "spec_labels": list(row["specs"]),
                    "safety_data_sheet_pdf": row.get("safety_data_sheet_pdf"),
                },
            )
        )

    paragraphs = [
        p
        for p in (p.strip() for p in row["description"].split("\n\n"))
        if p and p not in boilerplate
    ]
    for i, window in enumerate(pack(paragraphs, NARRATIVE_TARGET_CHARS)):
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}#narrative-{i}",
                doc_id=doc_id,
                source_type="manufacturer_datasheet",
                chunk_type="narrative",
                title=name,
                text=f"{header}. {window}",
                url=row["url"],
                metadata={"category": row["category"], "window": i},
            )
        )
    return chunks


def chunk_literature(row: dict) -> list[Chunk]:
    doc_id = row["doc_id"]
    journal = row.get("journal") or "unknown journal"
    year = row.get("year") or "n.d."
    header = f'"{row["title"]}" ({journal}, {year})'
    abstract = clean_abstract(row["abstract"])

    if len(abstract) <= ABSTRACT_MAX_CHARS:
        windows = [abstract]
    else:
        windows = pack(
            split_sentences(abstract),
            ABSTRACT_MAX_CHARS,
            overlap=ABSTRACT_OVERLAP_SENTENCES,
        )

    meta = {
        "journal": journal,
        "year": year,
        "doi": row.get("doi"),
        "domain": row.get("relevance", {}).get("domain"),
        "screened": row.get("relevance", {}).get("screened"),
    }
    return [
        Chunk(
            chunk_id=f"{doc_id}#abstract-{i}" if len(windows) > 1 else f"{doc_id}#abstract",
            doc_id=doc_id,
            source_type=row["source_type"],
            chunk_type="abstract",
            title=row["title"],
            text=f"{header}. {window}",
            url=row.get("landing_url") or row.get("doi"),
            metadata=meta,
        )
        for i, window in enumerate(windows)
    ]


def chunk_precedent(row: dict) -> list[Chunk]:
    # `to_passage()` already emits a self-contained paragraph with its own
    # scope disclaimer, so re-wrapping it here would only duplicate the header.
    return [
        Chunk(
            chunk_id=f"{row['doc_id']}#precedent",
            doc_id=row["doc_id"],
            source_type="collection_precedent",
            chunk_type="precedent",
            title=f"{row['material']} in the Tate collection",
            text=row["passage"],
            url="https://github.com/tategallery/collection",
            metadata={
                "material": row["material"],
                "artwork_count": row["artwork_count"],
                "first_year": row["first_year"],
                "last_year": row["last_year"],
            },
        )
    ]


# --- driver -----------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def build(raw_dir: Path, keep_filtered: bool = False) -> list[Chunk]:
    chunks: list[Chunk] = []

    products = read_jsonl(raw_dir / "smoothon.jsonl")
    boilerplate = boilerplate_paragraphs([p["description"] for p in products])
    for product in products:
        chunks.extend(chunk_manufacturer(product, boilerplate))

    dropped = 0
    for work in read_jsonl(raw_dir / "openalex_screened.jsonl"):
        # The gate's verdict is applied here, not in the gate itself, so the
        # unfiltered corpus stays on disk and the retrieval evaluation can be
        # re-run with `--keep-filtered` to measure what the screening bought.
        if not (keep_filtered or work["relevance"]["relevant"]):
            continue
        if is_degenerate(clean_abstract(work["abstract"])):
            dropped += 1
            continue
        chunks.extend(chunk_literature(work))
    if dropped:
        print(f"  dropped {dropped} works with unreadable reconstructed abstracts")

    for precedent in read_jsonl(raw_dir / "tate.jsonl"):
        chunks.extend(chunk_precedent(precedent))

    return chunks


if __name__ == "__main__":
    import argparse
    import statistics

    parser = argparse.ArgumentParser(description="Chunk the raw corpora")
    parser.add_argument("--raw", default="data/raw", type=Path)
    parser.add_argument("--out", default="data/chunks.jsonl", type=Path)
    parser.add_argument("--keep-filtered", action="store_true")
    args = parser.parse_args()

    chunks = build(args.raw, args.keep_filtered)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    print(f"wrote {len(chunks)} chunks -> {args.out}")
    by_type = Counter(c.chunk_type for c in chunks)
    by_source = Counter(c.source_type for c in chunks)
    lengths = [len(c.text) for c in chunks]
    for label, counter in (("chunk_type", by_type), ("source_type", by_source)):
        print(f"  {label}:")
        for key, count in counter.most_common():
            print(f"    {count:>6}  {key}")
    print(
        f"  chars: median {int(statistics.median(lengths))}, "
        f"p90 {int(statistics.quantiles(lengths, n=10)[8])}, max {max(lengths)}"
    )
