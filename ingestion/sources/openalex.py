"""OpenAlex: the *conservation and materials science* layer.

This is the voice that answers back. Where a manufacturer datasheet reports
500 hours of accelerated weathering, this layer reports what a polystyrene
adhesive bond looks like after light ageing, and why museums pick Paraloid B72
(reversibility) over anything stronger.

Two distinct source types are collected, because they carry different
authority and should be weighted differently at rerank time:

  conservation_literature -- Heritage Science, Studies in Conservation,
      Journal of Cultural Heritage. Answers "what happened to real objects,
      and what do institutions actually do".
  materials_science -- Polymer Degradation and Stability, Progress in Organic
      Coatings, Construction and Building Materials. Answers "why, and how
      fast" with quantitative ageing data.

Limitation, stated plainly: OpenAlex exposes abstracts, not full text. Answers
grounded in this layer are therefore grounded in abstracts. This is a real
ceiling on depth and is documented in the README rather than papered over.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path

USER_AGENT = (
    "artmat-rag/0.1 (educational RAG project; +mailto:today_is_yihong@outlook.com)"
)
# OpenAlex asks callers to identify themselves via `mailto`; doing so puts the
# request in their faster, more reliable "polite pool".
MAILTO = "today_is_yihong@outlook.com"
API = "https://api.openalex.org/works"

# An API key lifts the daily allowance from $0.10 (anonymous) to $1.00, i.e.
# from 1,000 requests to 10,000, and any prepaid balance stacks on top.
#
# It travels in the Authorization header rather than the documented `api_key`
# query parameter: a key in a query string leaks into access logs, proxy
# records, and -- here specifically -- the on-disk response cache's filenames.
API_KEY = os.environ.get("OPENALEX_API_KEY")

# Pages of PER_PAGE results to pull per query. One page was a concession to the
# anonymous quota; with a key, deeper paging costs 1 request each and directly
# raises corpus density. That matters for the retrieval evaluation: on a thin
# corpus hit-rate is inflated by having too few candidates to choose between,
# and the three retrieval strategies stop being distinguishable.
MAX_PAGES = int(os.environ.get("OPENALEX_MAX_PAGES", "5"))

# Two collection strategies, because the two layers behave differently in
# OpenAlex.
#
# Conservation journals are whitelisted: the field is small, the whitelist is
# high-precision, and every hit is on-topic.
#
# Materials science cannot be whitelisted the same way. The obvious journals --
# Polymer Degradation and Stability, Progress in Organic Coatings,
# Construction and Building Materials -- are all Elsevier, and Elsevier does
# not deposit abstracts to Crossref/OpenAlex: 48 of 50 sampled records came
# back with no `abstract_inverted_index` at all. Filtering on
# `has_abstract:true, open_access.is_oa:true` instead routes us to publishers
# that do deposit (MDPI, Springer, Frontiers, PLOS) and lifts the usable rate
# from 4% to ~98%.
#
# The cost of dropping the journal whitelist is keyword collision across
# disciplines: "resin yellowing" pulls in dental composites, "adhesive" pulls
# in sea-urchin biology. A field filter helps but does not fix it, so this
# strategy is paired with an LLM relevance gate (see `relevance_gate.py`).
CONSERVATION_JOURNALS = [
    "S2736968195",   # Heritage Science (fully open access)
    "S12492028",     # Studies in Conservation
    "S1016481467",   # Journal of Cultural Heritage
]

# Materials Science, Engineering, Chemistry, Arts & Humanities, Chemical Eng.
MATERIALS_FIELDS = "25|22|16|12|15"

# Each term is here because a working artist asks a question that needs it.
# Kept explicit rather than generated, so the corpus's coverage is auditable:
# if the system cannot answer something, you can see whether the gap is
# retrieval or simply an unsearched topic.
QUERIES = [
    # "which resin yellows slowest?"
    "resin yellowing", "epoxy UV degradation", "polyurethane photodegradation",
    "polymer discolouration ageing", "light ageing transparent polymer",
    # "what happens to outdoor work after ten years?"
    "outdoor sculpture weathering", "outdoor bronze corrosion",
    "accelerated weathering coating", "public art deterioration",
    # "why don't museums use that glue?"
    "adhesive reversibility conservation", "Paraloid B72",
    "adhesive ageing conservation", "consolidant contemporary art",
    # "cast aluminium or stainless steel?"
    "metal sculpture corrosion patina", "stainless steel atmospheric corrosion",
    # "which concrete can do thin walls?"
    "ultra high performance concrete", "glass fibre reinforced concrete",
    "thin section cementitious composite",
    # mould and casting materials as objects of study
    "silicone rubber mould degradation", "polyester resin casting deterioration",
    "plastics conservation contemporary art", "acrylic sheet crazing",
    "polyurethane foam degradation museum",
]

PER_PAGE = 50


@dataclass
class Work:
    openalex_id: str
    doi: str | None
    title: str
    abstract: str
    journal: str
    year: int | None
    is_oa: bool
    landing_url: str | None
    matched_queries: list[str] = field(default_factory=list)
    source_type: str = "conservation_literature"

    @property
    def doc_id(self) -> str:
        return f"openalex:{self.openalex_id.rsplit('/', 1)[-1]}"


# OpenAlex now meters the free tier by daily credits, not just burst rate:
# `x-ratelimit-limit: 1000` requests against an `x-ratelimit-limit-usd: 0.1`
# budget, resetting once a day. When the budget is spent, a 429 comes back with
# a `Retry-After` measured in *hours* -- 33787 seconds in testing.
#
# Honouring that header literally is a trap: the collector goes to sleep for
# most of a day and looks like it is still working. So the wait is capped, and
# anything longer is raised as a QuotaExhausted the caller can act on.
MAX_BACKOFF_S = 120


class QuotaExhausted(RuntimeError):
    """Daily credit budget spent; retrying today will not help."""

    def __init__(self, reset_seconds: float):
        self.reset_seconds = reset_seconds
        hours = reset_seconds / 3600
        super().__init__(
            f"OpenAlex daily quota exhausted; resets in {hours:.1f}h "
            f"({reset_seconds:.0f}s). Re-run after the reset, or add prepaid "
            f"credits. Collected work already on disk is unaffected."
        )


def _get(url: str, retries: int = 5) -> dict:
    """GET with capped backoff and daily-quota awareness.

    Returns the decoded payload. Raises QuotaExhausted rather than sleeping
    through a multi-hour reset window.
    """
    for attempt in range(retries):
        try:
            headers = {"User-Agent": USER_AGENT}
            if API_KEY:
                headers["Authorization"] = f"Bearer {API_KEY}"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                remaining = response.headers.get("x-ratelimit-remaining")
                if remaining is not None and int(remaining) < 20:
                    # Stop while there is still headroom, so a later run can
                    # probe the quota without immediately tripping the limit.
                    reset = float(response.headers.get("x-ratelimit-reset") or 0)
                    print(f"    quota nearly spent ({remaining} left)", flush=True)
                    raise QuotaExhausted(reset)
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 429:
                retry_after = float(error.headers.get("Retry-After") or 0)
                if retry_after > MAX_BACKOFF_S:
                    raise QuotaExhausted(retry_after) from error
                wait = retry_after or min(10 * (attempt + 1), MAX_BACKOFF_S)
                if attempt == retries - 1:
                    raise
                print(f"    rate limited, waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
            else:
                if attempt == retries - 1:
                    raise
                time.sleep(min(2 ** attempt, MAX_BACKOFF_S))
        except QuotaExhausted:
            raise
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, MAX_BACKOFF_S))
    raise RuntimeError("unreachable")


def record_doc_id(record: dict) -> str:
    """Stable join key for a serialised work.

    `Work.doc_id` is a property, so `asdict()` never emitted it and early runs
    wrote records without the field. Records already on disk cannot be
    regenerated cheaply, so the key is derived when absent rather than assumed.
    """
    doc_id = record.get("doc_id")
    if doc_id:
        return doc_id
    return "openalex:" + str(record["openalex_id"]).rsplit("/", 1)[-1]


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]} to sidestep copyright
    on contiguous text. Rebuilding it is legitimate and expected -- their own
    docs describe the format -- but note a missing index means no abstract,
    not an empty one, and such records are dropped rather than indexed blank.
    """
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, indices in inverted_index.items():
        for index in indices:
            positions[index] = word
    return " ".join(positions[i] for i in sorted(positions))


def fetch_works(source_type: str, query: str) -> list[Work]:
    if source_type == "conservation_literature":
        filters = (
            f"primary_location.source.id:{'|'.join(CONSERVATION_JOURNALS)},"
            f"title_and_abstract.search:{query}"
        )
    else:
        filters = (
            "has_abstract:true,open_access.is_oa:true,"
            f"topics.field.id:{MATERIALS_FIELDS},"
            f"title_and_abstract.search:{query}"
        )
    records = []
    for page in range(1, MAX_PAGES + 1):
        params = urllib.parse.urlencode(
            {
                "filter": filters,
                "per-page": PER_PAGE,
                "page": page,
                "mailto": MAILTO,
            }
        )
        payload = _get(f"{API}?{params}")
        batch = payload.get("results", [])
        records.extend(batch)
        # Short page means the result set is exhausted; paging further would
        # spend quota on empty responses.
        if len(batch) < PER_PAGE:
            break
        time.sleep(0.3)

    works = []
    for record in records:
        abstract = reconstruct_abstract(record.get("abstract_inverted_index"))
        if len(abstract) < 150:
            # Too short to be a usable retrieval unit; a bare title would
            # match on keywords and then give the LLM nothing to ground on.
            continue
        location = record.get("primary_location") or {}
        source = location.get("source") or {}
        works.append(
            Work(
                openalex_id=record["id"],
                doi=record.get("doi"),
                title=record.get("title") or "",
                abstract=abstract,
                journal=source.get("display_name") or "",
                year=record.get("publication_year"),
                is_oa=(record.get("open_access") or {}).get("is_oa", False),
                landing_url=location.get("landing_page_url"),
                matched_queries=[query],
                source_type=source_type,
            )
        )
    return works


def collect(done_queries: set[tuple[str, str]] | None = None) -> tuple[list[Work], bool]:
    """Run every query against both source groups and merge by work id.

    Returns (works, complete). `complete` is False when the daily quota ran
    out partway through -- the caller should still persist what came back and
    resume after the reset. Losing a partial harvest to an exception would mean
    re-spending quota on queries that already succeeded.
    """
    done = done_queries or set()
    merged: dict[str, Work] = {}
    complete = True
    try:
        for source_type in ("conservation_literature", "materials_science"):
            for query in QUERIES:
                if (source_type, query) in done:
                    continue
                for work in fetch_works(source_type, query):
                    existing = merged.get(work.doc_id)
                    if existing:
                        # A work can surface under several queries; keep them
                        # all, they signal what the record is "about".
                        if query not in existing.matched_queries:
                            existing.matched_queries.append(query)
                    else:
                        merged[work.doc_id] = work
                time.sleep(0.6)
            print(f"  {source_type}: {len(merged)} unique works so far", flush=True)
    except QuotaExhausted as exhausted:
        print(f"\n  STOPPED: {exhausted}", flush=True)
        complete = False
    return list(merged.values()), complete


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect conservation literature")
    parser.add_argument("--out", default="data/raw/openalex.jsonl", type=Path)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Merge with anything a previous (possibly quota-truncated) run collected,
    # so re-running after the reset accumulates rather than overwrites.
    existing: dict[str, dict] = {}
    if args.out.exists():
        for line in args.out.open(encoding="utf-8"):
            record = json.loads(line)
            key = record_doc_id(record)
            record["doc_id"] = key  # backfill so the next run needs no derivation
            existing[key] = record
        print(f"resuming with {len(existing)} works already on disk")

    works, complete = collect()
    for work in works:
        # `doc_id` is a property, and asdict() only serialises fields -- it has
        # to be written in explicitly or every downstream consumer loses the
        # join key.
        record = asdict(work)
        record["doc_id"] = work.doc_id
        existing[work.doc_id] = record

    with args.out.open("w", encoding="utf-8") as handle:
        for record in existing.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(existing)} works -> {args.out}")
    if not complete:
        print("run incomplete -- re-run this script after the quota resets")
        raise SystemExit(2)
