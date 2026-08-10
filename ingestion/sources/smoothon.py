"""Smooth-On product datasheets: the *manufacturer* layer of the knowledge base.

This is the voice that says "UV resistant" and "Shore 30A". It is precise,
quantitative, and structurally optimistic -- it reports accelerated-ageing
numbers, not what a piece looks like after ten winters outdoors. The
conservation layer (see `tate.py`, `openalex.py`) is what argues back.

Scraping policy: smooth-on.com/robots.txt disallows only `/compare/*`; product
and category pages are explicitly permitted. We identify ourselves honestly,
rate-limit, and cache every response to disk so that re-running the pipeline
during development costs the site nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import httpx
from selectolax.parser import HTMLParser
from tenacity import retry, stop_after_attempt, wait_exponential

BASE = "https://www.smooth-on.com"

# Contact address included per common crawling etiquette: a site owner who wants
# us to stop should be able to reach us without having to block an IP range.
# Read from the environment, because the same address hardcoded into a public
# repository is one a scraper harvests long before a site owner ever needs it.
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")
USER_AGENT = "artmat-rag/0.1 (educational RAG project" + (
    f"; +mailto:{CONTACT_EMAIL})" if CONTACT_EMAIL else ")"
)

REQUEST_DELAY_S = 0.7

# The 14 categories that constitute mould-making and casting. `tools-equipment`
# and the FX/lifecasting lines are deliberately excluded: they are about
# process hardware and skin contact rather than material selection, and they
# would dilute the retrieval corpus without answering any question in scope.
CATEGORIES = [
    # mould side
    "platinum-silicone",
    "tin-silicone",
    "urethane-rubber",
    "polysulfide-rubber",
    # casting side
    "urethane-resin",
    "epoxy-casting-and-laminating-resins",
    "concrete-gypsum-additives",
    "urethane-expanding-foams",
    "silicone-expanding-foam-platinum-cure",
    # process
    "sealers-release-agents",
    "adhesives",
    "color-and-fillers",
    "epoxy-putties",
    # durability
    "epoxy-urethane-coatings",
]


@dataclass
class Product:
    """One manufacturer datasheet, flattened.

    `specs` stays a dict rather than being exploded into columns because the
    label set is ragged across product families -- a silicone reports Shore A
    and a rigid resin reports Shore D, and neither has the other's field.
    """

    slug: str
    url: str
    name: str
    category: str
    description: str
    specs: dict[str, str] = field(default_factory=dict)
    technical_bulletin_pdf: str | None = None
    safety_data_sheet_pdf: str | None = None
    source_type: str = "manufacturer_datasheet"

    @property
    def doc_id(self) -> str:
        return f"smoothon:{self.slug}"


class Fetcher:
    """Rate-limited, disk-cached, retrying HTTP client."""

    def __init__(self, cache_dir: Path, delay: float = REQUEST_DELAY_S):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self._last_request = 0.0
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=30.0,
        )
        self.hits = 0
        self.misses = 0

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:16]
        slug = re.sub(r"[^a-z0-9]+", "-", url.lower().split(BASE)[-1]).strip("-")[:60]
        return self.cache_dir / f"{slug}-{digest}.html"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
    def _get(self, url: str) -> str:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.monotonic()
        response = self._client.get(url)
        response.raise_for_status()
        return response.text

    def get(self, url: str) -> str:
        path = self._cache_path(url)
        if path.exists():
            self.hits += 1
            return path.read_text(encoding="utf-8")
        html = self._get(url)
        path.write_text(html, encoding="utf-8")
        self.misses += 1
        return html

    def close(self) -> None:
        self._client.close()


def list_category(fetcher: Fetcher, category: str) -> list[str]:
    """Return product slugs listed under one category page."""
    html = fetcher.get(f"{BASE}/category/{category}/")
    slugs = re.findall(r'href="(?:https://www\.smooth-on\.com)?/products/([^"#?/]+)/?"', html)
    # dict.fromkeys preserves page order while de-duplicating; order is stable
    # across runs, which keeps the ingested corpus diffable.
    return list(dict.fromkeys(slugs))


def _clean(text: str) -> str:
    # &thinsp; and friends survive selectolax's unescaping as exotic whitespace
    # and would otherwise show up mid-number as "9 minutes" with a U+2009.
    return re.sub(r"\s+", " ", text.replace(" ", " ").replace("\xa0", " ")).strip()


def parse_product(html: str, slug: str, category: str) -> Product:
    tree = HTMLParser(html)

    name_node = tree.css_first("h1")
    name = _clean(name_node.text()) if name_node else slug

    specs: dict[str, str] = {}
    table = tree.css_first("#specs-table")
    if table:
        for row in table.css("tr"):
            cells = row.css("td")
            if len(cells) >= 2:
                label = _clean(cells[0].text())
                value = _clean(cells[1].text())
                if label and value:
                    specs[label] = value

    # The narrative sits between the <h1> and the technical data block. Rather
    # than guess at a container class, take every <p> that precedes the specs
    # table -- resilient to the template shuffling its wrappers around.
    paragraphs: list[str] = []
    for node in tree.css("p"):
        text = _clean(node.text())
        if len(text) > 40:
            paragraphs.append(text)
    description = "\n\n".join(dict.fromkeys(paragraphs))[:6000]

    # The datasheet links live inside an HTML comment -- the template ships a
    # commented-out block and renders the visible widget with JS. A DOM query
    # therefore returns nothing, so scan the raw markup instead. The URLs are
    # live (verified 200 application/pdf); losing them would cost us the entire
    # safety layer, which is the only source for toxicity questions.
    tb_pdf = sds_pdf = None
    for href in re.findall(r'href="(https?://[^"]+\.pdf)"', html):
        if "/tb/" in href:
            tb_pdf = tb_pdf or href
        elif "/msds/" in href:
            sds_pdf = sds_pdf or href

    return Product(
        slug=slug,
        url=f"{BASE}/products/{slug}/",
        name=name,
        category=category,
        description=description,
        specs=specs,
        technical_bulletin_pdf=tb_pdf,
        safety_data_sheet_pdf=sds_pdf,
    )


def iter_products(cache_dir: Path, categories: list[str] | None = None):
    """Yield every product across the in-scope categories.

    A product listed under two categories is emitted once, under the first
    category that claims it.
    """
    fetcher = Fetcher(cache_dir)
    seen: set[str] = set()
    try:
        for category in categories or CATEGORIES:
            for slug in list_category(fetcher, category):
                if slug in seen:
                    continue
                seen.add(slug)
                html = fetcher.get(f"{BASE}/products/{slug}/")
                yield parse_product(html, slug, category)
    finally:
        fetcher.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Smooth-On datasheets")
    parser.add_argument("--cache", default="data/cache/smoothon", type=Path)
    parser.add_argument("--out", default="data/raw/smoothon.jsonl", type=Path)
    parser.add_argument("--categories", nargs="*", default=None)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for product in iter_products(args.cache, args.categories):
            handle.write(json.dumps(asdict(product), ensure_ascii=False) + "\n")
            count += 1
            if count % 25 == 0:
                print(f"  ... {count} products", flush=True)
    print(f"wrote {count} products -> {args.out}")
