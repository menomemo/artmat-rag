"""Query rewriting: from what the user wants to what the corpus calls it.

This exists because of a specific, reproducible failure. Ask the system

    "how long can I work with the silicone before it starts to set"

and the answer is sitting in a spec chunk under the label **Pot Life**. The
question does not contain that phrase, and neither does any paraphrase a person
would produce who did not already know the term -- which is precisely the
person asking. BM25 has nothing to match; the dense model gets closer but
retrieves narrative about mixing rather than the table with the number in it.

The vocabulary gap here is not incidental, it is the problem domain. Three
communities wrote this corpus and none of them use the artist's words:
manufacturers say "pot life" and "Shore A", conservation scientists say
"photo-oxidative degradation" and "Paraloid B72", a catalogue says "polychromed
aluminium". A studio question says "how long have I got", "how squishy", "will
it go yellow", "what did people actually build with".

So the rewrite is a translation step, not a paraphrase step. It emits the terms
the *documents* would use, and retrieval runs over the original and the rewrite
together -- fused, not substituted. Substitution would be a regression: the
rewrite is a guess, and when the guess is wrong the original query is the only
thing standing between the user and a confidently irrelevant answer.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import anthropic
from qdrant_client import QdrantClient, models

import rag.env  # noqa: F401  -- loads .env on import

from rag.index import COLLECTION, DENSE, SPARSE, dense_doc, sparse_doc
from rag.search import (
    CANDIDATES,
    MMR_CANDIDATES,
    MMR_RELEVANCE_HEAD,
    Hit,
    _reranker,
    _source_filter,
    candidates_from_points,
    diversify,
)

# Rewriting is a per-query, user-facing call, so it runs on the cheapest model
# that can do it. It is a vocabulary lookup with a little judgement, not
# reasoning -- and it sits in the latency path of every search, where Sonnet
# would add roughly a second for no measurable gain.
MODEL = os.environ.get("REWRITE_MODEL", "claude-haiku-4-5")
MAX_TOKENS = 400

SYSTEM_PROMPT = """\
You translate studio questions into the vocabulary of technical documents.

The corpus has three voices, none of which is the asker's:
- manufacturer datasheets: pot life, demould time, Shore A/D hardness, mixed \
viscosity, tensile strength, elongation at break, exotherm, cure schedule, \
mix ratio, shrinkage, inhibition
- conservation and materials science: photo-oxidative degradation, yellowing \
index, UV stabiliser, hydrolytic stability, freeze-thaw cycling, carbonation, \
chloride-induced pitting, consolidant, reversibility, Paraloid B72
- a museum catalogue: material names as accessioned, e.g. polychromed \
aluminium, glass-reinforced polyester

Given a question, return:

1. `rewritten`: the same question restated using the terms a document would \
use. Keep it a question. Do not invent specifics the asker did not imply -- \
if they did not name a material, do not choose one for them.
2. `terms`: 3 to 8 individual technical terms or phrases likely to appear \
verbatim in a relevant document. These feed a keyword retriever, so prefer \
exact document vocabulary over descriptions.

If the question already uses technical vocabulary, say so by returning it \
nearly unchanged and listing its key terms. Do not pad.

Example. Question: "how long can I work with the silicone before it sets?"
rewritten: "What is the pot life and working time of platinum-cure silicone \
rubber before it begins to gel?"
terms: ["pot life", "working time", "gel time", "cure time", "platinum cure"]\
"""

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten": {
            "type": "string",
            "description": "The question restated in document vocabulary.",
        },
        "terms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Technical terms likely to appear verbatim.",
        },
    },
    "required": ["rewritten", "terms"],
    "additionalProperties": False,
}


@dataclass
class Rewrite:
    original: str
    rewritten: str
    terms: list[str] = field(default_factory=list)
    used: bool = True

    @property
    def keyword_query(self) -> str:
        """What the lexical arm searches for.

        The original text stays in front of the extracted terms. If the rewrite
        guessed badly, the user's own words still carry weight in BM25 rather
        than being replaced by a wrong guess.
        """
        return " ".join([self.original, *self.terms])


def rewrite(query: str, client: anthropic.Anthropic | None = None) -> Rewrite:
    """Translate a query, or fall back to the original.

    A rewriting failure must never be a search failure. If the API errors or
    returns something unusable, the caller gets a `Rewrite` marked `used=False`
    that behaves exactly like no rewriting at all -- degraded, not broken.
    """
    client = client or anthropic.Anthropic()
    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": REWRITE_SCHEMA}},
            messages=[{"role": "user", "content": query}],
        )
        text = next((b.text for b in message.content if b.type == "text"), None)
        if not text:
            return Rewrite(original=query, rewritten=query, used=False)
        parsed = json.loads(text)
        return Rewrite(
            original=query,
            rewritten=parsed.get("rewritten") or query,
            terms=[t for t in parsed.get("terms", []) if isinstance(t, str)],
        )
    except (anthropic.APIError, json.JSONDecodeError, KeyError):
        return Rewrite(original=query, rewritten=query, used=False)


def search_rewritten(
    client: QdrantClient,
    rw: Rewrite,
    limit: int = 5,
    rerank: bool = True,
    source_types: list[str] | None = None,
    diversify_results: bool = True,
) -> list[Hit]:
    """Hybrid retrieval over the original and the rewrite together.

    Three prefetch arms, fused by RRF inside Qdrant:

      dense(original)   -- what the user actually meant
      dense(rewritten)  -- the same intent in document vocabulary
      sparse(original + extracted terms) -- the exact strings to match

    The dense arm keeps the original because embeddings tolerate paraphrase and
    the user's phrasing is the ground truth for intent. The sparse arm keeps it
    too, but appended with the technical terms, since that is the arm that
    could not match anything before.
    """
    diversity_active = diversify_results and limit > MMR_RELEVANCE_HEAD
    points = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=dense_doc(rw.original), using=DENSE, limit=CANDIDATES
            ),
            models.Prefetch(
                query=dense_doc(rw.rewritten), using=DENSE, limit=CANDIDATES
            ),
            models.Prefetch(
                query=sparse_doc(rw.keyword_query), using=SPARSE, limit=CANDIDATES
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=(
            CANDIDATES
            if rerank
            else min(CANDIDATES, max(limit, MMR_CANDIDATES))
        ),
        query_filter=_source_filter(source_types),
        with_payload=True,
        with_vectors=[DENSE] if diversity_active and not rerank else False,
    ).points
    points.sort(
        key=lambda point: (
            -point.score, (point.payload or {}).get("chunk_id", "")
        )
    )

    if diversity_active and not rerank:
        return diversify(candidates_from_points(points), limit)

    hits = [Hit.from_point(p) for p in points]
    if not rerank or not hits:
        return hits[:limit]

    # Reranked against the *original* question. The rewrite is a retrieval aid;
    # scoring relevance against it would judge documents by how well they match
    # a machine's paraphrase rather than what the person asked.
    scores = list(_reranker().rerank(rw.original, [h.text for h in hits]))
    for hit, score in zip(hits, scores):
        hit.score = float(score)
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]


if __name__ == "__main__":
    import argparse

    from rag.index import connect

    parser = argparse.ArgumentParser(description="Rewrite a query and search")
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    rw = rewrite(args.query)
    print(f"original : {rw.original}")
    print(f"rewritten: {rw.rewritten}")
    print(f"terms    : {rw.terms}")
    if not rw.used:
        print("(rewrite failed; falling back to the original query)")
    print()

    client = connect()
    for i, hit in enumerate(
        search_rewritten(client, rw, args.limit, rerank=not args.no_rerank), 1
    ):
        print(f"{i}. [{hit.source_type}] {hit.title[:66]}  ({hit.score:.4f})")
        print(f"   {hit.text[:170].strip()}...")
