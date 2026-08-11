"""Conservative, zero-cost model routing after retrieval.

The cheap route is intentionally narrow. A false "complex" decision costs a
few cents; a false "simple" decision can flatten the source disagreement this
project exists to surface. Rules use only the question and retrieved metadata,
so routing adds no model call and every decision is auditable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from rag.search import Hit

COMPLEX_MODEL = os.environ.get("GENERATE_MODEL", "claude-sonnet-5")
SIMPLE_MODEL = os.environ.get("SIMPLE_GENERATE_MODEL", "claude-haiku-4-5")

LOOKUP_CUES = re.compile(
    r"\b(pot life|working time|cure time|demou?ld time|mix ratio|shore [ad]?|"
    r"hardness|viscosity|tensile strength|tear strength|elongation|shrinkage|"
    r"specific gravity|temperature range|colour|color)\b",
    re.I,
)
COMPLEX_CUES = re.compile(
    r"\b(compare|versus|vs\.?|which|better|best|choose|recommend(?:ed|ation)?|"
    r"should|why|avoid|safe|safety|toxic|gloves?|compatible|compatibility|"
    r"inhibit|release agent|failed|failure|won't|will not|outdoors?|outside|"
    r"weather|"
    r"uv|sunlight|ageing|aging|years?|long[- ]term|yellow|degrad|corrosion|"
    r"coast|sea|freeze|humidity)\b",
    re.I,
)


@dataclass(frozen=True)
class ModelRoute:
    model: str
    tier: str
    reason: str


def routing_enabled() -> bool:
    return os.environ.get("MODEL_ROUTING_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off"
    }


def route_question(question: str, hits: list[Hit], variant: str) -> ModelRoute:
    if not routing_enabled():
        return ModelRoute(COMPLEX_MODEL, "complex", "routing disabled")
    if SIMPLE_MODEL == COMPLEX_MODEL:
        return ModelRoute(COMPLEX_MODEL, "complex", "simple and complex models match")
    if variant == "no_context":
        return ModelRoute(COMPLEX_MODEL, "complex", "no-context control")
    if not hits:
        return ModelRoute(COMPLEX_MODEL, "complex", "no retrieved evidence")
    if len(question) > 180:
        return ModelRoute(COMPLEX_MODEL, "complex", "long or multi-part question")
    if COMPLEX_CUES.search(question):
        return ModelRoute(COMPLEX_MODEL, "complex", "comparison, risk, or ageing cue")

    source_types = {hit.source_type for hit in hits}
    if source_types != {"manufacturer_datasheet"}:
        return ModelRoute(COMPLEX_MODEL, "complex", "multiple or independent sources")
    if hits[0].chunk_type != "spec":
        return ModelRoute(COMPLEX_MODEL, "complex", "top evidence is not a spec table")
    if not LOOKUP_CUES.search(question):
        return ModelRoute(COMPLEX_MODEL, "complex", "no explicit specification cue")

    return ModelRoute(SIMPLE_MODEL, "simple", "single-source specification lookup")
