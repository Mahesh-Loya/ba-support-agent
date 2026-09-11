"""Intent classification. Returns a self-reported confidence whose calibration
we later MEASURE rather than trust (see src/eval/risk_coverage.py)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src import config, llm, taxonomy


@dataclass
class Classification:
    intent: str
    confidence: float


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def _definitions() -> dict:
    return json.loads(taxonomy.TAXONOMY_PATH.read_text(encoding="utf-8"))["definitions"]


def build_prompt(text: str, intents: list[str], definitions: dict) -> str:
    lines = "\n".join(f"- {i}: {definitions.get(i, '')}" for i in intents)
    return (
        "You are triaging a customer support tweet sent to British Airways.\n"
        "Classify it into exactly one intent.\n\n"
        f"Intents:\n{lines}\n\n"
        f"Message:\n\"\"\"{text}\"\"\"\n\n"
        'Reply with JSON only: {"intent": "<one of the intents above>", '
        '"confidence": <0.0-1.0>}\n'
        "confidence is your probability that this label is correct."
    )


def parse_response(raw: str, intents: list[str]) -> Classification:
    cleaned = _FENCE.sub("", raw.strip())
    m = re.search(r"\{.*\}", cleaned, re.S)
    if not m:
        return Classification("other", 0.0)
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Classification("other", 0.0)

    intent = str(data.get("intent", "")).strip()
    if intent not in intents:
        return Classification("other", 0.0)
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return Classification(intent, max(0.0, min(1.0, conf)))


def classify(text: str, use_cache: bool = True) -> Classification:
    intents = taxonomy.load_intents()
    raw = llm.complete(
        build_prompt(text, intents, _definitions()),
        provider=config.DRAFTER_PROVIDER, model=config.DRAFTER_MODEL,
        max_tokens=100, use_cache=use_cache,
    )
    return parse_response(raw, intents)
