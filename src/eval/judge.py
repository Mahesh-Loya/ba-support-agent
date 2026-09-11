"""LLM-as-judge for reply quality.

Deliberately a DIFFERENT model family from the drafter - a model grading its own
output inflates the score. How well this judge agrees with a human is measured
in src/eval/agreement.py and reported, not assumed.

Unparseable responses default to FAILING every axis, so judge flakiness can only
ever make our numbers look worse, never better.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src import config, llm
from src.agent.retrieve import Case

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)
AXES = ("grounded", "helpful", "on_brand", "no_overpromise")


@dataclass
class Verdict:
    grounded: bool
    helpful: bool
    on_brand: bool
    no_overpromise: bool
    rationale: str = ""

    @property
    def is_harmful(self) -> bool:
        """A reply is harmful if it is ungrounded or overpromises."""
        return (not self.grounded) or (not self.no_overpromise)

    def axes(self) -> dict:
        return {a: bool(getattr(self, a)) for a in AXES}


def build_judge_prompt(customer: str, reply: str, cases: list[Case]) -> str:
    hist = "\n".join(f"- {c.agent_text}" for c in cases)
    return (
        "You are auditing a draft reply from an automated British Airways support "
        "agent. Judge it strictly.\n\n"
        f"Customer message:\n\"\"\"{customer}\"\"\"\n\n"
        f"Draft reply:\n\"\"\"{reply}\"\"\"\n\n"
        f"How British Airways actually handled similar messages:\n{hist}\n\n"
        "Score four axes as true/false:\n"
        "- grounded: consistent with how BA actually handled similar cases above.\n"
        "- helpful: advances the customer's problem rather than stalling.\n"
        "- on_brand: matches BA's tone and length (short, polite, specific).\n"
        "- no_overpromise: invents NO commitment (refund, compensation, upgrade, "
        "rebooking) that the historical replies do not support.\n\n"
        'Reply with JSON only: {"grounded": bool, "helpful": bool, '
        '"on_brand": bool, "no_overpromise": bool, "rationale": "one sentence"}'
    )


_TRUE_STRINGS = {"true", "yes"}
_FALSE_STRINGS = {"false", "no"}


def _coerce_bool(value: object) -> bool:
    """Strictly coerce a judge-supplied axis value to bool.

    Fail-closed: only real booleans, 0/1 ints, and the strings
    true/false/yes/no (case-insensitive) are recognised. Anything else -
    a different string, a list, null, a float, etc. - defaults to False so a
    judge that emits a stringified boolean (e.g. "false") can never be
    silently read as a pass.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
        return False
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _TRUE_STRINGS:
            return True
        if s in _FALSE_STRINGS:
            return False
        return False
    return False


def parse_verdict(raw: str) -> Verdict:
    cleaned = _FENCE.sub("", raw.strip())
    m = re.search(r"\{.*\}", cleaned, re.S)
    if not m:
        return Verdict(False, False, False, False, "unparseable judge response")
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Verdict(False, False, False, False, "unparseable judge response")
    return Verdict(
        grounded=_coerce_bool(d.get("grounded", False)),
        helpful=_coerce_bool(d.get("helpful", False)),
        on_brand=_coerce_bool(d.get("on_brand", False)),
        no_overpromise=_coerce_bool(d.get("no_overpromise", False)),
        rationale=str(d.get("rationale", "")),
    )


def judge_reply(customer: str, reply: str, cases: list[Case],
                use_cache: bool = True) -> Verdict:
    raw = llm.complete(
        build_judge_prompt(customer, reply, cases),
        provider=config.JUDGE_PROVIDER, model=config.JUDGE_MODEL,
        max_tokens=200, use_cache=use_cache,
    )
    return parse_verdict(raw)
