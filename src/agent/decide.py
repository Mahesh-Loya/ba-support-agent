"""Auto-handle vs. escalate.

Two layers, because they answer different questions:

  1. HARD RULES - categories a business must never automate, regardless of how
     confident the model is. This is policy, not prediction.
  2. CONFIDENCE GATE - everything else routes on a score, and abstains below a
     threshold chosen from an explicit cost model (see eval/risk_coverage.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# These hard rules are DELIBERATELY over-inclusive. The cost of a false
# escalation is one extra human minute; the cost of a missed escalation on
# crisis or legal/safety language is unacceptable. When in doubt, the rule
# should fire.
HARD_RULES: list[tuple[str, str]] = [
    ("crisis_wellbeing", r"\b(suicid\w*|self[- ]?harm\w*|kill(?:ing)?\s+myself|"
                         r"end(?:ing)?\s+(?:it all|my life|it)|"
                         r"want(?:ed)?\s+to\s+die|"
                         r"don'?t\s+want\s+to\s+(?:live|be here)(?:\s+anymore)?|"
                         r"can'?t\s+(?:cope|go on|take\s+(?:it|this)(?:\s+any\s*more)?)|"
                         r"(?:having|have)\s+a\s+(?:mental\s+|nervous\s+)?breakdown|"
                         r"panic\s+attack|"
                         r"give\s+up\s+on\s+(?:life|everything)|"
                         r"no\s+(?:point|reason)\s+(?:in\s+)?(?:living|going\s+on))\b"),
    ("money_claim", r"\b(refund|compensat\w*|reimburs\w*|eu ?261|voucher|"
                    r"money back|charge(d)? me|overcharg\w*)\b"),
    ("legal_or_safety", r"\b(solicitor|lawyer|legal action|sue|court|ombudsman|"
                        r"caa|dangerous|unsafe|smoke|fire|injur\w*|assault\w*|"
                        r"discriminat\w*|racist|taking this further|"
                        r"escalat\w*\s+this|trading standards|small claims|"
                        r"my rights|regulator\w*|cedr|file a complaint|"
                        r"report you to|the press|bbc watchdog|watchdog)\b"),
    ("lost_baggage", r"\b(lost|missing|never arrived|didn'?t arrive|no sign of)\b"
                     r"[^.]{0,40}\b(bag|bags|baggage|luggage|suitcase|case)\b"
                     r"|\b(bag|bags|baggage|luggage|suitcase)\b[^.]{0,40}"
                     r"\b(lost|missing|never arrived|didn'?t arrive)\b"),
    ("account_lookup", r"\b(booking reference|pnr|my account|log ?in|password|"
                       r"executive club number|membership number)\b"),
    ("severe_distress", r"\b(disgust\w*|appall\w*|worst|never fly\w* again|"
                        r"disgrace\w*|furious|outrage\w*)\b"),
]

ALWAYS_ESCALATE_INTENTS = {"refund_compensation"}

_COMPILED = [(name, re.compile(p, re.I)) for name, p in HARD_RULES]


@dataclass
class Decision:
    action: str          # "auto" | "escalate"
    reason: str
    score: float
    rule_fired: str | None


def check_hard_rules(text: str, intent: str) -> str | None:
    if intent in ALWAYS_ESCALATE_INTENTS:
        return "money_claim"
    for name, pat in _COMPILED:
        if pat.search(text):
            return name
    return None


def routing_score(confidence: float, top_similarity: float,
                  clf_margin: float) -> float:
    """Blend three signals. Weights are deliberate and reported; which signal is
    actually calibrated is measured in eval/risk_coverage.py, not assumed."""
    sim = max(0.0, min(1.0, top_similarity))
    return max(0.0, min(1.0, 0.5 * confidence + 0.3 * sim + 0.2 * clf_margin))


def decide(text: str, intent: str, confidence: float, top_similarity: float,
           clf_margin: float, threshold: float = 0.5) -> Decision:
    score = routing_score(confidence, top_similarity, clf_margin)

    rule = check_hard_rules(text, intent)
    if rule:
        return Decision("escalate", rule, score, rule)

    if score < threshold:
        return Decision("escalate", "low_confidence_ambiguous", score, None)

    return Decision("auto", "high_confidence_routine", score, None)
