import pytest
from src.agent import decide


@pytest.mark.parametrize("text,rule", [
    ("I want a refund for my cancelled flight", "money_claim"),
    ("I am claiming compensation under EU261", "money_claim"),
    ("My solicitor will be in touch about this", "legal_or_safety"),
    ("There was smoke in the cabin, this was dangerous", "legal_or_safety"),
    ("My suitcase is lost, it never arrived at all", "lost_baggage"),
])
def test_hard_rules_fire_on_never_automate_categories(text, rule):
    assert decide.check_hard_rules(text, "other") == rule


def test_hard_rules_do_not_fire_on_a_plain_question():
    assert decide.check_hard_rules("What is the cabin baggage allowance?",
                                   "general_enquiry") is None


def test_refund_intent_always_escalates_even_at_max_confidence():
    d = decide.decide("please refund me", "refund_compensation",
                      confidence=1.0, top_similarity=1.0, clf_margin=1.0,
                      threshold=0.0)
    assert d.action == "escalate"
    assert d.rule_fired == "money_claim"


def test_high_confidence_safe_message_is_auto_handled():
    d = decide.decide("What is the cabin baggage allowance?", "general_enquiry",
                      confidence=0.95, top_similarity=0.9, clf_margin=0.8,
                      threshold=0.5)
    assert d.action == "auto"
    assert d.rule_fired is None


def test_low_score_escalates_with_a_confidence_reason():
    d = decide.decide("What is the cabin baggage allowance?", "general_enquiry",
                      confidence=0.2, top_similarity=0.1, clf_margin=0.05,
                      threshold=0.5)
    assert d.action == "escalate"
    assert d.reason == "low_confidence_ambiguous"


def test_routing_score_is_bounded_and_monotonic():
    lo = decide.routing_score(0.1, 0.1, 0.1)
    hi = decide.routing_score(0.9, 0.9, 0.9)
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
    assert hi > lo


def test_every_decision_carries_a_non_empty_reason():
    for args in [("refund me", "refund_compensation", 0.9, 0.9, 0.9),
                 ("baggage allowance?", "general_enquiry", 0.9, 0.9, 0.9),
                 ("baggage allowance?", "general_enquiry", 0.1, 0.1, 0.1)]:
        d = decide.decide(*args, threshold=0.5)
        assert d.reason
