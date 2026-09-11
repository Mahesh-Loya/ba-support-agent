import pytest
from src.agent import classify

INTENTS = ["baggage", "flight_disruption", "other"]


def test_parse_response_reads_well_formed_json():
    c = classify.parse_response('{"intent": "baggage", "confidence": 0.91}', INTENTS)
    assert c.intent == "baggage"
    assert c.confidence == pytest.approx(0.91)


def test_parse_response_tolerates_markdown_fences():
    raw = '```json\n{"intent": "baggage", "confidence": 0.8}\n```'
    assert classify.parse_response(raw, INTENTS).intent == "baggage"


def test_parse_response_falls_back_to_other_on_unknown_label():
    c = classify.parse_response('{"intent": "spaceflight", "confidence": 0.9}', INTENTS)
    assert c.intent == "other"
    assert c.confidence == 0.0


def test_parse_response_falls_back_to_other_on_garbage():
    c = classify.parse_response("I think it's about bags maybe?", INTENTS)
    assert c.intent == "other"
    assert c.confidence == 0.0


def test_confidence_is_clamped_to_unit_interval():
    assert classify.parse_response('{"intent":"baggage","confidence":5}', INTENTS).confidence == 1.0
    assert classify.parse_response('{"intent":"baggage","confidence":-2}', INTENTS).confidence == 0.0


def test_prompt_contains_every_intent_and_the_message():
    p = classify.build_prompt("my bag is gone", INTENTS, {"baggage": "lost bags"})
    for i in INTENTS:
        assert i in p
    assert "my bag is gone" in p


def test_classify_routes_through_the_cached_llm_wrapper(monkeypatch):
    calls = []

    def fake(prompt, **kw):
        calls.append(kw)
        return '{"intent": "baggage", "confidence": 0.7}'

    monkeypatch.setattr(classify.llm, "complete", fake)
    monkeypatch.setattr(classify.taxonomy, "load_intents", lambda: INTENTS)
    monkeypatch.setattr(classify, "_definitions", lambda: {})
    out = classify.classify("lost bag")
    assert out.intent == "baggage"
    assert len(calls) == 1
