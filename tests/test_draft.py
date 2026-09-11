from src.agent import draft
from src.agent.retrieve import Case

CASES = [
    Case("my flight was cancelled", "Sorry - we can rebook you free of charge.", 0.9),
    Case("flight cancelled last minute", "We'll get you on the next available flight.", 0.8),
]


def test_prompt_includes_every_retrieved_historical_reply():
    p = draft.build_draft_prompt("they cancelled my flight", "flight_disruption", CASES)
    for c in CASES:
        assert c.agent_text in p


def test_prompt_states_the_no_invented_commitments_rule():
    p = draft.build_draft_prompt("x", "flight_disruption", CASES)
    low = p.lower()
    assert "do not" in low or "never" in low
    assert "compensation" in low or "promise" in low or "commit" in low


def test_clean_reply_strips_fences_and_wrapping_quotes():
    assert draft.clean_reply('```\n"Sorry about that."\n```') == "Sorry about that."


def test_clean_reply_removes_invented_agent_signature():
    # The agent must not sign as a human who does not exist.
    assert draft.clean_reply("We can rebook you. ^Jane") == "We can rebook you."


def test_clean_reply_strips_a_leading_reply_label():
    assert draft.clean_reply("Reply: We can help with that.") == "We can help with that."


def test_draft_reply_uses_the_cached_wrapper_and_returns_clean_text(monkeypatch):
    monkeypatch.setattr(draft.llm, "complete",
                        lambda prompt, **kw: '"We can rebook you free of charge." ^Sam')
    out = draft.draft_reply("cancelled!", "flight_disruption", CASES)
    assert out == "We can rebook you free of charge."
