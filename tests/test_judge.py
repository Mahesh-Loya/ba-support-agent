# tests/test_judge.py
from src.eval import judge
from src.agent.retrieve import Case

CASES = [Case("flight cancelled", "We can rebook you free of charge.", 0.9)]

GOOD = ('{"grounded": true, "helpful": true, "on_brand": true, '
        '"no_overpromise": true, "rationale": "matches past handling"}')
BAD = ('{"grounded": false, "helpful": true, "on_brand": true, '
       '"no_overpromise": false, "rationale": "invents a refund"}')


def test_parse_verdict_reads_all_four_axes():
    v = judge.parse_verdict(GOOD)
    assert v.grounded and v.helpful and v.on_brand and v.no_overpromise


def test_parse_verdict_tolerates_markdown_fences():
    assert judge.parse_verdict(f"```json\n{GOOD}\n```").grounded is True


def test_parse_verdict_defaults_to_failing_on_garbage():
    # An unparseable judge response must never be scored as a pass.
    v = judge.parse_verdict("the reply seems fine to me")
    assert not v.grounded and not v.helpful
    assert v.is_harmful


def test_is_harmful_when_ungrounded_or_overpromising():
    assert judge.parse_verdict(BAD).is_harmful
    assert not judge.parse_verdict(GOOD).is_harmful


def test_axes_returns_the_four_booleans():
    a = judge.parse_verdict(GOOD).axes()
    assert set(a) == {"grounded", "helpful", "on_brand", "no_overpromise"}


def test_judge_prompt_includes_reply_message_and_historical_cases():
    p = judge.build_judge_prompt("my flight was cancelled",
                                 "We can rebook you.", CASES)
    assert "my flight was cancelled" in p
    assert "We can rebook you." in p
    assert CASES[0].agent_text in p


def test_judge_reply_uses_the_judge_model_not_the_drafter(monkeypatch):
    seen = {}

    def fake(prompt, **kw):
        seen.update(kw)
        return GOOD

    monkeypatch.setattr(judge.llm, "complete", fake)
    judge.judge_reply("c", "r", CASES)
    assert seen["model"] == judge.config.JUDGE_MODEL
    assert seen["model"] != judge.config.DRAFTER_MODEL
