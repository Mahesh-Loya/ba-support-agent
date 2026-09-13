import time

import pandas as pd
import pytest

from src import pipeline


def test_run_agent_produces_one_row_per_input_with_all_columns(monkeypatch):
    from src.agent import classify as C, decide as D, draft as F
    from src.agent.retrieve import Case

    golden = pd.DataFrame(dict(
        customer_tweet_id=[1, 2],
        customer_text=["my bag is lost and never arrived", "what is the baggage allowance?"],
        agent_text=["a", "b"],
        intent=["baggage", "general_enquiry"],
        action=["escalate", "auto"],
    ))

    class FakeRetriever:
        def search(self, text, k=5):
            return [Case("past", "BA replied something", 0.8)]

    monkeypatch.setattr(pipeline, "_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(C, "classify", lambda t, **k: C.Classification("baggage", 0.8))
    monkeypatch.setattr(F, "draft_reply", lambda *a, **k: "drafted reply")
    monkeypatch.setattr(pipeline, "_clf_margin", lambda texts: [0.5, 0.5])

    out = pipeline.run_agent(golden, threshold=0.5)
    assert len(out) == 2
    for col in ["pred_intent", "confidence", "score", "action", "reason", "reply"]:
        assert col in out.columns


def test_run_agent_escalates_the_lost_baggage_row_via_hard_rule(monkeypatch):
    from src.agent import classify as C, draft as F
    from src.agent.retrieve import Case

    golden = pd.DataFrame(dict(
        customer_tweet_id=[1],
        customer_text=["my suitcase is lost, it never arrived"],
        agent_text=["a"], intent=["baggage"], action=["escalate"],
    ))

    class FakeRetriever:
        def search(self, text, k=5):
            return [Case("p", "r", 0.99)]

    monkeypatch.setattr(pipeline, "_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(C, "classify", lambda t, **k: C.Classification("baggage", 1.0))
    monkeypatch.setattr(F, "draft_reply", lambda *a, **k: "reply")
    monkeypatch.setattr(pipeline, "_clf_margin", lambda texts: [1.0])

    out = pipeline.run_agent(golden, threshold=0.0)
    assert out.iloc[0].action == "escalate"
    assert out.iloc[0].rule_fired == "lost_baggage"


# --- Ruling A: judge-vs-human agreement -------------------------------------

def test_judge_agreement_is_none_when_quality_jsonl_is_absent(monkeypatch, tmp_path):
    # No data/golden/quality.jsonl at all in this empty tmp dir.
    monkeypatch.setattr(pipeline.config, "GOLDEN_DIR", tmp_path)
    assert pipeline._judge_agreement() is None


def test_judge_agreement_judges_the_real_historical_reply_not_a_draft(monkeypatch, tmp_path):
    from src.eval import judge as J
    from src.agent.retrieve import Case

    quality_path = tmp_path / "quality.jsonl"
    quality_path.write_text(
        pd.DataFrame(dict(
            customer_tweet_id=[1],
            customer_text=["my bag is lost"],
            agent_text=["BA's real historical reply"],
            grounded=[True], helpful=[True], on_brand=[False], no_overpromise=[True],
        )).to_json(orient="records", lines=True, force_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline.config, "GOLDEN_DIR", tmp_path)

    class FakeRetriever:
        def search(self, text, k=5):
            return [Case("past", "past reply", 0.9)]

    monkeypatch.setattr(pipeline, "_retriever", lambda: FakeRetriever())

    seen = {}

    def fake_judge_reply(customer, reply, cases, **kw):
        seen["reply"] = reply
        return J.Verdict(True, True, False, True, "matches human")

    monkeypatch.setattr(J, "judge_reply", fake_judge_reply)

    out = pipeline._judge_agreement()
    assert seen["reply"] == "BA's real historical reply"
    assert out is not None
    axes = {row["axis"] for row in out}
    assert axes == set(J.AXES)
    # perfect agreement on every axis in this fixture
    for row in out:
        assert row["raw_agreement"] == 1.0


# --- Known wrinkle: golden/preds merge suffixes a shared `action` column ---

def test_escalation_metric_compares_human_action_against_agent_action():
    # Simulates golden.merge(preds, on="customer_tweet_id") where both frames
    # carry an `action` column: pandas suffixes them action_x (human, from
    # golden) and action_y (agent, from preds). The escalation metric must
    # diff those two columns against each other, never a column against
    # itself (which would silently report perfect/trivial agreement).
    merged = pd.DataFrame({
        "customer_tweet_id": [1, 2, 3],
        "action_x": ["auto", "escalate", "auto"],       # human/golden action
        "action_y": ["escalate", "escalate", "auto"],   # agent's predicted action
    })
    human, agent = pipeline._human_agent_actions(merged)
    assert human == ["auto", "escalate", "auto"]
    assert agent == ["escalate", "escalate", "auto"]
    assert human != agent  # sanity: the two axes genuinely differ


def test_escalation_metric_falls_back_to_plain_action_column_when_unsuffixed():
    # Defensive path for the (unlikely) case pandas does not need to suffix.
    merged = pd.DataFrame({"customer_tweet_id": [1], "action": ["auto"]})
    human, agent = pipeline._human_agent_actions(merged)
    assert human == ["auto"]
    assert agent == ["auto"]


# --- Ruling S: retryable, rate-limit tolerant LLM calls ---------------------

def test_with_retry_retries_transient_errors_then_succeeds(monkeypatch):
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("rate limited")
        return "ok"

    assert pipeline._with_retry(flaky) == "ok"
    assert calls["n"] == 3


def test_with_retry_does_not_retry_a_missing_api_key(monkeypatch):
    from src import llm

    def should_not_sleep(_):
        raise AssertionError("NoAPIKeyError must not be retried")

    monkeypatch.setattr(pipeline.time, "sleep", should_not_sleep)

    def boom():
        raise llm.NoAPIKeyError("no key")

    with pytest.raises(llm.NoAPIKeyError):
        pipeline._with_retry(boom)


def test_with_retry_gives_up_after_max_attempts_and_raises_last_error(monkeypatch):
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise RuntimeError(f"boom {calls['n']}")

    with pytest.raises(RuntimeError, match="boom"):
        pipeline._with_retry(always_fails)
    assert calls["n"] == pipeline._MAX_RETRIES


# --- Ruling P: bounded concurrency, order preservation ----------------------

def test_parallel_map_preserves_input_order_despite_uneven_sleeps():
    # Deliberately make early items sleep longer than later ones, so a naive
    # "collect in completion order" implementation would visibly reorder the
    # output. Every downstream metric pairs predictions against golden rows
    # positionally, so this ordering guarantee is load-bearing.
    delays = [0.2, 0.01, 0.15, 0.01, 0.1, 0.01, 0.05, 0.01]

    def slow_double(i):
        time.sleep(delays[i])
        return i * 2

    out = pipeline._parallel_map(slow_double, range(len(delays)), max_workers=8)
    assert out == [i * 2 for i in range(len(delays))]


def test_parallel_map_matches_serial_result_with_a_single_worker():
    out = pipeline._parallel_map(lambda x: x + 1, range(10), max_workers=1)
    assert out == [x + 1 for x in range(10)]


def test_parallel_map_propagates_a_worker_exception_instead_of_swallowing_it():
    def boom(i):
        if i == 3:
            raise ValueError(f"worker {i} exploded")
        return i

    with pytest.raises(ValueError, match="worker 3 exploded"):
        pipeline._parallel_map(boom, range(6), max_workers=4)


def test_parallel_map_reports_progress_under_the_hood(monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "_PROGRESS_EVERY", 2)
    out = pipeline._parallel_map(lambda x: x, range(5), max_workers=4, label="widgets")
    assert out == [0, 1, 2, 3, 4]
    captured = capsys.readouterr().out
    assert "widgets: 5/5 processed" in captured
