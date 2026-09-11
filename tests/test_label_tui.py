"""Unit tests for the pure, non-interactive logic in src.label_tui.

These deliberately do NOT simulate the interactive loop (no input() mocking,
no stdin driving) - they test the small parsing/loading helpers in isolation,
per the standing rule that a typo on the ground-truth path must never
silently record the wrong value or silently skip an example.
"""
import json

from src import label_tui


# --------------------------------------------------------------------
# parse_action
# --------------------------------------------------------------------

def test_parse_action_accepts_a_and_auto():
    assert label_tui.parse_action("a") == "auto"
    assert label_tui.parse_action("A") == "auto"
    assert label_tui.parse_action(" auto ") == "auto"


def test_parse_action_accepts_e_and_escalate():
    assert label_tui.parse_action("e") == "escalate"
    assert label_tui.parse_action("Escalate") == "escalate"


def test_parse_action_rejects_typos_instead_of_defaulting_to_auto():
    # This is the exact bug from Finding 1: a fat-fingered "w" or stray
    # input must NOT silently become "auto".
    assert label_tui.parse_action("w") is None
    assert label_tui.parse_action("") is None
    assert label_tui.parse_action(" ") is None
    assert label_tui.parse_action("x") is None


# --------------------------------------------------------------------
# parse_intent_input
# --------------------------------------------------------------------

def test_parse_intent_input_valid_index():
    assert label_tui.parse_intent_input("0", 5) == ("index", 0)
    assert label_tui.parse_intent_input("4", 5) == ("index", 4)


def test_parse_intent_input_quit_and_skip():
    assert label_tui.parse_intent_input("q", 5) == ("quit", None)
    assert label_tui.parse_intent_input("Q", 5) == ("quit", None)
    assert label_tui.parse_intent_input("s", 5) == ("skip", None)


def test_parse_intent_input_out_of_range_is_invalid_not_skip():
    # Finding 2: an out-of-range or unparseable keystroke must be
    # distinguishable from a deliberate skip.
    assert label_tui.parse_intent_input("5", 5) == ("invalid", None)
    assert label_tui.parse_intent_input("99", 5) == ("invalid", None)


def test_parse_intent_input_garbage_is_invalid_not_skip():
    assert label_tui.parse_intent_input("", 5) == ("invalid", None)
    assert label_tui.parse_intent_input("x", 5) == ("invalid", None)
    assert label_tui.parse_intent_input(" ", 5) == ("invalid", None)
    assert label_tui.parse_intent_input("-1", 5) == ("invalid", None)


# --------------------------------------------------------------------
# parse_escalation_reason
# --------------------------------------------------------------------

def test_parse_escalation_reason_valid_codes():
    assert label_tui.parse_escalation_reason("1") == "needs_account_lookup"
    assert label_tui.parse_escalation_reason(" 6 ") == "multi_issue"


def test_parse_escalation_reason_invalid_returns_none_not_other():
    # Must not silently default to "other" - "other" is not one of the six
    # enumerated reasons.
    assert label_tui.parse_escalation_reason("9") is None
    assert label_tui.parse_escalation_reason("") is None
    assert label_tui.parse_escalation_reason("other") is None


# --------------------------------------------------------------------
# parse_yn
# --------------------------------------------------------------------

def test_parse_yn():
    assert label_tui.parse_yn("y") is True
    assert label_tui.parse_yn("Y") is True
    assert label_tui.parse_yn("n") is False
    assert label_tui.parse_yn("N") is False
    assert label_tui.parse_yn("") is None
    assert label_tui.parse_yn("yes") is None
    assert label_tui.parse_yn("maybe") is None


# --------------------------------------------------------------------
# _load_done
# --------------------------------------------------------------------

def test_load_done_missing_file_returns_empty(tmp_path):
    assert label_tui._load_done(tmp_path / "nope.jsonl") == {}


def test_load_done_skips_corrupt_line_without_raising(tmp_path, capsys):
    path = tmp_path / "golden.jsonl"
    good = {"customer_tweet_id": 42, "intent": "baggage_issue"}
    path.write_text(
        json.dumps(good) + "\n" + '{"customer_tweet_id": 43, "intent": tru\n',
        encoding="utf-8",
    )

    done = label_tui._load_done(path)

    assert done == {42: good}
    captured = capsys.readouterr()
    assert "line 2" in captured.out
    assert str(path) in captured.out


def test_load_done_reports_but_skips_a_line_missing_the_key(tmp_path, capsys):
    path = tmp_path / "golden.jsonl"
    good = {"customer_tweet_id": 1, "intent": "x"}
    path.write_text(
        json.dumps(good) + "\n" + json.dumps({"intent": "no id here"}) + "\n",
        encoding="utf-8",
    )

    done = label_tui._load_done(path)

    assert done == {1: good}
    assert "line 2" in capsys.readouterr().out
