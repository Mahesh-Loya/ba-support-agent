# tests/test_baselines.py
import numpy as np
from src import baselines

# Used only for TrivialBaseline: a clear majority class is the whole point,
# so this fixture is deliberately imbalanced.
TRIVIAL_TEXTS = ["flight cancelled", "flight delayed again", "lost my bag",
                 "bag missing", "flight cancelled today", "delayed flight"]
TRIVIAL_LABELS = ["flight_disruption", "flight_disruption", "baggage",
                   "baggage", "flight_disruption", "flight_disruption"]

# Used for SimpleBaseline: class-balanced (6/6) with varied vocabulary per
# class, so the classifier has real word evidence to learn from rather than
# an intercept dominated by class imbalance (a 6-row, 4:2, one-word-per-class
# fixture previously made the toy example fail against real word evidence -
# fixed here in the fixture, not by tuning the model's regularization).
TEXTS = [
    "flight was cancelled today",
    "flight delayed for hours",
    "my departure got cancelled",
    "cancelled flight this morning",
    "delayed departure again",
    "flight cancelled without notice",
    "lost my bag at the airport",
    "my suitcase never arrived",
    "baggage went missing",
    "luggage did not arrive",
    "my bag is delayed",
    "missing suitcase again",
]
LABELS = ["flight_disruption"] * 6 + ["baggage"] * 6


def test_trivial_baseline_always_predicts_the_majority_class():
    m = baselines.TrivialBaseline().fit(TRIVIAL_TEXTS, TRIVIAL_LABELS)
    assert m.predict(["anything at all", "something else"]) == \
        ["flight_disruption", "flight_disruption"]


def test_trivial_baseline_reply_is_the_single_most_common_response():
    m = baselines.TrivialBaseline().fit(TRIVIAL_TEXTS, TRIVIAL_LABELS)
    m.fit_replies(["we are sorry", "we are sorry", "please DM us"])
    assert m.reply("literally anything") == "we are sorry"


def test_simple_baseline_learns_the_two_classes():
    m = baselines.SimpleBaseline().fit(TEXTS, LABELS)
    assert m.predict(["my bag is lost"])[0] == "baggage"
    assert m.predict(["my flight is cancelled"])[0] == "flight_disruption"


def test_simple_baseline_proba_rows_sum_to_one():
    m = baselines.SimpleBaseline().fit(TEXTS, LABELS)
    p = m.predict_proba(["lost bag", "cancelled flight"])
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert p.shape == (2, len(m.classes_))


def test_simple_baseline_reply_copies_a_real_historical_reply():
    m = baselines.SimpleBaseline().fit(TEXTS, LABELS)
    replies = ["r-cancel1", "r-cancel2", "r-cancel3", "r-cancel4",
               "r-cancel5", "r-cancel6", "r-bag1", "r-bag2",
               "r-bag3", "r-bag4", "r-bag5", "r-bag6"]
    m.fit_replies(TEXTS, replies)
    assert m.reply("my bag has gone missing") in {
        "r-bag1", "r-bag2", "r-bag3", "r-bag4", "r-bag5", "r-bag6",
    }
