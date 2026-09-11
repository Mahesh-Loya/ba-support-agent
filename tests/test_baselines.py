# tests/test_baselines.py
import numpy as np
from src import baselines

TEXTS = ["flight cancelled", "flight delayed again", "lost my bag",
         "bag missing", "flight cancelled today", "delayed flight"]
LABELS = ["flight_disruption", "flight_disruption", "baggage",
          "baggage", "flight_disruption", "flight_disruption"]


def test_trivial_baseline_always_predicts_the_majority_class():
    m = baselines.TrivialBaseline().fit(TEXTS, LABELS)
    assert m.predict(["anything at all", "something else"]) == \
        ["flight_disruption", "flight_disruption"]


def test_trivial_baseline_reply_is_the_single_most_common_response():
    m = baselines.TrivialBaseline().fit(TEXTS, LABELS)
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
    m.fit_replies(TEXTS, ["r-cancel", "r-delay", "r-bag",
                          "r-bag2", "r-cancel2", "r-delay2"])
    assert m.reply("my bag has gone missing") in {"r-bag", "r-bag2"}
