import numpy as np
import pytest
from src.eval import metrics


def test_accuracy_is_exact_on_a_known_case():
    assert metrics.accuracy(["a", "b", "c", "d"], ["a", "b", "c", "x"]) == 0.75


def test_macro_f1_is_one_for_a_perfect_prediction():
    assert metrics.macro_f1(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)


def test_macro_f1_punishes_ignoring_the_rare_class():
    # Always predicting the majority class: high accuracy, poor macro-F1.
    y = ["a"] * 9 + ["b"]
    p = ["a"] * 10
    assert metrics.accuracy(y, p) == pytest.approx(0.9)
    assert metrics.macro_f1(y, p) < 0.5


def test_bootstrap_ci_brackets_the_point_estimate():
    y = ["a"] * 50 + ["b"] * 50
    p = ["a"] * 45 + ["b"] * 5 + ["b"] * 45 + ["a"] * 5
    lo, hi = metrics.bootstrap_ci(y, p, metrics.accuracy, n=2000, seed=42)
    point = metrics.accuracy(y, p)
    assert lo <= point <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_bootstrap_ci_is_wider_for_a_smaller_sample():
    y_big = ["a"] * 100 + ["b"] * 100
    p_big = ["a"] * 90 + ["b"] * 10 + ["b"] * 90 + ["a"] * 10
    lo_b, hi_b = metrics.bootstrap_ci(y_big, p_big, metrics.accuracy, n=2000, seed=1)
    # [:20] is all-matching (accuracy always 1.0) and produces a degenerate
    # zero-width interval; [80:100] straddles the mismatch boundary so it has
    # genuine resampling variance.
    lo_s, hi_s = metrics.bootstrap_ci(y_big[80:100], p_big[80:100], metrics.accuracy,
                                      n=2000, seed=1)
    assert (hi_s - lo_s) > (hi_b - lo_b)


def test_bootstrap_ci_is_deterministic_under_a_seed():
    y = ["a", "b"] * 25
    p = ["a"] * 50
    assert metrics.bootstrap_ci(y, p, metrics.accuracy, n=500, seed=7) == \
           metrics.bootstrap_ci(y, p, metrics.accuracy, n=500, seed=7)


def test_summary_reports_n_and_both_intervals():
    s = metrics.summary(["a", "b", "a", "b"], ["a", "b", "a", "a"])
    assert s["n"] == 4
    assert len(s["accuracy_ci"]) == 2
    assert len(s["macro_f1_ci"]) == 2
