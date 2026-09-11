import numpy as np
import pytest
from src.eval import risk_coverage as rc


def test_coverage_falls_as_the_threshold_rises():
    scores = np.linspace(0, 1, 100)
    harms = np.zeros(100, dtype=bool)
    curve = rc.risk_coverage_curve(scores, harms)
    assert curve.coverage.is_monotonic_decreasing


def test_harm_rate_falls_as_threshold_rises_when_score_is_informative():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 500)
    harms = rng.uniform(0, 1, 500) > scores      # low score => likely harmful
    curve = rc.risk_coverage_curve(scores, harms)
    lo = curve[curve.threshold <= 0.2].harm_rate.mean()
    hi = curve[curve.threshold >= 0.8].harm_rate.mean()
    assert hi < lo


def test_forced_escalations_are_excluded_from_coverage():
    scores = np.array([0.9, 0.9, 0.9, 0.9])
    harms = np.array([False] * 4)
    forced = np.array([True, True, False, False])
    curve = rc.risk_coverage_curve(scores, harms, forced_escalate=forced)
    assert curve.coverage.max() <= 0.5 + 1e-9


def test_expected_cost_weights_harm_by_k():
    cheap = rc.expected_cost(coverage=0.5, harm_rate=0.1, k=1)
    dear = rc.expected_cost(coverage=0.5, harm_rate=0.1, k=50)
    assert dear > cheap


def test_pick_threshold_gets_stricter_as_harm_gets_more_expensive():
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, 800)
    harms = rng.uniform(0, 1, 800) > scores
    curve = rc.risk_coverage_curve(scores, harms)
    assert rc.pick_threshold(curve, k=50)["threshold"] >= \
           rc.pick_threshold(curve, k=2)["threshold"]


def test_sensitivity_returns_one_row_per_k():
    rng = np.random.default_rng(2)
    scores = rng.uniform(0, 1, 300)
    curve = rc.risk_coverage_curve(scores, rng.uniform(0, 1, 300) > scores)
    out = rc.sensitivity(curve, ks=[5, 10, 20, 50])
    assert len(out) == 4
    assert set(["k", "threshold", "coverage", "harm_rate"]).issubset(out.columns)


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    scores = np.array([0.0] * 50 + [1.0] * 50)
    correct = np.array([False] * 50 + [True] * 50)
    assert rc.ece(scores, correct, bins=2) == pytest.approx(0.0, abs=1e-9)


def test_ece_is_large_for_a_confidently_wrong_predictor():
    scores = np.full(100, 0.99)
    correct = np.zeros(100, dtype=bool)
    assert rc.ece(scores, correct, bins=5) > 0.9


def test_reliability_bins_have_the_expected_columns():
    rel = rc.reliability(np.linspace(0, 1, 100),
                         np.linspace(0, 1, 100) > 0.5, bins=5)
    assert set(["bin_lo", "bin_hi", "mean_score", "accuracy", "n"]).issubset(rel.columns)
