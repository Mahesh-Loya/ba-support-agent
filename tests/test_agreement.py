import pytest
from src.eval import agreement


def test_kappa_is_one_for_perfect_agreement():
    assert agreement.cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_kappa_is_zero_for_chance_level_agreement():
    a = [1, 1, 0, 0]
    b = [1, 0, 1, 0]
    assert agreement.cohens_kappa(a, b) == pytest.approx(0.0, abs=1e-9)


def test_kappa_is_negative_for_systematic_disagreement():
    assert agreement.cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) < 0


def test_kappa_handles_a_degenerate_constant_rater():
    # Both raters say "true" for everything: agreement is total but uninformative.
    k = agreement.cohens_kappa([1, 1, 1, 1], [1, 1, 1, 1])
    assert k == 0.0 or k == pytest.approx(1.0)


def test_interpret_kappa_uses_landis_koch_bands():
    assert "poor" in agreement.interpret_kappa(-0.1).lower()
    assert "slight" in agreement.interpret_kappa(0.1).lower()
    assert "moderate" in agreement.interpret_kappa(0.5).lower()
    assert "substantial" in agreement.interpret_kappa(0.7).lower()
    assert "almost perfect" in agreement.interpret_kappa(0.9).lower()


def test_agreement_report_covers_every_axis():
    human = {"grounded": [1, 0, 1, 1], "helpful": [1, 1, 0, 0]}
    judged = {"grounded": [1, 0, 1, 0], "helpful": [1, 1, 0, 1]}
    r = agreement.agreement_report(human, judged)
    assert set(r.axis) == {"grounded", "helpful"}
    assert set(["axis", "kappa", "raw_agreement", "interpretation", "n"]).issubset(r.columns)
    assert (r.n == 4).all()
