import pandas as pd
import pytest
from src import sampling


def _skewed():
    # 200 head, 30 mid, 5 tail - the exact shape that makes random sampling useless
    return pd.DataFrame(dict(
        i=range(235),
        stratum=["head"] * 200 + ["mid"] * 30 + ["tail"] * 5,
    ))


def test_stratified_sample_respects_the_floor_for_rare_strata():
    out = sampling.stratified_sample(_skewed(), "stratum", n_total=60,
                                     floor_per_stratum=15, seed=42)
    counts = out.stratum.value_counts()
    assert counts["tail"] == 5      # can't exceed what exists
    assert counts["mid"] >= 15
    assert len(out) <= 60


def test_stratified_sample_is_deterministic_under_a_seed():
    a = sampling.stratified_sample(_skewed(), "stratum", 60, 15, seed=42)
    b = sampling.stratified_sample(_skewed(), "stratum", 60, 15, seed=42)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_stratified_sample_beats_random_on_tail_coverage():
    df = _skewed()
    strat = sampling.stratified_sample(df, "stratum", 60, 15, seed=42)
    rand = df.sample(60, random_state=42)
    assert strat.stratum.nunique() >= rand.stratum.nunique()
    assert (strat.stratum == "mid").sum() > (rand.stratum == "mid").sum()


def test_sample_never_returns_duplicates():
    out = sampling.stratified_sample(_skewed(), "stratum", 100, 15, seed=42)
    assert out.i.is_unique
