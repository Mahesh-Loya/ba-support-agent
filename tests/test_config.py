from pathlib import Path
from src import config


def test_brand_is_british_airways():
    assert config.BRAND == "British_Airways"


def test_seed_is_fixed():
    assert config.SEED == 42


def test_corpus_end_splits_october_from_november():
    assert config.CORPUS_END == "2017-11-01"


def test_paths_are_absolute_and_rooted_in_project():
    assert config.PROJECT_ROOT.is_absolute()
    for p in (config.RAW_CSV, config.INTERIM_DIR, config.GOLDEN_DIR,
              config.CACHE_DB, config.FIGURES_DIR):
        assert config.PROJECT_ROOT in p.parents or p == config.PROJECT_ROOT


def test_drafter_and_judge_are_different_models():
    # A model grading its own output inflates the score.
    assert config.DRAFTER_MODEL != config.JUDGE_MODEL
