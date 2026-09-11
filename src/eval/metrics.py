"""Metrics with confidence intervals.

With n=200, a 3-point difference between two systems is noise. Every headline
number in this project carries a bootstrap CI so that is visible rather than
hidden.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src import config


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    return float(accuracy_score(list(y_true), list(y_pred)))


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    return float(f1_score(list(y_true), list(y_pred), average="macro", zero_division=0))


def bootstrap_ci(y_true: Sequence, y_pred: Sequence,
                 stat_fn: Callable[[Sequence, Sequence], float],
                 n: int = config.BOOTSTRAP_N, seed: int = config.SEED,
                 alpha: float = 0.05) -> tuple[float, float]:
    yt, yp = np.asarray(list(y_true), dtype=object), np.asarray(list(y_pred), dtype=object)
    rng = np.random.default_rng(seed)
    size = len(yt)
    stats = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, size, size)
        stats[i] = stat_fn(yt[idx], yp[idx])
    return (float(np.quantile(stats, alpha / 2)),
            float(np.quantile(stats, 1 - alpha / 2)))


def confusion(y_true: Sequence, y_pred: Sequence, labels: list[str]) -> pd.DataFrame:
    m = confusion_matrix(list(y_true), list(y_pred), labels=labels)
    return pd.DataFrame(m, index=[f"true_{l}" for l in labels],
                        columns=[f"pred_{l}" for l in labels])


def summary(y_true: Sequence, y_pred: Sequence, n_boot: int = 2000) -> dict:
    return {
        "n": len(list(y_true)),
        "accuracy": accuracy(y_true, y_pred),
        "accuracy_ci": bootstrap_ci(y_true, y_pred, accuracy, n=n_boot),
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_f1_ci": bootstrap_ci(y_true, y_pred, macro_f1, n=n_boot),
    }
