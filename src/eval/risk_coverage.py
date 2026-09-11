"""Turn accuracy into a deployment decision.

The headline claim of this project is NOT "the classifier is X% accurate". It is
"at threshold T the agent safely auto-handles X% of volume at Y% harm rate".
This module produces that number, and shows how it moves with the cost model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def risk_coverage_curve(scores, harms, forced_escalate=None,
                        n_points: int = 101) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=float)
    harms = np.asarray(harms, dtype=bool)
    forced = (np.zeros(len(scores), dtype=bool) if forced_escalate is None
              else np.asarray(forced_escalate, dtype=bool))

    rows = []
    for t in np.linspace(0.0, 1.0, n_points):
        auto = (scores >= t) & ~forced
        n_auto = int(auto.sum())
        rows.append(dict(
            threshold=float(t),
            coverage=n_auto / len(scores),
            harm_rate=float(harms[auto].mean()) if n_auto else 0.0,
            n_auto=n_auto,
        ))
    return pd.DataFrame(rows)


def expected_cost(coverage: float, harm_rate: float, k: float) -> float:
    """Cost per incoming message, in units of one human escalation.

    k = cost(bad auto-reply) / cost(unnecessary escalation). Its true value is
    unknown, which is exactly why it is a parameter and the report shows a
    sensitivity sweep instead of one convenient number.
    """
    return (1.0 - coverage) * 1.0 + coverage * harm_rate * k


def pick_threshold(curve: pd.DataFrame, k: float) -> dict:
    c = curve.copy()
    c["expected_cost"] = [expected_cost(r.coverage, r.harm_rate, k)
                          for r in c.itertuples()]
    best = c.loc[c.expected_cost.idxmin()]
    return dict(threshold=float(best.threshold), coverage=float(best.coverage),
                harm_rate=float(best.harm_rate),
                expected_cost=float(best.expected_cost))


def sensitivity(curve: pd.DataFrame, ks=(2, 5, 10, 20, 50)) -> pd.DataFrame:
    return pd.DataFrame([{"k": k, **pick_threshold(curve, k)} for k in ks])


def reliability(scores, correct, bins: int = 10) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= 1.0)
        rows.append(dict(
            bin_lo=float(lo), bin_hi=float(hi),
            mean_score=float(scores[m].mean()) if m.any() else float("nan"),
            accuracy=float(correct[m].mean()) if m.any() else float("nan"),
            n=int(m.sum()),
        ))
    return pd.DataFrame(rows)


def ece(scores, correct, bins: int = 10) -> float:
    """Expected calibration error - how far self-reported confidence is from truth."""
    rel = reliability(scores, correct, bins)
    rel = rel[rel.n > 0]
    total = rel.n.sum()
    if total == 0:
        return 0.0
    return float((rel.n / total * (rel.mean_score - rel.accuracy).abs()).sum())


def plot_risk_coverage(curve: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve.coverage, curve.harm_rate, marker=".", lw=1)
    ax.set_xlabel("coverage (share of volume auto-handled)")
    ax.set_ylabel("harm rate among auto-handled")
    ax.set_title("Risk-coverage: what can we safely automate?")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(rel: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = rel[rel.n > 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="perfect calibration")
    ax.plot(r.mean_score, r.accuracy, marker="o", label="observed")
    ax.set_xlabel("mean self-reported confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Reliability diagram")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
