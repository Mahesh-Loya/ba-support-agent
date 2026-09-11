"""Validate the validator.

An LLM judge that disagrees with the human is not a measurement instrument. We
report Cohen's kappa per axis, honestly, and every judge-derived number in the
report inherits that uncertainty.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def cohens_kappa(a: Sequence, b: Sequence) -> float:
    a = np.asarray(list(a), dtype=int)
    b = np.asarray(list(b), dtype=int)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("raters must be the same non-zero length")

    po = float((a == b).mean())
    labels = sorted(set(a.tolist()) | set(b.tolist()))
    pe = sum(float((a == l).mean()) * float((b == l).mean()) for l in labels)

    if np.isclose(pe, 1.0):
        return 0.0 if np.isclose(po, 1.0) else float("nan")
    return float((po - pe) / (1 - pe))


def interpret_kappa(k: float) -> str:
    if k != k:            # NaN
        return "undefined (degenerate)"
    if k < 0:
        return "poor (worse than chance)"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def agreement_report(human: dict[str, list], judge: dict[str, list]) -> pd.DataFrame:
    rows = []
    for axis in human:
        h, j = list(human[axis]), list(judge[axis])
        k = cohens_kappa(h, j)
        rows.append(dict(
            axis=axis,
            kappa=round(k, 3),
            raw_agreement=round(float(np.mean(np.asarray(h) == np.asarray(j))), 3),
            interpretation=interpret_kappa(k),
            n=len(h),
        ))
    return pd.DataFrame(rows)
