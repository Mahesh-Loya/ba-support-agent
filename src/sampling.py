"""Stratified sampling for the golden set.

Random sampling over this corpus yields ~90 examples of the head intent and 2
of the tail, which makes macro-F1 meaningless. We allocate a floor per stratum
and fill the remainder proportionally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, embed, taxonomy


def stratified_sample(df: pd.DataFrame, strata_col: str, n_total: int,
                      floor_per_stratum: int, seed: int = config.SEED) -> pd.DataFrame:
    groups = {k: g for k, g in df.groupby(strata_col)}
    alloc = {k: min(floor_per_stratum, len(g)) for k, g in groups.items()}

    remaining = n_total - sum(alloc.values())
    if remaining > 0:
        headroom = {k: len(g) - alloc[k] for k, g in groups.items()}
        total_head = sum(headroom.values())
        if total_head > 0:
            for k in groups:
                extra = int(round(remaining * headroom[k] / total_head))
                alloc[k] = min(len(groups[k]), alloc[k] + extra)

    parts = [groups[k].sample(alloc[k], random_state=seed) for k in groups if alloc[k] > 0]
    out = pd.concat(parts).sample(frac=1.0, random_state=seed)
    return out.head(n_total).reset_index(drop=True)


def build_golden_pool(n_total: int = config.GOLDEN_TARGET) -> pd.DataFrame:
    """Sample from the EVAL split only, stratified by provisional cluster,
    with a deliberate hard/ambiguous stratum."""
    df = pd.read_parquet(config.INTERIM_DIR / "ba_pairs.parquet")
    df = df[df.split == "eval"].reset_index(drop=True)

    n_intents = len(taxonomy.load_intents())
    vec = embed.embed(df.customer_text.astype(str).tolist())
    labels = taxonomy.cluster(vec, k=n_intents)
    df["stratum"] = [f"c{int(l)}" for l in labels]

    # "hard" = far from its own cluster centroid, i.e. genuinely ambiguous
    centroids = np.stack([vec[labels == l].mean(axis=0) for l in sorted(set(labels))])
    sim_own = np.array([float(vec[i] @ centroids[labels[i]]) for i in range(len(df))])
    hard_cut = np.quantile(sim_own, 0.10)
    df.loc[sim_own <= hard_cut, "stratum"] = "hard"

    n_main = n_total - 25
    main = stratified_sample(df[df.stratum != "hard"], "stratum", n_main, 15)
    hard = df[df.stratum == "hard"].sample(min(25, (df.stratum == "hard").sum()),
                                           random_state=config.SEED)
    pool = pd.concat([main, hard]).sample(frac=1.0, random_state=config.SEED)
    pool = pool.reset_index(drop=True)

    config.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    pool.to_json(config.GOLDEN_DIR / "golden_pool.jsonl",
                 orient="records", lines=True, force_ascii=False)
    return pool
