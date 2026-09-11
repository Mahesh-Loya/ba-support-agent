"""Derive an intent taxonomy from the data instead of inventing one.

Cluster -> inspect -> name by hand. The naming step is deliberately human:
the report has to defend why these intents and not others.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config, embed

TAXONOMY_PATH = config.INTERIM_DIR / "taxonomy.json"


def cluster(vectors: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10)
    return km.fit_predict(vectors)


def top_terms(texts: list[str], labels: np.ndarray, k: int = 8) -> dict[int, list[str]]:
    vec = TfidfVectorizer(stop_words="english", min_df=1, max_features=5000)
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    out: dict[int, list[str]] = {}
    for lab in sorted(set(int(x) for x in labels)):
        mask = np.asarray(labels) == lab
        if mask.sum() == 0:
            out[lab] = []
            continue
        mean = np.asarray(X[mask].mean(axis=0)).ravel()
        out[lab] = list(vocab[mean.argsort()[::-1][:k]])
    return out


def propose(n_clusters: int = 12, sample: int = 4000) -> dict:
    """Cluster the corpus split and dump an inspection report for hand-naming."""
    df = pd.read_parquet(config.INTERIM_DIR / "ba_pairs.parquet")
    df = df[df.split == "corpus"].sample(
        min(sample, len(df[df.split == "corpus"])), random_state=config.SEED)
    texts = df.customer_text.astype(str).tolist()
    labels = cluster(embed.embed(texts), n_clusters)
    terms = top_terms(texts, labels)

    report = {}
    for lab in sorted(set(int(x) for x in labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        report[str(lab)] = {
            "size": len(idx),
            "top_terms": terms[lab],
            "examples": [texts[i][:180] for i in idx[:10]],
        }
    out = config.INTERIM_DIR / "clusters.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def load_intents() -> list[str]:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))["intents"]


try:
    INTENTS: list[str] = load_intents()
except FileNotFoundError:
    INTENTS = []
