"""Retrieve how BA historically handled similar messages.

Hard rule: the retriever only ever sees the CORPUS split. If it could see the
eval split it would retrieve the very thread being evaluated, and every number
downstream would be inflated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config, embed


@dataclass
class Case:
    customer_text: str
    agent_text: str
    similarity: float


class Retriever:
    def __init__(self, corpus: pd.DataFrame):
        if "split" not in corpus.columns:
            raise ValueError(
                "Retriever requires a 'split' column so it can enforce the "
                "corpus-only anti-leakage guarantee; refusing to treat a "
                "frame without one as safe corpus data."
            )
        corpus = corpus[corpus.split == "corpus"]
        self.corpus = corpus.reset_index(drop=True)
        self._vec = embed.embed(self.corpus.customer_text.astype(str).tolist())

    @classmethod
    def from_corpus_split(cls) -> "Retriever":
        df = pd.read_parquet(config.INTERIM_DIR / "ba_pairs.parquet")
        return cls(df)

    def search(self, text: str, k: int = 5) -> list[Case]:
        q = embed.embed([text])[0]
        sims = self._vec @ q
        idx = np.argsort(sims)[::-1][:k]
        return [
            Case(customer_text=str(self.corpus.iloc[i].customer_text),
                 agent_text=str(self.corpus.iloc[i].agent_text),
                 similarity=float(sims[i]))
            for i in idx
        ]
