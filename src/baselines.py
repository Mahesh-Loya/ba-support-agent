"""Two baselines the brief demands: a trivial one and a simple one.

If the LLM agent does not clearly beat SimpleBaseline, that IS the report's
finding, and it gets stated plainly rather than buried.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from src import config


class TrivialBaseline:
    """Majority intent; single most common canned reply."""

    def fit(self, texts: list[str], labels: list[str]) -> "TrivialBaseline":
        self.classes_ = sorted(set(labels))
        self.majority_ = Counter(labels).most_common(1)[0][0]
        self.canned_ = ""
        return self

    def fit_replies(self, replies: list[str]) -> "TrivialBaseline":
        self.canned_ = Counter(replies).most_common(1)[0][0]
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return [self.majority_] * len(texts)

    def reply(self, text: str) -> str:
        return self.canned_


class SimpleBaseline:
    """TF-IDF + logistic regression; reply = nearest historical reply. No LLM."""

    def fit(self, texts: list[str], labels: list[str]) -> "SimpleBaseline":
        self.vec_ = TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                    sublinear_tf=True, stop_words="english")
        X = self.vec_.fit_transform(texts)
        self.clf_ = LogisticRegression(max_iter=2000, random_state=config.SEED)
        self.clf_.fit(X, labels)
        self.classes_ = list(self.clf_.classes_)
        return self

    def fit_replies(self, texts: list[str], replies: list[str]) -> "SimpleBaseline":
        self.rvec_ = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.rX_ = self.rvec_.fit_transform(texts)
        self.replies_ = list(replies)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return list(self.clf_.predict(self.vec_.transform(texts)))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.clf_.predict_proba(self.vec_.transform(texts))

    def reply(self, text: str) -> str:
        sims = cosine_similarity(self.rvec_.transform([text]), self.rX_).ravel()
        return self.replies_[int(sims.argmax())]
