"""Local sentence embeddings. No API, no rate limit, fully reproducible."""
from __future__ import annotations

import hashlib

import numpy as np

from src import config

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(config.EMBED_MODEL)
    return _MODEL


def embed(texts: list[str], use_cache: bool = True) -> np.ndarray:
    key = hashlib.sha256(
        (config.EMBED_MODEL + "\x00" + "\x00".join(texts)).encode("utf-8")
    ).hexdigest()[:16]
    path = config.INTERIM_DIR / f"emb_{key}.npy"
    if use_cache and path.exists():
        return np.load(path)

    vec = _model().encode(texts, normalize_embeddings=True,
                          show_progress_bar=len(texts) > 500)
    vec = np.asarray(vec, dtype=np.float32)
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, vec)
    return vec
