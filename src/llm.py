"""Single choke point for every LLM call in this project.

Responses are cached in a SQLite file that is COMMITTED TO THE REPO. That is
what lets a reviewer reproduce the headline numbers with no API key in about
two minutes. Pass use_cache=False to force live calls.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

from src import config

_STATS = {"hits": 0, "misses": 0}

_KEY_ENV = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}


class NoAPIKeyError(RuntimeError):
    """Raised on a cache miss when no provider key is configured."""


def cache_key(provider: str, model: str, prompt: str,
              temperature: float, max_tokens: int) -> str:
    blob = f"{provider}\x00{model}\x00{temperature}\x00{max_tokens}\x00{prompt}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _init_db(db: Path) -> None:
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  k TEXT PRIMARY KEY,"
            "  provider TEXT, model TEXT, prompt TEXT, response TEXT,"
            "  created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )


def _cache_get(db: Path, k: str) -> str | None:
    if not db.exists():
        return None
    with sqlite3.connect(db) as con:
        row = con.execute("SELECT response FROM cache WHERE k = ?", (k,)).fetchone()
    return row[0] if row else None


def _cache_put(db: Path, k: str, response: str,
               provider: str = "", model: str = "", prompt: str = "") -> None:
    _init_db(db)
    with sqlite3.connect(db) as con:
        con.execute(
            "INSERT OR REPLACE INTO cache (k, provider, model, prompt, response)"
            " VALUES (?, ?, ?, ?, ?)",
            (k, provider, model, prompt, response),
        )


def _call_provider(prompt: str, *, provider: str, model: str,
                   temperature: float, max_tokens: int) -> str:
    """Live API call. Isolated so tests can monkeypatch it."""
    env_var = _KEY_ENV.get(provider, "")
    if env_var and not os.environ.get(env_var):
        raise NoAPIKeyError(
            f"Cache miss for {provider}/{model} and {env_var} is not set.\n"
            f"Reproducing published results should never miss the cache - if you "
            f"see this during `python -m src.cli reproduce`, the committed cache "
            f"is stale. To generate fresh results, set {env_var}."
        )

    if provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature, max_output_tokens=max_tokens
            ),
        )
        return (resp.text or "").strip()

    if provider == "groq":
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    raise ValueError(f"unknown provider: {provider}")


def complete(prompt: str, *, provider: str, model: str,
             temperature: float = config.TEMPERATURE,
             max_tokens: int = 512, use_cache: bool = True) -> str:
    db = config.CACHE_DB
    k = cache_key(provider, model, prompt, temperature, max_tokens)

    if use_cache:
        hit = _cache_get(db, k)
        if hit is not None:
            _STATS["hits"] += 1
            return hit

    _STATS["misses"] += 1

    out = _call_provider(prompt, provider=provider, model=model,
                         temperature=temperature, max_tokens=max_tokens)
    _cache_put(db, k, out, provider, model, prompt)
    return out


def cache_stats() -> dict:
    entries = 0
    if config.CACHE_DB.exists():
        with sqlite3.connect(config.CACHE_DB) as con:
            entries = con.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    return {"entries": entries, **_STATS}
