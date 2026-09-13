"""Single choke point for every LLM call in this project.

Responses are cached in a SQLite file that is COMMITTED TO THE REPO. That is
what lets a reviewer reproduce the headline numbers with no API key in about
two minutes. Pass use_cache=False to force live calls.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from pathlib import Path

from src import config

_STATS = {"hits": 0, "misses": 0}

_KEY_ENV = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}

# --- concurrency ------------------------------------------------------------
# The cache file is read/written from many threads in this process (parallel
# LLM calls in src/pipeline.py) AND from other OS processes running the same
# pipeline against the same committed cache file. Two things make that safe:
#
# 1. WAL journal mode. Once set, it is persisted in the database file itself,
#    so every connection that opens this file afterwards - in this process or
#    any other, old code or new - uses WAL automatically. WAL lets readers
#    (cache hits, the overwhelming majority of calls) proceed concurrently
#    with a writer instead of being blocked by the default rollback journal's
#    exclusive write lock. Only writer-vs-writer contention remains, and that
#    window is a single small INSERT.
# 2. A generous busy_timeout (set both via connect(timeout=...) and via the
#    PRAGMA, belt-and-suspenders) so a connection that does lose that brief
#    writer-vs-writer race *waits* for the lock instead of raising
#    "database is locked" immediately. That is what a bare 5s default timeout
#    under real concurrency would risk, and a raised/swallowed write here is
#    exactly the failure this project cannot afford (see cache_key docstring
#    below): it would either force a repeated paid API call or, worse, silently
#    fail to extend the committed cache.
#
# A module-level lock additionally serialises the write itself *within this
# process* (cheap - it's one INSERT) so this process's own worker threads
# never even attempt to race each other for the write lock. It is not needed
# for cross-process correctness (WAL + busy_timeout already guarantee that);
# it just avoids pointless contention/backoff among our own threads. Reads
# are deliberately NOT put behind this lock - cache hits must stay nearly
# free and concurrent with each other and with the in-flight write.
_WRITE_LOCK = threading.Lock()
_BUSY_TIMEOUT_S = 30.0


class NoAPIKeyError(RuntimeError):
    """Raised on a cache miss when no provider key is configured."""


def cache_key(provider: str, model: str, prompt: str,
              temperature: float, max_tokens: int) -> str:
    blob = f"{provider}\x00{model}\x00{temperature}\x00{max_tokens}\x00{prompt}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _connect(db: Path) -> sqlite3.Connection:
    """Every cache connection goes through here so WAL + a generous busy
    timeout are applied uniformly, whether this is a read or a write, and
    regardless of how many threads/processes are touching the file."""
    con = sqlite3.connect(db, timeout=_BUSY_TIMEOUT_S)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(f"PRAGMA busy_timeout={int(_BUSY_TIMEOUT_S * 1000)}")
    return con


def _init_db(db: Path) -> None:
    db.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  k TEXT PRIMARY KEY,"
            "  provider TEXT, model TEXT, prompt TEXT, response TEXT,"
            "  created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )


def _cache_get(db: Path, k: str) -> str | None:
    if not db.exists():
        return None
    with _connect(db) as con:
        row = con.execute("SELECT response FROM cache WHERE k = ?", (k,)).fetchone()
    return row[0] if row else None


def _cache_put(db: Path, k: str, response: str,
               provider: str = "", model: str = "", prompt: str = "") -> None:
    _init_db(db)
    with _WRITE_LOCK:
        with _connect(db) as con:
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
        with _connect(config.CACHE_DB) as con:
            entries = con.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    return {"entries": entries, **_STATS}
