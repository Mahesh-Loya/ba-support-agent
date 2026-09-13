import threading

import pytest
from src import llm


def test_cache_key_is_stable_for_same_inputs():
    a = llm.cache_key("gemini", "m", "hello", 0.0, 512)
    b = llm.cache_key("gemini", "m", "hello", 0.0, 512)
    assert a == b


def test_cache_key_changes_with_every_input():
    base = llm.cache_key("gemini", "m", "hello", 0.0, 512)
    assert llm.cache_key("groq", "m", "hello", 0.0, 512) != base
    assert llm.cache_key("gemini", "n", "hello", 0.0, 512) != base
    assert llm.cache_key("gemini", "m", "world", 0.0, 512) != base
    assert llm.cache_key("gemini", "m", "hello", 0.7, 512) != base
    assert llm.cache_key("gemini", "m", "hello", 0.0, 256) != base


def test_cache_hit_returns_stored_value_without_calling_provider(tmp_path, monkeypatch):
    db = tmp_path / "c.sqlite"
    monkeypatch.setattr(llm.config, "CACHE_DB", db)
    llm._init_db(db)
    llm._cache_put(db, llm.cache_key("gemini", "m", "p", 0.0, 512), "cached reply")

    def explode(*a, **k):
        raise AssertionError("provider must not be called on a cache hit")

    monkeypatch.setattr(llm, "_call_provider", explode)
    out = llm.complete("p", provider="gemini", model="m")
    assert out == "cached reply"


def test_cache_miss_without_api_key_raises_actionable_error(tmp_path, monkeypatch):
    db = tmp_path / "c.sqlite"
    monkeypatch.setattr(llm.config, "CACHE_DB", db)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(llm.NoAPIKeyError) as e:
        llm.complete("uncached prompt", provider="gemini", model="m")
    assert "GEMINI_API_KEY" in str(e.value)


def test_miss_then_store_then_hit(tmp_path, monkeypatch):
    db = tmp_path / "c.sqlite"
    monkeypatch.setattr(llm.config, "CACHE_DB", db)
    monkeypatch.setattr(llm, "_call_provider", lambda *a, **k: "fresh")
    first = llm.complete("p2", provider="gemini", model="m")
    assert first == "fresh"

    def explode(*a, **k):
        raise AssertionError("second call should hit cache")

    monkeypatch.setattr(llm, "_call_provider", explode)
    assert llm.complete("p2", provider="gemini", model="m") == "fresh"


def test_cache_db_uses_wal_journal_mode(tmp_path):
    # WAL is what lets concurrent cache reads proceed without being blocked by
    # an in-flight write, and is what a second OS process running the same
    # pipeline against the same file automatically inherits (it is stored in
    # the database file itself, not per-connection).
    db = tmp_path / "c.sqlite"
    llm._init_db(db)
    with llm._connect(db) as con:
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_concurrent_cache_writes_from_many_threads_are_not_dropped(tmp_path, monkeypatch):
    # Simulates many parallel worker threads each producing a distinct
    # never-before-seen prompt at roughly the same time. None of these writes
    # may be lost (a dropped write means a silently repeated paid API call)
    # and none may raise "database is locked".
    db = tmp_path / "c.sqlite"
    monkeypatch.setattr(llm.config, "CACHE_DB", db)
    llm._init_db(db)

    n = 40
    errors = []

    def write_one(i):
        try:
            k = llm.cache_key("groq", "m", f"prompt-{i}", 0.0, 512)
            llm._cache_put(db, k, f"response-{i}")
        except Exception as exc:  # noqa: BLE001
            errors.append((i, exc))

    threads = [threading.Thread(target=write_one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    for i in range(n):
        k = llm.cache_key("groq", "m", f"prompt-{i}", 0.0, 512)
        assert llm._cache_get(db, k) == f"response-{i}"


def test_no_module_bypasses_the_llm_wrapper():
    """Every LLM call must route through src/llm.py so it gets cached."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "src"
    offenders = []
    for py in root.rglob("*.py"):
        if py.name == "llm.py":
            continue
        text = py.read_text(encoding="utf-8")
        if "from groq import" in text or "from google import genai" in text:
            offenders.append(py.name)
    assert not offenders, f"these bypass the cache: {offenders}"
