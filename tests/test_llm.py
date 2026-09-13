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


def test_cache_key_reasoning_effort_differs_from_legacy_none():
    # The 7 old openai/gpt-oss-120b rows in the committed cache were written
    # with the pre-fix cache_key, which never had this dimension at all -
    # equivalent to reasoning_effort=None here. A judge call that now passes
    # reasoning_effort="low" must compute a genuinely different key so it can
    # never collide with (or be shadowed by) one of those broken entries.
    legacy = llm.cache_key("groq", "openai/gpt-oss-120b", "p", 0.0, 200)
    fixed = llm.cache_key("groq", "openai/gpt-oss-120b", "p", 0.0, 200,
                          reasoning_effort="low")
    assert legacy != fixed
    explicit_none = llm.cache_key("groq", "openai/gpt-oss-120b", "p", 0.0, 200,
                                  reasoning_effort=None)
    assert explicit_none == legacy


def test_complete_passes_reasoning_effort_through_to_call_provider(tmp_path, monkeypatch):
    db = tmp_path / "c.sqlite"
    monkeypatch.setattr(llm.config, "CACHE_DB", db)
    seen = {}

    def fake(prompt, **kw):
        seen.update(kw)
        return "resp"

    monkeypatch.setattr(llm, "_call_provider", fake)
    llm.complete("p", provider="groq", model="openai/gpt-oss-120b",
                 reasoning_effort="low")
    assert seen["reasoning_effort"] == "low"


def test_complete_default_omits_reasoning_effort_kwarg(tmp_path, monkeypatch):
    # The qwen classify/draft call path never passes reasoning_effort, so
    # _call_provider must receive no such kwarg at all - proving that path's
    # behaviour is byte-for-byte unchanged by this fix.
    db = tmp_path / "c.sqlite"
    monkeypatch.setattr(llm.config, "CACHE_DB", db)
    seen = {}

    def fake(prompt, **kw):
        seen.update(kw)
        return "resp"

    monkeypatch.setattr(llm, "_call_provider", fake)
    llm.complete("p", provider="groq", model="qwen/qwen3.8-27b")
    assert "reasoning_effort" not in seen


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
