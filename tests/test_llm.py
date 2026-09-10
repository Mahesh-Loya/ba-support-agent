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
