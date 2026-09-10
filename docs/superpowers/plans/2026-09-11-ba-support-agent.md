# BA Support Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an AI customer-support agent for British Airways that classifies intent, drafts a grounded reply, and decides auto-handle vs. escalate — plus the evaluation harness that proves how much volume it can safely automate.

**Architecture:** A linear CLI pipeline (`ingest → taxonomy → label → run → eval → report`). Every LLM call goes through a caching wrapper backed by a committed SQLite file, so reviewers reproduce headline numbers with no API key. Retrieval uses local sentence-transformer embeddings (no rate limits). Evaluation is the centre of gravity: bootstrap CIs on every number, an LLM judge validated against human labels via Cohen's kappa, and a risk–coverage curve that turns accuracy into a deployment decision.

**Tech Stack:** Python 3.13, pandas, scikit-learn, numpy, scipy, sentence-transformers (all-MiniLM-L6-v2), google-genai (Gemini 2.5 Flash, drafter), groq (Llama 3.3 70B, judge), typer, rich, pytest, matplotlib.

**Spec:** `docs/superpowers/specs/2026-09-11-hiver-support-agent-design.md`

## Global Constraints

- **Brand is `British_Airways`.** Hard-coded as `BRAND` in `src/config.py`. Never parameterised — the whole submission is about one brand.
- **Temporal split is non-negotiable.** Corpus = `created_at < 2017-11-01`; eval pool = `created_at >= 2017-11-01`. BA data is Oct 2017 (14,976), Nov 2017 (13,031), Dec 2017 (1,100). A test asserts zero tweet_id overlap.
- **Every LLM call goes through `src/llm.py`.** No direct provider SDK calls anywhere else. Enforced by a test that greps the source tree.
- **`cache/llm_cache.sqlite` is committed to git.** It is the reproducibility mechanism, not a build artifact. Never gitignored.
- **`data/raw/twcs.csv` is gitignored** (493 MB). `data/interim/*.parquet` and `data/golden/*.jsonl` ARE committed.
- **All file I/O uses `encoding="utf-8"` explicitly.** Windows defaults to cp1252 and the dataset contains emoji — this crashes otherwise. Set `PYTHONIOENCODING=utf-8` for console output.
- **`make` is not available.** The documented entry point is `python -m src.cli reproduce`.
- **Random seed is 42**, set in `src/config.py` and passed to every sampler, splitter and bootstrap.
- **Temperature is 0.0** for all LLM calls, so cache keys are meaningful and results are deterministic.
- Target ~8 intents plus `other`. Golden set target 200 examples, floor 150.

---

### Task 1: Project scaffold, config, and dependency lock

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/__init__.py`, `src/config.py`, `tests/__init__.py`, `tests/test_config.py`
- Create: `data/raw/.gitkeep`, `data/interim/.gitkeep`, `data/golden/.gitkeep`, `cache/.gitkeep`, `reports/figures/.gitkeep`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `src.config` module exposing `BRAND: str`, `SEED: int`, `CORPUS_END: str`, `PROJECT_ROOT: Path`, `RAW_CSV: Path`, `INTERIM_DIR: Path`, `GOLDEN_DIR: Path`, `CACHE_DB: Path`, `FIGURES_DIR: Path`, `DRAFTER_MODEL: str`, `JUDGE_MODEL: str`, `EMBED_MODEL: str`

- [ ] **Step 1: Initialise git and create the directory skeleton**

```bash
cd "A:/WebDev/Hiver SDE intern"
git init
mkdir -p src tests data/raw data/interim data/golden cache reports/figures
touch src/__init__.py tests/__init__.py
touch data/raw/.gitkeep data/interim/.gitkeep data/golden/.gitkeep cache/.gitkeep reports/figures/.gitkeep
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
# Raw dataset is 493MB - reviewers download it themselves
data/raw/twcs.csv
*.pyc
__pycache__/
.pytest_cache/
.venv/
venv/
*.egg-info/
.env

# NOTE: cache/llm_cache.sqlite is deliberately NOT ignored.
# It is how reviewers reproduce headline numbers without an API key.
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "ba-support-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.26",
    "scipy>=1.11",
    "scikit-learn>=1.4",
    "pyarrow>=15.0",
    "sentence-transformers>=3.0",
    "google-genai>=1.0",
    "groq>=0.11",
    "typer>=0.12",
    "rich>=13.0",
    "tqdm>=4.66",
    "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
from src import config


def test_brand_is_british_airways():
    assert config.BRAND == "British_Airways"


def test_seed_is_fixed():
    assert config.SEED == 42


def test_corpus_end_splits_october_from_november():
    assert config.CORPUS_END == "2017-11-01"


def test_paths_are_absolute_and_rooted_in_project():
    assert config.PROJECT_ROOT.is_absolute()
    for p in (config.RAW_CSV, config.INTERIM_DIR, config.GOLDEN_DIR,
              config.CACHE_DB, config.FIGURES_DIR):
        assert config.PROJECT_ROOT in p.parents or p == config.PROJECT_ROOT


def test_drafter_and_judge_are_different_models():
    # A model grading its own output inflates the score.
    assert config.DRAFTER_MODEL != config.JUDGE_MODEL
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 6: Write `src/config.py`**

```python
"""Project-wide constants. Every magic value in this repo lives here."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data ---------------------------------------------------------------
BRAND = "British_Airways"
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "twcs.csv"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
GOLDEN_DIR = PROJECT_ROOT / "data" / "golden"
CACHE_DB = PROJECT_ROOT / "cache" / "llm_cache.sqlite"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# BA's usable data is Oct-Dec 2017. Corpus = October, eval = Nov/Dec.
# This is a temporal split so the agent can never retrieve the thread
# it is being evaluated on.
CORPUS_END = "2017-11-01"

# --- Determinism --------------------------------------------------------
SEED = 42
TEMPERATURE = 0.0

# --- Models -------------------------------------------------------------
# Drafter and judge are deliberately different families.
DRAFTER_MODEL = "gemini-2.5-flash"
DRAFTER_PROVIDER = "gemini"
JUDGE_MODEL = "llama-3.3-70b-versatile"
JUDGE_PROVIDER = "groq"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Eval ---------------------------------------------------------------
GOLDEN_TARGET = 200
GOLDEN_FLOOR = 150
BOOTSTRAP_N = 10_000
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 5 passed

- [ ] **Step 8: Install dependencies**

```bash
python -m pip install -e ".[dev]"
```

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "chore: project scaffold, config constants, dependency lock"
```

---

### Task 2: Cached LLM wrapper

This is the reproducibility mechanism. Get it right before anything calls an LLM.

**Files:**
- Create: `src/llm.py`, `tests/test_llm.py`

**Interfaces:**
- Consumes: `src.config` (CACHE_DB, TEMPERATURE)
- Produces:
  - `cache_key(provider: str, model: str, prompt: str, temperature: float, max_tokens: int) -> str`
  - `complete(prompt: str, *, provider: str, model: str, temperature: float = 0.0, max_tokens: int = 512, use_cache: bool = True) -> str`
  - `cache_stats() -> dict` returning `{"entries": int, "hits": int, "misses": int}`
  - Raises `NoAPIKeyError` when a cache miss occurs and no key is configured.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.llm'`

- [ ] **Step 3: Write `src/llm.py`**

```python
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

    env_var = _KEY_ENV.get(provider, "")
    if env_var and not os.environ.get(env_var):
        raise NoAPIKeyError(
            f"Cache miss for {provider}/{model} and {env_var} is not set.\n"
            f"Reproducing published results should never miss the cache - if you "
            f"see this during `python -m src.cli reproduce`, the committed cache "
            f"is stale. To generate fresh results, set {env_var}."
        )

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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_llm.py -v`
Expected: 5 passed

- [ ] **Step 5: Add the architectural guard test**

```python
# append to tests/test_llm.py
from pathlib import Path


def test_no_module_bypasses_the_llm_wrapper():
    """Every LLM call must route through src/llm.py so it gets cached."""
    root = Path(__file__).resolve().parent.parent / "src"
    offenders = []
    for py in root.rglob("*.py"):
        if py.name == "llm.py":
            continue
        text = py.read_text(encoding="utf-8")
        if "from groq import" in text or "from google import genai" in text:
            offenders.append(py.name)
    assert not offenders, f"these bypass the cache: {offenders}"
```

- [ ] **Step 6: Run the full test file**

Run: `python -m pytest tests/test_llm.py -v`
Expected: 6 passed

- [ ] **Step 7: Commit**

```bash
git add src/llm.py tests/test_llm.py
git commit -m "feat: cached LLM wrapper with committed SQLite cache"
```

---

### Task 3: Ingest — BA pairs, multi-part reassembly, temporal split

31.4% of BA replies are split across tweets. Getting this wrong corrupts every downstream number.

**Files:**
- Create: `src/ingest.py`, `tests/test_ingest.py`

**Interfaces:**
- Consumes: `src.config`
- Produces:
  - `strip_signature(text: str) -> tuple[str, str | None]` → `(body, signature)`
  - `strip_part_marker(text: str) -> str`
  - `reassemble(parts: list[str]) -> str`
  - `build_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame` with columns `customer_tweet_id, customer_text, agent_text, agent_signature, created_at, n_parts`
  - `assign_split(pairs: pd.DataFrame, corpus_end: str) -> pd.DataFrame` adding column `split` ∈ {`corpus`, `eval`}
  - `run_ingest() -> Path` writing `data/interim/ba_pairs.parquet`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest.py
import pandas as pd
import pytest
from src import ingest


def test_strip_signature_extracts_caret_handle():
    body, sig = ingest.strip_signature("Sorry about that, we'll check. ^Jane")
    assert body == "Sorry about that, we'll check."
    assert sig == "Jane"


def test_strip_signature_extracts_asterisk_handle():
    body, sig = ingest.strip_signature("The earlier flight was unavailable. *TMT")
    assert body == "The earlier flight was unavailable."
    assert sig == "TMT"


def test_strip_signature_returns_none_when_unsigned():
    body, sig = ingest.strip_signature("No signature here.")
    assert body == "No signature here."
    assert sig is None


def test_strip_part_marker_removes_trailing_and_leading_markers():
    assert ingest.strip_part_marker("Hello there 1/2") == "Hello there"
    assert ingest.strip_part_marker("2/2 and the rest") == "and the rest"
    assert ingest.strip_part_marker("Nothing to strip") == "Nothing to strip"


def test_reassemble_joins_parts_in_order_with_single_space():
    assert ingest.reassemble(["We're sorry for the delay", "caused. ^Jane"]) == \
        "We're sorry for the delay caused. ^Jane"


def test_build_pairs_reassembles_a_split_reply():
    df = pd.DataFrame([
        # customer opening message (not a reply to anything)
        dict(tweet_id=1, author_id="115892", inbound=True,
             created_at="Wed Nov 01 10:00:00 +0000 2017",
             text="@British_Airways my flight was cancelled, what now?",
             response_tweet_id="2", in_response_to_tweet_id=None),
        dict(tweet_id=2, author_id="British_Airways", inbound=False,
             created_at="Wed Nov 01 10:05:00 +0000 2017",
             text="@115892 Sorry for the disruption 1/2",
             response_tweet_id="3", in_response_to_tweet_id=1.0),
        dict(tweet_id=3, author_id="British_Airways", inbound=False,
             created_at="Wed Nov 01 10:06:00 +0000 2017",
             text="@115892 we can rebook you free of charge. ^Jane 2/2",
             response_tweet_id=None, in_response_to_tweet_id=2.0),
    ])
    pairs = ingest.build_pairs(df, "British_Airways")
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row.n_parts == 2
    assert row.agent_text == "Sorry for the disruption we can rebook you free of charge."
    assert row.agent_signature == "Jane"
    assert "flight was cancelled" in row.customer_text


def test_build_pairs_ignores_replies_to_other_brands():
    df = pd.DataFrame([
        dict(tweet_id=1, author_id="115892", inbound=True,
             created_at="Wed Nov 01 10:00:00 +0000 2017", text="@Delta hi",
             response_tweet_id="2", in_response_to_tweet_id=None),
        dict(tweet_id=2, author_id="Delta", inbound=False,
             created_at="Wed Nov 01 10:05:00 +0000 2017", text="@115892 hello",
             response_tweet_id=None, in_response_to_tweet_id=1.0),
    ])
    assert len(ingest.build_pairs(df, "British_Airways")) == 0


def test_assign_split_is_temporal_and_disjoint():
    pairs = pd.DataFrame(dict(
        customer_tweet_id=[1, 2],
        created_at=pd.to_datetime(["2017-10-15T00:00:00Z", "2017-11-15T00:00:00Z"]),
    ))
    out = ingest.assign_split(pairs, "2017-11-01")
    assert out.set_index("customer_tweet_id").loc[1, "split"] == "corpus"
    assert out.set_index("customer_tweet_id").loc[2, "split"] == "eval"
    corpus_ids = set(out[out.split == "corpus"].customer_tweet_id)
    eval_ids = set(out[out.split == "eval"].customer_tweet_id)
    assert corpus_ids & eval_ids == set()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ingest'`

- [ ] **Step 3: Write `src/ingest.py`**

```python
"""Turn the raw 3M-row CSV into clean (customer message -> BA reply) pairs.

Two non-obvious jobs:
  1. 31.4% of BA replies are split across tweets ("1/2", "2/2"). We rejoin them,
     otherwise a third of our ground-truth replies are truncated fragments.
  2. BA agents sign with ^Jane / *TMT. That is style, not content. We split it
     out so retrieval does not teach the model to impersonate a named human.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src import config

_SIG = re.compile(r"\s*[\^*]([A-Za-z]{2,12})\s*$")
_PART = re.compile(r"(?:^\s*\(?[1-9]/[1-9]\)?\s*)|(?:\s*\(?[1-9]/[1-9]\)?\s*$)")
_HANDLE = re.compile(r"@\w+")
_TWITTER_TS = "%a %b %d %H:%M:%S %z %Y"


def strip_signature(text: str) -> tuple[str, str | None]:
    m = _SIG.search(text)
    if not m:
        return text.strip(), None
    return _SIG.sub("", text).strip(), m.group(1)


def strip_part_marker(text: str) -> str:
    return _PART.sub(" ", text).strip()


def reassemble(parts: list[str]) -> str:
    return " ".join(p.strip() for p in parts if p.strip()).strip()


def _clean(text: str) -> str:
    return _HANDLE.sub("", str(text)).strip()


def build_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """First-contact pairs: a customer's opening message and BA's full reply."""
    df = df.copy()
    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"], errors="coerce")
    by_id = df.set_index("tweet_id")

    agent = df[(df.author_id == brand) & (df.inbound == False)]  # noqa: E712
    agent_ids = set(agent.tweet_id)

    # A reply chain is one or more consecutive agent tweets. The head of the
    # chain answers a customer tweet; later parts answer the previous part.
    heads = agent[
        agent.in_response_to_tweet_id.notna()
        & ~agent.in_response_to_tweet_id.isin(agent_ids)
    ]

    # child map: agent tweet -> the agent tweet that continues it
    cont: dict[float, int] = {}
    for _, r in agent.iterrows():
        p = r.in_response_to_tweet_id
        if pd.notna(p) and p in agent_ids:
            cont[p] = r.tweet_id

    rows = []
    for _, head in heads.iterrows():
        parent_id = head.in_response_to_tweet_id
        if parent_id not in by_id.index:
            continue
        parent = by_id.loc[parent_id]
        if parent.inbound != True:  # noqa: E712
            continue
        if pd.notna(parent.in_response_to_tweet_id):
            continue  # not a first-contact message

        parts, node = [], head
        while True:
            parts.append(strip_part_marker(_clean(node.text)))
            nxt = cont.get(node.tweet_id)
            if nxt is None:
                break
            node = by_id.loc[nxt]

        body, sig = strip_signature(reassemble(parts))
        rows.append(dict(
            customer_tweet_id=int(parent_id),
            customer_text=_clean(parent.text),
            agent_text=body,
            agent_signature=sig,
            created_at=parent.created_at,
            n_parts=len(parts),
        ))

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "customer_tweet_id", "customer_text", "agent_text",
            "agent_signature", "created_at", "n_parts"])
    return out.drop_duplicates("customer_tweet_id").reset_index(drop=True)


def assign_split(pairs: pd.DataFrame, corpus_end: str) -> pd.DataFrame:
    out = pairs.copy()
    ts = pd.to_datetime(out.created_at, format=_TWITTER_TS, errors="coerce", utc=True)
    if ts.isna().all():
        ts = pd.to_datetime(out.created_at, errors="coerce", utc=True)
    out["created_at"] = ts
    cutoff = pd.Timestamp(corpus_end, tz="UTC")
    out["split"] = (out.created_at >= cutoff).map({True: "eval", False: "corpus"})
    return out


def run_ingest() -> Path:
    df = pd.read_csv(config.RAW_CSV, dtype={"author_id": str, "text": str})
    pairs = build_pairs(df, config.BRAND)
    pairs = assign_split(pairs, config.CORPUS_END)
    pairs = pairs.dropna(subset=["created_at"])
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = config.INTERIM_DIR / "ba_pairs.parquet"
    pairs.to_parquet(out, index=False)
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: 8 passed

- [ ] **Step 5: Run ingest on the real data and sanity-check**

```bash
cd "A:/WebDev/Hiver SDE intern"
PYTHONIOENCODING=utf-8 python -c "
from src import ingest
import pandas as pd
p = ingest.run_ingest(); print('wrote', p)
d = pd.read_parquet(p)
print('pairs:', len(d))
print(d.split.value_counts().to_string())
print('multi-part share:', round((d.n_parts>1).mean()*100,1), '%')
print('signed share:', round(d.agent_signature.notna().mean()*100,1), '%')
print(d[['customer_text','agent_text']].head(3).to_string())
"
```

Expected: roughly 19k pairs, both `corpus` and `eval` non-empty, multi-part share in the 20–35% range, signed share near 64%.

- [ ] **Step 6: Hand-verify five reassembled replies**

```bash
PYTHONIOENCODING=utf-8 python -c "
import pandas as pd
d = pd.read_parquet('data/interim/ba_pairs.parquet')
for _, r in d[d.n_parts>1].head(5).iterrows():
    print('CUST:', r.customer_text[:160]); print('AGENT:', r.agent_text[:320]); print()
"
```

Read them. Each reassembled reply must be a coherent sentence, not a fragment. If not, fix `build_pairs` before continuing — every downstream number depends on this.

- [ ] **Step 7: Commit**

```bash
git add src/ingest.py tests/test_ingest.py data/interim/ba_pairs.parquet
git commit -m "feat: ingest BA pairs with multi-part reassembly and temporal split"
```

---

### Task 4: Intent taxonomy via clustering

**Files:**
- Create: `src/embed.py`, `src/taxonomy.py`, `tests/test_taxonomy.py`
- Create: `data/interim/taxonomy.json` (output, hand-edited)

**Interfaces:**
- Consumes: `src.config`, `data/interim/ba_pairs.parquet`
- Produces:
  - `src.embed.embed(texts: list[str]) -> np.ndarray` (L2-normalised, cached to `data/interim/emb_<hash>.npy`)
  - `src.taxonomy.cluster(vectors, k) -> np.ndarray` (labels)
  - `src.taxonomy.top_terms(texts, labels, k) -> dict[int, list[str]]`
  - `src.taxonomy.propose(n_clusters: int) -> dict` written to `data/interim/clusters.json`
  - `src.taxonomy.INTENTS: list[str]` loaded from `data/interim/taxonomy.json`
  - `src.taxonomy.load_intents() -> list[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_taxonomy.py
import numpy as np
from src import embed, taxonomy


def test_embed_returns_normalised_vectors():
    v = embed.embed(["flight cancelled", "lost my bag"])
    assert v.shape[0] == 2
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_embed_is_deterministic():
    a = embed.embed(["flight cancelled"])
    b = embed.embed(["flight cancelled"])
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_similar_texts_are_closer_than_dissimilar_ones():
    v = embed.embed(["my flight was cancelled",
                     "my flight got cancelled today",
                     "how do I join the Executive Club"])
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])


def test_cluster_separates_two_obvious_groups():
    v = embed.embed(["lost baggage", "my bag is missing", "baggage never arrived",
                     "flight delayed", "flight is late", "delayed departure"])
    labels = taxonomy.cluster(v, k=2)
    assert len(set(labels)) == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_top_terms_surfaces_discriminative_words():
    texts = ["lost baggage", "missing baggage", "flight delayed", "flight late"]
    labels = np.array([0, 0, 1, 1])
    terms = taxonomy.top_terms(texts, labels, k=3)
    assert "baggage" in terms[0]
    assert "flight" in terms[1] or "delayed" in terms[1]


def test_load_intents_includes_other_and_is_reasonably_small():
    intents = taxonomy.load_intents()
    assert "other" in intents
    assert 5 <= len(intents) <= 12
    assert len(intents) == len(set(intents))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_taxonomy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.embed'`

- [ ] **Step 3: Write `src/embed.py`**

```python
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
```

- [ ] **Step 4: Write `src/taxonomy.py`**

```python
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
```

- [ ] **Step 5: Generate the cluster report and read it**

```bash
PYTHONIOENCODING=utf-8 python -c "
from src import taxonomy
r = taxonomy.propose(n_clusters=12)
for k, v in r.items():
    print(f\"--- cluster {k}  (n={v['size']}) ---\")
    print('terms:', ', '.join(v['top_terms']))
    for e in v['examples'][:4]: print('  *', e)
    print()
"
```

- [ ] **Step 6: Hand-write `data/interim/taxonomy.json` from what you read**

Merge clusters that are not genuinely distinct. Target ~8 plus `other`. Starting point — **edit this to match the actual clusters you saw**:

```json
{
  "intents": [
    "flight_disruption",
    "baggage",
    "booking_change",
    "refund_compensation",
    "checkin_boarding",
    "loyalty_account",
    "complaint_feedback",
    "general_enquiry",
    "other"
  ],
  "definitions": {
    "flight_disruption": "Delays, cancellations, diversions, missed connections.",
    "baggage": "Lost, delayed, damaged baggage; allowance and excess questions.",
    "booking_change": "Change/cancel a booking, seats, upgrades, name changes.",
    "refund_compensation": "Requests for money back, EU261 claims, vouchers.",
    "checkin_boarding": "Online check-in failures, boarding passes, documents.",
    "loyalty_account": "Executive Club, Avios, tier status, account access.",
    "complaint_feedback": "Service complaints and praise with no actionable request.",
    "general_enquiry": "Policy/informational questions with no active problem.",
    "other": "Not a support request, spam, or unintelligible."
  },
  "notes": "Merged during naming: <record what you merged and why>."
}
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest tests/test_taxonomy.py -v`
Expected: 6 passed

- [ ] **Step 8: Commit**

```bash
git add src/embed.py src/taxonomy.py tests/test_taxonomy.py data/interim/taxonomy.json data/interim/clusters.json
git commit -m "feat: derive intent taxonomy from clustered BA messages"
```

---

### Task 5: Golden set sampling and labelling tool

**Files:**
- Create: `src/sampling.py`, `src/label_tui.py`, `tests/test_sampling.py`
- Create: `data/golden/golden.jsonl` (your labels)

**Interfaces:**
- Consumes: `src.config`, `src.taxonomy`, `data/interim/ba_pairs.parquet`
- Produces:
  - `src.sampling.stratified_sample(df, strata_col, n_total, floor_per_stratum, seed) -> pd.DataFrame`
  - `src.sampling.build_golden_pool(n_total: int) -> pd.DataFrame` written to `data/golden/golden_pool.jsonl`
  - `src.label_tui.main()` — writes/updates `data/golden/golden.jsonl` with schema:
    `{customer_tweet_id, customer_text, agent_text, intent, action, escalation_reason, round}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sampling.py
import pandas as pd
import pytest
from src import sampling


def _skewed():
    # 200 head, 30 mid, 5 tail - the exact shape that makes random sampling useless
    return pd.DataFrame(dict(
        i=range(235),
        stratum=["head"] * 200 + ["mid"] * 30 + ["tail"] * 5,
    ))


def test_stratified_sample_respects_the_floor_for_rare_strata():
    out = sampling.stratified_sample(_skewed(), "stratum", n_total=60,
                                     floor_per_stratum=15, seed=42)
    counts = out.stratum.value_counts()
    assert counts["tail"] == 5      # can't exceed what exists
    assert counts["mid"] >= 15
    assert len(out) <= 60


def test_stratified_sample_is_deterministic_under_a_seed():
    a = sampling.stratified_sample(_skewed(), "stratum", 60, 15, seed=42)
    b = sampling.stratified_sample(_skewed(), "stratum", 60, 15, seed=42)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_stratified_sample_beats_random_on_tail_coverage():
    df = _skewed()
    strat = sampling.stratified_sample(df, "stratum", 60, 15, seed=42)
    rand = df.sample(60, random_state=42)
    assert strat.stratum.nunique() >= rand.stratum.nunique()
    assert (strat.stratum == "mid").sum() > (rand.stratum == "mid").sum()


def test_sample_never_returns_duplicates():
    out = sampling.stratified_sample(_skewed(), "stratum", 100, 15, seed=42)
    assert out.i.is_unique
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_sampling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.sampling'`

- [ ] **Step 3: Write `src/sampling.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_sampling.py -v`
Expected: 4 passed

- [ ] **Step 5: Write `src/label_tui.py`**

```python
"""Keyboard-driven labelling tool. 200 examples in ~45 minutes.

Shows ONLY the customer message while labelling intent and action - BA's actual
reply stays hidden so it cannot anchor your judgement. Resumable: rerun and it
picks up where you stopped.
"""
from __future__ import annotations

import json
import sys

import pandas as pd
from rich.console import Console
from rich.panel import Panel

from src import config, taxonomy

GOLDEN = config.GOLDEN_DIR / "golden.jsonl"

ESCALATION_REASONS = {
    "1": "needs_account_lookup",
    "2": "money_claim",
    "3": "legal_or_safety",
    "4": "low_confidence_ambiguous",
    "5": "severe_distress",
    "6": "multi_issue",
}


def _load_done() -> dict:
    if not GOLDEN.exists():
        return {}
    done = {}
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            done[r["customer_tweet_id"]] = r
    return done


def _append(rec: dict) -> None:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(round_id: int = 1) -> None:
    con = Console()
    intents = taxonomy.load_intents()
    pool = pd.read_json(config.GOLDEN_DIR / "golden_pool.jsonl", lines=True)
    done = _load_done()

    todo = [r for _, r in pool.iterrows()
            if r.customer_tweet_id not in done or round_id != 1]
    con.print(f"[bold]{len(done)} labelled, {len(todo)} to go[/bold]\n")

    for n, row in enumerate(todo, 1):
        con.print(Panel(str(row.customer_text), title=f"[{n}/{len(todo)}] customer message"))
        con.print("  " + "  ".join(f"[cyan]{i}[/cyan]={s}" for i, s in enumerate(intents)))
        raw = input("intent # (or 's' skip, 'q' quit) > ").strip().lower()
        if raw == "q":
            break
        if raw == "s" or not raw.isdigit() or int(raw) >= len(intents):
            continue
        intent = intents[int(raw)]

        act = input("  [a]uto-handle / [e]scalate > ").strip().lower()
        action = "escalate" if act.startswith("e") else "auto"

        reason = ""
        if action == "escalate":
            con.print("  " + "  ".join(f"[cyan]{k}[/cyan]={v}"
                                       for k, v in ESCALATION_REASONS.items()))
            reason = ESCALATION_REASONS.get(input("  reason # > ").strip(), "other")

        _append(dict(
            customer_tweet_id=int(row.customer_tweet_id),
            customer_text=str(row.customer_text),
            agent_text=str(row.agent_text),
            intent=intent, action=action,
            escalation_reason=reason, round=round_id,
        ))
        con.print("[green]saved[/green]\n")


if __name__ == "__main__":
    main(round_id=int(sys.argv[1]) if len(sys.argv) > 1 else 1)
```

- [ ] **Step 6: Build the pool and label**

```bash
PYTHONIOENCODING=utf-8 python -c "from src import sampling; p=sampling.build_golden_pool(200); print(len(p), 'pooled'); print(p.stratum.value_counts().to_string())"
PYTHONIOENCODING=utf-8 python -m src.label_tui 1
```

This is the grind. Do it in two or three sittings. **Do not look at any model output before finishing.**

- [ ] **Step 7: Re-label 25 examples days later to measure your own consistency**

```bash
PYTHONIOENCODING=utf-8 python -c "
import pandas as pd
from src import config
p = pd.read_json(config.GOLDEN_DIR/'golden_pool.jsonl', lines=True).sample(25, random_state=7)
p.to_json(config.GOLDEN_DIR/'golden_pool.jsonl'.replace('.jsonl','_round2.jsonl'), orient='records', lines=True, force_ascii=False)
print('round-2 slice written')
"
```

Then rerun the TUI with `round_id=2` over that slice. Intra-annotator agreement is the honest ceiling on any accuracy number you report.

- [ ] **Step 8: Verify the golden set is usable**

```bash
PYTHONIOENCODING=utf-8 python -c "
import pandas as pd
g = pd.read_json('data/golden/golden.jsonl', lines=True)
g1 = g[g['round']==1]
print('labelled:', len(g1)); assert len(g1) >= 150, 'below floor'
print(g1.intent.value_counts().to_string())
print(g1.action.value_counts().to_string())
assert g1.intent.nunique() >= 5
"
```

- [ ] **Step 9: Commit**

```bash
git add src/sampling.py src/label_tui.py tests/test_sampling.py data/golden/
git commit -m "feat: stratified golden pool and labelling TUI; add hand labels"
```

---

### Task 6: Baselines

**Files:**
- Create: `src/baselines.py`, `tests/test_baselines.py`

**Interfaces:**
- Consumes: `src.config`, `src.taxonomy`, `data/interim/ba_pairs.parquet`
- Produces:
  - `TrivialBaseline.fit(texts, labels)` / `.predict(texts) -> list[str]` / `.reply(text) -> str`
  - `SimpleBaseline.fit(texts, labels)` / `.predict(texts) -> list[str]` / `.predict_proba(texts) -> np.ndarray` / `.reply(text) -> str`
  - Both expose `.classes_: list[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines.py
import numpy as np
from src import baselines

TEXTS = ["flight cancelled", "flight delayed again", "lost my bag",
         "bag missing", "flight cancelled today", "delayed flight"]
LABELS = ["flight_disruption", "flight_disruption", "baggage",
          "baggage", "flight_disruption", "flight_disruption"]


def test_trivial_baseline_always_predicts_the_majority_class():
    m = baselines.TrivialBaseline().fit(TEXTS, LABELS)
    assert m.predict(["anything at all", "something else"]) == \
        ["flight_disruption", "flight_disruption"]


def test_trivial_baseline_reply_is_the_single_most_common_response():
    m = baselines.TrivialBaseline().fit(TEXTS, LABELS)
    m.fit_replies(["we are sorry", "we are sorry", "please DM us"])
    assert m.reply("literally anything") == "we are sorry"


def test_simple_baseline_learns_the_two_classes():
    m = baselines.SimpleBaseline().fit(TEXTS, LABELS)
    assert m.predict(["my bag is lost"])[0] == "baggage"
    assert m.predict(["my flight is cancelled"])[0] == "flight_disruption"


def test_simple_baseline_proba_rows_sum_to_one():
    m = baselines.SimpleBaseline().fit(TEXTS, LABELS)
    p = m.predict_proba(["lost bag", "cancelled flight"])
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert p.shape == (2, len(m.classes_))


def test_simple_baseline_reply_copies_a_real_historical_reply():
    m = baselines.SimpleBaseline().fit(TEXTS, LABELS)
    m.fit_replies(TEXTS, ["r-cancel", "r-delay", "r-bag",
                          "r-bag2", "r-cancel2", "r-delay2"])
    assert m.reply("my bag has gone missing") in {"r-bag", "r-bag2"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.baselines'`

- [ ] **Step 3: Write `src/baselines.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_baselines.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/baselines.py tests/test_baselines.py
git commit -m "feat: trivial and simple (TF-IDF) baselines"
```

---

### Task 7: Retrieval

**Files:**
- Create: `src/agent/__init__.py`, `src/agent/retrieve.py`, `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `src.config`, `src.embed`, `data/interim/ba_pairs.parquet`
- Produces:
  - `Case` dataclass: `customer_text: str`, `agent_text: str`, `similarity: float`
  - `Retriever(corpus: pd.DataFrame)` with `.search(text: str, k: int = 5) -> list[Case]`
  - `Retriever.from_corpus_split() -> Retriever` (loads `split == "corpus"` only)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retrieve.py
import pandas as pd
import pytest
from src.agent import retrieve


@pytest.fixture
def corpus():
    return pd.DataFrame(dict(
        customer_tweet_id=[1, 2, 3],
        customer_text=["my flight was cancelled",
                       "my suitcase never arrived",
                       "how many Avios for an upgrade"],
        agent_text=["Sorry - we can rebook you free of charge.",
                    "Please file a report at the baggage desk.",
                    "Upgrades start at 10,000 Avios."],
        split=["corpus"] * 3,
    ))


def test_search_returns_the_topically_matching_case_first(corpus):
    r = retrieve.Retriever(corpus)
    top = r.search("they cancelled my flight!", k=1)[0]
    assert "rebook" in top.agent_text


def test_search_respects_k(corpus):
    r = retrieve.Retriever(corpus)
    assert len(r.search("baggage lost", k=2)) == 2


def test_similarity_is_bounded_and_descending(corpus):
    r = retrieve.Retriever(corpus)
    cases = r.search("lost luggage", k=3)
    sims = [c.similarity for c in cases]
    assert sims == sorted(sims, reverse=True)
    assert all(-1.01 <= s <= 1.01 for s in sims)


def test_retriever_only_ever_holds_corpus_split_rows():
    df = pd.DataFrame(dict(
        customer_tweet_id=[1, 2],
        customer_text=["corpus row", "eval row"],
        agent_text=["a", "b"],
        split=["corpus", "eval"],
    ))
    r = retrieve.Retriever(df)
    assert len(r.corpus) == 1
    assert r.corpus.iloc[0].customer_text == "corpus row"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_retrieve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent'`

- [ ] **Step 3: Write `src/agent/retrieve.py`**

```python
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
        if "split" in corpus.columns:
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_retrieve.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/ tests/test_retrieve.py
git commit -m "feat: corpus-split-only retriever over local embeddings"
```

---

### Task 8: Classifier

**Files:**
- Create: `src/agent/classify.py`, `tests/test_classify.py`

**Interfaces:**
- Consumes: `src.llm`, `src.taxonomy`
- Produces:
  - `Classification` dataclass: `intent: str`, `confidence: float`
  - `build_prompt(text: str, intents: list[str], definitions: dict) -> str`
  - `parse_response(raw: str, intents: list[str]) -> Classification`
  - `classify(text: str, use_cache: bool = True) -> Classification`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_classify.py
import pytest
from src.agent import classify

INTENTS = ["baggage", "flight_disruption", "other"]


def test_parse_response_reads_well_formed_json():
    c = classify.parse_response('{"intent": "baggage", "confidence": 0.91}', INTENTS)
    assert c.intent == "baggage"
    assert c.confidence == pytest.approx(0.91)


def test_parse_response_tolerates_markdown_fences():
    raw = '```json\n{"intent": "baggage", "confidence": 0.8}\n```'
    assert classify.parse_response(raw, INTENTS).intent == "baggage"


def test_parse_response_falls_back_to_other_on_unknown_label():
    c = classify.parse_response('{"intent": "spaceflight", "confidence": 0.9}', INTENTS)
    assert c.intent == "other"
    assert c.confidence == 0.0


def test_parse_response_falls_back_to_other_on_garbage():
    c = classify.parse_response("I think it's about bags maybe?", INTENTS)
    assert c.intent == "other"
    assert c.confidence == 0.0


def test_confidence_is_clamped_to_unit_interval():
    assert classify.parse_response('{"intent":"baggage","confidence":5}', INTENTS).confidence == 1.0
    assert classify.parse_response('{"intent":"baggage","confidence":-2}', INTENTS).confidence == 0.0


def test_prompt_contains_every_intent_and_the_message():
    p = classify.build_prompt("my bag is gone", INTENTS, {"baggage": "lost bags"})
    for i in INTENTS:
        assert i in p
    assert "my bag is gone" in p


def test_classify_routes_through_the_cached_llm_wrapper(monkeypatch):
    calls = []

    def fake(prompt, **kw):
        calls.append(kw)
        return '{"intent": "baggage", "confidence": 0.7}'

    monkeypatch.setattr(classify.llm, "complete", fake)
    monkeypatch.setattr(classify.taxonomy, "load_intents", lambda: INTENTS)
    monkeypatch.setattr(classify, "_definitions", lambda: {})
    out = classify.classify("lost bag")
    assert out.intent == "baggage"
    assert len(calls) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_classify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.classify'`

- [ ] **Step 3: Write `src/agent/classify.py`**

```python
"""Intent classification. Returns a self-reported confidence whose calibration
we later MEASURE rather than trust (see src/eval/risk_coverage.py)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src import config, llm, taxonomy


@dataclass
class Classification:
    intent: str
    confidence: float


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def _definitions() -> dict:
    return json.loads(taxonomy.TAXONOMY_PATH.read_text(encoding="utf-8"))["definitions"]


def build_prompt(text: str, intents: list[str], definitions: dict) -> str:
    lines = "\n".join(f"- {i}: {definitions.get(i, '')}" for i in intents)
    return (
        "You are triaging a customer support tweet sent to British Airways.\n"
        "Classify it into exactly one intent.\n\n"
        f"Intents:\n{lines}\n\n"
        f"Message:\n\"\"\"{text}\"\"\"\n\n"
        'Reply with JSON only: {"intent": "<one of the intents above>", '
        '"confidence": <0.0-1.0>}\n'
        "confidence is your probability that this label is correct."
    )


def parse_response(raw: str, intents: list[str]) -> Classification:
    cleaned = _FENCE.sub("", raw.strip())
    m = re.search(r"\{.*\}", cleaned, re.S)
    if not m:
        return Classification("other", 0.0)
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Classification("other", 0.0)

    intent = str(data.get("intent", "")).strip()
    if intent not in intents:
        return Classification("other", 0.0)
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return Classification(intent, max(0.0, min(1.0, conf)))


def classify(text: str, use_cache: bool = True) -> Classification:
    intents = taxonomy.load_intents()
    raw = llm.complete(
        build_prompt(text, intents, _definitions()),
        provider=config.DRAFTER_PROVIDER, model=config.DRAFTER_MODEL,
        max_tokens=100, use_cache=use_cache,
    )
    return parse_response(raw, intents)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_classify.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/classify.py tests/test_classify.py
git commit -m "feat: LLM intent classifier with defensive parsing"
```

---

### Task 9: Reply drafting

**Files:**
- Create: `src/agent/draft.py`, `tests/test_draft.py`

**Interfaces:**
- Consumes: `src.llm`, `src.agent.retrieve.Case`
- Produces:
  - `build_draft_prompt(text: str, intent: str, cases: list[Case]) -> str`
  - `clean_reply(raw: str) -> str` (strips quotes, fences, invented `^Signature`)
  - `draft_reply(text: str, intent: str, cases: list[Case], use_cache: bool = True) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_draft.py
from src.agent import draft
from src.agent.retrieve import Case

CASES = [
    Case("my flight was cancelled", "Sorry - we can rebook you free of charge.", 0.9),
    Case("flight cancelled last minute", "We'll get you on the next available flight.", 0.8),
]


def test_prompt_includes_every_retrieved_historical_reply():
    p = draft.build_draft_prompt("they cancelled my flight", "flight_disruption", CASES)
    for c in CASES:
        assert c.agent_text in p


def test_prompt_states_the_no_invented_commitments_rule():
    p = draft.build_draft_prompt("x", "flight_disruption", CASES)
    low = p.lower()
    assert "do not" in low or "never" in low
    assert "compensation" in low or "promise" in low or "commit" in low


def test_clean_reply_strips_fences_and_wrapping_quotes():
    assert draft.clean_reply('```\n"Sorry about that."\n```') == "Sorry about that."


def test_clean_reply_removes_invented_agent_signature():
    # The agent must not sign as a human who does not exist.
    assert draft.clean_reply("We can rebook you. ^Jane") == "We can rebook you."


def test_clean_reply_strips_a_leading_reply_label():
    assert draft.clean_reply("Reply: We can help with that.") == "We can help with that."


def test_draft_reply_uses_the_cached_wrapper_and_returns_clean_text(monkeypatch):
    monkeypatch.setattr(draft.llm, "complete",
                        lambda prompt, **kw: '"We can rebook you free of charge." ^Sam')
    out = draft.draft_reply("cancelled!", "flight_disruption", CASES)
    assert out == "We can rebook you free of charge."
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_draft.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.draft'`

- [ ] **Step 3: Write `src/agent/draft.py`**

```python
"""Draft a reply grounded in how BA actually handled similar messages."""
from __future__ import annotations

import re

from src import config, llm
from src.agent.retrieve import Case

_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.M)
_SIG = re.compile(r"\s*[\^*][A-Za-z]{2,12}\s*$")
_LABEL = re.compile(r"^(?:reply|response|draft)\s*:\s*", re.I)


def build_draft_prompt(text: str, intent: str, cases: list[Case]) -> str:
    examples = "\n\n".join(
        f"Past customer: {c.customer_text}\nBritish Airways replied: {c.agent_text}"
        for c in cases
    )
    return (
        "You are a British Airways customer support agent replying on Twitter.\n"
        f"The customer's intent has been classified as: {intent}\n\n"
        "Here is how British Airways handled similar messages in the past:\n\n"
        f"{examples}\n\n"
        "Rules:\n"
        "- Match the tone and length of the past replies (usually 1-2 short sentences).\n"
        "- Ground your reply in what BA actually did above.\n"
        "- NEVER promise compensation, refunds, upgrades or rebooking unless the "
        "past replies show BA committing to exactly that.\n"
        "- Do not invent booking references, amounts, dates or policies.\n"
        "- Do not sign the message with a name.\n"
        "- Do not include a greeting like 'Hi there' unless the examples do.\n\n"
        f"New customer message:\n\"\"\"{text}\"\"\"\n\n"
        "Write only the reply text, nothing else."
    )


def clean_reply(raw: str) -> str:
    out = _FENCE.sub("", raw.strip()).strip()
    out = _LABEL.sub("", out).strip()
    out = _SIG.sub("", out).strip()
    if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'":
        out = out[1:-1].strip()
    out = _SIG.sub("", out).strip()
    return out


def draft_reply(text: str, intent: str, cases: list[Case],
                use_cache: bool = True) -> str:
    raw = llm.complete(
        build_draft_prompt(text, intent, cases),
        provider=config.DRAFTER_PROVIDER, model=config.DRAFTER_MODEL,
        max_tokens=200, use_cache=use_cache,
    )
    return clean_reply(raw)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_draft.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/agent/draft.py tests/test_draft.py
git commit -m "feat: grounded reply drafting with anti-overpromise rules"
```

---

### Task 10: Escalation decision — hard rules then confidence

**Files:**
- Create: `src/agent/decide.py`, `tests/test_decide.py`

**Interfaces:**
- Consumes: `src.agent.classify.Classification`, `src.agent.retrieve.Case`
- Produces:
  - `Decision` dataclass: `action: str` (`"auto"`/`"escalate"`), `reason: str`, `score: float`, `rule_fired: str | None`
  - `check_hard_rules(text: str, intent: str) -> str | None`
  - `routing_score(confidence: float, top_similarity: float, clf_margin: float) -> float`
  - `decide(text, intent, confidence, top_similarity, clf_margin, threshold) -> Decision`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decide.py
import pytest
from src.agent import decide


@pytest.mark.parametrize("text,rule", [
    ("I want a refund for my cancelled flight", "money_claim"),
    ("I am claiming compensation under EU261", "money_claim"),
    ("My solicitor will be in touch about this", "legal_or_safety"),
    ("There was smoke in the cabin, this was dangerous", "legal_or_safety"),
    ("My suitcase is lost, it never arrived at all", "lost_baggage"),
])
def test_hard_rules_fire_on_never_automate_categories(text, rule):
    assert decide.check_hard_rules(text, "other") == rule


def test_hard_rules_do_not_fire_on_a_plain_question():
    assert decide.check_hard_rules("What is the cabin baggage allowance?",
                                   "general_enquiry") is None


def test_refund_intent_always_escalates_even_at_max_confidence():
    d = decide.decide("please refund me", "refund_compensation",
                      confidence=1.0, top_similarity=1.0, clf_margin=1.0,
                      threshold=0.0)
    assert d.action == "escalate"
    assert d.rule_fired == "money_claim"


def test_high_confidence_safe_message_is_auto_handled():
    d = decide.decide("What is the cabin baggage allowance?", "general_enquiry",
                      confidence=0.95, top_similarity=0.9, clf_margin=0.8,
                      threshold=0.5)
    assert d.action == "auto"
    assert d.rule_fired is None


def test_low_score_escalates_with_a_confidence_reason():
    d = decide.decide("What is the cabin baggage allowance?", "general_enquiry",
                      confidence=0.2, top_similarity=0.1, clf_margin=0.05,
                      threshold=0.5)
    assert d.action == "escalate"
    assert d.reason == "low_confidence_ambiguous"


def test_routing_score_is_bounded_and_monotonic():
    lo = decide.routing_score(0.1, 0.1, 0.1)
    hi = decide.routing_score(0.9, 0.9, 0.9)
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
    assert hi > lo


def test_every_decision_carries_a_non_empty_reason():
    for args in [("refund me", "refund_compensation", 0.9, 0.9, 0.9),
                 ("baggage allowance?", "general_enquiry", 0.9, 0.9, 0.9),
                 ("baggage allowance?", "general_enquiry", 0.1, 0.1, 0.1)]:
        d = decide.decide(*args, threshold=0.5)
        assert d.reason
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_decide.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent.decide'`

- [ ] **Step 3: Write `src/agent/decide.py`**

```python
"""Auto-handle vs. escalate.

Two layers, because they answer different questions:

  1. HARD RULES - categories a business must never automate, regardless of how
     confident the model is. This is policy, not prediction.
  2. CONFIDENCE GATE - everything else routes on a score, and abstains below a
     threshold chosen from an explicit cost model (see eval/risk_coverage.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HARD_RULES: list[tuple[str, str]] = [
    ("money_claim", r"\b(refund|compensat\w*|reimburs\w*|eu ?261|voucher|"
                    r"money back|charge(d)? me|overcharg\w*)\b"),
    ("legal_or_safety", r"\b(solicitor|lawyer|legal action|sue|court|ombudsman|"
                        r"caa|dangerous|unsafe|smoke|fire|injur\w*|assault\w*|"
                        r"discriminat\w*|racist)\b"),
    ("lost_baggage", r"\b(lost|missing|never arrived|didn'?t arrive|no sign of)\b"
                     r"[^.]{0,40}\b(bag|bags|baggage|luggage|suitcase|case)\b"
                     r"|\b(bag|bags|baggage|luggage|suitcase)\b[^.]{0,40}"
                     r"\b(lost|missing|never arrived|didn'?t arrive)\b"),
    ("account_lookup", r"\b(booking reference|pnr|my account|log ?in|password|"
                       r"executive club number|membership number)\b"),
    ("severe_distress", r"\b(disgust\w*|appall\w*|worst|never fly\w* again|"
                        r"disgrace\w*|furious|outrage\w*)\b"),
]

ALWAYS_ESCALATE_INTENTS = {"refund_compensation"}

_COMPILED = [(name, re.compile(p, re.I)) for name, p in HARD_RULES]


@dataclass
class Decision:
    action: str          # "auto" | "escalate"
    reason: str
    score: float
    rule_fired: str | None


def check_hard_rules(text: str, intent: str) -> str | None:
    if intent in ALWAYS_ESCALATE_INTENTS:
        return "money_claim"
    for name, pat in _COMPILED:
        if pat.search(text):
            return name
    return None


def routing_score(confidence: float, top_similarity: float,
                  clf_margin: float) -> float:
    """Blend three signals. Weights are deliberate and reported; which signal is
    actually calibrated is measured in eval/risk_coverage.py, not assumed."""
    sim = max(0.0, min(1.0, top_similarity))
    return max(0.0, min(1.0, 0.5 * confidence + 0.3 * sim + 0.2 * clf_margin))


def decide(text: str, intent: str, confidence: float, top_similarity: float,
           clf_margin: float, threshold: float = 0.5) -> Decision:
    score = routing_score(confidence, top_similarity, clf_margin)

    rule = check_hard_rules(text, intent)
    if rule:
        return Decision("escalate",
                        f"hard_rule:{rule}" if False else rule,
                        score, rule)

    if score < threshold:
        return Decision("escalate", "low_confidence_ambiguous", score, None)

    return Decision("auto", "high_confidence_routine", score, None)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_decide.py -v`
Expected: 11 passed

- [ ] **Step 5: Simplify the leftover dead conditional**

Replace `f"hard_rule:{rule}" if False else rule` with just `rule`. Re-run the tests.

Run: `python -m pytest tests/test_decide.py -v`
Expected: 11 passed

- [ ] **Step 6: Commit**

```bash
git add src/agent/decide.py tests/test_decide.py
git commit -m "feat: escalation policy - hard rules plus confidence gate"
```

---

### Task 11: Metrics with bootstrap confidence intervals

**Files:**
- Create: `src/eval/__init__.py`, `src/eval/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing beyond numpy/sklearn
- Produces:
  - `accuracy(y_true, y_pred) -> float`
  - `macro_f1(y_true, y_pred) -> float`
  - `bootstrap_ci(y_true, y_pred, stat_fn, n=10000, seed=42) -> tuple[float, float]`
  - `confusion(y_true, y_pred, labels) -> pd.DataFrame`
  - `summary(y_true, y_pred) -> dict` with keys `accuracy`, `accuracy_ci`, `macro_f1`, `macro_f1_ci`, `n`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import numpy as np
import pytest
from src.eval import metrics


def test_accuracy_is_exact_on_a_known_case():
    assert metrics.accuracy(["a", "b", "c", "d"], ["a", "b", "c", "x"]) == 0.75


def test_macro_f1_is_one_for_a_perfect_prediction():
    assert metrics.macro_f1(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)


def test_macro_f1_punishes_ignoring_the_rare_class():
    # Always predicting the majority class: high accuracy, poor macro-F1.
    y = ["a"] * 9 + ["b"]
    p = ["a"] * 10
    assert metrics.accuracy(y, p) == pytest.approx(0.9)
    assert metrics.macro_f1(y, p) < 0.5


def test_bootstrap_ci_brackets_the_point_estimate():
    y = ["a"] * 50 + ["b"] * 50
    p = ["a"] * 45 + ["b"] * 5 + ["b"] * 45 + ["a"] * 5
    lo, hi = metrics.bootstrap_ci(y, p, metrics.accuracy, n=2000, seed=42)
    point = metrics.accuracy(y, p)
    assert lo <= point <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_bootstrap_ci_is_wider_for_a_smaller_sample():
    y_big = ["a"] * 100 + ["b"] * 100
    p_big = ["a"] * 90 + ["b"] * 10 + ["b"] * 90 + ["a"] * 10
    lo_b, hi_b = metrics.bootstrap_ci(y_big, p_big, metrics.accuracy, n=2000, seed=1)
    lo_s, hi_s = metrics.bootstrap_ci(y_big[:20], p_big[:20], metrics.accuracy,
                                      n=2000, seed=1)
    assert (hi_s - lo_s) > (hi_b - lo_b)


def test_bootstrap_ci_is_deterministic_under_a_seed():
    y = ["a", "b"] * 25
    p = ["a"] * 50
    assert metrics.bootstrap_ci(y, p, metrics.accuracy, n=500, seed=7) == \
           metrics.bootstrap_ci(y, p, metrics.accuracy, n=500, seed=7)


def test_summary_reports_n_and_both_intervals():
    s = metrics.summary(["a", "b", "a", "b"], ["a", "b", "a", "a"])
    assert s["n"] == 4
    assert len(s["accuracy_ci"]) == 2
    assert len(s["macro_f1_ci"]) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval'`

- [ ] **Step 3: Write `src/eval/metrics.py`**

```python
"""Metrics with confidence intervals.

With n=200, a 3-point difference between two systems is noise. Every headline
number in this project carries a bootstrap CI so that is visible rather than
hidden.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src import config


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    return float(accuracy_score(list(y_true), list(y_pred)))


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    return float(f1_score(list(y_true), list(y_pred), average="macro", zero_division=0))


def bootstrap_ci(y_true: Sequence, y_pred: Sequence,
                 stat_fn: Callable[[Sequence, Sequence], float],
                 n: int = config.BOOTSTRAP_N, seed: int = config.SEED,
                 alpha: float = 0.05) -> tuple[float, float]:
    yt, yp = np.asarray(list(y_true), dtype=object), np.asarray(list(y_pred), dtype=object)
    rng = np.random.default_rng(seed)
    size = len(yt)
    stats = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, size, size)
        stats[i] = stat_fn(yt[idx], yp[idx])
    return (float(np.quantile(stats, alpha / 2)),
            float(np.quantile(stats, 1 - alpha / 2)))


def confusion(y_true: Sequence, y_pred: Sequence, labels: list[str]) -> pd.DataFrame:
    m = confusion_matrix(list(y_true), list(y_pred), labels=labels)
    return pd.DataFrame(m, index=[f"true_{l}" for l in labels],
                        columns=[f"pred_{l}" for l in labels])


def summary(y_true: Sequence, y_pred: Sequence, n_boot: int = 2000) -> dict:
    return {
        "n": len(list(y_true)),
        "accuracy": accuracy(y_true, y_pred),
        "accuracy_ci": bootstrap_ci(y_true, y_pred, accuracy, n=n_boot),
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_f1_ci": bootstrap_ci(y_true, y_pred, macro_f1, n=n_boot),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/eval/ tests/test_metrics.py
git commit -m "feat: metrics with bootstrap confidence intervals"
```

---

### Task 12: LLM-as-judge

**Files:**
- Create: `src/eval/judge.py`, `tests/test_judge.py`

**Interfaces:**
- Consumes: `src.llm` (JUDGE provider — deliberately not the drafter)
- Produces:
  - `Verdict` dataclass: `grounded: bool`, `helpful: bool`, `on_brand: bool`, `no_overpromise: bool`, `rationale: str`, with `.is_harmful` property and `.axes() -> dict`
  - `build_judge_prompt(customer, reply, cases) -> str`
  - `parse_verdict(raw: str) -> Verdict`
  - `judge_reply(customer, reply, cases, use_cache=True) -> Verdict`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_judge.py
from src.eval import judge
from src.agent.retrieve import Case

CASES = [Case("flight cancelled", "We can rebook you free of charge.", 0.9)]

GOOD = ('{"grounded": true, "helpful": true, "on_brand": true, '
        '"no_overpromise": true, "rationale": "matches past handling"}')
BAD = ('{"grounded": false, "helpful": true, "on_brand": true, '
       '"no_overpromise": false, "rationale": "invents a refund"}')


def test_parse_verdict_reads_all_four_axes():
    v = judge.parse_verdict(GOOD)
    assert v.grounded and v.helpful and v.on_brand and v.no_overpromise


def test_parse_verdict_tolerates_markdown_fences():
    assert judge.parse_verdict(f"```json\n{GOOD}\n```").grounded is True


def test_parse_verdict_defaults_to_failing_on_garbage():
    # An unparseable judge response must never be scored as a pass.
    v = judge.parse_verdict("the reply seems fine to me")
    assert not v.grounded and not v.helpful
    assert v.is_harmful


def test_is_harmful_when_ungrounded_or_overpromising():
    assert judge.parse_verdict(BAD).is_harmful
    assert not judge.parse_verdict(GOOD).is_harmful


def test_axes_returns_the_four_booleans():
    a = judge.parse_verdict(GOOD).axes()
    assert set(a) == {"grounded", "helpful", "on_brand", "no_overpromise"}


def test_judge_prompt_includes_reply_message_and_historical_cases():
    p = judge.build_judge_prompt("my flight was cancelled",
                                 "We can rebook you.", CASES)
    assert "my flight was cancelled" in p
    assert "We can rebook you." in p
    assert CASES[0].agent_text in p


def test_judge_reply_uses_the_judge_model_not_the_drafter(monkeypatch):
    seen = {}

    def fake(prompt, **kw):
        seen.update(kw)
        return GOOD

    monkeypatch.setattr(judge.llm, "complete", fake)
    judge.judge_reply("c", "r", CASES)
    assert seen["model"] == judge.config.JUDGE_MODEL
    assert seen["model"] != judge.config.DRAFTER_MODEL
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.judge'`

- [ ] **Step 3: Write `src/eval/judge.py`**

```python
"""LLM-as-judge for reply quality.

Deliberately a DIFFERENT model family from the drafter - a model grading its own
output inflates the score. How well this judge agrees with a human is measured
in src/eval/agreement.py and reported, not assumed.

Unparseable responses default to FAILING every axis, so judge flakiness can only
ever make our numbers look worse, never better.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src import config, llm
from src.agent.retrieve import Case

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)
AXES = ("grounded", "helpful", "on_brand", "no_overpromise")


@dataclass
class Verdict:
    grounded: bool
    helpful: bool
    on_brand: bool
    no_overpromise: bool
    rationale: str = ""

    @property
    def is_harmful(self) -> bool:
        """A reply is harmful if it is ungrounded or overpromises."""
        return (not self.grounded) or (not self.no_overpromise)

    def axes(self) -> dict:
        return {a: bool(getattr(self, a)) for a in AXES}


def build_judge_prompt(customer: str, reply: str, cases: list[Case]) -> str:
    hist = "\n".join(f"- {c.agent_text}" for c in cases)
    return (
        "You are auditing a draft reply from an automated British Airways support "
        "agent. Judge it strictly.\n\n"
        f"Customer message:\n\"\"\"{customer}\"\"\"\n\n"
        f"Draft reply:\n\"\"\"{reply}\"\"\"\n\n"
        f"How British Airways actually handled similar messages:\n{hist}\n\n"
        "Score four axes as true/false:\n"
        "- grounded: consistent with how BA actually handled similar cases above.\n"
        "- helpful: advances the customer's problem rather than stalling.\n"
        "- on_brand: matches BA's tone and length (short, polite, specific).\n"
        "- no_overpromise: invents NO commitment (refund, compensation, upgrade, "
        "rebooking) that the historical replies do not support.\n\n"
        'Reply with JSON only: {"grounded": bool, "helpful": bool, '
        '"on_brand": bool, "no_overpromise": bool, "rationale": "one sentence"}'
    )


def parse_verdict(raw: str) -> Verdict:
    cleaned = _FENCE.sub("", raw.strip())
    m = re.search(r"\{.*\}", cleaned, re.S)
    if not m:
        return Verdict(False, False, False, False, "unparseable judge response")
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Verdict(False, False, False, False, "unparseable judge response")
    return Verdict(
        grounded=bool(d.get("grounded", False)),
        helpful=bool(d.get("helpful", False)),
        on_brand=bool(d.get("on_brand", False)),
        no_overpromise=bool(d.get("no_overpromise", False)),
        rationale=str(d.get("rationale", "")),
    )


def judge_reply(customer: str, reply: str, cases: list[Case],
                use_cache: bool = True) -> Verdict:
    raw = llm.complete(
        build_judge_prompt(customer, reply, cases),
        provider=config.JUDGE_PROVIDER, model=config.JUDGE_MODEL,
        max_tokens=200, use_cache=use_cache,
    )
    return parse_verdict(raw)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_judge.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/eval/judge.py tests/test_judge.py
git commit -m "feat: LLM-as-judge on a different model family from the drafter"
```

---

### Task 13: Judge–human agreement

**Files:**
- Create: `src/eval/agreement.py`, `tests/test_agreement.py`

**Interfaces:**
- Consumes: `src.eval.judge`
- Produces:
  - `cohens_kappa(a: Sequence, b: Sequence) -> float`
  - `interpret_kappa(k: float) -> str`
  - `agreement_report(human: dict[str, list[bool]], judge: dict[str, list[bool]]) -> pd.DataFrame` with columns `axis, kappa, raw_agreement, interpretation, n`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agreement.py
import pytest
from src.eval import agreement


def test_kappa_is_one_for_perfect_agreement():
    assert agreement.cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_kappa_is_zero_for_chance_level_agreement():
    a = [1, 1, 0, 0]
    b = [1, 0, 1, 0]
    assert agreement.cohens_kappa(a, b) == pytest.approx(0.0, abs=1e-9)


def test_kappa_is_negative_for_systematic_disagreement():
    assert agreement.cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) < 0


def test_kappa_handles_a_degenerate_constant_rater():
    # Both raters say "true" for everything: agreement is total but uninformative.
    k = agreement.cohens_kappa([1, 1, 1, 1], [1, 1, 1, 1])
    assert k == 0.0 or k == pytest.approx(1.0)


def test_interpret_kappa_uses_landis_koch_bands():
    assert "poor" in agreement.interpret_kappa(-0.1).lower()
    assert "slight" in agreement.interpret_kappa(0.1).lower()
    assert "moderate" in agreement.interpret_kappa(0.5).lower()
    assert "substantial" in agreement.interpret_kappa(0.7).lower()
    assert "almost perfect" in agreement.interpret_kappa(0.9).lower()


def test_agreement_report_covers_every_axis():
    human = {"grounded": [1, 0, 1, 1], "helpful": [1, 1, 0, 0]}
    judged = {"grounded": [1, 0, 1, 0], "helpful": [1, 1, 0, 1]}
    r = agreement.agreement_report(human, judged)
    assert set(r.axis) == {"grounded", "helpful"}
    assert set(["axis", "kappa", "raw_agreement", "interpretation", "n"]).issubset(r.columns)
    assert (r.n == 4).all()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_agreement.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.agreement'`

- [ ] **Step 3: Write `src/eval/agreement.py`**

```python
"""Validate the validator.

An LLM judge that disagrees with the human is not a measurement instrument. We
report Cohen's kappa per axis, honestly, and every judge-derived number in the
report inherits that uncertainty.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def cohens_kappa(a: Sequence, b: Sequence) -> float:
    a = np.asarray(list(a), dtype=int)
    b = np.asarray(list(b), dtype=int)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("raters must be the same non-zero length")

    po = float((a == b).mean())
    labels = sorted(set(a.tolist()) | set(b.tolist()))
    pe = sum(float((a == l).mean()) * float((b == l).mean()) for l in labels)

    if np.isclose(pe, 1.0):
        return 0.0 if np.isclose(po, 1.0) else float("nan")
    return float((po - pe) / (1 - pe))


def interpret_kappa(k: float) -> str:
    if k != k:            # NaN
        return "undefined (degenerate)"
    if k < 0:
        return "poor (worse than chance)"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def agreement_report(human: dict[str, list], judge: dict[str, list]) -> pd.DataFrame:
    rows = []
    for axis in human:
        h, j = list(human[axis]), list(judge[axis])
        k = cohens_kappa(h, j)
        rows.append(dict(
            axis=axis,
            kappa=round(k, 3),
            raw_agreement=round(float(np.mean(np.asarray(h) == np.asarray(j))), 3),
            interpretation=interpret_kappa(k),
            n=len(h),
        ))
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_agreement.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/eval/agreement.py tests/test_agreement.py
git commit -m "feat: Cohen's kappa judge-human agreement reporting"
```

---

### Task 14: Risk–coverage, calibration, and threshold selection

This produces the headline number.

**Files:**
- Create: `src/eval/risk_coverage.py`, `tests/test_risk_coverage.py`

**Interfaces:**
- Consumes: numpy/pandas/matplotlib
- Produces:
  - `risk_coverage_curve(scores, harms, forced_escalate) -> pd.DataFrame` with columns `threshold, coverage, harm_rate, n_auto`
  - `expected_cost(coverage, harm_rate, k) -> float`
  - `pick_threshold(curve, k) -> dict` with keys `threshold, coverage, harm_rate, expected_cost`
  - `sensitivity(curve, ks) -> pd.DataFrame`
  - `reliability(scores, correct, bins=10) -> pd.DataFrame`, `ece(scores, correct, bins=10) -> float`
  - `plot_risk_coverage(curve, path)`, `plot_reliability(rel, path)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_risk_coverage.py
import numpy as np
import pytest
from src.eval import risk_coverage as rc


def test_coverage_falls_as_the_threshold_rises():
    scores = np.linspace(0, 1, 100)
    harms = np.zeros(100, dtype=bool)
    curve = rc.risk_coverage_curve(scores, harms)
    assert curve.coverage.is_monotonic_decreasing


def test_harm_rate_falls_as_threshold_rises_when_score_is_informative():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 500)
    harms = rng.uniform(0, 1, 500) > scores      # low score => likely harmful
    curve = rc.risk_coverage_curve(scores, harms)
    lo = curve[curve.threshold <= 0.2].harm_rate.mean()
    hi = curve[curve.threshold >= 0.8].harm_rate.mean()
    assert hi < lo


def test_forced_escalations_are_excluded_from_coverage():
    scores = np.array([0.9, 0.9, 0.9, 0.9])
    harms = np.array([False] * 4)
    forced = np.array([True, True, False, False])
    curve = rc.risk_coverage_curve(scores, harms, forced_escalate=forced)
    assert curve.coverage.max() <= 0.5 + 1e-9


def test_expected_cost_weights_harm_by_k():
    cheap = rc.expected_cost(coverage=0.5, harm_rate=0.1, k=1)
    dear = rc.expected_cost(coverage=0.5, harm_rate=0.1, k=50)
    assert dear > cheap


def test_pick_threshold_gets_stricter_as_harm_gets_more_expensive():
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, 800)
    harms = rng.uniform(0, 1, 800) > scores
    curve = rc.risk_coverage_curve(scores, harms)
    assert rc.pick_threshold(curve, k=50)["threshold"] >= \
           rc.pick_threshold(curve, k=2)["threshold"]


def test_sensitivity_returns_one_row_per_k():
    rng = np.random.default_rng(2)
    scores = rng.uniform(0, 1, 300)
    curve = rc.risk_coverage_curve(scores, rng.uniform(0, 1, 300) > scores)
    out = rc.sensitivity(curve, ks=[5, 10, 20, 50])
    assert len(out) == 4
    assert set(["k", "threshold", "coverage", "harm_rate"]).issubset(out.columns)


def test_ece_is_zero_for_a_perfectly_calibrated_predictor():
    scores = np.array([0.0] * 50 + [1.0] * 50)
    correct = np.array([False] * 50 + [True] * 50)
    assert rc.ece(scores, correct, bins=2) == pytest.approx(0.0, abs=1e-9)


def test_ece_is_large_for_a_confidently_wrong_predictor():
    scores = np.full(100, 0.99)
    correct = np.zeros(100, dtype=bool)
    assert rc.ece(scores, correct, bins=5) > 0.9


def test_reliability_bins_have_the_expected_columns():
    rel = rc.reliability(np.linspace(0, 1, 100),
                         np.linspace(0, 1, 100) > 0.5, bins=5)
    assert set(["bin_lo", "bin_hi", "mean_score", "accuracy", "n"]).issubset(rel.columns)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_risk_coverage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.risk_coverage'`

- [ ] **Step 3: Write `src/eval/risk_coverage.py`**

```python
"""Turn accuracy into a deployment decision.

The headline claim of this project is NOT "the classifier is X% accurate". It is
"at threshold T the agent safely auto-handles X% of volume at Y% harm rate".
This module produces that number, and shows how it moves with the cost model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def risk_coverage_curve(scores, harms, forced_escalate=None,
                        n_points: int = 101) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=float)
    harms = np.asarray(harms, dtype=bool)
    forced = (np.zeros(len(scores), dtype=bool) if forced_escalate is None
              else np.asarray(forced_escalate, dtype=bool))

    rows = []
    for t in np.linspace(0.0, 1.0, n_points):
        auto = (scores >= t) & ~forced
        n_auto = int(auto.sum())
        rows.append(dict(
            threshold=float(t),
            coverage=n_auto / len(scores),
            harm_rate=float(harms[auto].mean()) if n_auto else 0.0,
            n_auto=n_auto,
        ))
    return pd.DataFrame(rows)


def expected_cost(coverage: float, harm_rate: float, k: float) -> float:
    """Cost per incoming message, in units of one human escalation.

    k = cost(bad auto-reply) / cost(unnecessary escalation). Its true value is
    unknown, which is exactly why it is a parameter and the report shows a
    sensitivity sweep instead of one convenient number.
    """
    return (1.0 - coverage) * 1.0 + coverage * harm_rate * k


def pick_threshold(curve: pd.DataFrame, k: float) -> dict:
    c = curve.copy()
    c["expected_cost"] = [expected_cost(r.coverage, r.harm_rate, k)
                          for r in c.itertuples()]
    best = c.loc[c.expected_cost.idxmin()]
    return dict(threshold=float(best.threshold), coverage=float(best.coverage),
                harm_rate=float(best.harm_rate),
                expected_cost=float(best.expected_cost))


def sensitivity(curve: pd.DataFrame, ks=(2, 5, 10, 20, 50)) -> pd.DataFrame:
    return pd.DataFrame([{"k": k, **pick_threshold(curve, k)} for k in ks])


def reliability(scores, correct, bins: int = 10) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= 1.0)
        rows.append(dict(
            bin_lo=float(lo), bin_hi=float(hi),
            mean_score=float(scores[m].mean()) if m.any() else float("nan"),
            accuracy=float(correct[m].mean()) if m.any() else float("nan"),
            n=int(m.sum()),
        ))
    return pd.DataFrame(rows)


def ece(scores, correct, bins: int = 10) -> float:
    """Expected calibration error - how far self-reported confidence is from truth."""
    rel = reliability(scores, correct, bins)
    rel = rel[rel.n > 0]
    total = rel.n.sum()
    if total == 0:
        return 0.0
    return float((rel.n / total * (rel.mean_score - rel.accuracy).abs()).sum())


def plot_risk_coverage(curve: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve.coverage, curve.harm_rate, marker=".", lw=1)
    ax.set_xlabel("coverage (share of volume auto-handled)")
    ax.set_ylabel("harm rate among auto-handled")
    ax.set_title("Risk-coverage: what can we safely automate?")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(rel: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = rel[rel.n > 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", label="perfect calibration")
    ax.plot(r.mean_score, r.accuracy, marker="o", label="observed")
    ax.set_xlabel("mean self-reported confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Reliability diagram")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_risk_coverage.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/eval/risk_coverage.py tests/test_risk_coverage.py
git commit -m "feat: risk-coverage curve, calibration, cost-based threshold selection"
```

---

### Task 15: Pipeline runner, CLI, and reproduction path

**Files:**
- Create: `src/pipeline.py`, `src/cli.py`, `tests/test_pipeline.py`
- Create: `reports/results.json` (output)

**Interfaces:**
- Consumes: everything above
- Produces:
  - `src.pipeline.run_agent(golden: pd.DataFrame, threshold: float) -> pd.DataFrame` with columns `customer_tweet_id, pred_intent, confidence, top_similarity, clf_margin, score, action, reason, rule_fired, reply`
  - `src.pipeline.run_all(k: float) -> dict` writing `reports/results.json`
  - `src.cli` typer app with commands `ingest`, `taxonomy`, `pool`, `run`, `evaluate`, `reproduce`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import pandas as pd
from src import pipeline


def test_run_agent_produces_one_row_per_input_with_all_columns(monkeypatch):
    from src.agent import classify as C, decide as D, draft as F
    from src.agent.retrieve import Case

    golden = pd.DataFrame(dict(
        customer_tweet_id=[1, 2],
        customer_text=["my bag is lost and never arrived", "what is the baggage allowance?"],
        agent_text=["a", "b"],
        intent=["baggage", "general_enquiry"],
        action=["escalate", "auto"],
    ))

    class FakeRetriever:
        def search(self, text, k=5):
            return [Case("past", "BA replied something", 0.8)]

    monkeypatch.setattr(pipeline, "_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(C, "classify", lambda t, **k: C.Classification("baggage", 0.8))
    monkeypatch.setattr(F, "draft_reply", lambda *a, **k: "drafted reply")
    monkeypatch.setattr(pipeline, "_clf_margin", lambda texts: [0.5, 0.5])

    out = pipeline.run_agent(golden, threshold=0.5)
    assert len(out) == 2
    for col in ["pred_intent", "confidence", "score", "action", "reason", "reply"]:
        assert col in out.columns


def test_run_agent_escalates_the_lost_baggage_row_via_hard_rule(monkeypatch):
    from src.agent import classify as C, draft as F
    from src.agent.retrieve import Case

    golden = pd.DataFrame(dict(
        customer_tweet_id=[1],
        customer_text=["my suitcase is lost, it never arrived"],
        agent_text=["a"], intent=["baggage"], action=["escalate"],
    ))

    class FakeRetriever:
        def search(self, text, k=5):
            return [Case("p", "r", 0.99)]

    monkeypatch.setattr(pipeline, "_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(C, "classify", lambda t, **k: C.Classification("baggage", 1.0))
    monkeypatch.setattr(F, "draft_reply", lambda *a, **k: "reply")
    monkeypatch.setattr(pipeline, "_clf_margin", lambda texts: [1.0])

    out = pipeline.run_agent(golden, threshold=0.0)
    assert out.iloc[0].action == "escalate"
    assert out.iloc[0].rule_fired == "lost_baggage"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.pipeline'`

- [ ] **Step 3: Write `src/pipeline.py`**

```python
"""Wire the stages together and produce reports/results.json."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import baselines, config, taxonomy
from src.agent import classify as C
from src.agent import decide as D
from src.agent import draft as F
from src.agent.retrieve import Retriever
from src.eval import agreement, judge, metrics
from src.eval import risk_coverage as rc

_RETRIEVER = None
_SIMPLE = None


def _retriever() -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever.from_corpus_split()
    return _RETRIEVER


def _simple_model() -> baselines.SimpleBaseline:
    """Trained on corpus-split messages labelled by the LLM classifier's cached
    predictions - the golden set is never used for training."""
    global _SIMPLE
    if _SIMPLE is None:
        df = pd.read_parquet(config.INTERIM_DIR / "ba_pairs.parquet")
        corpus = df[df.split == "corpus"].sample(2000, random_state=config.SEED)
        labels = [C.classify(t).intent for t in corpus.customer_text.astype(str)]
        m = baselines.SimpleBaseline().fit(corpus.customer_text.astype(str).tolist(), labels)
        m.fit_replies(corpus.customer_text.astype(str).tolist(),
                      corpus.agent_text.astype(str).tolist())
        _SIMPLE = m
    return _SIMPLE


def _clf_margin(texts: list[str]) -> list[float]:
    p = _simple_model().predict_proba(texts)
    s = np.sort(p, axis=1)
    return list(s[:, -1] - s[:, -2]) if p.shape[1] > 1 else [1.0] * len(texts)


def run_agent(golden: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    r = _retriever()
    texts = golden.customer_text.astype(str).tolist()
    margins = _clf_margin(texts)

    rows = []
    for (_, row), margin in zip(golden.iterrows(), margins):
        text = str(row.customer_text)
        cls = C.classify(text)
        cases = r.search(text, k=5)
        top_sim = cases[0].similarity if cases else 0.0
        reply = F.draft_reply(text, cls.intent, cases)
        d = D.decide(text, cls.intent, cls.confidence, top_sim, float(margin), threshold)
        rows.append(dict(
            customer_tweet_id=int(row.customer_tweet_id),
            pred_intent=cls.intent, confidence=cls.confidence,
            top_similarity=top_sim, clf_margin=float(margin),
            score=d.score, action=d.action, reason=d.reason,
            rule_fired=d.rule_fired, reply=reply,
        ))
    return pd.DataFrame(rows)


def run_all(k: float = 20.0) -> dict:
    golden = pd.read_json(config.GOLDEN_DIR / "golden.jsonl", lines=True)
    golden = golden[golden["round"] == 1].reset_index(drop=True)

    preds = run_agent(golden, threshold=0.5)
    merged = golden.merge(preds, on="customer_tweet_id")

    # --- intent: ours vs two baselines --------------------------------------
    simple = _simple_model()
    trivial = baselines.TrivialBaseline().fit(
        golden.customer_text.tolist(), golden.intent.tolist())

    y = merged.intent.tolist()
    results = {
        "intent": {
            "agent": metrics.summary(y, merged.pred_intent.tolist()),
            "simple": metrics.summary(y, simple.predict(merged.customer_text.tolist())),
            "trivial": metrics.summary(y, trivial.predict(merged.customer_text.tolist())),
        }
    }

    # --- reply quality via judge --------------------------------------------
    r = _retriever()
    verdicts = [judge.judge_reply(str(t), str(rep), r.search(str(t), k=3))
                for t, rep in zip(merged.customer_text, merged.reply)]
    harms = np.array([v.is_harmful for v in verdicts])
    for axis in judge.AXES:
        results.setdefault("reply", {})[axis] = float(
            np.mean([getattr(v, axis) for v in verdicts]))
    results["reply"]["harm_rate_all"] = float(harms.mean())

    # --- deployment decision -------------------------------------------------
    forced = merged.rule_fired.notna().to_numpy()
    curve = rc.risk_coverage_curve(merged.score.to_numpy(), harms, forced)
    chosen = rc.pick_threshold(curve, k)
    results["deployment"] = {
        "k": k, **chosen,
        "sensitivity": rc.sensitivity(curve).to_dict(orient="records"),
    }

    # --- calibration ---------------------------------------------------------
    correct = (merged.pred_intent == merged.intent).to_numpy()
    rel = rc.reliability(merged.confidence.to_numpy(), correct)
    results["calibration"] = {
        "ece_llm_confidence": rc.ece(merged.confidence.to_numpy(), correct),
        "ece_routing_score": rc.ece(merged.score.to_numpy(), correct),
    }

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rc.plot_risk_coverage(curve, config.FIGURES_DIR / "risk_coverage.png")
    rc.plot_reliability(rel, config.FIGURES_DIR / "reliability.png")

    # --- escalation decision vs human ---------------------------------------
    results["escalation"] = metrics.summary(
        merged.action_x.tolist() if "action_x" in merged else merged.action.tolist(),
        merged.action_y.tolist() if "action_y" in merged else merged.action.tolist())

    out = config.PROJECT_ROOT / "reports" / "results.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    merged.to_json(config.PROJECT_ROOT / "reports" / "predictions.jsonl",
                   orient="records", lines=True, force_ascii=False)
    return results
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: 2 passed

- [ ] **Step 5: Write `src/cli.py`**

```python
"""Single entry point. `python -m src.cli reproduce` regenerates headline numbers."""
from __future__ import annotations

import json

import typer
from rich.console import Console

from src import config, llm, pipeline, sampling, taxonomy
from src import ingest as ing

app = typer.Typer(add_completion=False, help="BA support agent pipeline")
con = Console()


@app.command()
def ingest():
    """Build data/interim/ba_pairs.parquet from the raw CSV."""
    con.print(f"wrote {ing.run_ingest()}")


@app.command()
def taxonomy_propose(clusters: int = 12):
    """Cluster corpus messages and dump data/interim/clusters.json for naming."""
    taxonomy.propose(n_clusters=clusters)
    con.print("wrote data/interim/clusters.json - now hand-write taxonomy.json")


@app.command()
def pool(n: int = config.GOLDEN_TARGET):
    """Build the stratified golden pool to label."""
    p = sampling.build_golden_pool(n)
    con.print(f"pooled {len(p)} examples")


@app.command()
def evaluate(k: float = 20.0):
    """Run the agent over the golden set and write reports/results.json."""
    res = pipeline.run_all(k=k)
    con.print_json(json.dumps(res, default=float))


@app.command()
def reproduce():
    """Reproduce headline results from the committed cache. No API key needed."""
    stats = llm.cache_stats()
    con.print(f"[dim]cache entries: {stats['entries']}[/dim]")
    res = pipeline.run_all(k=20.0)
    d = res["deployment"]
    i = res["intent"]["agent"]
    con.print("\n[bold]HEADLINE[/bold]")
    con.print(f"  intent accuracy : {i['accuracy']:.3f} "
              f"[{i['accuracy_ci'][0]:.3f}, {i['accuracy_ci'][1]:.3f}]")
    con.print(f"  macro-F1        : {i['macro_f1']:.3f}")
    con.print(f"  auto-handled    : {d['coverage']:.1%} of volume")
    con.print(f"  harm rate       : {d['harm_rate']:.1%} among auto-handled")
    con.print(f"  at threshold    : {d['threshold']:.2f} (cost ratio k={d['k']})")


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Run the whole pipeline end to end**

```bash
PYTHONIOENCODING=utf-8 python -m src.cli evaluate --k 20
```

Expected: `reports/results.json` and both figures written. This is the run that POPULATES the cache — it will make live API calls and take a while under free-tier limits.

- [ ] **Step 7: Verify reproduction works with no API key**

```bash
PYTHONIOENCODING=utf-8 env -u GEMINI_API_KEY -u GROQ_API_KEY \
  python -m src.cli reproduce
```

Expected: headline numbers print, **no `NoAPIKeyError`**. If it raises, the cache is incomplete — rerun step 6 until every call is cached. This is the reviewer's exact experience; it must work.

- [ ] **Step 8: Time the reproduction**

```bash
time PYTHONIOENCODING=utf-8 python -m src.cli reproduce
```

Expected: well under 15 minutes. Record the actual number for the README.

- [ ] **Step 9: Run the full test suite**

Run: `python -m pytest -v`
Expected: all tests pass (~70 tests).

- [ ] **Step 10: Commit**

```bash
git add src/pipeline.py src/cli.py tests/test_pipeline.py reports/ cache/llm_cache.sqlite
git commit -m "feat: pipeline runner, CLI, and no-API-key reproduction path"
```

---

### Task 16: Failure analysis, report, and decision log

**Files:**
- Create: `src/failure_analysis.py`, `reports/REPORT.md`, `reports/DECISIONS.md`, `README.md`

**Interfaces:**
- Consumes: `reports/predictions.jsonl`, `reports/results.json`
- Produces: `src.failure_analysis.top_failures(n: int = 5) -> pd.DataFrame`

- [ ] **Step 1: Write `src/failure_analysis.py`**

```python
"""Surface real failing examples for the report. No hand-picking."""
from __future__ import annotations

import pandas as pd

from src import config


def load() -> pd.DataFrame:
    return pd.read_json(config.PROJECT_ROOT / "reports" / "predictions.jsonl", lines=True)


def misclassifications(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.pred_intent != df.intent]


def confusion_pairs(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    m = misclassifications(df)
    pairs = (m.groupby(["intent", "pred_intent"]).size()
             .reset_index(name="count").sort_values("count", ascending=False))
    return pairs.head(n)


def escalation_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    """Where the agent auto-handled something the human said to escalate.
    These are the expensive errors."""
    if "action_x" in df.columns:
        human, agent = df.action_x, df.action_y
    else:
        human = agent = df.action
    return df[(human == "escalate") & (agent == "auto")]


def top_failures(n: int = 5) -> pd.DataFrame:
    df = load()
    pairs = confusion_pairs(df, n)
    rows = []
    for _, p in pairs.iterrows():
        ex = df[(df.intent == p.intent) & (df.pred_intent == p.pred_intent)].head(2)
        for _, e in ex.iterrows():
            rows.append(dict(true=p.intent, pred=p.pred_intent, count=p["count"],
                             text=str(e.customer_text)[:200], reply=str(e.reply)[:200]))
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Generate the failure material**

```bash
PYTHONIOENCODING=utf-8 python -c "
from src import failure_analysis as fa
df = fa.load()
print('=== TOP CONFUSION PAIRS ==='); print(fa.confusion_pairs(df).to_string())
print(); print('=== DANGEROUS: auto-handled but human said escalate ===')
print(fa.escalation_disagreements(df)[['customer_text','reply','reason']].head(10).to_string())
print(); print('=== EXAMPLES ==='); print(fa.top_failures().to_string())
"
```

Read the output. **Write hypotheses for each of the top 5 failure modes yourself** — the brief asks for hypotheses, not just counts, and you will be asked about them live.

- [ ] **Step 3: Write `reports/REPORT.md`**

Sections, in this order, filling every number from `reports/results.json`:

1. **Problem framing** — what "good" means for BA (short, specific, no invented commitments); what you chose *not* to build (multi-turn, tool use, fine-tuning, multilingual) and why.
2. **Data** — brand selection table with the measured deflection/link/support-density numbers; the multi-part reassembly problem and how you solved it; the temporal split.
3. **Taxonomy** — the intents, what you merged during naming and why.
4. **Results vs. baselines** — table of trivial / simple / agent with accuracy, macro-F1, **and CIs**. State plainly whether the agent beats the simple baseline and whether the gap exceeds the interval.
5. **The headline** — "at threshold T we auto-handle X% at Y% harm rate", with the risk–coverage figure and the k-sensitivity table.
6. **Judge validation** — per-axis kappa table and what it means for every judge-derived number above.
7. **Failure analysis** — top 5 modes, each with a real example and your hypothesis.
8. **What is misleading about my headline number?** — the seven points from spec Section 10, each stated plainly.
9. **What I'd do with one more week.**

- [ ] **Step 4: Write `reports/DECISIONS.md`**

At least 12 entries, each one line of decision plus one line of why. Start from these (all genuinely non-obvious, all made during this build):

1. Chose British Airways over the intuitive AppleSupport — measured 52% DM-deflection and 75% link-dumping in Apple's replies, which makes "grounded reply" vacuous.
2. Rejected AmazonHelp despite 4× the volume — 8% Japanese and only 22% support-dense.
3. Split temporally (Oct = corpus, Nov/Dec = eval) rather than randomly — a random split lets the retriever find the very thread being evaluated.
4. Reassembled 31% multi-part BA replies — otherwise a third of ground-truth replies are truncated fragments.
5. Stripped `^Jane` signatures into a separate field — style, not content; leaking them teaches the agent to impersonate a named human.
6. Committed the SQLite LLM cache to git — the only way to hit "reproduce in 15 minutes" with no API key on a free tier.
7. Used a different model family for judging than for drafting — a model grading its own output inflates the score.
8. Ran embeddings locally rather than via API — removes rate limits and makes retrieval exactly reproducible.
9. Made the harm/escalation cost ratio `k` a CLI parameter with a sensitivity sweep — the true ratio is unknown and picking one silently would be dishonest.
10. Layered hard rules *before* the confidence gate — refunds and legal threats must never be automated no matter how confident the model is.
11. Unparseable judge output defaults to failing every axis — flakiness should only ever make our numbers look worse.
12. Stratified the golden set with a floor per intent plus a deliberate hard stratum — random sampling would give ~2 examples of the tail intents.
13. Labelled the golden set before seeing any model output — prevents anchoring.
14. Double-labelled 25 examples days apart — intra-annotator agreement is the honest ceiling on any accuracy claim.
15. Reported bootstrap CIs on every headline number — at n=200 a 3-point gap is noise.

- [ ] **Step 5: Write `README.md`**

Must contain, in this order: one-paragraph what-this-is; **Quickstart** (`pip install -e .` then `python -m src.cli reproduce`, stating the measured wall-clock time and that no API key is needed); the headline result; repo layout; how to regenerate from scratch with keys; a **Borrowed** section citing the Kaggle dataset, sentence-transformers/all-MiniLM-L6-v2, scikit-learn, and any code patterns you took from elsewhere; and a link to `reports/REPORT.md`.

- [ ] **Step 6: Verify the reviewer's path from a clean clone**

```bash
cd /tmp && rm -rf repro-check
git clone "A:/WebDev/Hiver SDE intern" repro-check
cd repro-check && python -m pip install -e . -q
PYTHONIOENCODING=utf-8 env -u GEMINI_API_KEY -u GROQ_API_KEY \
  timeout 900 python -m src.cli reproduce
```

Expected: headline numbers print within 15 minutes with no key. **If this fails, the submission fails** — fix before submitting.

- [ ] **Step 7: Commit**

```bash
git add README.md reports/ src/failure_analysis.py
git commit -m "docs: report, decision log, README with reproduction instructions"
```

---

## Self-Review

**Spec coverage:** every spec section maps to a task — framing/non-goals → Task 16 report; brand selection → already decided, recorded in Task 16 decision log; architecture → Task 1; committed cache → Task 2; data pipeline incl. multi-part and temporal split → Task 3; taxonomy → Task 4; agent stages → Tasks 7–10; hard rules + confidence gate → Task 10; baselines → Task 6; golden set protocol → Task 5; metrics/judge/agreement/risk-coverage → Tasks 11–14; cost model + sensitivity → Task 14; "what is misleading" → Task 16 step 3 section 8; deliverable mapping → Task 15 (repro) and Task 16 (docs).

**Known gaps to watch during execution:**

- `run_all` merges golden and predictions, which both carry an `action` column; pandas will suffix them `action_x` (human) and `action_y` (agent). The code handles both shapes, but **verify the suffixes on the first real run** and simplify once confirmed.
- The `escalation` summary in `run_all` compares human vs. agent action; confirm the merge produced two distinct columns before trusting that number.
- `_simple_model` labels its training data with the LLM classifier, so the simple baseline inherits the LLM's biases. That is a deliberate trade-off (no other labels exist at corpus scale) and **must be disclosed in the report** — it makes the baseline comparison friendlier to the agent than a clean comparison would be.
