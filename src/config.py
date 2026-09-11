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
# Drafter and judge are deliberately different families so the judge never
# grades its own output. Both live on the Groq free tier (the only key this
# project has). qwen/qwen3.6-27b was tried and rejected: it emits <think>
# blocks that break JSON parsing. The stronger model (gpt-oss-120b) is
# deliberately the judge, not the drafter.
DRAFTER_MODEL = "qwen/qwen3.8-27b"
DRAFTER_PROVIDER = "groq"
JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_PROVIDER = "groq"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Eval ---------------------------------------------------------------
GOLDEN_TARGET = 200
GOLDEN_FLOOR = 150
BOOTSTRAP_N = 10_000
