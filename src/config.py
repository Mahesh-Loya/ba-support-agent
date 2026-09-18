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
# grades its own output - now doubly true, since they are on different
# COMPANIES' models, not just different models on one provider.
#
# History: both originally lived on Groq's free tier (qwen for drafter,
# gpt-oss-120b for judge; qwen/qwen3.6-27b was tried first and rejected -
# it emits <think> blocks that break JSON parsing). Groq's free tier caps
# at 200,000 tokens/day account-wide, which made the ~2,200-call drafter
# workload (baseline training + golden-set classify/draft) a multi-day
# bottleneck. The drafter moved to a paid OpenAI key (200,000 tokens/MINUTE,
# no observed daily cap) to remove that bottleneck; gpt-4o-mini was chosen
# over anything newer on the account (gpt-5.x, gpt-6-astra) because those
# postdate this assistant's training and an unfamiliar model's behaviour
# should never be gambled on a 2,000-call unattended run - gpt-4o-mini is
# well-understood and was verified live before use. The judge stays on Groq
# (gpt-oss-120b): it never hit a rate limit all session, so there was no
# reason to move it, and keeping it on a separate provider from the drafter
# is strictly better for the never-self-grade guarantee than moving both.
DRAFTER_MODEL = "gpt-4o-mini"
DRAFTER_PROVIDER = "openai"
JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_PROVIDER = "groq"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Eval ---------------------------------------------------------------
GOLDEN_TARGET = 200
GOLDEN_FLOOR = 150
BOOTSTRAP_N = 10_000
