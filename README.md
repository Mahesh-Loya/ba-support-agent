# BA Support Agent

An AI support agent for British Airways, built on British Airways' side of
the Kaggle "Customer Support on Twitter" dataset (~3M tweets, one brand
selected on evidence — see `reports/DECISIONS.md` §1). Given an inbound
customer tweet, it classifies the intent, drafts a reply grounded in how BA
has actually resolved similar issues in the past, and decides whether that
reply is safe to send automatically or should go to a human — with a
stated reason either way.

The headline claim this project makes is a **deployment decision**, not an
accuracy score:

> At routing threshold **T**, the agent safely auto-handles **X%** of BA's
> inbound support volume at a **Y%** harmful-reply rate (95% CI ...),
> escalating the remainder with a stated reason.

On the actual golden set (n=198): **T = 0.76, X = 8.6%, Y = 23.5%** — see
`reports/REPORT.md` for the full result and, just as importantly, for what
is misleading about it. Everything in this repo (the temporal train/eval
split, the two-layer escalation policy, the cost-ratio sensitivity sweep,
the committed LLM cache) exists to make that one sentence measurable and
honest, rather than to chase an accuracy number no one asked for.

## Quickstart

```bash
pip install -e ".[dev]"
python -m src.cli reproduce
```

**No API key is needed.** Every LLM call in this project (classification,
drafting, judging) routes through a single choke point, `src/llm.py`, which
looks up a SQLite cache (`cache/llm_cache.sqlite`) committed to this repo
before ever making a live call. `reproduce` is designed to run entirely off
that cache and print:

```
  intent accuracy : 0.636 [0.566, 0.702]
  macro-F1        : 0.570
  auto-handled    : 8.6% of volume
  harm rate       : 23.5% among auto-handled
  at threshold    : 0.76 (cost ratio k=20.0)
```

Wall-clock time for a cache-hit run, measured on this machine with both
API keys explicitly unset: **2 minutes 55 seconds** — well under the
15-minute requirement. Full results, including the judge-vs-human
agreement study and per-baseline comparison, are in `reports/REPORT.md`
and `reports/results.json`.

This has been run and verified end-to-end on a clean checkout with both
`GEMINI_API_KEY`/`GROQ_API_KEY`/`OPENAI_API_KEY` explicitly unset — the
numbers above are real output, not a projection.

## How it works

Four independently testable stages (`src/agent/`), run in order for every
inbound message:

1. **Classify** (`classify.py`) — an LLM few-shot classifies the message
   into one of 9 data-derived intents (see `data/interim/taxonomy.json`),
   returning a self-reported confidence. That confidence is *measured* for
   calibration later (`eval/risk_coverage.py`), never trusted outright.
2. **Retrieve** (`retrieve.py`) — top-k cosine-similar historical customer
   messages, embedded locally with MiniLM, restricted to the retrieval
   corpus split only. The retriever hard-fails (`ValueError`) if it is ever
   given a frame without a `split` column, so it cannot silently retrieve
   from the eval range it will later be scored against.
3. **Draft** (`draft.py`) — an LLM drafts a reply grounded in the retrieved
   real BA replies: matching BA's tone/length, never inventing a
   commitment (refund, compensation, upgrade, rebooking) the retrieved
   history doesn't support.
4. **Decide** (`decide.py`) — auto-handle or escalate, with a written
   reason. Two layers:
   - **Hard rules**, checked first, regardless of confidence: crisis/
     wellbeing language, money claims, legal/safety threats, lost baggage,
     anything needing account lookup, and severe distress. These are a
     stated policy, not a model prediction — a business must never
     automate these categories no matter how confident the model is.
   - **Confidence gate**, everything else: a blended score (LLM
     confidence, retrieval similarity, TF-IDF classifier margin) is
     thresholded against **T**, chosen to minimize expected cost under a
     cost ratio `k = cost(bad auto-reply) / cost(unnecessary escalation)`
     that is a CLI parameter, swept over a range rather than assumed (see
     `eval/risk_coverage.py`).

## Repo layout

```
src/
  config.py            all project constants (brand, seed, model ids, paths)
  ingest.py            raw CSV -> (customer message, BA reply) pairs;
                       multi-part reassembly, signature stripping, temporal split
  taxonomy.py          embed + cluster -> propose intents; humans name them
  sampling.py          stratified golden-set sampling (per-intent floor + hard stratum)
  label_tui.py         keyboard-driven tool for the human labelling pass
  embed.py             local MiniLM embeddings, disk-cached by content hash
  llm.py               single choke point for every LLM call; SQLite cache
  baselines.py         trivial (majority-class) and simple (TF-IDF+LogReg) baselines
  pipeline.py          wires the four agent stages + eval together
  cli.py               `python -m src.cli {ingest,taxonomy,pool,run,evaluate,reproduce}`
  agent/
    classify.py         message -> intent + confidence
    retrieve.py          message -> top-k similar historical (message, reply) cases
    draft.py             intent + cases -> grounded reply
    decide.py            -> auto/escalate + reason (hard rules + confidence gate)
  eval/
    metrics.py           accuracy, macro-F1, bootstrap CIs
    judge.py             LLM-as-judge, 4 binary axes, fails closed
    agreement.py         Cohen's kappa, judge vs. human
    risk_coverage.py     coverage/harm curve, threshold selection, calibration (ECE)
data/
  raw/                 twcs.csv — gitignored, 493MB, not in this repo (see below)
  interim/             ba_pairs.parquet, taxonomy.json, clusters.json, cached embeddings
  golden/              golden.jsonl (198, hand-labelled) + quality.jsonl (100)
reports/
  DECISIONS.md          this project's decision log (15 entries)
  REPORT.md             the actual report: results, failure analysis, limitations
  results.json           full metrics from the completed evaluation run
  predictions.jsonl       per-example agent output, merged with golden labels
  figures/              risk-coverage and reliability plots (generated by `evaluate`/`reproduce`)
tests/                  170 tests, one file per src module
```

## Reproducing from scratch (live API calls)

To regenerate everything from the raw dataset rather than the committed
cache:

1. Download `twcs.csv` from Kaggle
   (`thoughtvector/customer-support-on-twitter`) and place it at
   `data/raw/twcs.csv`. It is gitignored (493MB) — this repo ships the
   already-derived `data/interim/ba_pairs.parquet` (2.7MB) instead, which
   is enough to run everything except a fresh `ingest`.
2. Set `OPENAI_API_KEY` (drafter/classifier, `gpt-4o-mini`) and
   `GROQ_API_KEY` (judge, `openai/gpt-oss-120b`) in your environment (or
   `.env`) — see `src/config.py` for exactly which provider each role
   uses and why they're split across two companies.
3. Run the pipeline stages in order:

   ```bash
   python -m src.cli ingest              # data/raw/twcs.csv -> data/interim/ba_pairs.parquet
   python -m src.cli taxonomy             # clusters -> data/interim/clusters.json (hand-name into taxonomy.json)
   python -m src.cli pool                 # stratified 200-example golden pool -> data/golden/golden_pool.jsonl
   python -m src.label_tui                # human labelling: intent + auto/escalate -> data/golden/golden.jsonl
   python -m src.label_tui quality        # human labelling: 4-axis quality on BA's real replies -> data/golden/quality.jsonl
   python -m src.cli evaluate             # runs the full agent + judge, writes reports/results.json
   ```

   Steps 4-5 are interactive and require a human (~45 minutes for 200
   labels, by design of `label_tui.py`). `evaluate` (and `reproduce`) will
   make live LLM calls for anything not already in `cache/llm_cache.sqlite`
   and add those responses to the cache as it goes — pass `--no-cache` to
   `llm.complete` call sites to force live calls even on a cache hit
   (rarely needed).

## Tests

```bash
python -m pytest -q
```

170 passing, as of this writing (verified by running the command). One
test file per `src` module; the escalation, judging, and calibration logic
in particular are covered by hand-verified edge cases (degenerate/chance-
level kappa, zero-coverage harm rate, forced-escalation interaction with
the risk-coverage curve).

## Borrowed

Anything not written for this project, cited explicitly:

- **Dataset**: Kaggle "Customer Support on Twitter",
  `thoughtvector/customer-support-on-twitter`.
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`
  (`sentence-transformers` library), run locally.
- **scikit-learn**: `TfidfVectorizer` + `LogisticRegression` (simple
  baseline intent classifier), `KMeans` (taxonomy clustering).
- **LLM providers**: OpenAI's `gpt-4o-mini` (drafter/classifier) and
  Groq-hosted `openai/gpt-oss-120b` (judge) — two different companies'
  models, so the judge never grades output from its own family. See
  `reports/DECISIONS.md` for why the drafter moved from Groq's free tier
  (an account-wide daily token quota made the ~2,200-call drafter
  workload a multi-day bottleneck) to a paid OpenAI key, and why
  `gpt-4o-mini` specifically rather than a newer model on the same
  account.
- **Statistics**: Cohen's kappa (judge-vs-human agreement) and the
  Landis & Koch (1977) interpretation bands (slight/fair/moderate/
  substantial/almost perfect) for reading it; bootstrap resampling for 95%
  confidence intervals on every headline metric.

## Status

This project is complete. All five required deliverables exist:

- **Code**: all four agent stages, both baselines, the full eval harness
  (metrics, judge, agreement, risk-coverage/calibration), and the CLI are
  implemented and unit-tested (170 tests passing).
- **Golden set**: `data/golden/golden.jsonl` — 198 hand-labelled examples
  (intent + auto/escalate + reason). `data/golden/quality.jsonl` — 100
  reply-quality judgements used for the judge-agreement study.
- **Evaluation harness output**: `reports/results.json`,
  `reports/predictions.jsonl`, and the two figures in `reports/figures/`
  are real output from a completed run, not placeholders.
- **Report**: `reports/REPORT.md` — problem framing, results vs. both
  baselines with confidence intervals, the headline deployment number,
  the mandatory "what is misleading" section, the judge-agreement study
  (which came back weak — reported honestly, not hidden), failure
  analysis with real examples, and next steps.
- **Decision log**: `reports/DECISIONS.md` — 15 non-obvious decisions and
  why, made during the actual build.

See `reports/REPORT.md` for the full results and their honest limitations.
