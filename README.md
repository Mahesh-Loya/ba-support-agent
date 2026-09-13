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

`X`, `Y`, and `T` are not yet known — see [Status](#status). Everything in
this repo (the temporal train/eval split, the two-layer escalation policy,
the cost-ratio sensitivity sweep, the committed LLM cache) exists to make
that one sentence measurable and honest, rather than to chase an accuracy
number no one asked for.

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
  intent accuracy : <TBD after full run> [<TBD>, <TBD>]
  macro-F1        : <TBD after full run>
  auto-handled    : <TBD after full run> of volume
  harm rate       : <TBD after full run> among auto-handled
  at threshold    : <TBD after full run> (cost ratio k=20.0)
```

Wall-clock time for a cache-hit run: **<TBD after full run>** (expected to
be well under 15 minutes — the entire point of the committed cache is that
this step does no network I/O).

**Current caveat:** the human-labelled golden set
(`data/golden/golden.jsonl`) and the populated LLM cache do not exist in
this repo yet — see [Status](#status). Until both exist, `reproduce` cannot
produce output (it reads `golden.jsonl` first). The command above is the
intended, tested reproduction path; it has not yet been run end-to-end
against real API calls because that step is blocked on human labelling.

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
  golden/              hand-labelled golden set (not yet populated — see Status)
reports/
  DECISIONS.md          this project's decision log (15 entries)
  REPORT.md             not yet written — needs real numbers (see Status)
  figures/              risk-coverage and reliability plots (generated by `evaluate`/`reproduce`)
tests/                  149 tests, one file per src module
```

## Reproducing from scratch (live API calls)

To regenerate everything from the raw dataset rather than the committed
cache:

1. Download `twcs.csv` from Kaggle
   (`thoughtvector/customer-support-on-twitter`) and place it at
   `data/raw/twcs.csv`. It is gitignored (493MB) — this repo ships the
   already-derived `data/interim/ba_pairs.parquet` (2.7MB) instead, which
   is enough to run everything except a fresh `ingest`.
2. Set `GROQ_API_KEY` in your environment (or `.env`) — this project uses
   Groq-hosted models for both the drafter/classifier and the judge (see
   `src/config.py`).
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

149 passing, as of this writing (verified by running the command). One
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
- **LLM providers**: Groq-hosted `qwen/qwen3.8-27b` (drafter/classifier)
  and `openai/gpt-oss-120b` (judge) — see `reports/DECISIONS.md` §7 for why
  these two and not the design spec's original Gemini/Llama pairing (the
  only API key available exposes neither model family).
- **Statistics**: Cohen's kappa (judge-vs-human agreement) and the
  Landis & Koch (1977) interpretation bands (slight/fair/moderate/
  substantial/almost perfect) for reading it; bootstrap resampling for 95%
  confidence intervals on every headline metric.

## Status

Honest state of this repo as of this writing:

- **Code**: all four agent stages, both baselines, the full eval harness
  (metrics, judge, agreement, risk-coverage/calibration), and the CLI are
  implemented and unit-tested (149 tests passing).
- **Not done**: the human's 200 golden-set labels
  (`data/golden/golden.jsonl`) and ~100 reply-quality labels
  (`data/golden/quality.jsonl`) have not been collected yet — only the
  unlabelled 200-example stratified pool (`data/golden/golden_pool.jsonl`)
  exists. Labelling is explicitly a human task (the brief requires it) and
  is next.
- **Not done**: the full evaluation run (~2,400 LLM calls: classification
  + drafting + judging over the golden set, plus judge-agreement scoring on
  the quality subset) has not happened, so `cache/llm_cache.sqlite` and
  `reports/results.json` do not exist yet, and every number in this README
  and in `reports/DECISIONS.md` that depends on them is marked
  `<TBD after full run>` rather than invented.
- **Not done**: `reports/REPORT.md` (problem framing, baseline comparison
  with CIs, the headline risk-coverage result, judge-validation kappa
  table, failure analysis, and the "what is misleading about my headline
  number" section) is not yet written — it needs the numbers above to
  exist first.

`reports/DECISIONS.md` is complete and does not depend on the run above;
it documents *how* the code got here, not what the numbers turned out to
be.
