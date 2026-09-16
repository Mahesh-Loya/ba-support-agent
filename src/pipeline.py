"""Wire the stages together and produce reports/results.json."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src import baselines, config, llm
from src.agent import classify as C
from src.agent import decide as D
from src.agent import draft as F
from src.agent.retrieve import Retriever
from src.eval import agreement, judge, metrics
from src.eval import risk_coverage as rc

_RETRIEVER = None
_SIMPLE = None

# --- Ruling S: resumable, rate-limit tolerant API calls ---------------------
# ~2,400 calls against a free tier, unattended, will hit transient rate-limit
# / timeout errors. Every successful call is already cached by src/llm.py, so
# retrying here never re-pays for a call that already landed - it only
# protects the (uncached) call that just failed.
_MAX_RETRIES = 5
_BASE_DELAY_S = 2.0
_MAX_DELAY_S = 30.0
_PROGRESS_EVERY = 25

# --- Ruling P: bounded concurrency for network-bound LLM calls --------------
# The measured bottleneck is round-trip latency (and rate-limit backoff), not
# model compute (0.45-1.1s per call directly, ~3.7s observed serially) - so
# threads are the right tool, not multiprocessing (no CPU-bound work here,
# and multiprocessing would only add pickling overhead for I/O-bound calls).
# Default of 8 is conservative for a free-tier provider; tune down via the
# `max_workers` parameter (or by editing this constant) if the provider
# starts rate-limiting harder under concurrency than it does serially.
MAX_WORKERS = 8


def _with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) with capped exponential backoff on transient
    provider errors. NoAPIKeyError means the committed cache is incomplete and
    there is no key to fall back on - that is not transient, so it is raised
    immediately instead of being retried."""
    delay = _BASE_DELAY_S
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except llm.NoAPIKeyError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider errors are opaque here
            last_exc = exc
            if attempt == _MAX_RETRIES:
                break
            print(f"  [retry {attempt}/{_MAX_RETRIES}] {type(exc).__name__}: {exc} "
                  f"- backing off {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, _MAX_DELAY_S)
    raise last_exc


def _progress(i: int, n: int, label: str) -> None:
    if i % _PROGRESS_EVERY == 0 or i == n:
        print(f"{label}: {i}/{n} processed")


def _parallel_map(fn, items, *, max_workers: int = MAX_WORKERS, label: str = ""):
    """Map fn over items with bounded concurrency, preserving input order.

    `executor.map` returns results in the order its inputs were given,
    regardless of which worker happens to finish first - completion order
    never leaks into the output - so no manual index bookkeeping is needed
    to stay ordered. Downstream code pairs these results positionally against
    golden rows, so that ordering guarantee is load-bearing.

    Progress is a lock-protected counter incremented as each call *completes*
    (so it may tick slightly out of input order under concurrency - that's
    fine, it is only a heartbeat so a long unattended run never goes silent
    for minutes).

    Any exception raised inside a worker propagates out of `list(...)` below
    instead of being swallowed - a silently-eaten worker exception would mean
    a missing/garbage row silently corrupting every downstream metric.
    """
    items = list(items)
    n = len(items)
    count = 0
    lock = threading.Lock()

    def _call(x):
        nonlocal count
        result = fn(x)
        with lock:
            count += 1
            _progress(count, n, label)
        return result

    if max_workers <= 1 or n <= 1:
        return [_call(x) for x in items]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_call, items))


def _retriever() -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever.from_corpus_split()
    return _RETRIEVER


def _simple_model() -> baselines.SimpleBaseline:
    """Trained on corpus-split messages labelled by the LLM classifier's cached
    predictions - the golden set is never used for training.

    Deliberately still trained on the full 2,000-message sample: shrinking it
    would speed this up but would weaken the TF-IDF baseline relative to the
    LLM agent, which is the dishonest direction. Speed comes only from running
    those 2,000 classify calls concurrently, never from sampling fewer of
    them."""
    global _SIMPLE
    if _SIMPLE is None:
        df = pd.read_parquet(config.INTERIM_DIR / "ba_pairs.parquet")
        corpus = df[df.split == "corpus"].sample(2000, random_state=config.SEED)
        texts = corpus.customer_text.astype(str).tolist()
        classifications = _parallel_map(
            lambda t: _with_retry(C.classify, t), texts,
            label="baseline training labels")
        labels = [c.intent for c in classifications]
        m = baselines.SimpleBaseline().fit(texts, labels)
        m.fit_replies(texts, corpus.agent_text.astype(str).tolist())
        _SIMPLE = m
    return _SIMPLE


def _clf_margin(texts: list[str]) -> list[float]:
    p = _simple_model().predict_proba(texts)
    s = np.sort(p, axis=1)
    return list(s[:, -1] - s[:, -2]) if p.shape[1] > 1 else [1.0] * len(texts)


def run_agent(golden: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    r = _retriever()
    texts = golden.customer_text.astype(str).tolist()

    # Phase 1: classify every message concurrently.
    classifications = _parallel_map(
        lambda t: _with_retry(C.classify, t), texts, label="run_agent classify")

    # Retrieval is local (embeddings + numpy), not a network call, so it is
    # not a bottleneck and is left serial for simplicity.
    cases_list = [r.search(t, k=5) for t in texts]
    top_sims = [cases[0].similarity if cases else 0.0 for cases in cases_list]

    # Phase 2: draft every reply concurrently. Each draft depends on its own
    # message's classification + retrieved cases (computed above), not on
    # any other row, so this is safe to run in a second parallel pass indexed
    # by position - order is preserved throughout.
    replies = _parallel_map(
        lambda i: _with_retry(F.draft_reply, texts[i], classifications[i].intent,
                               cases_list[i]),
        range(len(texts)), label="run_agent draft")

    # Ruling O: compute clf_margin LAST, only after every golden-set classify
    # and draft call has already landed (and been cached by src/llm.py).
    # `_clf_margin` trains the 2,000-call TF-IDF baseline on first use - doing
    # this FIRST (the original bug) meant that entire baseline-training step
    # ran before a single one of the 200 golden-set examples was classified
    # or drafted, so a quota cutoff during it meant zero golden-set calls
    # ever landed (or got cached) for the day. Moving it here means the
    # golden-set's own (comparatively cheap, ~400-call) classify+draft work
    # is done - and cached - before the expensive baseline step even starts.
    margins = _clf_margin(texts)

    rows = []
    for i, (_, row) in enumerate(golden.iterrows()):
        cls = classifications[i]
        d = D.decide(texts[i], cls.intent, cls.confidence, top_sims[i],
                     float(margins[i]), threshold)
        rows.append(dict(
            customer_tweet_id=int(row.customer_tweet_id),
            pred_intent=cls.intent, confidence=cls.confidence,
            top_similarity=top_sims[i], clf_margin=float(margins[i]),
            score=d.score, action=d.action, reason=d.reason,
            rule_fired=d.rule_fired, reply=replies[i],
        ))
    return pd.DataFrame(rows)


def _human_agent_actions(merged: pd.DataFrame) -> tuple[list, list]:
    """Isolate human (golden) vs. agent (predicted) action after the
    golden/preds merge on customer_tweet_id.

    Both frames carry an `action` column, so pandas suffixes the colliding
    one `action_x` (from the left frame, golden/human) and `action_y` (from
    the right frame, preds/agent). The escalation metric must compare those
    two different columns against each other - never a column against
    itself, which would silently report perfect agreement."""
    if "action_x" in merged.columns and "action_y" in merged.columns:
        return merged.action_x.tolist(), merged.action_y.tolist()
    return merged.action.tolist(), merged.action.tolist()


def _judge_agreement() -> list[dict] | None:
    """Ruling A: measure how well the LLM judge agrees with a human on BA's
    actual historical replies.

    Reads data/golden/quality.jsonl, written by
    `python -m src.label_tui quality` (customer_tweet_id, customer_text,
    agent_text, and four human booleans: grounded, helpful, on_brand,
    no_overpromise). Judges BA's actual reply (agent_text), not the agent's
    draft, so the comparison is judge-vs-human on the same artifact the
    human scored. Returns None (and does not crash) if that file does not
    exist yet - the human labelling step is a separate, optional deliverable
    that must not block the rest of the pipeline."""
    path = config.GOLDEN_DIR / "quality.jsonl"
    if not path.exists():
        return None

    quality = pd.read_json(path, lines=True)
    if quality.empty:
        return None

    r = _retriever()
    texts = quality.customer_text.astype(str).tolist()
    agent_texts = quality.agent_text.astype(str).tolist()
    cases_list = [r.search(t, k=5) for t in texts]

    verdicts = _parallel_map(
        lambda i: _with_retry(judge.judge_reply, texts[i], agent_texts[i], cases_list[i]),
        range(len(texts)), label="judge agreement")

    human = {axis: [] for axis in judge.AXES}
    judged = {axis: [] for axis in judge.AXES}
    for i, row in enumerate(quality.itertuples()):
        for axis in judge.AXES:
            human[axis].append(int(bool(getattr(row, axis))))
            judged[axis].append(int(bool(getattr(verdicts[i], axis))))

    report = agreement.agreement_report(human, judged)
    return report.to_dict(orient="records")


def _write_results(results: dict, merged: pd.DataFrame) -> None:
    """Durably persist the current state of `results` + `merged`.

    Called twice by `run_all`: once as an intermediate checkpoint right after
    the golden-set-only sections are computed (before baseline training even
    starts), and once more at the very end with the complete dict. Both calls
    write the exact same two files, so a checkpoint write and the final write
    are indistinguishable in shape - only the content differs depending on
    how far the run got."""
    out_dir = config.PROJECT_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8")
    merged.to_json(out_dir / "predictions.jsonl",
                   orient="records", lines=True, force_ascii=False)


def run_all(k: float = 20.0) -> dict:
    """Ruling O: golden-set path runs to completion (and is durably
    checkpointed) BEFORE the 2,000-call baseline-training step starts.

    `_simple_model()` alone makes 2,000 LLM calls to label its TF-IDF
    training corpus - on a free-tier quota that is an entire day's budget by
    itself. If that step ran first (as it once did, transitively, via
    `run_agent` -> `_clf_margin` -> `_simple_model`) and the quota cut out
    partway through it, the run would end with zero golden-set numbers to
    show for the whole day despite having spent the whole day. Reordering so
    the golden-set sections run and get written to disk first means a
    baseline-training failure or interruption still leaves a complete,
    real set of agent numbers on disk - only the baseline comparison fields
    are missing, never the whole file."""
    golden = pd.read_json(config.GOLDEN_DIR / "golden.jsonl", lines=True)
    golden = golden[golden["round"] == 1].reset_index(drop=True)

    # --- golden-set agent run (order-critical: first, and cheap relative to
    # baseline training - one classify + one draft call per golden row) -----
    preds = run_agent(golden, threshold=0.5)
    merged = golden.merge(preds, on="customer_tweet_id")

    y = merged.intent.tolist()
    results: dict = {
        "intent": {
            "agent": metrics.summary(y, merged.pred_intent.tolist()),
        },
        "_baseline_complete": False,
        "_judge_agreement_complete": False,
    }

    # --- reply quality via judge --------------------------------------------
    r = _retriever()
    texts = merged.customer_text.astype(str).tolist()
    replies = merged.reply.astype(str).tolist()
    cases_list = [r.search(t, k=3) for t in texts]
    verdicts = _parallel_map(
        lambda i: _with_retry(judge.judge_reply, texts[i], replies[i], cases_list[i]),
        range(len(texts)), label="judge (reply quality)")
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
    human_action, agent_action = _human_agent_actions(merged)
    results["escalation"] = metrics.summary(human_action, agent_action)

    # --- CHECKPOINT: everything above needs only `merged` - never the ------
    # baseline models. Write results.json + predictions.jsonl NOW, before
    # `_simple_model()` is touched at all, so a quota cutoff or crash during
    # baseline training still leaves real, complete golden-set numbers on
    # disk (intent.agent, reply, deployment, calibration, escalation) with
    # `_baseline_complete: False` marking the file as a checkpoint rather
    # than a full run.
    _write_results(results, merged)

    # --- intent: ours vs two baselines (the expensive, 2,000-call step) -----
    simple = _simple_model()
    trivial = baselines.TrivialBaseline().fit(
        golden.customer_text.tolist(), golden.intent.tolist())
    results["intent"]["simple"] = metrics.summary(
        y, simple.predict(merged.customer_text.tolist()))
    results["intent"]["trivial"] = metrics.summary(
        y, trivial.predict(merged.customer_text.tolist()))
    results["_baseline_complete"] = True

    # --- CHECKPOINT #2: the 2,000 real baseline-training calls just landed --
    # and cost real quota. Persist them NOW, before `_judge_agreement()` runs
    # - it goes through `_with_retry` against a real network stage and can
    # raise after exhausting retries. Without this write, a judge-agreement
    # failure would propagate out of run_all before the final write below,
    # and the baseline results that were just genuinely paid for would never
    # reach disk - the exact failure mode this whole fix exists to prevent,
    # just moved one stage later.
    _write_results(results, merged)

    # --- judge-vs-human agreement (Ruling A) ---------------------------------
    # `results` is always built fresh in memory here (run_all never reads
    # back a prior checkpoint to resume from), so `_judge_agreement_complete`
    # is always False at this point and this guard is always taken. It is
    # kept only as a documented no-op placeholder for a future
    # resume-from-checkpoint path - it is not real resume logic today.
    if not results["_judge_agreement_complete"]:
        judge_agreement = _judge_agreement()
        results["judge_agreement"] = judge_agreement
        results["_judge_agreement_complete"] = judge_agreement is not None

    # --- FINAL write: golden-set sections + baseline sections + judge ------
    # agreement, all present.
    _write_results(results, merged)
    return results
