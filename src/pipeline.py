"""Wire the stages together and produce reports/results.json."""
from __future__ import annotations

import json
import time

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
        texts = corpus.customer_text.astype(str).tolist()
        labels = []
        n = len(texts)
        for i, t in enumerate(texts, 1):
            labels.append(_with_retry(C.classify, t).intent)
            _progress(i, n, "baseline training labels")
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
    margins = _clf_margin(texts)

    rows = []
    n = len(golden)
    for i, ((_, row), margin) in enumerate(zip(golden.iterrows(), margins), 1):
        text = str(row.customer_text)
        cls = _with_retry(C.classify, text)
        cases = r.search(text, k=5)
        top_sim = cases[0].similarity if cases else 0.0
        reply = _with_retry(F.draft_reply, text, cls.intent, cases)
        d = D.decide(text, cls.intent, cls.confidence, top_sim, float(margin), threshold)
        rows.append(dict(
            customer_tweet_id=int(row.customer_tweet_id),
            pred_intent=cls.intent, confidence=cls.confidence,
            top_similarity=top_sim, clf_margin=float(margin),
            score=d.score, action=d.action, reason=d.reason,
            rule_fired=d.rule_fired, reply=reply,
        ))
        _progress(i, n, "run_agent")
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
    human = {axis: [] for axis in judge.AXES}
    judged = {axis: [] for axis in judge.AXES}

    n = len(quality)
    for i, row in enumerate(quality.itertuples(), 1):
        cases = r.search(str(row.customer_text), k=5)
        v = _with_retry(judge.judge_reply, str(row.customer_text),
                        str(row.agent_text), cases)
        for axis in judge.AXES:
            human[axis].append(int(bool(getattr(row, axis))))
            judged[axis].append(int(bool(getattr(v, axis))))
        _progress(i, n, "judge agreement")

    report = agreement.agreement_report(human, judged)
    return report.to_dict(orient="records")


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
    verdicts = []
    n = len(merged)
    for i, (t, rep) in enumerate(zip(merged.customer_text, merged.reply), 1):
        cases = r.search(str(t), k=3)
        verdicts.append(_with_retry(judge.judge_reply, str(t), str(rep), cases))
        _progress(i, n, "judge (reply quality)")
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

    # --- judge-vs-human agreement (Ruling A) ---------------------------------
    results["judge_agreement"] = _judge_agreement()

    out = config.PROJECT_ROOT / "reports" / "results.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    merged.to_json(config.PROJECT_ROOT / "reports" / "predictions.jsonl",
                   orient="records", lines=True, force_ascii=False)
    return results
