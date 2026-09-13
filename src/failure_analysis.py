"""Surface real failing examples for the report. No hand-picking.

`reports/predictions.jsonl` is written by `src.pipeline.run_all` as
`golden.merge(preds, on="customer_tweet_id")`. Both frames carry an `action`
column (golden's human-assigned action, preds' agent-decided action), so
pandas suffixes the collision `action_x` (left/golden/human) and `action_y`
(right/preds/agent) - see `src.pipeline._human_agent_actions`. `intent`
(human, from golden) and `pred_intent` (agent, from preds) never collide, so
they keep their bare names regardless of merge suffixing.
"""
from __future__ import annotations

import pandas as pd

from src import config


def load() -> pd.DataFrame:
    """Read the merged golden/prediction records written by run_all."""
    return pd.read_json(config.PROJECT_ROOT / "reports" / "predictions.jsonl", lines=True)


def misclassifications(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where the agent's predicted intent differs from the human label."""
    return df[df.pred_intent != df.intent]


def confusion_pairs(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The most frequent (true, predicted) intent mistake pairs."""
    m = misclassifications(df)
    pairs = (m.groupby(["intent", "pred_intent"]).size()
             .reset_index(name="count").sort_values("count", ascending=False))
    return pairs.head(n).reset_index(drop=True)


def escalation_disagreements(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where the agent auto-handled something the human said to escalate.

    These are the expensive errors and the most important output of this
    module: an unsafe message the agent let through on its own. Direction
    matters - this must be human == "escalate" AND agent == "auto", never
    the reverse (which would report the harmless, over-cautious error class
    instead of the dangerous one).
    """
    if "action_x" in df.columns and "action_y" in df.columns:
        human, agent = df.action_x, df.action_y
    else:
        human = agent = df.action
    return df[(human == "escalate") & (agent == "auto")]


def top_failures(n: int = 5, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """The top-n confusion pairs, each with up to 2 real example rows.

    `df` may be injected for testing; when omitted, `load()` reads
    `reports/predictions.jsonl` from disk.
    """
    if df is None:
        df = load()
    pairs = confusion_pairs(df, n)
    rows = []
    for _, p in pairs.iterrows():
        ex = df[(df.intent == p.intent) & (df.pred_intent == p.pred_intent)].head(2)
        for _, e in ex.iterrows():
            rows.append(dict(true=p.intent, pred=p.pred_intent, count=p["count"],
                             text=str(e.customer_text)[:200], reply=str(e.reply)[:200]))
    return pd.DataFrame(rows)
