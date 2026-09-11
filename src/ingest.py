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
_PART = re.compile(r"(?:^\s*\(?[1-9]/[1-9]\)?\.?\s*)|(?:\s*\(?[1-9]/[1-9]\)?\.?\s*$)")
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


def _strip_trailing_tokens(text: str) -> tuple[str, str | None]:
    """Peel trailing signature and part-marker tokens off `text`, in
    whichever order they appear (agents write both "^Jane 2/2" and
    "2/2. ^Jane"), until neither pattern matches any more.
    """
    sig = None
    while True:
        no_sig, s = strip_signature(text)
        if no_sig != text:
            text = no_sig
            if s is not None:
                sig = s
            continue
        no_marker = strip_part_marker(text)
        if no_marker != text:
            text = no_marker
            continue
        break
    return text, sig


def _keep_longest_reply(out: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per customer_tweet_id, keeping the longest
    agent_text.

    Ordinarily every agent reply chain rooted at the same customer tweet is
    already merged (concatenated) by the groupby in build_pairs, so this is
    normally a no-op safety net. But if a future data fix, dtype change, or
    edge case (e.g. very large tweet ids losing float precision) ever causes
    two distinct rows to share a customer_tweet_id, an arbitrary
    drop_duplicates pick would silently discard the richer reply. Keeping the
    longest agent_text instead ensures ground truth reflects the most
    informative reply BA actually sent, rather than whichever row happened
    to sort first.
    """
    return (out.assign(_len=out.agent_text.str.len())
               .sort_values("_len", ascending=False)
               .drop_duplicates("customer_tweet_id")
               .drop(columns="_len")
               .sort_values("customer_tweet_id")
               .reset_index(drop=True))


def build_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """First-contact pairs: a customer's opening message and BA's full reply.

    Multi-part replies are NOT a simple linked list in this dataset: most
    continuation tweets reply directly to the same parent (customer) tweet
    as the opening part, rather than to the previous part's tweet_id (a few
    do use that pattern too). So every agent tweet's `in_response_to_tweet_id`
    is walked upward through any intermediate agent tweets to find its
    ultimate root parent, and all agent tweets sharing a root are grouped as
    one reply, ordered chronologically by `created_at`.
    """
    df = df.copy()
    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"], errors="coerce")
    by_id = df.set_index("tweet_id", drop=False)
    if not by_id.index.is_unique:
        by_id = by_id[~by_id.index.duplicated(keep="first")]

    agent = df[(df.author_id == brand) & (df.inbound == False)].copy()  # noqa: E712
    agent_ids = set(agent.tweet_id)

    def root_of(parent_id):
        seen = set()
        while pd.notna(parent_id) and parent_id in agent_ids and parent_id not in seen:
            seen.add(parent_id)
            if parent_id not in by_id.index:
                return float("nan")
            parent_id = by_id.loc[parent_id].in_response_to_tweet_id
        return parent_id

    agent["root_id"] = [root_of(p) for p in agent.in_response_to_tweet_id]
    agent["_ts"] = pd.to_datetime(
        agent.created_at, format=_TWITTER_TS, errors="coerce", utc=True)

    rows = []
    for root_id, group in agent[agent.root_id.notna()].groupby("root_id"):
        if root_id not in by_id.index:
            continue
        parent = by_id.loc[root_id]
        if parent.inbound != True:  # noqa: E712
            continue
        if pd.notna(parent.in_response_to_tweet_id):
            continue  # not a first-contact message

        ordered = group.sort_values("_ts")
        parts = []
        sig = None
        for t in ordered.text:
            cleaned, s = _strip_trailing_tokens(_clean(t))
            parts.append(cleaned)
            if s is not None:
                sig = s
        body = reassemble(parts)
        rows.append(dict(
            customer_tweet_id=int(root_id),
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
    return _keep_longest_reply(out)


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
