"""Keyboard-driven labelling tool. 200 examples in ~45 minutes.

Two modes, selected by the first CLI argument:

  intent (default) - shows ONLY the customer message while labelling intent
    and action - BA's actual reply stays hidden so it cannot anchor your
    judgement. Resumable: rerun and it picks up where you stopped. Writes
    data/golden/golden.jsonl.

  quality - shows the customer message AND BA's actual historical reply, and
    collects four binary (y/n) judgements on that real reply: grounded,
    helpful, on_brand, no_overpromise. This is the deliberate exception to
    the "hide the reply" rule - here judging the real reply IS the task.
    Writes data/golden/quality.jsonl. Target ~100 examples; resumable like
    intent mode.

Usage:
    python -m src.label_tui                       # intent mode, round 1, default pool
    python -m src.label_tui intent 2 golden_pool_round2.jsonl
    python -m src.label_tui quality                # quality mode, round 1
    python -m src.label_tui quality 1 golden_pool_round2.jsonl
"""
from __future__ import annotations

import json
import sys

import pandas as pd
from rich.console import Console
from rich.panel import Panel

from src import config, taxonomy

GOLDEN = config.GOLDEN_DIR / "golden.jsonl"
QUALITY = config.GOLDEN_DIR / "quality.jsonl"
DEFAULT_POOL = config.GOLDEN_DIR / "golden_pool.jsonl"

ESCALATION_REASONS = {
    "1": "needs_account_lookup",
    "2": "money_claim",
    "3": "legal_or_safety",
    "4": "low_confidence_ambiguous",
    "5": "severe_distress",
    "6": "multi_issue",
}

QUALITY_AXES = ["grounded", "helpful", "on_brand", "no_overpromise"]
QUALITY_PROMPTS = {
    "grounded": "grounded (consistent with how BA handles such cases)?",
    "helpful": "helpful (advances the customer's problem, not stalling)?",
    "on_brand": "on_brand (BA tone/length: short, polite, specific)?",
    "no_overpromise": "no_overpromise (doesn't commit to refunds/comp/upgrades it shouldn't)?",
}


def _load_done(path) -> dict:
    if not path.exists():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            done[r["customer_tweet_id"]] = r
    return done


def _append(path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(round_id: int = 1, pool_path=DEFAULT_POOL) -> None:
    """Intent/action labelling mode. Customer message ONLY - agent_text is
    hidden from the labeller while making this judgement."""
    con = Console()
    intents = taxonomy.load_intents()
    pool = pd.read_json(pool_path, lines=True)
    done = _load_done(GOLDEN)

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

        _append(GOLDEN, dict(
            customer_tweet_id=int(row.customer_tweet_id),
            customer_text=str(row.customer_text),
            agent_text=str(row.agent_text),
            intent=intent, action=action,
            escalation_reason=reason, round=round_id,
        ))
        con.print("[green]saved[/green]\n")


def main_quality(round_id: int = 1, pool_path=DEFAULT_POOL, target: int = 100) -> None:
    """Quality mode (Ruling A). Shows the customer message AND BA's actual
    historical reply, and collects four y/n judgements on that reply.

    Writes data/golden/quality.jsonl. Resumable exactly like intent mode:
    re-reads the file on start and skips ids already done. Does not require
    all `target` examples to be useful - quit any time with 'q'.
    """
    con = Console()
    pool = pd.read_json(pool_path, lines=True)
    done = _load_done(QUALITY)

    remaining_target = max(target - len(done), 0)
    todo = [r for _, r in pool.iterrows()
            if r.customer_tweet_id not in done or round_id != 1]
    if round_id == 1:
        todo = todo[:remaining_target] if remaining_target else todo
    con.print(f"[bold]{len(done)} quality-labelled, {len(todo)} queued "
              f"(target {target})[/bold]\n")

    for n, row in enumerate(todo, 1):
        con.print(Panel(str(row.customer_text),
                        title=f"[{n}/{len(todo)}] customer message"))
        con.print(Panel(str(row.agent_text), title="BA's actual reply",
                        border_style="yellow"))

        raw = input("proceed? [enter]=yes, 's'=skip, 'q'=quit > ").strip().lower()
        if raw == "q":
            break
        if raw == "s":
            continue

        answers = {}
        for axis in QUALITY_AXES:
            while True:
                a = input(f"  {axis} - {QUALITY_PROMPTS[axis]} (y/n) > ").strip().lower()
                if a in ("y", "n"):
                    answers[axis] = (a == "y")
                    break
                con.print("  [red]please press y or n[/red]")

        _append(QUALITY, dict(
            customer_tweet_id=int(row.customer_tweet_id),
            customer_text=str(row.customer_text),
            agent_text=str(row.agent_text),
            **answers,
            round=round_id,
        ))
        con.print("[green]saved[/green]\n")


def _parse_args(argv: list[str]):
    """mode [round_id] [pool_path] - mode defaults to 'intent'."""
    mode = "intent"
    rest = argv
    if argv and argv[0] in ("intent", "quality"):
        mode = argv[0]
        rest = argv[1:]

    round_id = int(rest[0]) if len(rest) > 0 else 1
    pool_path = config.GOLDEN_DIR / rest[1] if len(rest) > 1 else DEFAULT_POOL
    return mode, round_id, pool_path


if __name__ == "__main__":
    _mode, _round_id, _pool_path = _parse_args(sys.argv[1:])
    if _mode == "quality":
        main_quality(round_id=_round_id, pool_path=_pool_path)
    else:
        main(round_id=_round_id, pool_path=_pool_path)
