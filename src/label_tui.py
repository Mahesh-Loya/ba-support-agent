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

Every keyboard input on the path that produces ground truth is validated and
re-prompted on ambiguous input rather than silently defaulted - a fat-fingered
keystroke must never silently record the wrong label or silently skip one.

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


# --------------------------------------------------------------------------
# Pure parsing helpers - no I/O, no console, unit-testable in isolation.
# Each one either returns an unambiguous result or a sentinel meaning
# "invalid, re-prompt" - none of them silently guess.
# --------------------------------------------------------------------------

def parse_action(raw: str) -> str | None:
    """'a'/'auto' -> 'auto', 'e'/'escalate' -> 'escalate', anything else -> None."""
    r = raw.strip().lower()
    if r in ("a", "auto"):
        return "auto"
    if r in ("e", "escalate"):
        return "escalate"
    return None


def parse_intent_input(raw: str, n_intents: int) -> tuple[str, int | None]:
    """Classify a raw keystroke for the intent prompt.

    Returns (kind, value):
      ("quit", None)   - labeller typed 'q', stop the session
      ("skip", None)   - labeller deliberately typed 's', skip this example
      ("index", i)     - a valid intent index in [0, n_intents)
      ("invalid", None) - anything else (typo, out-of-range, blank, ...);
                          the caller must re-prompt the SAME example, never
                          silently advance.
    """
    r = raw.strip().lower()
    if r == "q":
        return ("quit", None)
    if r == "s":
        return ("skip", None)
    if r.isdigit() and 0 <= int(r) < n_intents:
        return ("index", int(r))
    return ("invalid", None)


def parse_escalation_reason(raw: str) -> str | None:
    """Look up a reason code; None (not 'other') on invalid input so the
    caller re-prompts instead of recording a reason that isn't one of the
    six enumerated ones."""
    return ESCALATION_REASONS.get(raw.strip())


def parse_yn(raw: str) -> bool | None:
    """'y' -> True, 'n' -> False, anything else -> None (re-prompt)."""
    r = raw.strip().lower()
    if r == "y":
        return True
    if r == "n":
        return False
    return None


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def _load_done(path) -> dict:
    """Read already-labelled records keyed by customer_tweet_id.

    Tolerant of a corrupt trailing line (e.g. a Ctrl+C mid-write): a bad line
    is reported with its line number and file, then skipped, so the rest of
    the file - and the labeller's session - is never blocked by one bad row.
    """
    if not path.exists():
        return {}
    done = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            done[r["customer_tweet_id"]] = r
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[label_tui] warning: skipping corrupt line {lineno} in "
                  f"{path} ({exc}). That entry will need re-labelling; "
                  f"the rest of the file loaded fine.")
    return done


def _append(path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Interactive prompt loops - thin wrappers around the pure parsers above,
# re-prompting on anything invalid instead of guessing.
# --------------------------------------------------------------------------

def _prompt_action(con: Console) -> str:
    while True:
        act = input("  [a]uto-handle / [e]scalate > ")
        parsed = parse_action(act)
        if parsed is not None:
            return parsed
        con.print(f"  [red]'{act.strip()}' is not 'a'/'auto' or 'e'/'escalate' - try again[/red]")


def _prompt_escalation_reason(con: Console) -> str:
    con.print("  " + "  ".join(f"[cyan]{k}[/cyan]={v}"
                               for k, v in ESCALATION_REASONS.items()))
    while True:
        raw = input("  reason # > ")
        reason = parse_escalation_reason(raw)
        if reason is not None:
            return reason
        con.print(f"  [red]'{raw.strip()}' is not one of 1-6 - try again[/red]")


def _prompt_yn(con: Console, prompt: str) -> bool:
    while True:
        raw = input(prompt)
        val = parse_yn(raw)
        if val is not None:
            return val
        con.print("  [red]please press y or n[/red]")


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

    saved_count = 0
    try:
        n = 1
        while n <= len(todo):
            row = todo[n - 1]
            con.print(Panel(str(row.customer_text), title=f"[{n}/{len(todo)}] customer message"))
            con.print("  " + "  ".join(f"[cyan]{i}[/cyan]={s}" for i, s in enumerate(intents)))
            raw = input("intent # (or 's' skip, 'q' quit) > ")
            kind, value = parse_intent_input(raw, len(intents))

            if kind == "quit":
                break
            if kind == "invalid":
                con.print(f"  [red]'{raw.strip()}' is not a valid intent number - try again[/red]")
                continue  # re-prompt the SAME example, do not advance n
            n += 1
            if kind == "skip":
                con.print("[yellow]skipped[/yellow]\n")
                continue
            intent = intents[value]

            action = _prompt_action(con)

            reason = ""
            if action == "escalate":
                reason = _prompt_escalation_reason(con)

            _append(GOLDEN, dict(
                customer_tweet_id=int(row.customer_tweet_id),
                customer_text=str(row.customer_text),
                agent_text=str(row.agent_text),
                intent=intent, action=action,
                escalation_reason=reason, round=round_id,
            ))
            saved_count += 1
            con.print("[green]saved[/green]\n")
    except KeyboardInterrupt:
        con.print(f"\n[yellow]Interrupted - {saved_count} label(s) saved this session. "
                  f"Rerun the same command to resume.[/yellow]")


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
        if remaining_target == 0:
            con.print(f"[green]Target of {target} quality labels already reached "
                      f"({len(done)} done) - nothing to do.[/green]")
            return
        todo = todo[:remaining_target]
    con.print(f"[bold]{len(done)} quality-labelled, {len(todo)} queued "
              f"(target {target})[/bold]\n")

    saved_count = 0
    try:
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
                answers[axis] = _prompt_yn(con, f"  {axis} - {QUALITY_PROMPTS[axis]} (y/n) > ")

            _append(QUALITY, dict(
                customer_tweet_id=int(row.customer_tweet_id),
                customer_text=str(row.customer_text),
                agent_text=str(row.agent_text),
                **answers,
                round=round_id,
            ))
            saved_count += 1
            con.print("[green]saved[/green]\n")
    except KeyboardInterrupt:
        con.print(f"\n[yellow]Interrupted - {saved_count} label(s) saved this session. "
                  f"Rerun the same command to resume.[/yellow]")


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
