"""Draft a reply grounded in how BA actually handled similar messages."""
from __future__ import annotations

import re

from src import config, llm
from src.agent.retrieve import Case

_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.M)
_SIG = re.compile(r"\s*[\^*][A-Za-z]{2,12}\s*$")
_LABEL = re.compile(r"^(?:reply|response|draft)\s*:\s*", re.I)


def build_draft_prompt(text: str, intent: str, cases: list[Case]) -> str:
    examples = "\n\n".join(
        f"Past customer: {c.customer_text}\nBritish Airways replied: {c.agent_text}"
        for c in cases
    )
    return (
        "You are a British Airways customer support agent replying on Twitter.\n"
        f"The customer's intent has been classified as: {intent}\n\n"
        "Here is how British Airways handled similar messages in the past:\n\n"
        f"{examples}\n\n"
        "Rules:\n"
        "- Match the tone and length of the past replies (usually 1-2 short sentences).\n"
        "- Ground your reply in what BA actually did above.\n"
        "- NEVER promise compensation, refunds, upgrades or rebooking unless the "
        "past replies show BA committing to exactly that.\n"
        "- Do not invent booking references, amounts, dates or policies.\n"
        "- Do not sign the message with a name.\n"
        "- Do not include a greeting like 'Hi there' unless the examples do.\n\n"
        f"New customer message:\n\"\"\"{text}\"\"\"\n\n"
        "Write only the reply text, nothing else."
    )


def clean_reply(raw: str) -> str:
    out = _FENCE.sub("", raw.strip()).strip()
    out = _LABEL.sub("", out).strip()
    out = _SIG.sub("", out).strip()
    if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'":
        out = out[1:-1].strip()
    out = _SIG.sub("", out).strip()
    return out


def draft_reply(text: str, intent: str, cases: list[Case],
                use_cache: bool = True) -> str:
    raw = llm.complete(
        build_draft_prompt(text, intent, cases),
        provider=config.DRAFTER_PROVIDER, model=config.DRAFTER_MODEL,
        max_tokens=200, use_cache=use_cache,
    )
    return clean_reply(raw)
