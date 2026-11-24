"""Detect a dataset's schema and normalize rows into one internal shape.

Internal shapes
  sft:        {"messages": [{"role": "...", "content": "..."}, ...]}
  text:       {"text": "..."}                       (continued pretraining)
  preference: {"prompt": str, "chosen": str, "rejected": str}
"""

from __future__ import annotations

from typing import Any

from bitstraining.core.errors import DataError

SCHEMAS = ("messages", "sharegpt", "alpaca", "prompt_completion", "preference", "text")

_ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
             "bot": "assistant", "system": "system", "model": "assistant"}


def detect_schema(rows: list[dict[str, Any]], sample: int = 50) -> str:
    """Return the schema that fits the most sampled rows, or 'unknown'."""
    if not rows:
        return "unknown"
    votes: dict[str, int] = {}
    for r in rows[:sample]:
        s = _row_schema(r)
        votes[s] = votes.get(s, 0) + 1
    best = max(votes, key=lambda k: votes[k])
    return best


def _row_schema(r: dict[str, Any]) -> str:
    keys = set(r)
    if {"chosen", "rejected"} <= keys:
        return "preference"
    if "messages" in keys and isinstance(r["messages"], list):
        return "messages"
    if "conversations" in keys and isinstance(r["conversations"], list):
        return "sharegpt"
    if {"instruction", "output"} <= keys:
        return "alpaca"
    if {"prompt", "completion"} <= keys or {"prompt", "response"} <= keys:
        return "prompt_completion"
    if {"question", "answer"} <= keys:
        return "prompt_completion"
    if "text" in keys:
        return "text"
    return "unknown"


def normalize_row(r: dict[str, Any], schema: str) -> dict[str, Any]:
    """Convert a raw row to the internal shape. Raises DataError on malformed rows."""
    try:
        if schema == "messages":
            msgs = [{"role": m["role"], "content": str(m["content"])} for m in r["messages"]]
            return {"messages": msgs}
        if schema == "sharegpt":
            msgs = []
            for m in r["conversations"]:
                role = _ROLE_MAP.get(str(m.get("from", m.get("role", ""))).lower())
                if role is None:
                    raise DataError(f"unknown role in conversation: {m!r}")
                msgs.append({"role": role, "content": str(m.get("value", m.get("content", "")))})
            return {"messages": msgs}
        if schema == "alpaca":
            instr = str(r["instruction"])
            extra = str(r.get("input") or "")
            user = f"{instr}\n\n{extra}" if extra.strip() else instr
            return {"messages": [{"role": "user", "content": user},
                                 {"role": "assistant", "content": str(r["output"])}]}
        if schema == "prompt_completion":
            prompt = str(r.get("prompt", r.get("question", "")))
            comp = str(r.get("completion", r.get("response", r.get("answer", ""))))
            return {"messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": comp}]}
        if schema == "preference":
            return {"prompt": _as_text(r.get("prompt", r.get("question", ""))),
                    "chosen": _as_text(r["chosen"]), "rejected": _as_text(r["rejected"])}
        if schema == "text":
            return {"text": str(r["text"])}
    except (KeyError, TypeError, AttributeError) as e:
        raise DataError(f"row does not match schema '{schema}': missing/invalid {e}") from e
    raise DataError(
        f"cannot interpret rows with schema '{schema}'",
        fix="use columns like instruction/output, prompt/completion, messages, chosen/rejected, or text",
    )


def _as_text(v: Any) -> str:
    if isinstance(v, list):  # list of messages
        return "\n".join(str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in v)
    return str(v)


def row_sequences(norm: dict[str, Any]) -> list[str]:
    """Texts that are each processed as one training sequence (preference rows give two)."""
    if "messages" in norm:
        # ~4 tokens of chat-template overhead per message are added by the token counter
        return ["".join(f"<|{m['role']}|>{m['content']}" for m in norm["messages"])]
    if "text" in norm:
        return [norm["text"]]
    if "chosen" in norm:
        return [norm["prompt"] + norm["chosen"], norm["prompt"] + norm["rejected"]]
    return []


def canonical_text(norm: dict[str, Any]) -> str:
    """Whitespace/case-normalized text used for duplicate and leakage checks.

    Normalization is applied per message so stray spaces at message boundaries cannot make two
    otherwise identical rows look different."""

    def n(s: str) -> str:
        return " ".join(str(s).lower().split())

    if "messages" in norm:
        return "|".join(f"{m['role']}:{n(m['content'])}" for m in norm["messages"])
    if "text" in norm:
        return n(norm["text"])
    if "chosen" in norm:
        return "|".join(n(norm[k]) for k in ("prompt", "chosen", "rejected"))
    return ""
