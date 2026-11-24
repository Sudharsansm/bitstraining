"""Train/eval splitting that cannot leak duplicates across the boundary."""

from __future__ import annotations

import hashlib
import random
from typing import Any

from bitstraining.data.schema import canonical_text


def _key(norm: dict[str, Any]) -> str:
    return hashlib.blake2b(canonical_text(norm).encode(), digest_size=8).hexdigest()


def split_rows(
    rows: list[dict[str, Any]], eval_fraction: float, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic split of *normalized* rows. Rows with identical canonical text always
    land on the same side, so exact duplicates never straddle train and eval."""
    if eval_fraction <= 0 or len(rows) < 2:
        return list(rows), []
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(_key(r), []).append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, round(len(rows) * eval_fraction))
    eval_idx: set[int] = set()
    for k in keys:
        if len(eval_idx) >= target:
            break
        eval_idx.update(groups[k])
    train = [r for i, r in enumerate(rows) if i not in eval_idx]
    ev = [r for i, r in enumerate(rows) if i in eval_idx]
    return train, ev


def leakage(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> list[int]:
    """Indices of eval rows whose canonical text also appears in train."""
    train_keys = {_key(r) for r in train_rows}
    return [i for i, r in enumerate(eval_rows) if _key(r) in train_keys]
