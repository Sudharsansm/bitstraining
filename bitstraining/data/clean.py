"""Cleaning: malformed/empty/oversized rows, exact duplicates, optional MinHash near-duplicates."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from bitstraining.core.errors import DataError
from bitstraining.core.schemas import Serializable
from bitstraining.data.schema import canonical_text, normalize_row, row_sequences


@dataclass
class CleanReport(Serializable):
    input_rows: int = 0
    kept: int = 0
    malformed: int = 0
    empty: int = 0
    too_long: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    examples: dict[str, list[int]] = field(default_factory=dict)  # reason -> first row indices

    def note(self, reason: str, idx: int) -> None:
        lst = self.examples.setdefault(reason, [])
        if len(lst) < 5:
            lst.append(idx)


def _is_empty(norm: dict[str, Any]) -> bool:
    if "messages" in norm:
        msgs = norm["messages"]
        return not msgs or not any(m["role"] == "assistant" and m["content"].strip() for m in msgs)
    if "text" in norm:
        return not norm["text"].strip()
    return not (norm["chosen"].strip() and norm["rejected"].strip() and norm["prompt"].strip())


def clean_rows(
    rows: list[dict[str, Any]],
    schema: str,
    max_chars: int | None = None,
    dedup: bool = True,
    near_dup_threshold: float | None = None,
) -> tuple[list[dict[str, Any]], CleanReport]:
    """Return (normalized kept rows, report). Never mutates input."""
    rep = CleanReport(input_rows=len(rows))
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, raw in enumerate(rows):
        try:
            norm = normalize_row(raw, schema)
        except DataError:
            rep.malformed += 1
            rep.note("malformed", i)
            continue
        if _is_empty(norm):
            rep.empty += 1
            rep.note("empty", i)
            continue
        if max_chars and sum(len(s) for s in row_sequences(norm)) > max_chars:
            rep.too_long += 1
            rep.note("too_long", i)
            continue
        if dedup:
            h = hashlib.blake2b(canonical_text(norm).encode(), digest_size=16).hexdigest()
            if h in seen:
                rep.exact_duplicates += 1
                rep.note("exact_duplicate", i)
                continue
            seen.add(h)
        kept.append(norm)
    if near_dup_threshold and len(kept) > 1:
        groups = near_duplicate_groups([canonical_text(r) for r in kept], near_dup_threshold)
        drop = {idx for g in groups for idx in g[1:]}  # keep the first of each group
        rep.near_duplicates = len(drop)
        kept = [r for i, r in enumerate(kept) if i not in drop]
    rep.kept = len(kept)
    return kept, rep


# ----------------------------------------------------------------------------- near-duplicates
_P = (1 << 61) - 1


def _shingles(text: str, k: int = 3) -> set[int]:
    words = text.split()
    if len(words) <= k:
        grams = [" ".join(words)] if words else []
    else:
        grams = [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
    return {int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big") for g in grams}


def near_duplicate_groups(
    texts: list[str], threshold: float = 0.8, num_perm: int = 64, bands: int = 16, seed: int = 1
) -> list[list[int]]:
    """MinHash + LSH. Returns groups (sorted index lists, size >= 2) of likely near-duplicates.

    Candidate pairs come from band collisions and are confirmed by the estimated Jaccard
    similarity of their signatures, so `threshold` is the minimum estimated similarity.
    """
    if num_perm % bands:
        raise ValueError("num_perm must be divisible by bands")
    rows_per_band = num_perm // bands
    rng = random.Random(seed)
    coeffs = [(rng.randrange(1, _P), rng.randrange(0, _P)) for _ in range(num_perm)]
    sigs: list[tuple[int, ...] | None] = []
    for t in texts:
        sh = _shingles(t)
        sigs.append(tuple(min((a * x + b) % _P for x in sh) for a, b in coeffs) if sh else None)

    parent = list(range(len(texts)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for i, s in enumerate(sigs):
        if s is None:
            continue
        for b in range(bands):
            buckets.setdefault((b, s[b * rows_per_band : (b + 1) * rows_per_band]), []).append(i)
    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for ai in range(len(members)):
            for bi in range(ai + 1, len(members)):
                i, j = members[ai], members[bi]
                if (i, j) in checked:
                    continue
                checked.add((i, j))
                si, sj = sigs[i], sigs[j]
                assert si is not None and sj is not None
                est = sum(1 for x, y in zip(si, sj, strict=True) if x == y) / num_perm
                if est >= threshold:
                    parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(texts)):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]
