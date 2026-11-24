"""Dataset statistics that feed the planner, plus the human-readable `inspect` report."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bitstraining.core.schemas import DatasetStats
from bitstraining.data.clean import clean_rows
from bitstraining.data.loaders import read_rows
from bitstraining.data.pack import packing_efficiency, padding_efficiency, percentile
from bitstraining.data.schema import detect_schema, row_sequences

_CHAT_OVERHEAD = 4  # tokens of template framing per sequence segment (rough)

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Dependency-free estimate: ~3.8 characters per token for English-like text.

    This is an approximation; pass a real tokenizer's counter (see `tokenizer_counter`)
    for exact numbers. It tends to be within roughly 15-25% on English prose and is
    less reliable for code or non-Latin scripts.
    """
    return max(1, math.ceil(len(text) / 3.8)) if text else 0


def tokenizer_counter(name: str) -> TokenCounter | None:
    """Exact counter backed by a Hugging Face tokenizer, or None if unavailable offline."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name)
    except Exception:
        return None
    return lambda t: len(tok(t, add_special_tokens=False)["input_ids"])


def hash_rows(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(repr(sorted(r.items())).encode())
    return h.hexdigest()[:16]


def hash_file(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def compute_stats(
    norm_rows: list[dict[str, Any]],
    max_length: int = 2048,
    counter: TokenCounter | None = None,
    schema: str = "unknown",
    data_hash: str = "",
    sample_cap: int = 4000,
) -> DatasetStats:
    """Token statistics over *normalized* rows (see data.schema.normalize_row)."""
    count = counter or estimate_tokens
    lengths: list[int] = []
    for r in norm_rows:
        for seq in row_sequences(r):
            lengths.append(count(seq) + _CHAT_OVERHEAD)
    n = len(lengths)
    s = sorted(lengths)
    over = sum(1 for x in lengths if x > max_length)
    step = max(1, n // sample_cap)
    return DatasetStats(
        num_rows=len(norm_rows),
        total_tokens=sum(min(x, max_length) for x in lengths),
        mean_len=(sum(lengths) / n) if n else 0.0,
        p50_len=percentile(s, 0.5),
        p95_len=percentile(s, 0.95),
        max_len=s[-1] if s else 0,
        truncation_rate=(over / n) if n else 0.0,
        schema=schema,
        data_hash=data_hash,
        lengths_sample=lengths[::step][:sample_cap],
        token_counter="tokenizer" if counter else "chars/3.8 estimate",
    )


def load_and_profile(
    path: str,
    fmt: str = "auto",
    max_length: int = 2048,
    dedup: bool = True,
    counter: TokenCounter | None = None,
) -> tuple[list[dict[str, Any]], DatasetStats]:
    """Read, detect schema, clean, and compute stats in one call."""
    raw = read_rows(path)
    schema = detect_schema(raw) if fmt == "auto" else fmt
    kept, rep = clean_rows(raw, schema, dedup=dedup)
    stats = compute_stats(kept, max_length, counter, schema, hash_file(path) or hash_rows(kept))
    n = max(rep.input_rows, 1)
    stats.empty_rate = rep.empty / n
    stats.duplicate_rate = rep.exact_duplicates / n
    return kept, stats


def inspect_report(path: str, fmt: str = "auto", max_length: int = 2048) -> str:
    """Plain-text report for `bitstraining data inspect`."""
    raw = read_rows(path)
    schema = detect_schema(raw) if fmt == "auto" else fmt
    kept, rep = clean_rows(raw, schema, dedup=True)
    st = compute_stats(kept, max_length, None, schema, hash_file(path) or hash_rows(kept))
    lens = st.lengths_sample
    lines = [
        f"File:            {path}",
        f"Rows read:       {rep.input_rows}",
        f"Schema:          {schema}",
        f"Usable rows:     {rep.kept}",
        f"  malformed:     {rep.malformed}",
        f"  empty:         {rep.empty}",
        f"  duplicates:    {rep.exact_duplicates}",
        f"Tokens (est.):   {st.total_tokens:,} total ({st.token_counter})",
        f"Length tokens:   mean {st.mean_len:.0f} | p50 {st.p50_len} | p95 {st.p95_len} | max {st.max_len}",
        f"Truncated @{max_length}: {st.truncation_rate:.1%} of sequences",
        f"Content hash:    {st.data_hash}",
    ]
    if lens:
        ml = max_length
        lines += [
            "Padding waste (batch=8, padded to longest): "
            f"{1 - padding_efficiency(lens, 8, False, ml):.0%}",
            "Padding waste with length grouping:         "
            f"{1 - padding_efficiency(lens, 8, True, ml):.0%}",
            f"Padding waste with sequence packing:        {1 - packing_efficiency(lens, ml):.0%}",
        ]
    if rep.examples:
        lines.append("Examples of dropped rows (0-based index): " + str(rep.examples))
    return "\n".join(lines)
