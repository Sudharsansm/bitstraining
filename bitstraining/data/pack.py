"""Sequence packing and padding-waste models.

Padding is the largest hidden cost on short-example datasets: if a batch is padded to
its longest member, every shorter row burns compute on pad tokens. Packing concatenates
several examples into one fixed-length sequence to avoid that.
"""

from __future__ import annotations

import random


def ffd_pack(lengths: list[int], max_len: int) -> list[list[int]]:
    """First-fit-decreasing bin packing. Returns bins of indices; sequences longer than
    `max_len` are truncated to it (they occupy a bin alone). O(n log n)."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins: list[list[int]] = []
    free: list[int] = []  # remaining capacity per bin
    # For FFD we need "first bin with enough room"; a linear scan over open bins is O(n*b),
    # so we keep bins in creation order and scan only bins that can still fit the smallest item.
    smallest = min((min(n, max_len) for n in lengths), default=0)
    open_bins: list[int] = []
    for i in order:
        ln = min(lengths[i], max_len)
        placed = False
        for pos, b in enumerate(open_bins):
            if free[b] >= ln:
                bins[b].append(i)
                free[b] -= ln
                if free[b] < smallest:
                    open_bins.pop(pos)
                placed = True
                break
        if not placed:
            bins.append([i])
            free.append(max_len - ln)
            if free[-1] >= smallest:
                open_bins.append(len(bins) - 1)
    return bins


def packing_efficiency(lengths: list[int], max_len: int) -> float:
    """Useful tokens / total slots after FFD packing (1.0 = no padding at all)."""
    if not lengths:
        return 1.0
    bins = ffd_pack(lengths, max_len)
    useful = sum(min(n, max_len) for n in lengths)
    return useful / (len(bins) * max_len)


def padding_efficiency(
    lengths: list[int], batch_size: int, grouped: bool = False, max_len: int | None = None,
    seed: int = 0,
) -> float:
    """Useful tokens / padded tokens when batches are padded to their longest member.

    `grouped=True` models length-grouped batching (sort within large mega-chunks), which
    removes most but not all padding.
    """
    if not lengths:
        return 1.0
    ls = [min(n, max_len) if max_len else n for n in lengths]
    rng = random.Random(seed)
    order = list(range(len(ls)))
    rng.shuffle(order)
    if grouped:
        mega = max(batch_size * 50, batch_size)
        chunks = [sorted(order[i : i + mega], key=lambda k: -ls[k]) for i in range(0, len(order), mega)]
        order = [k for c in chunks for k in c]
    useful = padded = 0
    for i in range(0, len(order), batch_size):
        batch = [ls[k] for k in order[i : i + batch_size]]
        useful += sum(batch)
        padded += max(batch) * len(batch)
    return useful / padded if padded else 1.0


def pack_sequences(
    token_lists: list[list[int]], max_len: int
) -> list[tuple[list[int], list[int]]]:
    """Materialize packed sequences. Returns (input_ids, position_ids) per packed row; position
    ids restart at 0 for each member, which flash-attention varlen kernels use as boundaries."""
    lens = [len(t) for t in token_lists]
    out = []
    for b in ffd_pack(lens, max_len):
        ids: list[int] = []
        pos: list[int] = []
        for i in b:
            t = token_lists[i][:max_len]
            ids.extend(t)
            pos.extend(range(len(t)))
        out.append((ids, pos))
    return out


def percentile(sorted_vals: list[int], q: float) -> int:
    """Nearest-rank percentile of an ascending list (q in [0, 1])."""
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, max(0, int(q * (len(sorted_vals) - 1) + 0.5)))]
