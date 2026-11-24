"""OOM-safe search for the largest micro-batch that fits."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from bitstraining.engines.base import ProbeResult


def find_max_microbatch(
    probe: Callable[[int], ProbeResult],
    candidates: Sequence[int],
    hint: int | None = None,
) -> tuple[int | None, dict[int, ProbeResult]]:
    """Binary search over ascending `candidates`, assuming memory grows monotonically with batch.

    Returns (largest fitting candidate or None, every probe result obtained). Probing starts at
    `hint` (usually the planner's analytic answer) so a good estimate costs only 2-3 probes.
    """
    cands = sorted(set(candidates))
    results: dict[int, ProbeResult] = {}

    def fits(mb: int) -> bool:
        if mb not in results:
            results[mb] = probe(mb)
        return results[mb].fits

    lo, hi = 0, len(cands) - 1
    if hint in cands:
        i = cands.index(hint)  # type: ignore[arg-type]
        if fits(cands[i]):
            lo = i + 1
        else:
            hi = i - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(cands[mid]):
            lo = mid + 1
        else:
            hi = mid - 1
    fitting = [mb for mb, r in results.items() if r.fits]
    return (max(fitting) if fitting else None), results
