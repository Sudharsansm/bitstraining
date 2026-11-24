"""Autotune a plan against real (or simulated) probes and learn calibration from them."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from bitstraining.autotune import cache as cache_mod
from bitstraining.autotune.microbatch import find_max_microbatch
from bitstraining.core.schemas import Capabilities, ExecutionPlan
from bitstraining.engines.base import Engine, ProbeContext, ProbeResult
from bitstraining.planner import calibration as calib_mod
from bitstraining.planner.search import MICRO_BATCHES, estimate_plan
from bitstraining.planner.throughput import device_for

SAFETY = 0.95  # never choose a micro-batch whose measured peak exceeds this share of device memory


@dataclass
class TuneResult:
    plan: ExecutionPlan
    probes: dict[int, ProbeResult] = field(default_factory=dict)
    calibration: dict[str, float] | None = None
    from_cache: bool = False
    notes: list[str] = field(default_factory=list)


def autotune(
    plan: ExecutionPlan,
    ctx: ProbeContext,
    engine: Engine,
    caps: Capabilities | None = None,
    use_cache: bool = True,
    cache_path: Path | None = None,
    calibration_path: Path | None = None,
) -> TuneResult:
    """Pick the micro-batch that fits and runs fastest; update calibration from what was measured."""
    hw, g = ctx.hw, ctx.spec.goal
    caps = caps or engine.capabilities(hw)
    vram = device_for(hw).vram_gb
    n_gpus = max(plan.n_gpus, 1)
    cap_mb = max(1, g.global_batch_size // n_gpus)
    cands = [mb for mb in MICRO_BATCHES if mb <= cap_mb] or [1]
    key = cache_mod.cache_key(hw, ctx.model.name, plan, ctx.seq_len)
    notes: list[str] = []

    cached = cache_mod.get(key, cache_path) if use_cache else None
    if cached and cached.get("micro_batch") in cands:
        mb = int(cached["micro_batch"])
        verify = engine.probe(plan, ctx, mb)
        if verify.fits and (verify.peak_gb or 0) <= vram * SAFETY:
            tuned = _apply(plan, mb, g.global_batch_size, n_gpus)
            tuned.estimate = estimate_plan(tuned, ctx.spec, ctx.model, ctx.stats, hw, caps,
                                           calibration=calib_mod.load(calibration_path))
            return TuneResult(tuned, {mb: verify}, None, True, ["reused cached tuning (verified with one probe)"])
        notes.append("cached tuning no longer fits; re-tuning")

    best_mb, probes = find_max_microbatch(lambda mb: engine.probe(plan, ctx, mb), cands, hint=plan.micro_batch)
    if best_mb is None:
        raise_msg = "no micro-batch fits on this device even at 1; plan is infeasible"
        out = replace(plan)
        out.estimate.fits = False
        out.estimate.notes.append(raise_msg)
        return TuneResult(out, probes, None, False, [raise_msg])

    # Respect the safety margin, then take the fastest measured fitting candidate near the top.
    ok = sorted(
        mb for mb, r in probes.items()
        if r.fits and (r.peak_gb is None or r.peak_gb <= vram * SAFETY)
    )
    if not ok:
        ok = [best_mb]
        notes.append(f"largest fitting batch uses >{SAFETY:.0%} of memory; little headroom for long batches")
    top = ok[-2:]  # compare the two largest safe candidates
    for mb in top:
        if mb not in probes:
            probes[mb] = engine.probe(plan, ctx, mb)
    chosen = max(top, key=lambda mb: (probes[mb].tokens_per_s or 0.0, mb))
    tuned = _apply(plan, chosen, g.global_batch_size, n_gpus)

    # Learn: measured / predicted, using the *uncalibrated* prediction for the chosen batch.
    raw_est = estimate_plan(tuned, ctx.spec, ctx.model, ctx.stats, hw, caps, calibration={})
    pr = probes[chosen]
    cal = None
    if pr.peak_gb and raw_est.peak_gb_per_gpu > 0:
        mem_ratio = pr.peak_gb / raw_est.peak_gb_per_gpu
        speed_ratio = (pr.tokens_per_s / raw_est.raw_tokens_per_s) if pr.tokens_per_s and raw_est.raw_tokens_per_s else None
        cal = calib_mod.update(calib_mod.key(hw.key, plan.engine, plan.method),
                               mem_ratio=mem_ratio, speed_ratio=speed_ratio, path=calibration_path)
        notes.append(f"calibration: memory x{cal['mem_ratio']:.2f}, speed x{cal['speed_ratio']:.2f} (n={int(cal['n'])})")
    tuned.estimate = estimate_plan(tuned, ctx.spec, ctx.model, ctx.stats, hw, caps,
                                   calibration=calib_mod.load(calibration_path))
    cache_mod.put(key, {"micro_batch": chosen, "peak_gb": pr.peak_gb, "tokens_per_s": pr.tokens_per_s}, cache_path)
    if chosen != plan.micro_batch:
        notes.append(f"micro-batch {plan.micro_batch} -> {chosen} (grad_accum {plan.grad_accum} -> {tuned.grad_accum})")
    return TuneResult(tuned, probes, cal, False, notes)


def _apply(plan: ExecutionPlan, mb: int, global_batch: int, n_gpus: int) -> ExecutionPlan:
    return replace(
        plan,
        micro_batch=mb,
        grad_accum=max(1, math.ceil(global_batch / (mb * n_gpus))),
    )


def bake_off(
    plans: list[ExecutionPlan],
    ctx: ProbeContext,
    engines: dict[str, Engine],
    calibration_path: Path | None = None,
) -> list[tuple[ExecutionPlan, ProbeResult]]:
    """Probe one plan per engine at its own micro-batch and rank by measured tokens/s.

    This is how engine claims (e.g. "fused kernels are faster") become measurements on *your*
    hardware: each result also updates that engine's calibration ratios.
    """
    out: list[tuple[ExecutionPlan, ProbeResult]] = []
    for p in plans:
        eng = engines[p.engine]
        caps = eng.capabilities(ctx.hw)
        res = eng.probe(p, ctx, p.micro_batch)
        if res.fits and res.peak_gb:
            raw = estimate_plan(p, ctx.spec, ctx.model, ctx.stats, ctx.hw, caps, calibration={})
            calib_mod.update(
                calib_mod.key(ctx.hw.key, p.engine, p.method),
                mem_ratio=res.peak_gb / raw.peak_gb_per_gpu if raw.peak_gb_per_gpu else None,
                speed_ratio=(res.tokens_per_s / raw.raw_tokens_per_s) if res.tokens_per_s and raw.raw_tokens_per_s else None,
                path=calibration_path,
            )
        out.append((p, res))
    out.sort(key=lambda t: (not t[1].fits, -(t[1].tokens_per_s or 0.0)))
    return out
