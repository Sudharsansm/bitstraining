"""Strategy search: enumerate feasible execution plans, estimate them, rank them.

Search space (restricted by each engine's declared Capabilities and by the user's constraints):
  engine x method x layer-streaming x activation checkpointing x optimizer x micro-batch
  x packing x parallelism.  Precision, attention kernel and fused loss are chosen per hardware.

Ranking: plans inside the user's time/cost budget come first, then by the objective
(time | cost | memory), then by operational complexity. The result keeps the best plan per
(engine, method, streaming, parallelism) so the caller sees real alternatives rather than
five micro-batch variants of the same idea.
"""

from __future__ import annotations

import math
from dataclasses import replace

from bitstraining.core.errors import PlanningError
from bitstraining.core.schemas import (
    Capabilities,
    DatasetStats,
    Estimate,
    ExecutionPlan,
    HardwareProfile,
    ModelInfo,
    RunSpec,
)
from bitstraining.data.pack import packing_efficiency, padding_efficiency
from bitstraining.hardware.database import GPU_DB
from bitstraining.planner import calibration as calib_mod
from bitstraining.planner.memory import estimate_memory
from bitstraining.planner.throughput import device_for, estimate_throughput
from bitstraining.strategies.catalog import METHODS

MICRO_BATCHES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_HEADROOM = 0.92  # fraction of device memory the plan may use
_ATTN_PREF = ("flash", "sdpa", "eager")


def device_kind(hw: HardwareProfile) -> str:
    return {"nvidia": "cuda", "amd": "cuda", "apple": "mps"}.get(hw.gpu_vendor, "cpu")


def _pick_attention(caps: Capabilities, hw: HardwareProfile) -> str:
    gpu = device_for(hw)
    for a in _ATTN_PREF:
        if a in caps.attention and (a != "flash" or (gpu.bf16 and device_kind(hw) == "cuda")):
            return a
    return caps.attention[0] if caps.attention else "sdpa"


def _complexity(p: ExecutionPlan) -> float:
    return (
        3.0 * p.layer_stream
        + (1.0 if p.activation != "none" else 0.0)
        + (1.0 if p.optimizer.startswith("paged") else 0.0)
        + (0.5 if p.optimizer == "adamw8bit" else 0.0)
        + (0.5 if p.parallelism == "fsdp" else 0.0)
    )


def suggest_gpus(peak_gb: float, headroom: float = DEFAULT_HEADROOM, n: int = 3) -> list[str]:
    """Smallest catalog GPUs whose memory could hold `peak_gb`."""
    seen: dict[float, str] = {}
    for name, row in sorted(GPU_DB.items(), key=lambda kv: kv[1]["vram_gb"]):
        if row["vram_gb"] * headroom >= peak_gb and "laptop" not in name and not name.startswith("apple"):
            seen.setdefault(row["vram_gb"], name)
    return [f"{name} ({vram:g} GB)" for vram, name in list(seen.items())[:n]]


def make_plans(
    spec: RunSpec,
    model: ModelInfo,
    stats: DatasetStats,
    hw: HardwareProfile,
    engines: list[Capabilities],
    top_k: int = 5,
    headroom: float = DEFAULT_HEADROOM,
    calibration: dict[str, dict[str, float]] | None = None,
) -> list[ExecutionPlan]:
    """Return up to `top_k` ranked, feasible plans or raise PlanningError with advice."""
    g, d = spec.goal, spec.data
    kind = device_kind(hw)
    gpu = device_for(hw)
    n_gpus = max(spec.compute.n_gpus or hw.n_gpus or 1, 1)
    vram = gpu.vram_gb
    ram = hw.ram_gb or float("inf")
    calibration = calibration if calibration is not None else calib_mod.load()

    seq_len = max(8, min(d.max_length, stats.max_len or d.max_length))
    lengths = stats.lengths_sample
    mean_len = stats.mean_len or float(seq_len)
    mem_precision = "bf16" if gpu.bf16 else "fp16"

    methods = [m for m in (g.methods or list(METHODS)) if m in METHODS]
    if not g.allow_quantization:
        methods = [m for m in methods if METHODS[m].quant == "none"]
    if not methods:
        raise PlanningError(
            "no methods left after applying goal.methods / allow_quantization",
            fix="allow quantization or list at least one of full, lora, dora",
        )

    pad_cache: dict[tuple[int, bool], float] = {}

    def efficiency(mb: int, packing: bool) -> float:
        k = (mb, packing)
        if k not in pad_cache:
            if not lengths:
                pad_cache[k] = 1.0
            elif packing:
                pad_cache[k] = packing_efficiency(lengths, d.max_length)
            else:
                pad_cache[k] = padding_efficiency(lengths, mb, grouped=True, max_len=d.max_length)
        return pad_cache[k]

    feasible: list[ExecutionPlan] = []
    closest: tuple[float, ExecutionPlan] | None = None
    total_tokens = stats.total_tokens * g.epochs

    for caps in engines:
        if not caps.available or g.task not in caps.tasks or kind not in caps.platforms:
            continue
        attention = _pick_attention(caps, hw)
        precision = mem_precision if mem_precision in caps.precisions else caps.precisions[0]
        fused = caps.supports_fused_ce
        parallel_opts = ["single"] if n_gpus == 1 else [p for p in ("ddp", "fsdp") if p in caps.parallelism]
        if not parallel_opts:
            parallel_opts = ["single"]
            eff_gpus = 1
        else:
            eff_gpus = n_gpus
        pack_opts = [False]
        if caps.supports_packing and d.packing != "off":
            pack_opts = [True] if d.packing == "on" else [False, True]

        for method_name in methods:
            if method_name not in caps.methods:
                continue
            method = METHODS[method_name]
            stream_opts = [False] + (
                [True] if caps.supports_layer_stream and not method.full_weights_trainable else []
            )
            for stream in stream_opts:
                for par in parallel_opts:
                    if stream and par != "single":
                        continue
                    for act in caps.activation_modes:
                        for opt in caps.optimizers:
                            for mb in MICRO_BATCHES:
                                if mb * eff_gpus > g.global_batch_size and mb > 1:
                                    continue
                                for pack in pack_opts:
                                    plan = ExecutionPlan(
                                        engine=caps.name,
                                        method=method_name,
                                        precision=precision,
                                        quant=method.quant,
                                        layer_stream=stream,
                                        activation=act,
                                        optimizer=opt,
                                        micro_batch=mb,
                                        grad_accum=max(1, math.ceil(g.global_batch_size / (mb * eff_gpus))),
                                        n_gpus=eff_gpus,
                                        parallelism=par,
                                        packing=pack,
                                        max_length=d.max_length,
                                        attention=attention,
                                        fused_ce=fused,
                                        dataloader_workers=max(2, min(8, (hw.cpu_cores or 4) // 2)),
                                        lora_r=g.lora_r,
                                        lora_alpha=g.lora_alpha,
                                        lora_dropout=g.lora_dropout,
                                        target_modules=g.target_modules,
                                        epochs=g.epochs,
                                        learning_rate=g.learning_rate,
                                        seed=g.seed,
                                        hardware=hw.name,
                                        model=model.name,
                                    )
                                    est = _estimate(
                                        plan, model, stats, hw, caps, seq_len, mean_len,
                                        efficiency(mb, pack), total_tokens, vram, ram, headroom,
                                        calibration, kind,
                                    )
                                    plan.estimate = est
                                    peak = est.peak_gb_per_gpu
                                    if closest is None or peak < closest[0]:
                                        closest = (peak, plan)
                                    if est.fits:
                                        plan.total_steps = _total_steps(plan, stats, g.epochs, mean_len, efficiency(mb, pack))
                                        feasible.append(plan)

    if not feasible:
        raise _no_plan_error(closest, vram, ram, headroom, model, hw)

    def violates(p: ExecutionPlan) -> int:
        e = p.estimate
        over_time = g.time_budget_hours is not None and e.total_hours > g.time_budget_hours
        over_cost = g.budget_usd is not None and e.cost_usd is not None and e.cost_usd > g.budget_usd
        return int(over_time or over_cost)

    def key(p: ExecutionPlan) -> tuple[float, ...]:
        e = p.estimate
        if g.objective == "memory":
            obj = e.peak_gb_per_gpu
        elif g.objective == "cost":
            obj = e.cost_usd if e.cost_usd is not None else e.total_hours
        else:
            obj = e.total_hours
        return (violates(p), obj, _complexity(p), e.peak_gb_per_gpu)

    feasible.sort(key=key)
    best_per_group: dict[tuple[str, str, bool, str], ExecutionPlan] = {}
    for p in feasible:
        best_per_group.setdefault((p.engine, p.method, p.layer_stream, p.parallelism), p)
    out = sorted(best_per_group.values(), key=key)[:top_k]
    if violates(out[0]):
        out[0].estimate.notes.append(
            "no plan meets the time/cost budget; this is the closest. Consider a faster GPU, fewer epochs or less data"
        )
    return out


def estimate_plan(
    plan: ExecutionPlan,
    spec: RunSpec,
    model: ModelInfo,
    stats: DatasetStats,
    hw: HardwareProfile,
    caps: Capabilities,
    headroom: float = DEFAULT_HEADROOM,
    calibration: dict[str, dict[str, float]] | None = None,
) -> Estimate:
    """Re-estimate one (possibly modified) plan, e.g. a different micro-batch during autotune."""
    d = spec.data
    gpu = device_for(hw)
    seq_len = max(8, min(d.max_length, stats.max_len or d.max_length))
    mean_len = stats.mean_len or float(seq_len)
    lengths = stats.lengths_sample
    if not lengths:
        eff = 1.0
    elif plan.packing:
        eff = packing_efficiency(lengths, d.max_length)
    else:
        eff = padding_efficiency(lengths, plan.micro_batch, grouped=True, max_len=d.max_length)
    cal = calibration if calibration is not None else calib_mod.load()
    return _estimate(
        plan, model, stats, hw, caps, seq_len, mean_len, eff,
        stats.total_tokens * plan.epochs, gpu.vram_gb, hw.ram_gb or float("inf"), headroom, cal,
        device_kind(hw),
    )


def _total_steps(plan: ExecutionPlan, stats: DatasetStats, epochs: float, mean_len: float, eff: float) -> int:
    seqs_per_step = plan.micro_batch * plan.grad_accum * plan.n_gpus
    if plan.packing:
        # each packed row holds several examples
        seqs_per_step *= max(1.0, plan.max_length * eff / max(mean_len, 1.0))
    return max(1, math.ceil(stats.num_rows * epochs / seqs_per_step))


def _estimate(
    plan: ExecutionPlan,
    model: ModelInfo,
    stats: DatasetStats,
    hw: HardwareProfile,
    caps: Capabilities,
    seq_len: int,
    mean_len: float,
    eff: float,
    total_tokens: float,
    vram: float,
    ram: float,
    headroom: float,
    calibration: dict[str, dict[str, float]],
    kind: str,
) -> Estimate:
    ck = calib_mod.key(hw.key, plan.engine, plan.method)
    cal = calibration.get(ck)
    mem_ratio = cal["mem_ratio"] if cal else 1.0
    speed_ratio = cal["speed_ratio"] if cal else 1.0

    parts, host_gb = estimate_memory(model, plan, seq_len, kind)
    peak = parts["total"] * mem_ratio
    parts = {k: v * (mem_ratio if k != "total" else 1.0) for k, v in parts.items()}
    parts["total"] = peak

    tp = estimate_throughput(model, plan, hw, caps, mean_len, eff, speed_ratio)
    hours = total_tokens / max(tp.useful_tokens_per_s, 1e-9) / 3600.0
    cost = hours * hw.price_per_hour if hw.price_per_hour is not None else None

    fits_mem = peak <= vram * headroom
    fits_host = host_gb <= ram * 0.9
    notes: list[str] = []
    if not fits_mem:
        notes.append(f"needs {peak:.1f} GB > {vram * headroom:.1f} GB usable")
    if not fits_host:
        notes.append(f"needs {host_gb:.1f} GB host RAM > {ram * 0.9:.1f} GB")
    if plan.layer_stream:
        notes.append("layer streaming is modeled only; no bundled real engine implements it yet")
        if tp.bottleneck == "transfer":
            notes.append("layer streaming is transfer-bound: expect a large slowdown vs a resident model")
    if plan.method in ("qlora", "dora"):
        notes.append(f"{plan.method} time overhead is a heuristic until calibrated")
    if not cal and not caps.speed_prior_is_measured and caps.speed_prior != 1.0:
        notes.append(f"{caps.name} speed prior {caps.speed_prior:g}x is unverified; run `bitstraining autotune`")

    # Honest confidence: without a real probe (calibration) every number is a model prediction,
    # so it is "low". A probe on this hardware earns "medium"; measured hardware, an exactly
    # known architecture and comfortable headroom earn "high".
    gpu = device_for(hw)
    if not cal or gpu.peak_tflops_bf16 <= 0:
        confidence = "low"
    elif hw.measured and model.source != "name-heuristic" and peak / max(vram, 1e-6) <= 0.85:
        confidence = "high"
    else:
        confidence = "medium"
    if confidence == "low":
        notes.append("uncalibrated: run `bitstraining autotune` on this hardware to verify with a real probe")

    return Estimate(
        fits=fits_mem and fits_host,
        peak_gb_per_gpu=round(peak, 3),
        breakdown_gb={k: round(v, 3) for k, v in parts.items()},
        host_ram_gb=round(host_gb, 2),
        raw_tokens_per_s=round(tp.raw_tokens_per_s, 1),
        useful_tokens_per_s=round(tp.useful_tokens_per_s, 1),
        step_time_s=round(tp.micro_batch_time_s * plan.grad_accum, 3),
        total_hours=round(hours, 4),
        cost_usd=None if cost is None else round(cost, 2),
        confidence=confidence,
        bottleneck=tp.bottleneck,
        notes=notes,
    )


def _no_plan_error(
    closest: tuple[float, ExecutionPlan] | None,
    vram: float,
    ram: float,
    headroom: float,
    model: ModelInfo,
    hw: HardwareProfile,
) -> PlanningError:
    if closest is None:
        return PlanningError(
            "no engine can run this task on this hardware",
            fix="install an engine (`pip install 'bitstraining[train]'`) or change goal.task / compute target",
        )
    peak, plan = closest
    gpus = suggest_gpus(peak)
    fixes = [
        f"closest plan ({plan.describe()}) needs {peak:.1f} GB but only {vram * headroom:.1f} GB is usable on {hw.name}",
        "reduce data.max_length or goal.global_batch_size",
        "allow quantization (goal.allow_quantization: true) or use a smaller base model",
    ]
    if gpus:
        fixes.append("or plan for rented hardware: " + ", ".join(gpus) + " (use --gpu)")
    return PlanningError(
        f"no feasible plan for {model.name or 'model'} ({model.params / 1e9:.1f}B params) on {hw.name}",
        fix="; ".join(fixes),
    )


def with_overrides(plan: ExecutionPlan, **kw: object) -> ExecutionPlan:
    """Copy a plan with changes (estimates are NOT recomputed; callers should re-estimate)."""
    return replace(plan, **kw)  # type: ignore[arg-type]
