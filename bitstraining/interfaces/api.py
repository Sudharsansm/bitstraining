"""Python API. The CLI is a thin layer over these functions, so notebooks and pipelines get the same power.

    import bitstraining as bt
    spec = bt.core.config.load_spec("run.yaml")
    plans = bt.plan(spec)                  # ranked ExecutionPlans with predicted memory/speed/cost
    report = bt.doctor(spec)               # pre-flight checks with fix suggestions
    result = bt.train(spec)                # profile -> plan -> doctor -> run -> track
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bitstraining.autotune import TuneResult
from bitstraining.autotune import autotune as _autotune
from bitstraining.compute.base import ResourceRequest
from bitstraining.compute.bundle import make_bundle
from bitstraining.compute.orchestrator import run_job
from bitstraining.core import paths, registry
from bitstraining.core.errors import BitstrainingError, ConfigError, PlanningError
from bitstraining.core.schemas import (
    Capabilities,
    DatasetStats,
    ExecutionPlan,
    HardwareProfile,
    ModelInfo,
    RunSpec,
)
from bitstraining.data.clean import clean_rows
from bitstraining.data.loaders import read_rows
from bitstraining.data.schema import detect_schema
from bitstraining.data.split import split_rows
from bitstraining.data.stats import compute_stats, hash_file, hash_rows
from bitstraining.engines.base import Callbacks, ProbeContext, RunResult, available_engines
from bitstraining.hardware.database import synthetic_profile
from bitstraining.hardware.profiler import profile as _detect
from bitstraining.models.introspect import load_model_info
from bitstraining.planner.doctor import DoctorReport, run_doctor
from bitstraining.planner.search import make_plans
from bitstraining.tracking.registry import Registry


@dataclass
class Context:
    """Everything the planner needs, gathered once."""

    spec: RunSpec
    hw: HardwareProfile
    model: ModelInfo
    stats: DatasetStats
    train_rows: list[dict[str, Any]]
    eval_rows: list[dict[str, Any]] | None
    caps: list[Capabilities]
    remote_hw: bool = False
    notes: list[str] = field(default_factory=list)


def profile_hardware(
    benchmark: bool = False, gpu: str | None = None, n_gpus: int | None = None, price: float | None = None
) -> HardwareProfile:
    """Profile this machine, or build a profile for a named GPU (`gpu="a100-80gb"`) to plan ahead."""
    if gpu:
        return synthetic_profile(gpu, n_gpus or 1, price)
    hw = _detect(benchmark=benchmark)
    if price is not None:
        hw.price_per_hour = price
    return hw


def inspect_data(path: str, fmt: str = "auto", max_length: int = 2048) -> str:
    from bitstraining.data.stats import inspect_report

    return inspect_report(path, fmt, max_length)


def build_context(
    spec: RunSpec,
    gpu: str | None = None,
    n_gpus: int | None = None,
    price: float | None = None,
    benchmark: bool = False,
) -> Context:
    gpu = gpu or spec.compute.gpu
    n_gpus = n_gpus or spec.compute.n_gpus
    price = price if price is not None else spec.compute.options.get("price_per_hour")
    hw = profile_hardware(benchmark, gpu, n_gpus, price)
    remote = gpu is not None
    offline = os.environ.get("BITSTRAINING_OFFLINE") == "1"
    model = load_model_info(spec.model.base, allow_network=not offline)

    raw = read_rows(spec.data.train)
    schema = detect_schema(raw) if spec.data.format == "auto" else spec.data.format
    rows, rep = clean_rows(raw, schema, dedup=spec.data.dedup)
    eval_rows = None
    if spec.data.eval:
        eval_rows, _ = clean_rows(read_rows(spec.data.eval), schema, dedup=False)
    elif spec.data.eval_split > 0:
        rows, eval_rows = split_rows(rows, spec.data.eval_split, spec.goal.seed)
    stats = compute_stats(rows, spec.data.max_length, None, schema,
                          hash_file(spec.data.train) or hash_rows(rows))
    n = max(rep.input_rows, 1)
    stats.empty_rate = rep.empty / n
    stats.duplicate_rate = rep.exact_duplicates / n
    caps = available_engines(hw, spec.engine, assume_installed=remote)
    return Context(spec, hw, model, stats, rows, eval_rows, caps, remote)


def plan(spec: RunSpec, top_k: int = 5, ctx: Context | None = None, **kw: Any) -> list[ExecutionPlan]:
    """Ranked, feasible execution plans for this run on this (or the named) hardware."""
    ctx = ctx or build_context(spec, **kw)
    return make_plans(ctx.spec, ctx.model, ctx.stats, ctx.hw, ctx.caps, top_k=top_k)


def doctor(spec: RunSpec, ctx: Context | None = None, **kw: Any) -> DoctorReport:
    try:
        ctx = ctx or build_context(spec, **kw)
    except BitstrainingError as e:  # still report data/model problems as findings
        rep = DoctorReport()
        rep.add("error", "DR000", e.message, e.fix or "")
        return rep
    try:
        plans, err = plan(spec, ctx=ctx), None
    except PlanningError as e:
        plans, err = None, str(e)
    return run_doctor(spec, ctx.hw, ctx.model, ctx.stats, plans, err, ctx.train_rows, ctx.eval_rows,
                      check_imports=not ctx.remote_hw)


def autotune(spec: RunSpec, ctx: Context | None = None, use_cache: bool = True, **kw: Any) -> TuneResult:
    """Probe the top plan on the real device (or simulate) and return a measured, tuned plan."""
    ctx = ctx or build_context(spec, **kw)
    best = plan(spec, ctx=ctx, top_k=1)[0]
    engine = registry.create("engines", best.engine)
    seq = max(8, min(spec.data.max_length, ctx.stats.max_len or spec.data.max_length))
    pctx = ProbeContext(spec, ctx.model, ctx.hw, ctx.stats, seq)
    caps = next(c for c in ctx.caps if c.name == best.engine)
    return _autotune(best, pctx, engine, caps, use_cache=use_cache)


def train(
    spec: RunSpec,
    gpu: str | None = None,
    resume: bool = False,
    force: bool = False,
    tune: bool = False,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
    chosen_plan: ExecutionPlan | None = None,
) -> dict[str, Any]:
    """Profile -> plan -> doctor -> (autotune) -> run -> track. Returns a summary dict."""
    ctx = build_context(spec, gpu=gpu)
    plans = plan(spec, ctx=ctx) if chosen_plan is None else [chosen_plan]
    best = plans[0]
    report = run_doctor(spec, ctx.hw, ctx.model, ctx.stats, plans, None, ctx.train_rows, ctx.eval_rows,
                        check_imports=not ctx.remote_hw)
    log(report.render())
    if not report.ok and not force:
        raise ConfigError("pre-flight checks failed", fix="fix the errors above or pass force=True / --force")
    if tune and chosen_plan is None:
        tr = autotune(spec, ctx=ctx)
        best = tr.plan
        for n in tr.notes:
            log(f"autotune: {n}")
    if ctx.remote_hw and spec.compute.provider == "local":
        raise ConfigError(
            "a hardware profile was named (--gpu) but compute.provider is 'local'",
            fix="--gpu is for planning ahead; to run elsewhere set compute.provider (ssh, docker, notebook)",
        )
    e = best.estimate
    log(f"plan: {best.describe()}")
    log(f"predicted: {e.peak_gb_per_gpu:.1f} GB/GPU, {e.useful_tokens_per_s:,.0f} tok/s, "
        f"{e.total_hours:.2f} h, confidence {e.confidence}")
    out = Path(spec.output)
    summary: dict[str, Any] = {"plan": best.to_dict(), "doctor_ok": report.ok, "dry_run": dry_run}
    if dry_run:
        return summary

    out.mkdir(parents=True, exist_ok=True)
    with (out / "train_normalized.jsonl").open("w") as f:  # lets `spikes` show the offending rows
        for r in ctx.train_rows:
            f.write(json.dumps(r) + "\n")
    reg = Registry()
    simulated = best.engine == "simulate"
    run_id = reg.create_run(spec.name, spec.to_dict(), best.to_dict(), ctx.hw.to_dict(),
                            ctx.stats.data_hash, best.engine, simulated)
    summary["run_id"] = run_id
    try:
        if spec.compute.provider == "local":
            result = _run_local(spec, best, ctx, reg, run_id, out, resume, log)
        else:
            result = _run_remote(spec, best, ctx, reg, run_id, out, resume, log)
        reg.finish_run(run_id, result.status, result.to_dict())
        if result.adapter_path:
            reg.add_artifact(run_id, "adapter", result.adapter_path)
        summary["result"] = result.to_dict()
        if simulated:
            log("NOTE: simulated run. No model was trained.")
        return summary
    except BaseException as ex:
        reg.finish_run(run_id, "failed", {"error": str(ex)})
        raise
    finally:
        reg.close()


def _run_local(spec: RunSpec, plan_: ExecutionPlan, ctx: Context, reg: Registry, run_id: str,
               out: Path, resume: bool, log: Callable[[str], None]) -> RunResult:
    engine = registry.create("engines", plan_.engine)
    every = max(1, plan_.total_steps // 20)

    def on_step(step: int, metrics: dict[str, float]) -> None:
        reg.log_metrics(run_id, step, metrics)
        if step <= 3 or step % every == 0:
            log(f"step {step}: " + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()))

    cb = Callbacks(on_step=on_step, on_batch=lambda s, ids: reg.log_batch(run_id, s, ids), log=log)
    resume_from = None
    if resume:
        cks = sorted((p for p in out.glob("checkpoint-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[-1]))
        resume_from = str(cks[-1]) if cks else None
    return engine.train(plan_, spec, ctx.train_rows, ctx.eval_rows, cb, out, resume_from)


def _run_remote(spec: RunSpec, plan_: ExecutionPlan, ctx: Context, reg: Registry, run_id: str,
                out: Path, resume: bool, log: Callable[[str], None]) -> RunResult:
    from bitstraining.compute.base import JobSpec

    provider = registry.create("providers", spec.compute.provider, spec.compute.options)
    offers = provider.list_offers(ResourceRequest(gpu=spec.compute.gpu, n_gpus=spec.compute.n_gpus or 1))
    if not offers:
        raise ConfigError("the provider returned no offers", fix="relax compute constraints")
    offer = offers[0]
    bundle = make_bundle(spec, plan_, paths.home() / "jobs" / f"bundle-{run_id}")
    steps_seen = {"n": 0}

    def relay(line: str) -> None:
        if line.startswith("BT_METRIC "):
            try:
                m = json.loads(line[len("BT_METRIC "):])
                step = int(m.pop("step"))
                reg.log_metrics(run_id, step, {k: float(v) for k, v in m.items()})
                steps_seen["n"] = step
            except (ValueError, KeyError):
                pass
        log(line)

    outcome = run_job(provider, offer, bundle, JobSpec(name=spec.name, bundle_dir=str(bundle), resume=resume),
                      out, budget_usd=spec.goal.budget_usd, log=relay)
    log(f"job {outcome.status}; {outcome.hours:.3f} h" + (f", ${outcome.cost_usd:.2f}" if outcome.cost_usd is not None else ""))
    res_file = out / "result.json"
    if res_file.exists():
        d = json.loads(res_file.read_text())
        d.pop("wall_seconds", None)
        res = RunResult(**d)
    else:
        res = RunResult(status="stopped" if outcome.status == "manual" else "failed", steps=steps_seen["n"],
                        message=f"job ended with status {outcome.status}")
    if outcome.status in ("budget_exceeded", "timeout"):
        res.status = "stopped"
        res.message = outcome.status
    return res
