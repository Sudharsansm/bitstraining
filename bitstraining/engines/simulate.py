"""Simulation engine: no model is trained.

Purpose: exercise the whole pipeline (planning, orchestration, tracking, eval gates, CLI) on any
machine without a GPU, and give CI deterministic runs. It produces a synthetic loss curve and
measurements that deviate from the planner's analytic model by a fixed hidden bias per
hardware/method, which lets the calibration loop be tested end to end.

Everything it writes is labeled `simulated: true`. It must never be used to claim results.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

from bitstraining.core.schemas import Capabilities, ExecutionPlan, HardwareProfile, RunSpec
from bitstraining.engines.base import Callbacks, Engine, ProbeContext, ProbeResult, RunResult
from bitstraining.planner.search import estimate_plan
from bitstraining.planner.throughput import device_for


def _bias(key: str, lo: float, hi: float) -> float:
    h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    return lo + (hi - lo) * h


class SimulateEngine(Engine):
    name = "simulate"

    def capabilities(self, hw: HardwareProfile, assume_installed: bool = False) -> Capabilities:
        return Capabilities(
            name=self.name,
            methods=["full", "lora", "qlora", "dora"],
            precisions=["bf16", "fp16"],
            activation_modes=["none", "selective", "full"],
            optimizers=["adamw", "adamw8bit", "paged_adamw8bit"],
            attention=["flash", "sdpa", "eager"],
            parallelism=["single", "ddp", "fsdp"],
            supports_packing=True,
            supports_layer_stream=True,
            supports_fused_ce=True,
            supports_compile=True,
            tasks=["sft", "dpo", "orpo", "kto", "pretrain"],
            platforms=["cuda", "mps", "cpu"],
            speed_prior=1.0,
            speed_prior_is_measured=True,
        )

    def probe(self, plan: ExecutionPlan, ctx: ProbeContext, micro_batch: int | None = None) -> ProbeResult:
        p = replace(plan, micro_batch=micro_batch or plan.micro_batch)
        est = estimate_plan(p, ctx.spec, ctx.model, ctx.stats, ctx.hw,
                            self.capabilities(ctx.hw), calibration={})
        hk = ctx.hw.key
        peak = est.peak_gb_per_gpu * (1 + _bias(f"{hk}|{p.method}|mem", -0.06, 0.10))
        tps = est.raw_tokens_per_s * (1 + _bias(f"{hk}|{p.method}|speed", -0.20, 0.10))
        vram = device_for(ctx.hw).vram_gb
        fits = peak <= vram
        return ProbeResult(
            fits=fits,
            peak_gb=round(peak, 3),
            tokens_per_s=round(tps, 1) if fits else None,
            steps=3,
            note="simulated measurement",
        )

    def train(
        self,
        plan: ExecutionPlan,
        spec: RunSpec,
        train_rows: list[dict[str, Any]],
        eval_rows: list[dict[str, Any]] | None,
        callbacks: Callbacks,
        output_dir: Path,
        resume_from: str | None = None,
    ) -> RunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        total = int(spec.extras.get("max_steps") or plan.total_steps or 100)
        start = 0
        if resume_from and Path(resume_from, "state.json").exists():
            start = int(json.loads(Path(resume_from, "state.json").read_text())["step"])
        rng = random.Random(plan.seed)
        n_rows = max(len(train_rows), 1)
        per_step = max(1, plan.micro_batch * plan.grad_accum * plan.n_gpus)
        order = list(range(n_rows))
        rng.shuffle(order)
        spike_at = spec.extras.get("inject_spike_step")
        tau = max(total / 4, 1.0)
        loss = 0.0
        sim_seconds = 0.0
        stopped = False
        step = start
        for step in range(start + 1, total + 1):
            if callbacks.stop_requested():
                stopped = True
                break
            base = 0.9 + (2.8 - 0.9) * math.exp(-step / tau)
            loss = base + rng.gauss(0, 0.03)
            if spike_at and step == int(spike_at):
                loss += 1.5
            i0 = ((step - 1) * per_step) % n_rows
            batch = [order[(i0 + k) % n_rows] for k in range(per_step)]
            if callbacks.on_batch:
                callbacks.on_batch(step, batch)
            sim_seconds += plan.estimate.step_time_s or 0.0
            callbacks.step(step, {"loss": round(loss, 4), "lr": plan.learning_rate})
            if step % max(total // 5, 1) == 0:
                ck = output_dir / f"checkpoint-{step}"
                ck.mkdir(exist_ok=True)
                (ck / "state.json").write_text(json.dumps({"step": step, "simulated": True}))
                if callbacks.on_checkpoint:
                    callbacks.on_checkpoint(str(ck))
        done = step if not stopped else step - 1
        eval_loss = round(loss + 0.08, 4) if eval_rows else None
        summary = {
            "simulated": True,
            "warning": "No model was trained. This directory contains a simulation record only.",
            "steps": done,
            "final_loss": round(loss, 4),
            "plan": plan.to_dict(),
        }
        (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        return RunResult(
            status="stopped" if stopped else "completed",
            steps=done,
            final_loss=round(loss, 4),
            eval_loss=eval_loss,
            seconds=round(sim_seconds, 2),
            peak_gb=plan.estimate.peak_gb_per_gpu,
            tokens_per_s=plan.estimate.useful_tokens_per_s,
            adapter_path=None,
            simulated=True,
            message="simulated run (no model trained)",
        )
