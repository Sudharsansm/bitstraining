"""Apple Silicon engine: drives the official `mlx_lm.lora` trainer.

EXPERIMENTAL and not exercised by CI (needs macOS + mlx-lm). The adapter writes the dataset in
mlx-lm's directory layout (train.jsonl / valid.jsonl), runs the trainer as a subprocess, and parses
its "Iter N: Train loss X" lines into callbacks.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from bitstraining.core.errors import EngineUnavailable
from bitstraining.core.schemas import Capabilities, ExecutionPlan, HardwareProfile, RunSpec
from bitstraining.engines.base import Callbacks, Engine, ProbeContext, ProbeResult, RunResult

_LOSS = re.compile(r"Iter\s+(\d+):\s+Train loss\s+([0-9.]+)")
_PEAK = re.compile(r"Peak mem\s+([0-9.]+)\s*GB", re.I)


class MLXEngine(Engine):
    name = "mlx"

    def capabilities(self, hw: HardwareProfile, assume_installed: bool = False) -> Capabilities:
        apple = hw.gpu_vendor == "apple"
        have = assume_installed or importlib.util.find_spec("mlx_lm") is not None
        ok = apple and have
        return Capabilities(
            name=self.name,
            available=ok,
            reason="" if ok else ("needs Apple Silicon" if not apple else "mlx-lm not installed (pip install 'bitstraining[mlx]')"),
            methods=["lora", "qlora"],  # qlora = LoRA over a pre-quantized MLX model
            precisions=["bf16", "fp16"],
            activation_modes=["none", "full"],
            optimizers=["adamw"],
            attention=["sdpa"],
            parallelism=["single"],
            tasks=["sft"],
            platforms=["mps"],
            speed_prior=1.0,
            speed_prior_is_measured=False,
        )

    def probe(self, plan: ExecutionPlan, ctx: ProbeContext, micro_batch: int | None = None) -> ProbeResult:
        return ProbeResult(fits=True, note="probe not implemented for MLX; plan estimate is unverified")

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
        if importlib.util.find_spec("mlx_lm") is None:
            raise EngineUnavailable("mlx-lm is not installed", fix="pip install 'bitstraining[mlx]'")
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = output_dir / "mlx_data"
        data_dir.mkdir(exist_ok=True)
        valid = eval_rows or train_rows[: max(1, len(train_rows) // 20)]
        for name, rows in (("train", train_rows), ("valid", valid)):
            with (data_dir / f"{name}.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        iters = int(spec.extras.get("max_steps") or plan.total_steps * plan.grad_accum or 100)
        cmd = [
            sys.executable, "-m", "mlx_lm.lora", "--model", spec.model.base, "--train",
            "--data", str(data_dir), "--iters", str(iters),
            "--batch-size", str(plan.micro_batch), "--learning-rate", str(plan.learning_rate),
            "--max-seq-length", str(plan.max_length), "--adapter-path", str(output_dir),
            "--grad-accumulation-steps", str(plan.grad_accum),
        ]
        if plan.activation == "full":
            cmd.append("--grad-checkpoint")
        callbacks.log("running: " + " ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last_loss, last_step, peak = None, 0, None
        assert proc.stdout is not None
        for line in proc.stdout:
            callbacks.log(line.rstrip())
            if m := _LOSS.search(line):
                last_step, last_loss = int(m.group(1)), float(m.group(2))
                callbacks.step(last_step, {"loss": last_loss})
            if m := _PEAK.search(line):
                peak = float(m.group(1))
            if callbacks.stop_requested():
                proc.terminate()
                break
        rc = proc.wait()
        status = "completed" if rc == 0 else ("stopped" if callbacks.stop_requested() else "failed")
        return RunResult(status=status, steps=last_step, final_loss=last_loss, peak_gb=peak,
                         adapter_path=str(output_dir) if rc == 0 else None,
                         message="" if rc == 0 else f"mlx_lm.lora exited with code {rc}")
