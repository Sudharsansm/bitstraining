"""DeepSpeed / FSDP engine placeholder.

Not implemented. It is registered so the plugin surface and capability reporting are visible, but it
always reports `available=False`, so the planner never selects it. Multi-GPU data parallelism is
available today through `hf_peft` under torchrun; sharded (ZeRO-3/FSDP) training is roadmap Phase 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bitstraining.core.errors import EngineUnavailable
from bitstraining.core.schemas import Capabilities, ExecutionPlan, HardwareProfile, RunSpec
from bitstraining.engines.base import Callbacks, Engine, ProbeContext, ProbeResult, RunResult


class DeepSpeedEngine(Engine):
    name = "deepspeed"

    def capabilities(self, hw: HardwareProfile, assume_installed: bool = False) -> Capabilities:
        return Capabilities(
            name=self.name,
            available=False,
            reason="not implemented yet (roadmap Phase 4: sharded multi-GPU training)",
            methods=["full", "lora"],
            parallelism=["ddp", "fsdp"],
            tasks=["sft"],
        )

    def probe(self, plan: ExecutionPlan, ctx: ProbeContext, micro_batch: int | None = None) -> ProbeResult:
        raise EngineUnavailable("deepspeed engine is not implemented yet")

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
        raise EngineUnavailable("deepspeed engine is not implemented yet", fix="use engine: hf_peft")
