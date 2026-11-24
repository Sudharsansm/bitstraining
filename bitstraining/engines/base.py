"""The Engine interface every training backend implements.

Core code depends only on this interface, never on torch/transformers/mlx. An engine reports
what it can do on given hardware (`capabilities`), can verify a plan with a short real run
(`probe`), and can execute it (`train`). Engines are discovered through the plugin registry.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bitstraining.core import registry
from bitstraining.core.schemas import (
    Capabilities,
    DatasetStats,
    ExecutionPlan,
    HardwareProfile,
    ModelInfo,
    RunSpec,
    Serializable,
)


@dataclass
class ProbeContext:
    """Everything a probe needs to size a realistic batch."""

    spec: RunSpec
    model: ModelInfo
    hw: HardwareProfile
    stats: DatasetStats
    seq_len: int


@dataclass
class ProbeResult(Serializable):
    fits: bool = True
    peak_gb: float | None = None
    tokens_per_s: float | None = None
    steps: int = 0
    note: str = ""


@dataclass
class RunResult(Serializable):
    status: str = "completed"  # completed | stopped | failed
    steps: int = 0
    final_loss: float | None = None
    eval_loss: float | None = None
    seconds: float = 0.0
    peak_gb: float | None = None
    tokens_per_s: float | None = None
    adapter_path: str | None = None
    simulated: bool = False
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Callbacks:
    """Hooks the orchestrator gives an engine. All optional."""

    on_step: Callable[[int, dict[str, float]], None] | None = None
    on_batch: Callable[[int, list[int]], None] | None = None  # step -> training row indices
    on_checkpoint: Callable[[str], None] | None = None
    should_stop: Callable[[], bool] | None = None
    log: Callable[[str], None] = print

    def step(self, step: int, metrics: dict[str, float]) -> None:
        if self.on_step:
            self.on_step(step, metrics)

    def stop_requested(self) -> bool:
        return bool(self.should_stop and self.should_stop())


class Engine(abc.ABC):
    name: str = "engine"

    @abc.abstractmethod
    def capabilities(self, hw: HardwareProfile, assume_installed: bool = False) -> Capabilities:
        """What this engine can do on `hw`. With `assume_installed` (planning for remote
        hardware) library availability is assumed rather than checked locally."""

    @abc.abstractmethod
    def probe(self, plan: ExecutionPlan, ctx: ProbeContext, micro_batch: int | None = None) -> ProbeResult:
        """Run a few real (or simulated) steps to measure peak memory and throughput."""

    @abc.abstractmethod
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
        """Execute the plan on normalized rows (see bitstraining.data.schema)."""


REAL_ENGINES = ("hf_peft", "unsloth", "mlx", "deepspeed")


def available_engines(
    hw: HardwareProfile, which: str = "auto", assume_installed: bool = False
) -> list[Capabilities]:
    """Capabilities of the engines the planner may use. `auto` = real engines only; the
    `simulate` engine must be asked for by name so it never competes with real backends."""
    names = [which] if which != "auto" else list(REAL_ENGINES)
    out = []
    for n in names:
        eng: Engine = registry.create("engines", n)
        out.append(eng.capabilities(hw, assume_installed))
    return out
