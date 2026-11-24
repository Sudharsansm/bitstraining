"""Provider interface: the contract every compute backend implements.

A job is identical wherever it runs. Providers only move a *bundle* (spec, plan, data, source) to a
machine, start `python -m bitstraining.runner`, stream its log, and bring the output back.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bitstraining.core.schemas import HardwareProfile


@dataclass
class ResourceRequest:
    gpu: str | None = None
    n_gpus: int = 1
    min_vram_gb: float = 0.0
    max_price_per_hour: float | None = None
    spot_ok: bool = True


@dataclass
class Offer:
    provider: str
    name: str
    gpu: str
    n_gpus: int
    vram_gb: float
    price_per_hour: float | None  # whole machine, USD
    spot: bool = False
    region: str = ""
    hardware: HardwareProfile | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Instance:
    id: str
    provider: str
    offer: Offer | None = None
    workdir: str = ""
    host: str | None = None
    started: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobSpec:
    name: str
    bundle_dir: str
    command: list[str] = field(default_factory=list)  # empty -> provider default runner command
    env: dict[str, str] = field(default_factory=dict)
    max_hours: float | None = None
    resume: bool = False


@dataclass
class JobHandle:
    id: str
    instance: Instance
    log_path: str = ""
    started: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


# Statuses returned by ComputeProvider.status
RUNNING, SUCCEEDED, FAILED, PREEMPTED, MANUAL = "running", "succeeded", "failed", "preempted", "manual"


class ComputeProvider(abc.ABC):
    name = "provider"

    @abc.abstractmethod
    def list_offers(self, req: ResourceRequest) -> list[Offer]: ...

    @abc.abstractmethod
    def provision(self, offer: Offer) -> Instance: ...

    @abc.abstractmethod
    def sync_workspace(self, inst: Instance, bundle_dir: Path) -> None: ...

    @abc.abstractmethod
    def launch(self, inst: Instance, job: JobSpec) -> JobHandle: ...

    @abc.abstractmethod
    def status(self, job: JobHandle) -> str: ...

    @abc.abstractmethod
    def stream_logs(self, job: JobHandle) -> Iterator[str]:
        """Yield log lines until the job stops."""

    @abc.abstractmethod
    def fetch_artifacts(self, job: JobHandle, dest: Path) -> None: ...

    @abc.abstractmethod
    def cancel(self, job: JobHandle) -> None: ...

    @abc.abstractmethod
    def teardown(self, inst: Instance) -> None: ...
