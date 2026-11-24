"""Tuning cache keyed by (hardware, model, strategy) so repeat runs start instantly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bitstraining.core import paths
from bitstraining.core.schemas import ExecutionPlan, HardwareProfile


def cache_key(hw: HardwareProfile, model: str, plan: ExecutionPlan, seq_len: int) -> str:
    return "|".join([
        hw.key, model, plan.engine, plan.method, plan.precision, plan.activation, plan.optimizer,
        f"pack={int(plan.packing)}", f"stream={int(plan.layer_stream)}", f"seq={seq_len}",
    ])


def load(path: Path | None = None) -> dict[str, Any]:
    f = path or paths.tuning_cache()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return {}


def get(key: str, path: Path | None = None) -> dict[str, Any] | None:
    return load(path).get(key)


def put(key: str, value: dict[str, Any], path: Path | None = None) -> None:
    f = path or paths.tuning_cache()
    data = load(f)
    data[key] = value
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data, indent=2, sort_keys=True))
