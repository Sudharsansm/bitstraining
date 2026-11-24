"""Loading, validating and saving RunSpec YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from bitstraining.core.errors import ConfigError
from bitstraining.core.schemas import ExecutionPlan, RunSpec, from_dict

_TASKS = {"sft", "dpo", "orpo", "kto", "pretrain"}
_OBJECTIVES = {"time", "cost", "memory"}
_METHODS = {"full", "lora", "qlora", "dora"}


def load_spec(path: str | Path) -> RunSpec:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}", fix="run `bitstraining init` first")
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{p} is not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{p} must contain a mapping at the top level")
    return spec_from_dict(raw)


def spec_from_dict(raw: dict[str, Any]) -> RunSpec:
    spec = from_dict(RunSpec, raw)
    validate_spec(spec)
    return spec


def validate_spec(spec: RunSpec) -> None:
    """Hard validation: raises ConfigError. Softer advice lives in the doctor."""
    g = spec.goal
    if not spec.model.base:
        raise ConfigError("model.base is required", fix="set model.base to a Hugging Face id")
    if not spec.data.train:
        raise ConfigError("data.train is required", fix="point data.train at a JSONL/CSV file")
    if g.task not in _TASKS:
        raise ConfigError(f"goal.task '{g.task}' unsupported", fix=f"use one of {sorted(_TASKS)}")
    if g.objective not in _OBJECTIVES:
        raise ConfigError(f"goal.objective '{g.objective}' unsupported", fix=f"use {sorted(_OBJECTIVES)}")
    if g.methods:
        bad = [m for m in g.methods if m not in _METHODS]
        if bad:
            raise ConfigError(f"unknown methods {bad}", fix=f"use {sorted(_METHODS)}")
    if g.epochs <= 0:
        raise ConfigError("goal.epochs must be > 0")
    if g.global_batch_size < 1:
        raise ConfigError("goal.global_batch_size must be >= 1")
    if not 0 < g.learning_rate < 1:
        raise ConfigError(f"goal.learning_rate {g.learning_rate} is outside (0, 1)")
    if g.lora_r < 1:
        raise ConfigError("goal.lora_r must be >= 1")
    if spec.data.max_length < 8:
        raise ConfigError("data.max_length must be >= 8")
    if spec.data.packing not in {"auto", "on", "off"}:
        raise ConfigError("data.packing must be auto, on or off")
    if not 0 <= spec.data.eval_split < 0.5:
        raise ConfigError("data.eval_split must be in [0, 0.5)")
    if g.budget_usd is not None and g.budget_usd <= 0:
        raise ConfigError("goal.budget_usd must be > 0")
    if g.time_budget_hours is not None and g.time_budget_hours <= 0:
        raise ConfigError("goal.time_budget_hours must be > 0")


def dump_spec(spec: RunSpec, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False))


def load_plan(path: str | Path) -> ExecutionPlan:
    import json

    return from_dict(ExecutionPlan, json.loads(Path(path).read_text()))


def save_plan(plan: ExecutionPlan, path: str | Path) -> None:
    import json

    Path(path).write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
