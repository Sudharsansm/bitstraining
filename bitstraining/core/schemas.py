"""Typed, serializable schemas shared by every layer.

Everything the planner produces is plain data (`ExecutionPlan`), so plans can be
inspected, diffed, saved, shipped to a remote machine and replayed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import types
import typing
from dataclasses import dataclass, field
from typing import Any, TypeVar, Union, get_type_hints

from bitstraining.core.errors import ConfigError

T = TypeVar("T")


# --------------------------------------------------------------------------- helpers
def _build(tp: Any, value: Any, path: str) -> Any:
    """Recursively coerce `value` into annotation `tp` (dataclasses, lists, dicts, Optional)."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if value is None:
        return None
    if origin in (Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _build(non_none[0], value, path)
        return value
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"'{path}' must be a mapping, got {type(value).__name__}")
        return from_dict(tp, value, path)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"'{path}' must be a list, got {type(value).__name__}")
        return [_build(args[0], v, f"{path}[{i}]") for i, v in enumerate(value)] if args else value
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"'{path}' must be a mapping, got {type(value).__name__}")
        return dict(value)
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(
                f"'{path}' must be a number, got {value!r}",
                fix="remove quotes / booleans from numeric fields",
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigError(f"'{path}' must be finite, got {value}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"'{path}' must be an integer, got {value!r}")
        if isinstance(value, float):
            if not value.is_integer():
                raise ConfigError(f"'{path}' must be an integer, got {value}")
            return int(value)
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"'{path}' must be true/false, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"'{path}' must be a string, got {value!r}")
        return value
    return value


def from_dict(cls: type[T], data: dict[str, Any], path: str = "") -> T:
    """Build dataclass `cls` from a dict, rejecting unknown keys with a suggestion."""
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = set(data) - names
    if unknown:
        import difflib

        key = sorted(unknown)[0]
        close = difflib.get_close_matches(key, sorted(names), n=1)
        where = f"{path}." if path else ""
        raise ConfigError(
            f"unknown field '{where}{key}'",
            fix=f"did you mean '{close[0]}'?" if close else f"valid fields: {sorted(names)}",
        )
    kwargs = {}
    for name in names:
        if name in data:
            kwargs[name] = _build(hints[name], data[name], f"{path}.{name}" if path else name)
    return cls(**kwargs)  # type: ignore[arg-type]


class Serializable:
    """Mixin: dict/json round-tripping and stable content hashing."""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        return from_dict(cls, data)

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, **kw)

    def content_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- hardware
@dataclass
class GPUInfo(Serializable):
    name: str = "unknown"
    vram_gb: float = 0.0
    compute_capability: str | None = None
    bf16: bool = False
    fp8: bool = False
    peak_tflops_bf16: float = 0.0  # dense tensor-core peak
    mem_bw_gbs: float = 0.0
    h2d_gbs: float = 12.0  # measured/assumed host<->device bandwidth
    nvlink: bool = False
    source: str = "database"  # detected | database | measured


@dataclass
class HardwareProfile(Serializable):
    name: str = "local"
    os: str = ""
    cpu_model: str = ""
    cpu_cores: int = 1
    ram_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_read_gbs: float | None = None
    gpu_vendor: str = "none"  # nvidia | apple | amd | none
    unified_memory: bool = False
    gpus: list[GPUInfo] = field(default_factory=list)
    price_per_hour: float | None = None  # for the whole machine
    provider: str = "local"
    measured: bool = False

    @property
    def n_gpus(self) -> int:
        return len(self.gpus)

    @property
    def vram_gb(self) -> float:
        """Usable device memory per GPU (min across devices). Unified memory shares RAM."""
        if not self.gpus:
            return 0.0
        return min(g.vram_gb for g in self.gpus)

    @property
    def key(self) -> str:
        """Stable key for caches and calibration."""
        if self.gpus:
            return f"{self.gpus[0].name}x{self.n_gpus}".lower().replace(" ", "-")
        return f"cpu-{self.cpu_cores}c"


# --------------------------------------------------------------------------- model / data
@dataclass
class ModelInfo(Serializable):
    name: str = ""
    architecture: str = "llama"
    params: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    intermediate_size: int = 0
    vocab_size: int = 0
    tie_embeddings: bool = False
    max_position: int = 4096
    num_experts: int = 1
    active_experts: int = 1
    gated_mlp: bool = True  # SwiGLU-style (3 matrices) vs plain 2-matrix MLP
    head_dim_override: int = 0  # for architectures where head_dim != hidden/heads
    source: str = "preset"  # config | preset | name-heuristic | hub

    @property
    def head_dim(self) -> int:
        return self.head_dim_override or self.hidden_size // max(self.num_heads, 1)

    @property
    def embed_params(self) -> int:
        n = self.vocab_size * self.hidden_size
        return n if self.tie_embeddings else 2 * n

    @property
    def nonembed_params(self) -> int:
        return max(self.params - self.embed_params, 0)


@dataclass
class DatasetStats(Serializable):
    num_rows: int = 0
    total_tokens: int = 0
    mean_len: float = 0.0
    p50_len: int = 0
    p95_len: int = 0
    max_len: int = 0
    truncation_rate: float = 0.0
    empty_rate: float = 0.0
    duplicate_rate: float = 0.0
    schema: str = "unknown"
    data_hash: str = ""
    lengths_sample: list[int] = field(default_factory=list)
    token_counter: str = "chars/4 estimate"


# --------------------------------------------------------------------------- run spec
@dataclass
class ModelSpec(Serializable):
    base: str = ""
    trust_remote_code: bool = False
    revision: str | None = None


@dataclass
class DataSpec(Serializable):
    train: str = ""
    eval: str | None = None
    format: str = "auto"
    max_length: int = 2048
    eval_split: float = 0.0
    packing: str = "auto"  # auto | on | off
    dedup: bool = True


@dataclass
class GoalSpec(Serializable):
    task: str = "sft"  # sft | dpo | orpo | kto | pretrain
    objective: str = "time"  # time | cost | memory
    epochs: float = 3.0
    global_batch_size: int = 16
    learning_rate: float = 2e-4
    time_budget_hours: float | None = None
    budget_usd: float | None = None
    allow_quantization: bool = True
    methods: list[str] | None = None  # restrict: full | lora | qlora | dora
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    seed: int = 42


@dataclass
class ComputeSpec(Serializable):
    provider: str = "local"
    gpu: str | None = None  # plan for named hardware, e.g. "a100-80gb"
    n_gpus: int | None = None
    max_price_per_hour: float | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSpec(Serializable):
    """What the user asks for: a goal plus constraints, not a hand-tuned config."""

    name: str = "run"
    model: ModelSpec = field(default_factory=ModelSpec)
    data: DataSpec = field(default_factory=DataSpec)
    goal: GoalSpec = field(default_factory=GoalSpec)
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    output: str = "./output"
    engine: str = "auto"
    extras: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- plan
@dataclass
class Estimate(Serializable):
    fits: bool = True
    peak_gb_per_gpu: float = 0.0
    breakdown_gb: dict[str, float] = field(default_factory=dict)
    host_ram_gb: float = 0.0
    raw_tokens_per_s: float = 0.0
    useful_tokens_per_s: float = 0.0
    step_time_s: float = 0.0
    total_hours: float = 0.0
    cost_usd: float | None = None
    confidence: str = "low"  # low | medium | high
    bottleneck: str = "compute"  # compute | transfer
    notes: list[str] = field(default_factory=list)


@dataclass
class ExecutionPlan(Serializable):
    engine: str = "hf_peft"
    method: str = "lora"  # full | lora | qlora | dora
    precision: str = "bf16"  # bf16 | fp16 | fp32
    quant: str = "none"  # none | nf4
    layer_stream: bool = False
    activation: str = "none"  # none | selective | full
    optimizer: str = "adamw"  # adamw | adamw8bit | paged_adamw8bit
    micro_batch: int = 1
    grad_accum: int = 1
    n_gpus: int = 1
    parallelism: str = "single"  # single | ddp | fsdp
    packing: bool = False
    max_length: int = 2048
    attention: str = "sdpa"  # flash | sdpa | eager
    fused_ce: bool = False
    compile: bool = False
    dataloader_workers: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    epochs: float = 3.0
    total_steps: int = 0
    learning_rate: float = 2e-4
    seed: int = 42
    hardware: str = ""
    model: str = ""
    estimate: Estimate = field(default_factory=Estimate)

    def describe(self) -> str:
        bits = [
            f"{self.engine}",
            f"{self.method}{'+stream' if self.layer_stream else ''}",
            self.precision,
            f"ckpt={self.activation}",
            f"opt={self.optimizer}",
            f"mb={self.micro_batch}x{self.grad_accum}",
            f"gpus={self.n_gpus}/{self.parallelism}",
            f"pack={'on' if self.packing else 'off'}",
            f"attn={self.attention}",
        ]
        return " ".join(bits)


@dataclass
class Capabilities(Serializable):
    """What an engine can do on given hardware; the planner only searches inside this."""

    name: str = ""
    available: bool = True
    reason: str = ""  # why unavailable
    methods: list[str] = field(default_factory=lambda: ["lora"])
    precisions: list[str] = field(default_factory=lambda: ["bf16", "fp16"])
    activation_modes: list[str] = field(default_factory=lambda: ["none", "full"])
    optimizers: list[str] = field(default_factory=lambda: ["adamw"])
    attention: list[str] = field(default_factory=lambda: ["sdpa"])
    parallelism: list[str] = field(default_factory=lambda: ["single"])
    supports_packing: bool = False
    supports_layer_stream: bool = False
    supports_fused_ce: bool = False
    supports_compile: bool = False
    tasks: list[str] = field(default_factory=lambda: ["sft"])
    platforms: list[str] = field(default_factory=lambda: ["cuda"])
    speed_prior: float = 1.0  # relative throughput prior; replaced by calibration when probed
    speed_prior_is_measured: bool = False
