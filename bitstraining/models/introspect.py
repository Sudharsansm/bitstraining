"""Derive exact parameter counts and layer shapes from a model's config, never its weights.

Resolution order for `load_model_info(name_or_path)`:
1. a local directory containing config.json
2. a built-in preset table (public architecture hyper-parameters)
3. the Hugging Face Hub's config.json (small download; skipped when offline)
4. a size parsed from the name ("...-7b"), mapped onto the nearest preset's shape
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from bitstraining.core.errors import ConfigError
from bitstraining.core.schemas import ModelInfo

# name-fragment -> hyper-parameters
_PRESETS: dict[str, dict[str, Any]] = {
    "smollm2-135m": dict(h=576, L=30, heads=9, kv=3, inter=1536, vocab=49152, tied=True),
    "gpt2": dict(h=768, L=12, heads=12, kv=12, inter=3072, vocab=50257, tied=True, gated=False, arch="gpt2", max_pos=1024),
    "qwen2.5-0.5b": dict(h=896, L=24, heads=14, kv=2, inter=4864, vocab=151936, tied=True, arch="qwen2"),
    "tinyllama-1.1b": dict(h=2048, L=22, heads=32, kv=4, inter=5632, vocab=32000, tied=False),
    "llama-3.2-1b": dict(h=2048, L=16, heads=32, kv=8, inter=8192, vocab=128256, tied=True),
    "qwen2.5-1.5b": dict(h=1536, L=28, heads=12, kv=2, inter=8960, vocab=151936, tied=True, arch="qwen2"),
    "llama-3.2-3b": dict(h=3072, L=28, heads=24, kv=8, inter=8192, vocab=128256, tied=True),
    "qwen2.5-3b": dict(h=2048, L=36, heads=16, kv=2, inter=11008, vocab=151936, tied=True, arch="qwen2"),
    "phi-3-mini": dict(h=3072, L=32, heads=32, kv=32, inter=8192, vocab=32064, tied=False, arch="phi3"),
    "mistral-7b": dict(h=4096, L=32, heads=32, kv=8, inter=14336, vocab=32000, tied=False, arch="mistral"),
    "qwen2.5-7b": dict(h=3584, L=28, heads=28, kv=4, inter=18944, vocab=152064, tied=False, arch="qwen2"),
    "llama-3.1-8b": dict(h=4096, L=32, heads=32, kv=8, inter=14336, vocab=128256, tied=False, max_pos=131072),
    "llama-3-8b": dict(h=4096, L=32, heads=32, kv=8, inter=14336, vocab=128256, tied=False, max_pos=8192),
    "qwen2.5-14b": dict(h=5120, L=48, heads=40, kv=8, inter=13824, vocab=152064, tied=False, arch="qwen2"),
    "llama-3.1-70b": dict(h=8192, L=80, heads=64, kv=8, inter=28672, vocab=128256, tied=False, max_pos=131072),
}

_STRIP = re.compile(r"-(instruct|chat|it|base|hf|bnb-4bit|4bit|awq|gptq|v\d+(\.\d+)*)$")


def compute_params(
    h: int,
    L: int,
    heads: int,
    kv: int,
    inter: int,
    vocab: int,
    tied: bool,
    head_dim: int | None = None,
    gated: bool = True,
    experts: int = 1,
) -> int:
    """Exact parameter count of a decoder-only transformer from its shape (biases ignored)."""
    hd = head_dim or h // heads
    attn = h * heads * hd + 2 * h * kv * hd + heads * hd * h
    mlp_mats = 3 if gated else 2
    mlp = mlp_mats * h * inter * experts + (h * experts if experts > 1 else 0)
    per_layer = attn + mlp + 2 * h
    embed = vocab * h * (1 if tied else 2)
    return L * per_layer + h + embed


def _info(name: str, source: str, **p: Any) -> ModelInfo:
    experts = p.get("experts", 1)
    params = compute_params(
        p["h"], p["L"], p["heads"], p["kv"], p["inter"], p["vocab"], p["tied"],
        head_dim=p.get("head_dim"), gated=p.get("gated", True), experts=experts,
    )
    return ModelInfo(
        name=name,
        architecture=p.get("arch", "llama"),
        params=params,
        hidden_size=p["h"],
        num_layers=p["L"],
        num_heads=p["heads"],
        num_kv_heads=p["kv"],
        intermediate_size=p["inter"],
        vocab_size=p["vocab"],
        tie_embeddings=p["tied"],
        max_position=p.get("max_pos", 4096),
        num_experts=experts,
        active_experts=p.get("active", 1),
        gated_mlp=p.get("gated", True),
        head_dim_override=p.get("head_dim") or 0,
        source=source,
    )


def from_hf_config(cfg: dict[str, Any], name: str = "", source: str = "config") -> ModelInfo:
    """Build ModelInfo from a Hugging Face `config.json` dictionary."""
    mtype = cfg.get("model_type", "llama")
    if mtype == "gpt2":
        return _info(
            name, source, h=cfg["n_embd"], L=cfg["n_layer"], heads=cfg["n_head"], kv=cfg["n_head"],
            inter=cfg.get("n_inner") or 4 * cfg["n_embd"], vocab=cfg["vocab_size"],
            tied=True, gated=False, arch="gpt2", max_pos=cfg.get("n_positions", 1024),
        )
    try:
        h = cfg["hidden_size"]
        heads = cfg["num_attention_heads"]
        p = dict(
            h=h,
            L=cfg["num_hidden_layers"],
            heads=heads,
            kv=cfg.get("num_key_value_heads") or heads,
            inter=cfg.get("intermediate_size") or 4 * h,
            vocab=cfg["vocab_size"],
            tied=bool(cfg.get("tie_word_embeddings", False)),
            arch=mtype,
            max_pos=cfg.get("max_position_embeddings", 4096),
        )
    except KeyError as e:
        raise ConfigError(
            f"config.json for '{name}' lacks {e}", fix="is this a decoder-only transformer config?"
        ) from e
    if cfg.get("head_dim"):
        p["head_dim"] = int(cfg["head_dim"])
    experts = cfg.get("num_local_experts") or cfg.get("num_experts") or 1
    if experts and experts > 1:
        p["experts"] = int(experts)
        p["active"] = int(cfg.get("num_experts_per_tok", 2))
        p["inter"] = cfg.get("moe_intermediate_size", p["inter"])
    return _info(name, source, **p)


def _normalize(name: str) -> str:
    base = name.rstrip("/").split("/")[-1].lower().replace("_", "-")
    prev = None
    while prev != base:
        prev = base
        base = _STRIP.sub("", base)
    return base.replace("meta-llama-", "llama-")


def _preset_for(name: str) -> ModelInfo | None:
    norm = _normalize(name)
    for key in sorted(_PRESETS, key=len, reverse=True):
        if key in norm:
            return _info(name, "preset", **_PRESETS[key])
    return None


def _size_from_name(name: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])(?![a-zA-Z])", name.split("/")[-1])
    if not m:
        return None
    val = float(m.group(1))
    return val * 1e9 if m.group(2).lower() == "b" else val * 1e6


def _heuristic(name: str) -> ModelInfo | None:
    size = _size_from_name(name)
    if size is None:
        return None
    nearest = min(_PRESETS, key=lambda k: abs(_info(k, "x", **_PRESETS[k]).params - size))
    info = _info(name, "name-heuristic", **_PRESETS[nearest])
    info.params = int(size)  # trust the advertised size; shape comes from the nearest preset
    return info


def _hub_config(name: str, timeout: float = 5.0) -> dict[str, Any] | None:
    if os.environ.get("BITSTRAINING_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1":
        return None
    if "/" not in name:
        return None
    req = urllib.request.Request(f"https://huggingface.co/{name}/raw/main/config.json")
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def load_model_info(name_or_path: str, allow_network: bool = True) -> ModelInfo:
    p = Path(name_or_path)
    if p.is_dir() and (p / "config.json").exists():
        return from_hf_config(json.loads((p / "config.json").read_text()), name_or_path, "config")
    info = _preset_for(name_or_path)
    if info:
        return info
    if allow_network:
        cfg = _hub_config(name_or_path)
        if cfg:
            try:
                return from_hf_config(cfg, name_or_path, "hub")
            except ConfigError:
                pass
    info = _heuristic(name_or_path)
    if info:
        return info
    raise ConfigError(
        f"cannot determine the architecture of '{name_or_path}'",
        fix="use a local directory with config.json, a known model, or a name containing the size (e.g. '-7b')",
    )


def lora_param_count(m: ModelInfo, r: int, target: str = "all-linear") -> int:
    """Trainable LoRA parameters: r * (in + out) for each targeted linear layer."""
    hd = m.head_dim
    h, q_out, kv_out = m.hidden_size, m.num_heads * hd, m.num_kv_heads * hd
    attn = {
        "q_proj": h + q_out,
        "k_proj": h + kv_out,
        "v_proj": h + kv_out,
        "o_proj": q_out + h,
    }
    mlp_dim = h + m.intermediate_size
    mlp = {"up_proj": mlp_dim, "down_proj": mlp_dim}
    if m.gated_mlp:
        mlp["gate_proj"] = mlp_dim
    experts = max(m.num_experts, 1)
    if target == "all-linear":
        per_layer = sum(attn.values()) + sum(mlp.values()) * experts
    elif target == "attention":
        per_layer = sum(attn.values())
    elif target == "qv":
        per_layer = attn["q_proj"] + attn["v_proj"]
    else:
        names = [t.strip() for t in target.split(",")]
        per_layer = sum(attn.get(n, 0) + mlp.get(n, 0) * experts for n in names)
    return r * per_layer * m.num_layers
