"""Roofline-style throughput model.

  time per token = FLOPs per token / (peak FLOPs * MFU * batch-saturation)  x method overhead

FLOPs per token = multiplier * (2 * matmul params + attention) + adapter FLOPs, where the
multiplier is 3 for full fine-tuning (forward + both backward matmuls) and 2 for frozen-base
methods (the weight-gradient matmuls are skipped), plus recompute for activation checkpointing.

MFU (model FLOPs utilization) is a prior: 0.42 with a fused attention kernel on a modern GPU.
It is a heuristic, not a measurement; calibration against a real probe replaces it
(see planner.calibration).

Layer streaming adds a transfer bound: every micro-batch re-reads the frozen stack over PCIe
(forward + backward), so step time = max(compute, transfer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bitstraining.core.schemas import (
    Capabilities,
    ExecutionPlan,
    GPUInfo,
    HardwareProfile,
    ModelInfo,
)
from bitstraining.models.introspect import lora_param_count
from bitstraining.planner.memory import layer_params
from bitstraining.strategies.catalog import METHODS, weight_bytes_per_param

BASE_MFU = 0.42
EAGER_PENALTY = 0.70
FUSED_CE_BONUS = 1.03
FP32_PENALTY = 0.25
H2D_EFFICIENCY = 0.8
STREAM_OVERLAP_LOSS = 0.30  # heuristic: share of the shorter phase NOT hidden by double-buffering
SAT_REF_TOKENS = 16384
UNKNOWN_RAM_GB = 8.0  # assumed when system RAM could not be detected (the doctor warns)
FALLBACK_PEAK_TFLOPS = 20.0  # used when the GPU is not in the database

_SCALE_EFF = {  # per doubling of GPUs: (nvlink, pcie)
    "ddp": (0.96, 0.90),
    "fsdp": (0.92, 0.80),
}


@dataclass
class Throughput:
    raw_tokens_per_s: float  # includes padding tokens
    useful_tokens_per_s: float
    micro_batch_time_s: float
    bottleneck: str  # compute | transfer
    mfu: float
    used_peak_tflops: float


def matmul_params(m: ModelInfo) -> int:
    """Parameters that participate in per-token matmuls (active experts only, plus LM head)."""
    p = m.nonembed_params
    if m.num_experts > 1:
        mats = 3 if m.gated_mlp else 2
        per_expert = mats * m.hidden_size * m.intermediate_size
        p -= m.num_layers * per_expert * (m.num_experts - max(m.active_experts, 1))
    return p + m.hidden_size * m.vocab_size


def flops_per_token(m: ModelInfo, plan: ExecutionPlan, attn_len: float) -> float:
    method = METHODS[plan.method]
    attn_fwd = 2.0 * m.num_layers * attn_len * m.num_heads * m.head_dim  # causal
    fwd = 2.0 * matmul_params(m) + attn_fwd
    mult = method.compute_multiplier
    if plan.activation == "full":
        mult += 1.0
    elif plan.activation == "selective":
        mult += 0.5
    adapter = 0.0
    if not method.full_weights_trainable:
        adapter = 6.0 * lora_param_count(m, plan.lora_r, plan.target_modules)
    return mult * fwd + adapter


def device_for(hw: HardwareProfile) -> GPUInfo:
    """The device the planner models: GPU 0, or a CPU stand-in for GPU-less machines."""
    if hw.gpus:
        return hw.gpus[0]
    ram = hw.ram_gb if hw.ram_gb > 0 else UNKNOWN_RAM_GB  # undetected RAM must not mean "0 GB fits nothing"
    return GPUInfo(
        name="cpu", vram_gb=round(ram * 0.6, 2), peak_tflops_bf16=max(hw.cpu_cores * 0.04, 0.1),
        mem_bw_gbs=40.0, bf16=False, h2d_gbs=20.0, source="database",
    )


def estimate_throughput(
    m: ModelInfo,
    plan: ExecutionPlan,
    hw: HardwareProfile,
    caps: Capabilities,
    mean_len: float,
    pad_eff: float,
    speed_ratio: float = 1.0,
) -> Throughput:
    gpu = device_for(hw)
    peak = gpu.peak_tflops_bf16 or FALLBACK_PEAK_TFLOPS
    mfu = BASE_MFU
    if hw.gpu_vendor == "apple":
        mfu = 0.25
    if plan.attention == "eager":
        mfu *= EAGER_PENALTY
    if plan.fused_ce:
        mfu *= FUSED_CE_BONUS
    if plan.precision == "fp32":
        mfu *= FP32_PENALTY

    # Tokens actually computed per micro-batch (padding included unless packed).
    tokens_mb = plan.micro_batch * (plan.max_length if plan.packing else max(mean_len / max(pad_eff, 1e-3), 1))
    k_sat = max(64.0, 8.0 * peak)
    sat = (tokens_mb / (tokens_mb + k_sat)) / (SAT_REF_TOKENS / (SAT_REF_TOKENS + k_sat))
    sat = min(sat, 1.0)

    fpt = flops_per_token(m, plan, attn_len=mean_len)
    method = METHODS[plan.method]
    eff_flops = peak * 1e12 * mfu * sat
    tps_single = eff_flops / (fpt * method.time_overhead) * caps.speed_prior * speed_ratio

    n = max(plan.n_gpus, 1)
    scale = 1.0
    if n > 1:
        nv, pcie = _SCALE_EFF.get(plan.parallelism, _SCALE_EFF["ddp"])
        scale = n * (nv if gpu.nvlink else pcie) ** math.log2(n)
    t_compute = tokens_mb / tps_single

    bottleneck = "compute"
    t_mb = t_compute
    if plan.layer_stream:
        wb = weight_bytes_per_param(method.quant, plan.precision)
        stack_bytes = layer_params(m) * m.num_layers * wb
        h2d = (gpu.h2d_gbs or 12.0) * 1e9 * H2D_EFFICIENCY
        t_transfer = 2.0 * stack_bytes / h2d  # forward + backward pass over the stack
        # Perfect overlap would give max(compute, transfer). Real double-buffering never hides
        # everything, so a fraction of the shorter phase is added back (heuristic, see constant).
        t_mb = max(t_compute, t_transfer) + STREAM_OVERLAP_LOSS * min(t_compute, t_transfer)
        if t_transfer > t_compute:
            bottleneck = "transfer"
    raw = tokens_mb / t_mb * scale
    return Throughput(
        raw_tokens_per_s=raw,
        useful_tokens_per_s=raw * pad_eff,  # pad_eff = useful / computed tokens for this batching mode
        micro_batch_time_s=t_mb,
        bottleneck=bottleneck,
        mfu=mfu * sat,
        used_peak_tflops=peak,
    )
