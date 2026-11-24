"""Analytic GPU-memory model for fine-tuning.

Peak memory per device is the sum of:

  weights      base parameters at their storage precision (bf16 = 2 B, NF4 = 0.53 B)
  master       fp32 master weights (full fine-tuning under mixed precision only)
  gradients    per trainable parameter (adapters are kept in fp32)
  optimizer    per trainable parameter: AdamW 8 B, 8-bit Adam 2 B
  activations  per Korthikanti et al. (2022): s*b*h*(34 + 5*a*s/h) bytes per layer, where the
               5*a*s/h attention-score term disappears with fused (flash) attention. The 34 is
               split into an attention part (15) and an MLP part (19, scaled by intermediate/4h).
               Checkpointing keeps only 2*s*b*h per checkpointed layer plus one live layer.
  logits       b*s*V*10 B unfused (bf16 logits + fp32 upcast + fp32 grad); chunked fused loss ~1.5 B
  context      CUDA context and kernels workspace (~0.8 GB)

Sharding (FSDP/ZeRO-3) divides weights, gradients, master and optimizer state by the number of
GPUs and adds one gathered layer. Layer streaming keeps only embeddings, the LM head, adapters and
two layer buffers resident; the frozen stack lives in host RAM.

These are estimates. `planner.calibration` stores measured/predicted ratios from real probes
and the planner applies them, and the engine probe verifies the final plan before a long run.
"""

from __future__ import annotations

from bitstraining.core.schemas import ExecutionPlan, ModelInfo
from bitstraining.models.introspect import lora_param_count
from bitstraining.strategies.catalog import (
    METHODS,
    OPTIMIZER_STATE_BYTES,
    weight_bytes_per_param,
)

GB = 1024**3
CONTEXT_GB_CUDA = 0.8
CONTEXT_GB_OTHER = 0.3
FRAGMENTATION = 1.08


def layer_params(m: ModelInfo) -> int:
    return m.nonembed_params // max(m.num_layers, 1)


def trainable_params(m: ModelInfo, plan: ExecutionPlan) -> int:
    if METHODS[plan.method].full_weights_trainable:
        return m.params
    return lora_param_count(m, plan.lora_r, plan.target_modules)


def activation_bytes(m: ModelInfo, plan: ExecutionPlan, seq_len: int) -> float:
    """Saved-activation bytes for one micro-batch under the plan's checkpointing mode."""
    b, s, h = plan.micro_batch, seq_len, m.hidden_size
    mlp_scale = m.intermediate_size / (4 * h)
    mlp_term = 19 * mlp_scale * (max(m.active_experts, 1) if m.num_experts > 1 else 1)
    per_layer = s * b * h * (15 + mlp_term)
    if plan.attention == "eager":
        per_layer += 5 * m.num_heads * s * s * b  # materialized attention scores
    ckpt_layer = 2 * s * b * h  # saved layer input only
    if plan.activation == "full":
        return m.num_layers * ckpt_layer + per_layer
    if plan.activation == "selective":
        half = m.num_layers // 2
        return (m.num_layers - half) * per_layer + half * ckpt_layer + per_layer
    return m.num_layers * per_layer


def estimate_memory(
    m: ModelInfo,
    plan: ExecutionPlan,
    seq_len: int,
    device_kind: str = "cuda",
) -> tuple[dict[str, float], float]:
    """Return (per-device breakdown in GB including a 'total' key, host RAM needed in GB)."""
    method = METHODS[plan.method]
    n = max(plan.n_gpus, 1)
    shard = n if plan.parallelism == "fsdp" else 1
    wb = weight_bytes_per_param(method.quant, plan.precision)
    layer_stored = layer_params(m) * wb
    n_train = trainable_params(m, plan)

    notes_host = 0.0
    if plan.layer_stream:
        resident = m.embed_params * (2.0 if method.quant == "nf4" else wb) + 2 * layer_stored
        weights = resident
        notes_host = layer_params(m) * m.num_layers * wb + 2 * GB
    else:
        weights = m.params * wb / shard
        notes_host = m.params * 2.0 + 1 * GB  # bf16 staging while loading/quantizing

    master = 0.0
    if method.full_weights_trainable:
        master = n_train * 4.0 / shard
        grads = n_train * 2.0 / shard
        opt = n_train * OPTIMIZER_STATE_BYTES[plan.optimizer] / shard
        adapters = 0.0
    else:
        adapters = n_train * 4.0  # fp32 adapter weights
        grads = n_train * 4.0
        opt = n_train * OPTIMIZER_STATE_BYTES[plan.optimizer]

    gathered = layer_stored * 2 if (plan.parallelism == "fsdp" and n > 1) else 0.0
    dequant = layer_params(m) * 2.0 if method.quant == "nf4" else 0.0  # one bf16 layer workspace

    acts = activation_bytes(m, plan, seq_len)
    logit_bytes_per = 1.5 if plan.fused_ce else 10.0
    logits = plan.micro_batch * seq_len * m.vocab_size * logit_bytes_per

    context = (CONTEXT_GB_CUDA if device_kind == "cuda" else CONTEXT_GB_OTHER) * GB
    static = weights + master + adapters + grads + opt + gathered + dequant
    dynamic = (acts + logits) * FRAGMENTATION
    parts = {
        "weights": weights + gathered,
        "master_weights": master,
        "adapters": adapters,
        "gradients": grads,
        "optimizer": opt,
        "dequant_workspace": dequant,
        "activations": acts * FRAGMENTATION,
        "logits": logits * FRAGMENTATION,
        "context": context,
    }
    breakdown = {k: v / GB for k, v in parts.items()}
    breakdown["total"] = (static + dynamic + context) / GB
    if plan.optimizer == "paged_adamw8bit":
        notes_host += 0.0  # paging is opportunistic; host usage grows only under pressure
    return breakdown, notes_host / GB
