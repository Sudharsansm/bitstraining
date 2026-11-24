"""Method catalog: how each fine-tuning method changes memory and compute.

These are *modeling constants*, documented so they can be audited and corrected by the
calibration loop. Values marked (heuristic) are engineering estimates, not measurements.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    name: str
    full_weights_trainable: bool  # all base weights receive gradients
    quant: str  # "none" | "nf4"
    compute_multiplier: float  # x forward FLOPs for fwd+bwd (frozen base skips weight-grad matmuls)
    time_overhead: float  # extra wall-time factor (dequantization etc.)  (heuristic)
    description: str


METHODS: dict[str, Method] = {
    "full": Method("full", True, "none", 3.0, 1.0, "Update every weight. Highest memory, no adapter."),
    "lora": Method("lora", False, "none", 2.0, 1.0, "Low-rank adapters on a frozen bf16 base."),
    "qlora": Method("qlora", False, "nf4", 2.0, 1.20, "LoRA on a 4-bit NF4-quantized frozen base. (overhead: heuristic)"),
    "dora": Method("dora", False, "none", 2.0, 1.25, "Weight-decomposed LoRA. (overhead: heuristic)"),
}

# Bytes of optimizer state per *trainable* parameter.
OPTIMIZER_STATE_BYTES: dict[str, float] = {
    "adamw": 8.0,  # fp32 m and v
    "adamw8bit": 2.0,  # 8-bit m and v
    "paged_adamw8bit": 2.0,  # same, and pages to host memory under pressure
}

NF4_BYTES_PER_PARAM = 0.53  # 4 bits + per-block constants (double quantization ~0.516)


def weight_bytes_per_param(quant: str, precision: str) -> float:
    if quant == "nf4":
        return NF4_BYTES_PER_PARAM
    return {"bf16": 2.0, "fp16": 2.0, "fp32": 4.0}[precision]
