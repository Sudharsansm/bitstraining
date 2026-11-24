"""Unsloth adapter (single-GPU fast path).

EXPERIMENTAL and not exercised by CI. Reuses the baseline engine's data handling and Trainer loop;
only model loading differs. Whether it is actually faster on *your* GPU is not assumed: Bitstraining
gives it the same speed prior as the baseline (1.0) and relies on `bitstraining autotune --bake-off`
to measure both engines before the planner prefers one.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from bitstraining.core.errors import EngineUnavailable
from bitstraining.core.schemas import Capabilities, ExecutionPlan, HardwareProfile, RunSpec
from bitstraining.engines.hf_peft import HFPeftEngine

_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class UnslothEngine(HFPeftEngine):
    name = "unsloth"

    def capabilities(self, hw: HardwareProfile, assume_installed: bool = False) -> Capabilities:
        cuda = hw.gpu_vendor == "nvidia"
        have = assume_installed or importlib.util.find_spec("unsloth") is not None
        ok = cuda and have
        return Capabilities(
            name=self.name,
            available=ok,
            reason="" if ok else ("needs an NVIDIA GPU" if not cuda else "unsloth not installed (pip install 'bitstraining[fast]')"),
            methods=["lora", "qlora"],
            precisions=["bf16", "fp16"],
            activation_modes=["none", "full"],
            optimizers=["adamw", "adamw8bit"],
            attention=["flash", "sdpa"],
            parallelism=["single"],
            supports_packing=False,
            supports_layer_stream=False,
            supports_fused_ce=True,
            tasks=["sft"],
            platforms=["cuda"],
            speed_prior=1.0,
            speed_prior_is_measured=False,
        )

    def _load_model(self, plan: ExecutionPlan, spec: RunSpec) -> Any:
        try:
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise EngineUnavailable("unsloth is not installed", fix="pip install 'bitstraining[fast]'") from e
        model, _tok = FastLanguageModel.from_pretrained(
            model_name=spec.model.base,
            max_seq_length=plan.max_length,
            dtype=None,
            load_in_4bit=plan.quant == "nf4",
        )
        return FastLanguageModel.get_peft_model(
            model,
            r=plan.lora_r,
            target_modules=_TARGETS,
            lora_alpha=plan.lora_alpha,
            lora_dropout=plan.lora_dropout,
            use_gradient_checkpointing="unsloth" if plan.activation == "full" else False,
            random_state=plan.seed,
        )
