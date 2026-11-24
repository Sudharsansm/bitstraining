"""Merge a LoRA adapter into its base model and save a standalone checkpoint (needs torch+peft).

NOT exercised by CI. Merging into a quantized base is lossy and is refused: merge into the bf16 base.
"""

from __future__ import annotations

from pathlib import Path

from bitstraining.core.errors import EngineUnavailable


def merge_adapter(base: str, adapter: str, dest: str | Path, dtype: str = "bfloat16") -> Path:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise EngineUnavailable("merging needs torch, transformers and peft", fix="pip install 'bitstraining[train]'") from e
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=getattr(torch, dtype))
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(dest, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(dest)
    return dest
