"""Text generation for evaluation. Lazy: needs torch + transformers (+ peft for adapters).

Returns a callable `list[str] prompts -> list[str] completions`, which is all the audit needs. Tests and
custom backends can pass any function with this signature instead.
"""

from __future__ import annotations

from collections.abc import Callable

from bitstraining.core.errors import EngineUnavailable

Generator = Callable[[list[str]], list[str]]


def hf_generator(base: str, adapter: str | None = None, max_new_tokens: int = 128, batch_size: int = 8) -> Generator:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise EngineUnavailable("generation needs torch and transformers", fix="pip install 'bitstraining[train]'") from e
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def gen(prompts: list[str]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            texts = [
                tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                if getattr(tok, "chat_template", None) else p
                for p in chunk
            ]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            out += tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out

    return gen
