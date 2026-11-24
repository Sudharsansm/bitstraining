"""Baseline engine: Hugging Face Transformers + PEFT (+ bitsandbytes for QLoRA).

All heavy imports happen inside methods, so importing this module is free and the core works
without PyTorch installed.

STATUS: this adapter is written against the public Transformers/PEFT APIs but is NOT exercised by
this repository's CI (which has no GPU/torch). Run `bitstraining autotune --engine hf_peft` and a
short real run on your hardware before trusting it; report breakage as an issue.

Scope today: single process, one GPU per process; multi-GPU data parallelism by launching under
`torchrun` (the runner does this when plan.n_gpus > 1). Only the final assistant turn of each
conversation is trained (prompt tokens are masked). Layer streaming and FSDP are not implemented.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

from bitstraining.core.errors import EngineUnavailable
from bitstraining.core.schemas import Capabilities, ExecutionPlan, HardwareProfile, RunSpec
from bitstraining.engines.base import Callbacks, Engine, ProbeContext, ProbeResult, RunResult

GB = 1024**3
_OPTIM = {"adamw": "adamw_torch_fused", "adamw8bit": "adamw_bnb_8bit",
          "paged_adamw8bit": "paged_adamw_8bit"}
_ATTN = {"flash": "flash_attention_2", "sdpa": "sdpa", "eager": "eager"}
_FALLBACK_TEMPLATE = "### User:\n{user}\n\n### Assistant:\n"


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _require() -> None:
    missing = [m for m in ("torch", "transformers", "peft") if not _have(m)]
    if missing:
        raise EngineUnavailable(
            f"hf_peft engine needs: {', '.join(missing)}",
            fix="pip install 'bitstraining[train]'  (and 'bitstraining[quant]' for QLoRA)",
        )


class HFPeftEngine(Engine):
    name = "hf_peft"

    # ------------------------------------------------------------------ capabilities
    def capabilities(self, hw: HardwareProfile, assume_installed: bool = False) -> Capabilities:
        core = all(_have(m) for m in ("torch", "transformers", "peft"))
        cuda = hw.gpu_vendor == "nvidia"
        bnb = assume_installed or _have("bitsandbytes")
        flash = (assume_installed and cuda) or _have("flash_attn")
        available = assume_installed or core
        methods = ["full", "lora", "dora"] + (["qlora"] if cuda and bnb else [])
        optimizers = ["adamw"] + (["adamw8bit", "paged_adamw8bit"] if cuda and bnb else [])
        return Capabilities(
            name=self.name,
            available=available,
            reason="" if available else "torch/transformers/peft not installed",
            methods=methods,
            precisions=["bf16", "fp16"],
            activation_modes=["none", "full"],
            optimizers=optimizers,
            attention=(["flash"] if flash else []) + ["sdpa", "eager"],
            parallelism=["single", "ddp"] if cuda else ["single"],
            supports_packing=flash,  # padding-free collator needs flash-attention varlen
            supports_layer_stream=False,
            supports_fused_ce=False,
            supports_compile=False,
            tasks=["sft", "pretrain"],
            platforms=["cuda", "mps", "cpu"],
            speed_prior=1.0,
            speed_prior_is_measured=True,  # the baseline other engines are compared against
        )

    # ------------------------------------------------------------------ model loading
    def _load_model(self, plan: ExecutionPlan, spec: RunSpec) -> Any:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        dtype = torch.bfloat16 if plan.precision == "bf16" else torch.float16
        kwargs: dict[str, Any] = dict(
            torch_dtype=dtype,
            attn_implementation=_ATTN.get(plan.attention, "sdpa"),
            trust_remote_code=spec.model.trust_remote_code,
            revision=spec.model.revision,
        )
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", 0))}
        if plan.quant == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(spec.model.base, **kwargs)
        model.config.use_cache = False
        return self._wrap(model, plan)

    def _wrap(self, model: Any, plan: ExecutionPlan) -> Any:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        ckpt = plan.activation == "full"
        if plan.quant == "nf4":
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=ckpt,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        elif ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if plan.method != "full":
            cfg = LoraConfig(
                r=plan.lora_r, lora_alpha=plan.lora_alpha, lora_dropout=plan.lora_dropout,
                target_modules=plan.target_modules, task_type="CAUSAL_LM",
                use_dora=plan.method == "dora",
            )
            if ckpt:
                model.enable_input_require_grads()
            model = get_peft_model(model, cfg)
        return model

    # ------------------------------------------------------------------ data
    @staticmethod
    def _encode(tok: Any, row: dict[str, Any], max_len: int) -> dict[str, list[int]] | None:
        if "text" in row:
            ids = tok(row["text"], truncation=True, max_length=max_len)["input_ids"]
            return {"input_ids": ids, "labels": list(ids), "attention_mask": [1] * len(ids)}
        msgs = row["messages"]
        if not msgs or msgs[-1]["role"] != "assistant":
            return None
        if getattr(tok, "chat_template", None):
            prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            full = tok.apply_chat_template(msgs, tokenize=False)
        else:
            user = "\n".join(m["content"] for m in msgs[:-1])
            prefix = _FALLBACK_TEMPLATE.format(user=user)
            full = prefix + msgs[-1]["content"] + (tok.eos_token or "")
        if not full.startswith(prefix):  # template does not nest; train on everything
            prefix = ""
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"] if prefix else []
        f_ids = tok(full, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        f_ids, labels = f_ids[:max_len], labels[:max_len]
        if all(x == -100 for x in labels):
            return None
        return {"input_ids": f_ids, "labels": labels, "attention_mask": [1] * len(f_ids)}

    def _dataset(self, tok: Any, rows: list[dict[str, Any]], max_len: int) -> Any:
        from torch.utils.data import Dataset

        enc = [e for r in rows if (e := self._encode(tok, r, max_len)) is not None]

        class _DS(Dataset):  # type: ignore[type-arg]
            def __len__(self) -> int:
                return len(enc)

            def __getitem__(self, i: int) -> dict[str, list[int]]:
                return enc[i]

        return _DS()

    # ------------------------------------------------------------------ probe
    def probe(self, plan: ExecutionPlan, ctx: ProbeContext, micro_batch: int | None = None) -> ProbeResult:
        _require()
        import torch

        mb = micro_batch or plan.micro_batch
        if not torch.cuda.is_available():
            return ProbeResult(fits=True, note="no CUDA device: probe skipped")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = None
        try:
            model = self._load_model(plan, ctx.spec)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=1e-5)
            vocab = ctx.model.vocab_size or 32000
            x = torch.randint(0, vocab, (mb, ctx.seq_len), device="cuda")
            steps, t0 = 3, None
            for i in range(steps + 1):
                if i == 1:  # step 0 is warmup
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                out = model(input_ids=x, labels=x)
                out.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - (t0 or time.perf_counter())
            return ProbeResult(
                fits=True,
                peak_gb=round(torch.cuda.max_memory_allocated() / GB, 3),
                tokens_per_s=round(steps * mb * ctx.seq_len / max(dt, 1e-9), 1),
                steps=steps,
                note="measured",
            )
        except torch.cuda.OutOfMemoryError:
            return ProbeResult(fits=False, note="CUDA out of memory")
        finally:
            del model
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ train
    def train(
        self,
        plan: ExecutionPlan,
        spec: RunSpec,
        train_rows: list[dict[str, Any]],
        eval_rows: list[dict[str, Any]] | None,
        callbacks: Callbacks,
        output_dir: Path,
        resume_from: str | None = None,
    ) -> RunResult:
        _require()
        import torch
        from transformers import (
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )

        set_seed(plan.seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(spec.model.base, trust_remote_code=spec.model.trust_remote_code)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        train_ds = self._dataset(tok, train_rows, plan.max_length)
        if len(train_ds) == 0:  # type: ignore[arg-type]
            raise EngineUnavailable("no trainable rows after tokenization", fix="check the data format")
        eval_ds = self._dataset(tok, eval_rows, plan.max_length) if eval_rows else None

        collator: Any = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
        packing_ok = False
        if plan.packing:
            try:
                from transformers import DataCollatorWithFlattening

                collator, packing_ok = DataCollatorWithFlattening(), True
            except ImportError:
                callbacks.log("packing requested but DataCollatorWithFlattening is unavailable; using padding")

        model = self._load_model(plan, spec)
        total_steps = plan.total_steps or 0
        args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=plan.micro_batch,
            per_device_eval_batch_size=plan.micro_batch,
            gradient_accumulation_steps=plan.grad_accum,
            num_train_epochs=plan.epochs,
            learning_rate=plan.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            logging_steps=1,
            save_strategy="steps",
            save_steps=int(spec.extras.get("save_steps") or max(50, total_steps // 5 or 50)),
            save_total_limit=2,
            bf16=plan.precision == "bf16",
            fp16=plan.precision == "fp16",
            optim=_OPTIM.get(plan.optimizer, "adamw_torch"),
            group_by_length=not packing_ok and plan.micro_batch > 1,
            dataloader_num_workers=plan.dataloader_workers,
            report_to="none",
            seed=plan.seed,
            remove_unused_columns=False,
            eval_strategy="epoch" if eval_ds is not None else "no",
            ddp_find_unused_parameters=False,
            max_steps=int(spec.extras["max_steps"]) if spec.extras.get("max_steps") else -1,
        )

        t_start = time.perf_counter()
        tokens_seen = {"n": 0}

        class _Hook(TrainerCallback):
            def on_log(self, a: Any, state: Any, control: Any, logs: dict[str, float] | None = None, **kw: Any) -> None:
                if not logs or "loss" not in logs:
                    return
                elapsed = max(time.perf_counter() - t_start, 1e-9)
                m = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
                m["tokens_per_s"] = tokens_seen["n"] / elapsed
                callbacks.step(int(state.global_step), m)
                if callbacks.stop_requested():
                    control.should_training_stop = True

            def on_step_end(self, a: Any, state: Any, control: Any, **kw: Any) -> None:
                tokens_seen["n"] += a.per_device_train_batch_size * a.gradient_accumulation_steps * plan.max_length

            def on_save(self, a: Any, state: Any, control: Any, **kw: Any) -> None:
                if callbacks.on_checkpoint:
                    callbacks.on_checkpoint(str(output_dir / f"checkpoint-{state.global_step}"))

        trainer = Trainer(
            model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
            data_collator=collator, callbacks=[_Hook()],
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        trainer.train(resume_from_checkpoint=resume_from)
        trainer.save_model(str(output_dir))
        tok.save_pretrained(str(output_dir))

        hist = trainer.state.log_history
        losses = [h["loss"] for h in hist if "loss" in h]
        eval_losses = [h["eval_loss"] for h in hist if "eval_loss" in h]
        seconds = time.perf_counter() - t_start
        peak = torch.cuda.max_memory_allocated() / GB if torch.cuda.is_available() else None
        (output_dir / "bitstraining_run.json").write_text(
            json.dumps({"base": spec.model.base, "plan": plan.to_dict(), "steps": trainer.state.global_step}, indent=2)
        )
        return RunResult(
            status="stopped" if callbacks.stop_requested() else "completed",
            steps=int(trainer.state.global_step),
            final_loss=losses[-1] if losses else None,
            eval_loss=eval_losses[-1] if eval_losses else None,
            seconds=round(seconds, 1),
            peak_gb=round(peak, 3) if peak else None,
            tokens_per_s=round(tokens_seen["n"] / max(seconds, 1e-9), 1),
            adapter_path=str(output_dir),
            simulated=False,
        )
