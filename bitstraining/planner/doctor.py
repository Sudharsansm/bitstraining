"""Pre-flight doctor: catch problems before any GPU time is spent.

Every finding carries a fix suggestion. `errors` block a run; `warnings` should be read.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bitstraining.core.schemas import (
    DatasetStats,
    ExecutionPlan,
    HardwareProfile,
    ModelInfo,
    RunSpec,
)
from bitstraining.data.schema import row_sequences
from bitstraining.planner.throughput import device_for
from bitstraining.strategies.catalog import METHODS

ERROR, WARN, INFO = "error", "warning", "info"
_VALID_ROLES = {"system", "user", "assistant", "tool"}


@dataclass
class Finding:
    level: str
    code: str
    message: str
    fix: str = ""


@dataclass
class DoctorReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, level: str, code: str, message: str, fix: str = "") -> None:
        self.findings.append(Finding(level, code, message, fix))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == WARN]

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        if not self.findings:
            return "doctor: all checks passed"
        mark = {ERROR: "ERROR", WARN: "WARN ", INFO: "info "}
        lines = []
        for f in sorted(self.findings, key=lambda x: (x.level != ERROR, x.level != WARN, x.code)):
            lines.append(f"[{mark[f.level]}] {f.code}: {f.message}")
            if f.fix:
                lines.append(f"         fix: {f.fix}")
        lines.append(f"doctor: {len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        return "\n".join(lines)


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def run_doctor(
    spec: RunSpec,
    hw: HardwareProfile | None = None,
    model: ModelInfo | None = None,
    stats: DatasetStats | None = None,
    plans: list[ExecutionPlan] | None = None,
    plan_error: str | None = None,
    train_rows: list[dict[str, Any]] | None = None,
    eval_rows: list[dict[str, Any]] | None = None,
    check_imports: bool = True,
) -> DoctorReport:
    rep = DoctorReport()
    g, d = spec.goal, spec.data

    # --- config sanity -----------------------------------------------------------------
    if g.learning_rate > 1e-3:
        rep.add(WARN, "DR010", f"learning_rate {g.learning_rate:g} is very high for fine-tuning",
                "typical values: 1e-4 to 3e-4 for LoRA, 1e-5 to 5e-5 for full fine-tuning")
    methods = g.methods or []
    if methods == ["full"] and g.learning_rate > 1e-4:
        rep.add(WARN, "DR011", f"learning_rate {g.learning_rate:g} is high for full fine-tuning",
                "use <= 5e-5")
    if g.learning_rate < 1e-6:
        rep.add(WARN, "DR012", f"learning_rate {g.learning_rate:g} is extremely low; the model may not change",
                "raise it or check for a unit mistake")
    if g.lora_r > 256:
        rep.add(WARN, "DR013", f"lora_r {g.lora_r} is unusually large", "r of 8-64 covers most tasks")
    if g.lora_alpha < g.lora_r / 4:
        rep.add(WARN, "DR014", "lora_alpha is much smaller than lora_r, which scales the adapter down",
                "common choice: lora_alpha = 2 * lora_r")
    if Path(spec.output).resolve() == Path(d.train).resolve():
        rep.add(ERROR, "DR015", "output path equals the training data path", "choose a different output directory")

    # --- data --------------------------------------------------------------------------
    if not (d.train.startswith("hf://") or Path(d.train).exists()):
        rep.add(ERROR, "DR001", f"training data not found: {d.train}", "check data.train")
    if d.eval and not (d.eval.startswith("hf://") or Path(d.eval).exists()):
        rep.add(ERROR, "DR016", f"eval data not found: {d.eval}", "check data.eval")
    if stats is not None:
        if stats.num_rows == 0:
            rep.add(ERROR, "DR002", "no usable rows after cleaning", "run `bitstraining data inspect`")
        elif stats.num_rows < 50:
            rep.add(WARN, "DR003", f"only {stats.num_rows} usable rows", "expect overfitting; 500+ is a safer floor")
        if stats.schema == "unknown":
            rep.add(ERROR, "DR004", "dataset schema not recognized",
                    "use columns: instruction/output, prompt/completion, messages, chosen/rejected or text")
        if g.task in ("dpo", "orpo", "kto") and stats.schema != "preference":
            rep.add(ERROR, "DR017", f"task '{g.task}' needs preference data (prompt, chosen, rejected) but found '{stats.schema}'",
                    "provide chosen/rejected columns or use task: sft")
        if g.task == "sft" and stats.schema == "preference":
            rep.add(ERROR, "DR018", "task 'sft' selected but the data is preference pairs", "set goal.task to dpo, or convert the data")
        if stats.empty_rate > 0.02:
            rep.add(WARN, "DR005", f"{stats.empty_rate:.1%} of rows are empty or lack an assistant reply", "clean the source data")
        if stats.duplicate_rate > 0.05:
            rep.add(WARN, "DR006", f"{stats.duplicate_rate:.1%} of rows are exact duplicates", "keep data.dedup: true")
        if stats.truncation_rate > 0.05:
            sev = ERROR if stats.truncation_rate > 0.5 else WARN
            rep.add(sev, "DR007", f"{stats.truncation_rate:.0%} of sequences exceed max_length {d.max_length} and will be cut",
                    f"raise data.max_length (p95 is {stats.p95_len} tokens) or shorten examples")
        if stats.num_rows and g.global_batch_size > stats.num_rows:
            rep.add(ERROR, "DR019", f"global_batch_size {g.global_batch_size} exceeds the {stats.num_rows} rows available",
                    "lower goal.global_batch_size")
        elif stats.num_rows:
            steps = stats.num_rows * g.epochs / g.global_batch_size
            if steps < 10:
                rep.add(WARN, "DR020", f"only ~{steps:.0f} optimizer steps in total", "add data, epochs, or lower the batch size")

    if train_rows and eval_rows:
        from bitstraining.data.split import leakage

        leak = leakage(train_rows, eval_rows)
        if leak:
            frac = len(leak) / len(eval_rows)
            rep.add(ERROR if frac > 0.01 else WARN, "DR008",
                    f"{len(leak)} eval rows ({frac:.1%}) also appear in the training data",
                    "re-split with `bitstraining data split`, which keeps duplicates on one side")
    if train_rows:
        bad = 0
        for r in train_rows[:500]:
            msgs = r.get("messages")
            if msgs and (any(m["role"] not in _VALID_ROLES for m in msgs) or msgs[-1]["role"] != "assistant"):
                bad += 1
        if bad:
            rep.add(WARN, "DR021", f"{bad} sampled conversations have unknown roles or do not end with an assistant turn",
                    "end each conversation with the reply the model should learn")
        if not any(row_sequences(r) for r in train_rows[:5]):
            rep.add(WARN, "DR022", "rows contain no trainable text", "check column mapping")

    # --- model -------------------------------------------------------------------------
    if model is not None:
        if model.source == "name-heuristic":
            rep.add(WARN, "DR030", f"architecture of '{model.name}' was guessed from its name",
                    "point model.base at a local directory with config.json for exact estimates")
        if "meta-llama" in spec.model.base.lower() and not os.environ.get("HF_TOKEN") and not Path(spec.model.base).exists():
            rep.add(WARN, "DR031", "this looks like a gated Hugging Face model and HF_TOKEN is not set",
                    "export HF_TOKEN=... and accept the model license on the Hub")

    # --- hardware / plan ---------------------------------------------------------------
    if hw is not None:
        gpu = device_for(hw)
        if hw.ram_gb <= 0:
            rep.add(WARN, "DR050", "could not detect system RAM on this platform; assuming 8 GB for planning",
                    "report your OS to the project, or plan with --gpu <name> for a specific machine")
        if not hw.gpus and spec.compute.gpu is None and (model is None or model.params > 1.5e9):
            rep.add(WARN, "DR040", "no GPU detected; fine-tuning an LLM on CPU is impractically slow",
                    "use `--gpu <name>` to plan for rented hardware or `bitstraining notebook` for Colab/Kaggle")
        free = hw.disk_free_gb
        if model is not None and free:
            full = bool(methods) and methods == ["full"]
            need = (model.params * 2 * 2 / 1024**3) if full else 1.0  # ~2 checkpoints for full FT
            if full and free < need:
                rep.add(ERROR, "DR041", f"checkpoints may need ~{need:.0f} GB but only {free:.0f} GB is free",
                        "free disk space or point output at a larger volume")
        if gpu.peak_tflops_bf16 == 0 and hw.gpus:
            rep.add(INFO, "DR042", f"'{gpu.name}' is not in the GPU database; speed estimates use a generic prior",
                    "run `bitstraining profile --benchmark` to measure it")
    if plans:
        best = plans[0]
        vram = device_for(hw).vram_gb if hw else 0
        if vram and best.estimate.peak_gb_per_gpu / vram > 0.9:
            rep.add(WARN, "DR043", f"best plan uses {best.estimate.peak_gb_per_gpu / vram:.0%} of device memory",
                    "long or unusual batches may run out of memory; consider max_length or QLoRA")
        if g.time_budget_hours and best.estimate.total_hours > g.time_budget_hours:
            rep.add(WARN, "DR044", f"predicted {best.estimate.total_hours:.1f} h exceeds the {g.time_budget_hours:g} h budget",
                    "reduce epochs/data or choose faster hardware")
        if g.budget_usd and best.estimate.cost_usd and best.estimate.cost_usd > g.budget_usd:
            rep.add(WARN, "DR045", f"predicted ${best.estimate.cost_usd:.2f} exceeds the ${g.budget_usd:g} budget", "choose cheaper compute")
        if d.packing == "on" and best.attention != "flash":
            rep.add(WARN, "DR046", "packing is on but the attention kernel is not flash; packed examples may attend across boundaries",
                    "use an engine/GPU with flash attention or set data.packing: off")
        if check_imports:
            engine = best.engine
            need = {"hf_peft": ["torch", "transformers", "peft"], "unsloth": ["unsloth"],
                    "mlx": ["mlx_lm"], "deepspeed": ["deepspeed"]}.get(engine, [])
            missing = [m for m in need if not _have(m)]
            if missing:
                rep.add(WARN, "DR047", f"engine '{engine}' needs missing packages: {', '.join(missing)}",
                        "pip install 'bitstraining[train]'")
            if METHODS[best.method].quant == "nf4" and not _have("bitsandbytes"):
                rep.add(WARN, "DR048", "QLoRA needs bitsandbytes", "pip install 'bitstraining[quant]'")
    elif plan_error:
        rep.add(ERROR, "DR049", plan_error.split("\n")[0], plan_error.split("fix:", 1)[1].strip() if "fix:" in plan_error else "")
    return rep
