"""Command line: `bitstraining <command>` (alias `bt`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from bitstraining import __version__
from bitstraining.core import registry
from bitstraining.core.config import load_spec
from bitstraining.core.errors import BitstrainingError

TEMPLATES: dict[str, dict[str, Any]] = {
    "chat": {
        "name": "chat-assistant",
        "model": {"base": "Qwen/Qwen2.5-0.5B-Instruct"},
        "data": {"train": "./data/train.jsonl", "max_length": 1024, "eval_split": 0.05},
        "goal": {"task": "sft", "epochs": 3, "global_batch_size": 16, "learning_rate": 2e-4, "lora_r": 16},
        "output": "./output",
    },
    "code": {
        "name": "code-assistant",
        "model": {"base": "Qwen/Qwen2.5-1.5B-Instruct"},
        "data": {"train": "./data/train.jsonl", "max_length": 2048, "eval_split": 0.05},
        "goal": {"task": "sft", "epochs": 2, "global_batch_size": 32, "learning_rate": 1e-4, "lora_r": 32},
        "output": "./output",
    },
    "preference": {
        "name": "preference-tuning",
        "model": {"base": "Qwen/Qwen2.5-0.5B-Instruct"},
        "data": {"train": "./data/prefs.jsonl", "max_length": 1024},
        "goal": {"task": "dpo", "epochs": 1, "global_batch_size": 16, "learning_rate": 5e-6},
        "output": "./output",
    },
    "full-finetune": {
        "name": "full-finetune",
        "model": {"base": "Qwen/Qwen2.5-0.5B-Instruct"},
        "data": {"train": "./data/train.jsonl", "max_length": 1024},
        "goal": {"task": "sft", "epochs": 2, "global_batch_size": 16, "learning_rate": 2e-5, "methods": ["full"]},
        "output": "./output",
    },
}

_DEMO = [
    ("Translate to Spanish: Good morning", "Buenos días"),
    ("Translate to Spanish: Thank you very much", "Muchas gracias"),
    ("What is the capital of France?", "Paris"),
    ("What is 12 times 12?", "144"),
    ("Name a primary color.", "Red"),
    ("Summarize: The cat sat on the mat.", "A cat sat on a mat."),
]


def _demo_rows(n: int) -> list[dict[str, str]]:
    rows = []
    for i in range(n):
        q, a = _DEMO[i % len(_DEMO)]
        rows.append({"instruction": q if i < len(_DEMO) else f"{q} (variant {i})", "input": "", "output": a})
    return rows


def _spec(args: argparse.Namespace):
    spec = load_spec(args.config)
    if getattr(args, "engine", None):
        spec.engine = args.engine
    return spec


def _add_hw(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpu", help="plan for a named GPU instead of this machine, e.g. a100-80gb")
    p.add_argument("--n-gpus", type=int)
    p.add_argument("--price", type=float, help="USD per hour for the whole machine")


def cmd_init(a: argparse.Namespace) -> int:
    if a.template not in TEMPLATES:
        print(f"unknown template '{a.template}'. choose from: {', '.join(TEMPLATES)}", file=sys.stderr)
        return 2
    dest = Path(a.output)
    if dest.exists() and not a.force:
        print(f"{dest} exists; use --force to overwrite", file=sys.stderr)
        return 2
    dest.write_text(yaml.safe_dump(TEMPLATES[a.template], sort_keys=False))
    print(f"wrote {dest}  (template: {a.template})\nnext: bitstraining plan --config {dest}")
    return 0


def cmd_profile(a: argparse.Namespace) -> int:
    from bitstraining import profile_hardware

    hw = profile_hardware(a.benchmark, a.gpu, a.n_gpus, a.price)
    print(f"Machine:  {hw.name} ({hw.os})")
    print(f"CPU:      {hw.cpu_model} x{hw.cpu_cores}")
    print(f"RAM:      {hw.ram_gb:.1f} GB   Disk free: {hw.disk_free_gb:.0f} GB"
          + (f"   Disk read: {hw.disk_read_gbs} GB/s" if hw.disk_read_gbs else ""))
    if not hw.gpus:
        print("GPU:      none detected (CPU only)")
    for i, g in enumerate(hw.gpus):
        print(f"GPU {i}:    {g.name}  {g.vram_gb:g} GB  bf16={g.bf16}  peak~{g.peak_tflops_bf16:g} TFLOPs  [{g.source}]")
    if a.json:
        print(json.dumps(hw.to_dict(), indent=2))
    return 0


def cmd_data(a: argparse.Namespace) -> int:
    from bitstraining.data import clean_rows, detect_schema, read_rows

    if a.data_cmd == "inspect":
        from bitstraining import inspect_data

        print(inspect_data(a.path, a.format, a.max_length))
    elif a.data_cmd == "demo":
        dest = Path(a.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w") as f:
            for r in _demo_rows(a.rows):
                f.write(json.dumps(r) + "\n")
        print(f"wrote {a.rows} demo rows to {dest}")
    elif a.data_cmd == "clean":
        raw = read_rows(a.path)
        schema = detect_schema(raw)
        kept, rep = clean_rows(raw, schema, dedup=True, near_dup_threshold=a.near_dup)
        with Path(a.output).open("w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        print(f"kept {rep.kept}/{rep.input_rows} (malformed {rep.malformed}, empty {rep.empty}, "
              f"duplicates {rep.exact_duplicates}, near-duplicates {rep.near_duplicates}) -> {a.output}")
    elif a.data_cmd == "split":
        from bitstraining.data import split_rows

        raw = read_rows(a.path)
        kept, _ = clean_rows(raw, detect_schema(raw), dedup=True)
        tr, ev = split_rows(kept, a.eval_fraction, a.seed)
        for name, rows in (("train", tr), ("eval", ev)):
            with Path(f"{a.prefix}.{name}.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        print(f"train {len(tr)} rows, eval {len(ev)} rows (duplicates kept on one side)")
    return 0


def cmd_doctor(a: argparse.Namespace) -> int:
    from bitstraining import doctor

    rep = doctor(_spec(a), gpu=a.gpu, n_gpus=a.n_gpus, price=a.price)
    print(rep.render())
    return 0 if rep.ok else 1


def _print_plan(i: int, p) -> None:
    e = p.estimate
    cost = f"  ${e.cost_usd:.2f}" if e.cost_usd is not None else ""
    print(f"#{i}  {p.describe()}")
    print(f"     {e.peak_gb_per_gpu:.1f} GB/GPU | {e.useful_tokens_per_s:,.0f} useful tok/s | "
          f"{e.total_hours:.2f} h{cost} | confidence {e.confidence} | steps {p.total_steps}")
    for n in e.notes:
        print(f"     note: {n}")


def cmd_plan(a: argparse.Namespace) -> int:
    from bitstraining import plan
    from bitstraining.core.config import save_plan

    plans = plan(_spec(a), top_k=a.top, gpu=a.gpu, n_gpus=a.n_gpus, price=a.price)
    for i, p in enumerate(plans, 1):
        _print_plan(i, p)
        if a.verbose:
            print("     memory GB: " + ", ".join(f"{k}={v}" for k, v in p.estimate.breakdown_gb.items()))
    if a.save:
        save_plan(plans[0], a.save)
        print(f"saved plan #1 to {a.save}")
    print("\nThese are model-based estimates. `bitstraining autotune` verifies them with a real probe.")
    return 0


def cmd_autotune(a: argparse.Namespace) -> int:
    from bitstraining import tune

    tr = tune(_spec(a), use_cache=not a.no_cache, gpu=a.gpu, n_gpus=a.n_gpus, price=a.price)
    _print_plan(1, tr.plan)
    for n in tr.notes:
        print(f"autotune: {n}")
    for mb, r in sorted(tr.probes.items()):
        print(f"  probe mb={mb}: fits={r.fits} peak={r.peak_gb} GB tok/s={r.tokens_per_s} ({r.note})")
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    from bitstraining import train

    res = train(_spec(a), gpu=a.gpu, resume=a.resume, force=a.force, tune=a.tune, dry_run=a.dry_run)
    r = res.get("result")
    if r:
        print(f"run {res['run_id']}: {r['status']} steps={r['steps']} loss={r['final_loss']}")
    return 0


def cmd_runs(a: argparse.Namespace) -> int:
    from bitstraining.tracking import Registry

    with Registry() as reg:
        if a.runs_cmd == "list":
            for r in reg.list_runs(a.limit):
                sim = " [simulated]" if r["simulated"] else ""
                print(f"{r['id']}  {r['status']:<9} {r['engine']:<9} {r['name']}{sim}")
        elif a.runs_cmd == "lineage":
            print(json.dumps(reg.lineage(a.run or reg.latest_run_id() or ""), indent=2))
        elif a.runs_cmd == "spikes":
            rid = a.run or reg.latest_run_id()
            if not rid:
                print("no runs recorded")
                return 1
            run = reg.get_run(rid)
            rows_file = Path((run["spec"] or {}).get("output", "./output")) / "train_normalized.jsonl"
            lines = rows_file.read_text().splitlines() if rows_file.exists() else []
            spikes = reg.loss_spikes(rid, a.window, a.z)
            if not spikes:
                print("no loss spikes detected")
            for s in spikes:
                print(f"step {s['step']}: loss {s['loss']:.3f} (score {s['score']}) rows {s['rows']}")
                for ri in s["rows"][:3]:
                    if ri < len(lines):
                        print(f"    row {ri}: {lines[ri][:120]}")
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    from bitstraining.eval import GateCriteria, audit_rows, evaluate_gate

    def load(p: str) -> list[dict[str, Any]]:
        return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

    task = load(a.task)  # each: {"reference":..,"base":..,"tuned":..,"prompt":..}
    rep = audit_rows([r["reference"] for r in task], [r["base"] for r in task], [r["tuned"] for r in task],
                     a.metric, [r.get("prompt", "") for r in task])
    print("task:    " + rep.summary())
    if a.audit_out:
        rep.write(a.audit_out)
        print(f"per-row audit written to {a.audit_out}")
    gen = None
    if a.general:
        g = load(a.general)
        gen = audit_rows([r["reference"] for r in g], [r["base"] for r in g], [r["tuned"] for r in g], a.metric)
        print("general: " + gen.summary())
    res = evaluate_gate(rep, gen, GateCriteria(a.min_gain, a.max_forgetting, a.max_worse))
    print(res.render())
    return 0 if res.passed else 1


def cmd_card(a: argparse.Namespace) -> int:
    from bitstraining.deploy import write_model_card
    from bitstraining.tracking import Registry

    with Registry() as reg:
        rid = a.run or reg.latest_run_id()
        if not rid:
            print("no runs recorded", file=sys.stderr)
            return 1
        run = reg.get_run(rid)
        p = write_model_card(run, reg.lineage(rid), a.output)
    print(f"wrote {p}")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    import shlex
    import subprocess

    from bitstraining.deploy import serve_command

    cmd = serve_command(a.model, a.backend, a.port, a.host, a.max_len)
    print(" ".join(shlex.quote(c) for c in cmd))
    if a.run:
        return subprocess.call(cmd)
    print("(add --run to start it)")
    return 0


def cmd_compute(a: argparse.Namespace) -> int:
    from bitstraining.compute.base import ResourceRequest

    if a.compute_cmd == "offers":
        prov = registry.create("providers", a.provider, {})
        for o in prov.list_offers(ResourceRequest(gpu=a.gpu, n_gpus=a.n_gpus, min_vram_gb=a.min_vram,
                                                  max_price_per_hour=a.max_price)):
            price = f"${o.price_per_hour:.2f}/h" if o.price_per_hour is not None else "n/a"
            note = " (illustrative price)" if o.meta.get("illustrative") else ""
            print(f"{o.provider:<9} {o.name:<34} {o.vram_gb:>5g} GB  {price}{note}")
    elif a.compute_cmd == "fetch":
        from bitstraining.compute.base import Instance, JobHandle
        from bitstraining.compute.notebook import NotebookProvider

        h = JobHandle(id="manual", instance=Instance("manual", "notebook"), meta={"result_zip": a.zip})
        NotebookProvider().fetch_artifacts(h, Path(a.dest))
        print(f"extracted {a.zip} to {a.dest}")
    return 0


def cmd_notebook(a: argparse.Namespace) -> int:
    """Build a self-running Colab/Kaggle notebook for a run."""
    import json as _json

    from bitstraining import plan
    from bitstraining.compute.bundle import make_bundle
    from bitstraining.compute.notebook import build_notebook
    from bitstraining.core import paths

    spec = _spec(a)
    best = plan(spec, top_k=1, gpu="t4")[0]
    bundle = make_bundle(spec, best, paths.home() / "jobs" / "notebook-bundle")
    Path(a.output).write_text(_json.dumps(build_notebook(bundle)))
    print(f"wrote {a.output}; upload it to {a.target}, pick a GPU runtime, Run all, download output.zip")
    return 0


def cmd_plugins(a: argparse.Namespace) -> int:
    for group in ("engines", "providers"):
        print(f"{group}: {', '.join(registry.names(group))}")
    return 0


def cmd_quickstart(a: argparse.Namespace) -> int:
    """End-to-end smoke run with the simulation engine: no GPU, no downloads, no model trained."""
    import os
    import tempfile

    from bitstraining import train

    root = Path(tempfile.mkdtemp(prefix="bt-quickstart-"))
    data = root / "train.jsonl"
    with data.open("w") as f:
        for r in _demo_rows(200):
            f.write(json.dumps(r) + "\n")
    spec_d = dict(TEMPLATES["chat"])
    spec_d.update(engine="simulate", output=str(root / "out"), extras={"max_steps": 30})
    spec_d["data"] = {**spec_d["data"], "train": str(data)}
    os.environ.setdefault("BITSTRAINING_OFFLINE", "1")
    from bitstraining.core.config import spec_from_dict

    res = train(spec_from_dict(spec_d))
    print(f"\nquickstart finished in {root}. The simulation engine trains nothing; "
          f"see `bitstraining init` for a real config.")
    return 0 if res.get("result", {}).get("status") == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bitstraining", description="Train. Fine-tune. Optimize. Deploy.")
    p.add_argument("--version", action="version", version=f"bitstraining {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def cfg(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", "-c", default="bitstraining.yaml")
        sp.add_argument("--engine", help="force an engine (hf_peft, unsloth, mlx, simulate)")

    s = sub.add_parser("init", help="write a starter config")
    s.add_argument("--template", default="chat", help=f"one of: {', '.join(TEMPLATES)}")
    s.add_argument("--output", "-o", default="bitstraining.yaml")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("profile", help="detect hardware (or describe a named GPU)")
    s.add_argument("--benchmark", action="store_true", help="measure TFLOPs/bandwidth (needs PyTorch+CUDA)")
    s.add_argument("--json", action="store_true")
    _add_hw(s)
    s.set_defaults(fn=cmd_profile)

    s = sub.add_parser("data", help="inspect, clean, split, demo data")
    ds = s.add_subparsers(dest="data_cmd", required=True)
    x = ds.add_parser("inspect")
    x.add_argument("path")
    x.add_argument("--format", default="auto")
    x.add_argument("--max-length", type=int, default=2048)
    x = ds.add_parser("clean")
    x.add_argument("path")
    x.add_argument("--output", "-o", required=True)
    x.add_argument("--near-dup", type=float, default=None, help="MinHash similarity threshold, e.g. 0.8")
    x = ds.add_parser("split")
    x.add_argument("path")
    x.add_argument("--eval-fraction", type=float, default=0.1)
    x.add_argument("--prefix", default="data")
    x.add_argument("--seed", type=int, default=42)
    x = ds.add_parser("demo")
    x.add_argument("--output", "-o", default="data/train.jsonl")
    x.add_argument("--rows", type=int, default=100)
    s.set_defaults(fn=cmd_data)

    s = sub.add_parser("doctor", help="pre-flight checks")
    cfg(s)
    _add_hw(s)
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("plan", help="rank execution plans with predicted memory/speed/cost")
    cfg(s)
    _add_hw(s)
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--save", help="write the best plan to a JSON file")
    s.add_argument("--verbose", "-v", action="store_true")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("autotune", help="probe the best plan and learn calibration")
    cfg(s)
    _add_hw(s)
    s.add_argument("--no-cache", action="store_true")
    s.set_defaults(fn=cmd_autotune)

    s = sub.add_parser("train", help="plan, check, run, track")
    cfg(s)
    s.add_argument("--gpu", help="(planning only) named GPU; combine with compute.provider in the config")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--force", action="store_true", help="run even if doctor reports errors")
    s.add_argument("--tune", action="store_true", help="autotune before training")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("runs", help="run history, lineage, loss-spike tracing")
    rs = s.add_subparsers(dest="runs_cmd", required=True)
    x = rs.add_parser("list")
    x.add_argument("--limit", type=int, default=20)
    x = rs.add_parser("lineage")
    x.add_argument("run", nargs="?")
    x = rs.add_parser("spikes")
    x.add_argument("run", nargs="?")
    x.add_argument("--window", type=int, default=20)
    x.add_argument("--z", type=float, default=4.0)
    s.set_defaults(fn=cmd_runs)

    s = sub.add_parser("eval", help="per-row audit + forgetting check + gate over prediction files")
    s.add_argument("--task", required=True, help="JSONL rows: reference, base, tuned [, prompt]")
    s.add_argument("--general", help="JSONL of the same shape on general-capability prompts")
    s.add_argument("--metric", default="token_f1", choices=["exact_match", "contains", "token_f1"])
    s.add_argument("--min-gain", type=float, default=0.0)
    s.add_argument("--max-forgetting", type=float, default=0.05)
    s.add_argument("--max-worse", type=float, default=None, help="max fraction of task rows allowed to regress")
    s.add_argument("--audit-out")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("card", help="write a model card for a run")
    s.add_argument("run", nargs="?")
    s.add_argument("--output", "-o", default="MODEL_CARD.md")
    s.set_defaults(fn=cmd_card)

    s = sub.add_parser("serve", help="print/run the command that serves a model (vLLM or llama.cpp)")
    s.add_argument("--model", required=True)
    s.add_argument("--backend", default="vllm", choices=["vllm", "llamacpp"])
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--max-len", type=int)
    s.add_argument("--run", action="store_true")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("compute", help="providers and offers")
    cs = s.add_subparsers(dest="compute_cmd", required=True)
    x = cs.add_parser("offers")
    x.add_argument("--provider", default="catalog")
    x.add_argument("--gpu")
    x.add_argument("--n-gpus", type=int, default=1)
    x.add_argument("--min-vram", type=float, default=0.0)
    x.add_argument("--max-price", type=float)
    x = cs.add_parser("fetch")
    x.add_argument("--zip", required=True)
    x.add_argument("--dest", default="./output")
    s.set_defaults(fn=cmd_compute)

    s = sub.add_parser("notebook", help="build a self-running Colab/Kaggle notebook")
    cfg(s)
    s.add_argument("--target", default="colab", choices=["colab", "kaggle"])
    s.add_argument("--output", "-o", default="bitstraining_job.ipynb")
    s.set_defaults(fn=cmd_notebook)

    s = sub.add_parser("plugins", help="list engines and providers")
    s.set_defaults(fn=cmd_plugins)

    s = sub.add_parser("quickstart", help="no-GPU smoke run with the simulation engine")
    s.set_defaults(fn=cmd_quickstart)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except BitstrainingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
