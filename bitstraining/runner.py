"""Entry point executed on the compute target: `python -m bitstraining.runner --bundle DIR`.

Reads spec.yaml + plan.json from the bundle, loads and cleans the data, runs the chosen engine and writes
results into DIR/output. Progress is printed as `BT_METRIC {json}` lines, which the orchestrator relays.
It deliberately imports nothing heavy until the engine needs it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _latest_checkpoint(out: Path) -> str | None:
    cks = [p for p in out.glob("checkpoint-*") if p.is_dir()]
    if not cks:
        return None
    return str(max(cks, key=lambda p: int(p.name.split("-")[-1])))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bitstraining.runner")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)

    from bitstraining.core import registry
    from bitstraining.core.config import load_plan, load_spec
    from bitstraining.data.clean import clean_rows
    from bitstraining.data.loaders import read_rows
    from bitstraining.data.schema import detect_schema
    from bitstraining.data.split import split_rows
    from bitstraining.engines.base import Callbacks

    bundle = Path(a.bundle).resolve()
    spec = load_spec(bundle / "spec.yaml")
    plan = load_plan(bundle / "plan.json")
    out = (bundle / spec.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw = read_rows(bundle / spec.data.train)
    schema = detect_schema(raw) if spec.data.format == "auto" else spec.data.format
    rows, rep = clean_rows(raw, schema, dedup=spec.data.dedup)
    eval_rows = None
    if spec.data.eval:
        eval_rows, _ = clean_rows(read_rows(bundle / spec.data.eval), schema, dedup=False)
    elif spec.data.eval_split > 0:
        rows, eval_rows = split_rows(rows, spec.data.eval_split, spec.goal.seed)
    print(f"runner: {len(rows)} train rows ({rep.malformed} malformed, {rep.empty} empty, "
          f"{rep.exact_duplicates} duplicates dropped); engine={plan.engine}", flush=True)

    def on_step(step: int, m: dict[str, float]) -> None:
        print("BT_METRIC " + json.dumps({"step": step, **m}), flush=True)

    cb = Callbacks(on_step=on_step, log=lambda s: print(s, flush=True))
    engine = registry.create("engines", plan.engine)
    resume = _latest_checkpoint(out) if a.resume else None
    if resume:
        print(f"runner: resuming from {resume}", flush=True)
    t0 = time.time()
    res = engine.train(plan, spec, rows, eval_rows, cb, out, resume_from=resume)
    (out / "result.json").write_text(json.dumps({**res.to_dict(), "wall_seconds": time.time() - t0}, indent=2))
    print(f"runner: {res.status} steps={res.steps} loss={res.final_loss}", flush=True)
    return 0 if res.status in ("completed", "stopped") else 1


if __name__ == "__main__":
    sys.exit(main())
