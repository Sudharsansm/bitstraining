"""Bundles: a self-contained directory that can run anywhere.

    bundle/
      spec.yaml      RunSpec with data paths rewritten to bundle-relative paths
      plan.json      the ExecutionPlan to execute
      data/          copies of the training/eval files
      src/bitstraining/   a copy of this package, so remote machines need no pip install of it
      requirements.txt    ML dependencies for the engine
      bootstrap.sh        installs requirements and runs the job
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

import bitstraining
from bitstraining.core.config import save_plan
from bitstraining.core.schemas import ExecutionPlan, RunSpec

_REQS = {
    "hf_peft": ["pyyaml", "torch", "transformers>=4.44", "peft>=0.11", "accelerate>=0.30", "datasets"],
    "unsloth": ["pyyaml", "unsloth"],
    "mlx": ["pyyaml", "mlx-lm"],
    "simulate": ["pyyaml"],
}


def make_bundle(spec: RunSpec, plan: ExecutionPlan, dest: str | Path, include_source: bool = True) -> Path:
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "data").mkdir(parents=True)
    d = spec.to_dict()
    for field_name in ("train", "eval"):
        src = d["data"].get(field_name)
        if src and not str(src).startswith("hf://"):
            p = Path(src)
            if p.is_dir():
                shutil.copytree(p, dest / "data" / p.name)
            else:
                shutil.copy2(p, dest / "data" / p.name)
            d["data"][field_name] = f"data/{p.name}"
    d["output"] = "output"
    (dest / "spec.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    save_plan(plan, dest / "plan.json")
    reqs = list(_REQS.get(plan.engine, ["pyyaml"]))
    if plan.quant == "nf4":
        reqs.append("bitsandbytes>=0.43")
    (dest / "requirements.txt").write_text("\n".join(reqs) + "\n")
    (dest / "bootstrap.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'cd "$(dirname "$0")"\n'
        "python -m pip install -q -r requirements.txt\n"
        'PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" python -m bitstraining.runner --bundle .\n'
    )
    (dest / "bootstrap.sh").chmod(0o755)
    if include_source:
        pkg = Path(bitstraining.__file__).parent
        shutil.copytree(pkg, dest / "src" / "bitstraining",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return dest
