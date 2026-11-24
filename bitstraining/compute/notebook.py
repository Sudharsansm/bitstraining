"""Notebook provider: Google Colab, Kaggle, or any hosted Jupyter.

Honest scope: Bitstraining cannot log into or drive Colab/Kaggle for you (their terms and APIs do not
allow unattended GPU sessions). Instead this provider produces a *self-running notebook*: it embeds the
whole bundle (base64 zip) so you upload ONE file, press Run All, and download `output.zip` when done.
Then `bitstraining compute fetch --zip output.zip` (or `fetch_artifacts`) brings the result home.

Status is always MANUAL: the orchestrator prints the instructions and stops instead of polling.
"""

from __future__ import annotations

import base64
import io
import json
import secrets
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bitstraining.compute.base import (
    MANUAL,
    ComputeProvider,
    Instance,
    JobHandle,
    JobSpec,
    Offer,
    ResourceRequest,
)
from bitstraining.core import paths
from bitstraining.core.errors import ComputeError
from bitstraining.hardware.database import synthetic_profile

_TARGETS = {
    "colab": ("T4 16 GB (free tier, availability varies)", "t4"),
    "kaggle": ("T4 x2 or P100 (free weekly quota, availability varies)", "t4"),
}


def build_notebook(bundle_dir: Path) -> dict[str, Any]:
    """An .ipynb (nbformat 4) dict that unpacks the embedded bundle, trains, and zips the output."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(Path(bundle_dir).rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(bundle_dir))
    payload = base64.b64encode(buf.getvalue()).decode()
    if len(payload) > 40_000_000:
        raise ComputeError("bundle is too large to embed in a notebook (>40 MB base64)",
                           fix="shrink the dataset, or upload it separately and edit spec.yaml")

    def code(src: str) -> dict[str, Any]:
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": src.splitlines(keepends=True)}

    md = {"cell_type": "markdown", "metadata": {},
          "source": ["# Bitstraining job\n", "Runtime > Change runtime type > pick a GPU, then **Run all**. "
                     "When it finishes, download `output.zip` from the file browser."]}
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [
            md,
            code(f'import base64, io, zipfile, os\nPAYLOAD = "{payload}"\n'
                 'os.makedirs("job", exist_ok=True)\n'
                 'zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall("job")\n'
                 'print("bundle unpacked")'),
            code("!pip install -q -r job/requirements.txt"),
            code("import subprocess, sys\n"
                 "env = dict(os.environ, PYTHONPATH=os.path.abspath('job/src'))\n"
                 "subprocess.run([sys.executable, '-m', 'bitstraining.runner', '--bundle', 'job'], env=env, check=True)"),
            code("import shutil\nshutil.make_archive('output', 'zip', 'job/output')\nprint('now download output.zip')"),
        ],
    }


class NotebookProvider(ComputeProvider):
    name = "notebook"

    def __init__(self, options: dict[str, Any] | None = None):
        self.o = dict(options or {})
        self.target = self.o.get("target", "colab")

    def list_offers(self, req: ResourceRequest) -> list[Offer]:
        label, gpu = _TARGETS.get(self.target, _TARGETS["colab"])
        hw = synthetic_profile(gpu, 1, 0.0)
        hw.provider = self.name
        return [Offer(provider=self.name, name=f"{self.target}: {label}", gpu=gpu, n_gpus=1,
                      vram_gb=hw.vram_gb, price_per_hour=0.0, hardware=hw, meta={"manual": True})]

    def provision(self, offer: Offer) -> Instance:
        work = paths.home() / "jobs" / f"notebook-{secrets.token_hex(3)}"
        work.mkdir(parents=True, exist_ok=True)
        return Instance(id=work.name, provider=self.name, offer=offer, workdir=str(work), started=time.time())

    def sync_workspace(self, inst: Instance, bundle_dir: Path) -> None:
        nb = build_notebook(Path(bundle_dir))
        out = Path(inst.workdir) / "bitstraining_job.ipynb"
        out.write_text(json.dumps(nb))
        inst.meta["notebook"] = str(out)

    def launch(self, inst: Instance, job: JobSpec) -> JobHandle:
        return JobHandle(id=inst.id, instance=inst, started=time.time(), meta={"notebook": inst.meta["notebook"]})

    def status(self, job: JobHandle) -> str:
        return MANUAL

    def stream_logs(self, job: JobHandle) -> Iterator[str]:
        nb = job.meta["notebook"]
        yield f"Notebook written: {nb}"
        yield f"1. Upload it to {'colab.research.google.com' if self.target == 'colab' else 'kaggle.com (New Notebook > File > Import)'}"
        yield "2. Choose a GPU runtime and press Run all."
        yield "3. Download output.zip, then run: bitstraining compute fetch --zip output.zip --dest ./output"

    def fetch_artifacts(self, job: JobHandle, dest: Path) -> None:
        z = job.meta.get("result_zip")
        if not z:
            raise ComputeError("no result zip registered", fix="pass the downloaded output.zip via `compute fetch --zip`")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for member in zf.namelist():  # refuse path traversal
                if Path(member).is_absolute() or ".." in Path(member).parts:
                    raise ComputeError(f"unsafe path in archive: {member}")
            zf.extractall(dest)

    def cancel(self, job: JobHandle) -> None:
        pass

    def teardown(self, inst: Instance) -> None:
        pass
