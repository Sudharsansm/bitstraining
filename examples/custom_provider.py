"""A minimal custom compute provider (runs the bundle in a temp dir, like `local`).

Run it:  python examples/custom_provider.py
It registers the provider at runtime and runs a simulated job through the real orchestrator.
Copy it as a starting point for a cloud provider: replace the five marked spots.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

from bitstraining.compute.base import (
    FAILED,
    RUNNING,
    SUCCEEDED,
    ComputeProvider,
    Instance,
    JobHandle,
    JobSpec,
    Offer,
    ResourceRequest,
)
from bitstraining.core import registry


class TempDirProvider(ComputeProvider):
    name = "tempdir"

    def __init__(self, options=None):
        self.options = options or {}
        self._procs: dict[str, subprocess.Popen] = {}

    def list_offers(self, req: ResourceRequest) -> list[Offer]:  # (1) query your provider's catalog
        return [Offer(self.name, "temp dir on this machine", "cpu", 0, 0.0, price_per_hour=0.0)]

    def provision(self, offer: Offer) -> Instance:  # (2) create the machine / pod / VM
        return Instance(id="t1", provider=self.name, offer=offer, workdir=tempfile.mkdtemp(prefix="bt-"))

    def sync_workspace(self, inst: Instance, bundle_dir: Path) -> None:  # (3) upload the bundle
        shutil.copytree(bundle_dir, Path(inst.workdir) / "bundle")

    def launch(self, inst: Instance, job: JobSpec) -> JobHandle:  # (4) start the runner
        bundle = Path(inst.workdir) / "bundle"
        log = open(Path(inst.workdir) / "job.log", "wb")  # noqa: SIM115
        proc = subprocess.Popen(
            [sys.executable, "-m", "bitstraining.runner", "--bundle", str(bundle)],
            cwd=bundle, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(bundle / "src")},  # keep the full env: Windows needs SYSTEMROOT
        )
        self._procs[inst.id] = proc
        return JobHandle(inst.id, inst, log_path=str(Path(inst.workdir) / "job.log"), started=time.time())

    def status(self, job: JobHandle) -> str:
        rc = self._procs[job.id].poll()
        return RUNNING if rc is None else SUCCEEDED if rc == 0 else FAILED

    def stream_logs(self, job: JobHandle) -> Iterator[str]:
        self._procs[job.id].wait()
        yield from Path(job.log_path).read_text().splitlines()

    def fetch_artifacts(self, job: JobHandle, dest: Path) -> None:  # (5) download results
        shutil.copytree(Path(job.instance.workdir) / "bundle" / "output", dest, dirs_exist_ok=True)

    def cancel(self, job: JobHandle) -> None:
        self._procs[job.id].terminate()

    def teardown(self, inst: Instance) -> None:
        shutil.rmtree(inst.workdir, ignore_errors=True)


def main() -> None:
    from bitstraining.compute import make_bundle, run_job
    from bitstraining.core.config import spec_from_dict
    from bitstraining.core.schemas import ExecutionPlan

    registry.register("providers", "tempdir", TempDirProvider)
    work = Path(tempfile.mkdtemp(prefix="bt-example-"))
    data = work / "train.jsonl"
    data.write_text("\n".join(f'{{"instruction": "q{i}", "output": "a{i}"}}' for i in range(30)))
    spec = spec_from_dict({"model": {"base": "gpt2"}, "data": {"train": str(data)}, "engine": "simulate",
                           "extras": {"max_steps": 5}})
    bundle = make_bundle(spec, ExecutionPlan(engine="simulate"), work / "bundle")
    provider = registry.create("providers", "tempdir")
    outcome = run_job(provider, provider.list_offers(ResourceRequest())[0], bundle,
                      JobSpec("example", str(bundle)), work / "result", log=lambda s: None)
    if outcome.status != "succeeded":  # show why, instead of failing later on a missing result folder
        raise RuntimeError(f"job {outcome.status}; last log lines:\n" + "\n".join(outcome.log_tail[-20:]))
    print("job status:", outcome.status, "| result files:", sorted(p.name for p in (work / "result").iterdir()))


if __name__ == "__main__":
    main()
