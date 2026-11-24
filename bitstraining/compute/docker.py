"""Docker provider: run the bundle inside a container (local daemon or DOCKER_HOST).

Options: image (default "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"), gpus ("all" by default;
set "" for CPU-only), price_per_hour.

NOT exercised by CI (needs a Docker daemon). Command construction is unit-tested.
"""

from __future__ import annotations

import secrets
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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
from bitstraining.core import paths
from bitstraining.core.errors import ComputeError
from bitstraining.hardware.profiler import profile

DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"


class DockerProvider(ComputeProvider):
    name = "docker"

    def __init__(self, options: dict[str, Any] | None = None, runner: Any = None):
        self.o = dict(options or {})
        self._run = runner or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True, check=False))

    def run_command(self, name: str, workdir: str, job: JobSpec) -> list[str]:
        cmd = ["docker", "run", "-d", "--name", name, "-v", f"{workdir}/bundle:/job", "-w", "/job"]
        gpus = self.o.get("gpus", "all")
        if gpus:
            cmd += ["--gpus", str(gpus)]
        for k, v in job.env.items():
            cmd += ["-e", f"{k}={v}"]
        inner = "bash bootstrap.sh" + (" --resume" if job.resume else "")
        return [*cmd, self.o.get("image", DEFAULT_IMAGE), "bash", "-lc", inner]

    def list_offers(self, req: ResourceRequest) -> list[Offer]:
        hw = profile()
        g = hw.gpus[0] if hw.gpus else None
        return [Offer(provider=self.name, name="docker (local daemon)", gpu=g.name if g else "cpu",
                      n_gpus=hw.n_gpus, vram_gb=g.vram_gb if g else 0.0,
                      price_per_hour=self.o.get("price_per_hour", 0.0), hardware=hw)]

    def provision(self, offer: Offer) -> Instance:
        if not shutil.which("docker"):
            raise ComputeError("docker is not installed or not on PATH", fix="install Docker or use provider: local")
        work = paths.home() / "jobs" / f"docker-{secrets.token_hex(3)}"
        work.mkdir(parents=True, exist_ok=True)
        return Instance(id=work.name, provider=self.name, offer=offer, workdir=str(work), started=time.time())

    def sync_workspace(self, inst: Instance, bundle_dir: Path) -> None:
        target = Path(inst.workdir) / "bundle"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(bundle_dir, target)

    def launch(self, inst: Instance, job: JobSpec) -> JobHandle:
        name = f"bt-{inst.id}"
        r = self._run(self.run_command(name, inst.workdir, job))
        if r.returncode != 0:
            raise ComputeError(f"docker run failed: {r.stderr.strip()[:300]}")
        return JobHandle(id=name, instance=inst, log_path="", started=time.time(), meta={"container": name})

    def status(self, job: JobHandle) -> str:
        r = self._run(["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", job.id])
        if r.returncode != 0:
            return FAILED
        state, _, code = r.stdout.strip().partition(" ")
        if state in ("running", "created", "restarting"):
            return RUNNING
        return SUCCEEDED if code == "0" else FAILED

    def stream_logs(self, job: JobHandle) -> Iterator[str]:
        proc = subprocess.Popen(["docker", "logs", "-f", job.id], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        yield from (line.rstrip() for line in proc.stdout)
        proc.wait()

    def fetch_artifacts(self, job: JobHandle, dest: Path) -> None:
        src = Path(job.instance.workdir) / "bundle" / "output"
        if src.exists():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)

    def cancel(self, job: JobHandle) -> None:
        self._run(["docker", "stop", job.id])

    def teardown(self, inst: Instance) -> None:
        self._run(["docker", "rm", "-f", f"bt-{inst.id}"])
