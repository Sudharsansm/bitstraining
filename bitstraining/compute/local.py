"""Local provider: run the bundle as a subprocess on this machine."""

from __future__ import annotations

import os
import secrets
import shutil
import signal
import subprocess
import sys
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
from bitstraining.core import paths
from bitstraining.hardware.profiler import profile

_PROCS: dict[str, subprocess.Popen[bytes]] = {}


class LocalProvider(ComputeProvider):
    name = "local"

    def __init__(self, options: dict | None = None):
        self.o = dict(options or {})

    def list_offers(self, req: ResourceRequest) -> list[Offer]:
        hw = profile()
        g = hw.gpus[0] if hw.gpus else None
        return [Offer(
            provider=self.name, name="this machine", gpu=g.name if g else "cpu",
            n_gpus=hw.n_gpus, vram_gb=g.vram_gb if g else 0.0, price_per_hour=0.0, hardware=hw,
        )]

    def provision(self, offer: Offer) -> Instance:
        work = paths.home() / "jobs" / f"local-{secrets.token_hex(3)}"
        work.mkdir(parents=True, exist_ok=True)
        return Instance(id=work.name, provider=self.name, offer=offer, workdir=str(work), started=time.time())

    def sync_workspace(self, inst: Instance, bundle_dir: Path) -> None:
        target = Path(inst.workdir) / "bundle"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(bundle_dir, target)

    def launch(self, inst: Instance, job: JobSpec) -> JobHandle:
        bundle = Path(inst.workdir) / "bundle"
        log = Path(inst.workdir) / "job.log"
        cmd = job.command or [sys.executable, "-m", "bitstraining.runner", "--bundle", str(bundle)]
        if job.resume:
            cmd = [*cmd, "--resume"]
        env = {**os.environ, **job.env}
        src = bundle / "src"
        env["PYTHONPATH"] = f"{src}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(src)
        fh = log.open("ab")
        extra = {"start_new_session": True} if os.name == "posix" else {}  # new process group (POSIX only)
        proc = subprocess.Popen(cmd, cwd=bundle, stdout=fh, stderr=subprocess.STDOUT, env=env, **extra)
        h = JobHandle(id=f"{inst.id}-{proc.pid}", instance=inst, log_path=str(log), started=time.time())
        _PROCS[h.id] = proc
        return h

    def status(self, job: JobHandle) -> str:
        proc = _PROCS.get(job.id)
        if proc is None:
            return FAILED
        rc = proc.poll()
        if rc is None:
            return RUNNING
        return SUCCEEDED if rc == 0 else FAILED

    def stream_logs(self, job: JobHandle) -> Iterator[str]:
        log = Path(job.log_path)
        pos = 0
        while True:
            if log.exists():
                with log.open("rb") as f:
                    f.seek(pos)
                    chunk = f.read()
                    pos = f.tell()
                if chunk:
                    yield from chunk.decode(errors="replace").splitlines()
            if self.status(job) != RUNNING:
                if log.exists():  # drain what was written after the last read
                    with log.open("rb") as f:
                        f.seek(pos)
                        tail = f.read()
                    if tail:
                        yield from tail.decode(errors="replace").splitlines()
                return
            time.sleep(0.2)

    def fetch_artifacts(self, job: JobHandle, dest: Path) -> None:
        src = Path(job.instance.workdir) / "bundle" / "output"
        if src.exists():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest, dirs_exist_ok=True)

    def cancel(self, job: JobHandle) -> None:
        proc = _PROCS.get(job.id)
        if proc and proc.poll() is None:
            try:
                if hasattr(os, "killpg"):  # POSIX: stop the whole process group
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:  # Windows
                    proc.terminate()
            except (ProcessLookupError, PermissionError):
                proc.terminate()

    def teardown(self, inst: Instance) -> None:
        pass  # keep the workdir: it holds logs and checkpoints for resume
