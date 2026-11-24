"""SSH provider: any GPU box you can reach with `ssh` (your own server, a rented VM, a pod).

Uses the system `ssh`/`rsync`/`scp` binaries, so your ~/.ssh/config, keys and agents just work.
No credentials are ever stored or logged by Bitstraining.

Options (compute.options in the run spec):
  host (required)   e.g. "ubuntu@203.0.113.7" or an ssh-config alias
  port              ssh port
  identity_file     path to a private key
  remote_root       default "~/bitstraining"
  python            remote python executable, default "python3"
  price_per_hour    USD/hour for the whole machine, for cost estimates and budget caps
  gpu, n_gpus       what is installed there (used for planning without logging in)

NOT exercised by CI (needs a real remote host). The command construction is unit-tested.
"""

from __future__ import annotations

import secrets
import shlex
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
from bitstraining.core.errors import ComputeError
from bitstraining.hardware.database import synthetic_profile


class SSHProvider(ComputeProvider):
    name = "ssh"

    def __init__(self, options: dict[str, Any] | None = None, runner: Any = None):
        self.o = dict(options or {})
        self._run = runner or self._subprocess_run  # injectable for tests

    # ---- command builders (pure, unit-tested) -------------------------------------------
    def _ssh_base(self) -> list[str]:
        host = self.o.get("host")
        if not host:
            raise ComputeError("ssh provider needs compute.options.host", fix="e.g. host: ubuntu@203.0.113.7")
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30"]
        if self.o.get("port"):
            cmd += ["-p", str(self.o["port"])]
        if self.o.get("identity_file"):
            cmd += ["-i", str(self.o["identity_file"])]
        return [*cmd, host]

    def ssh_command(self, remote_cmd: str) -> list[str]:
        return [*self._ssh_base(), remote_cmd]

    def rsync_command(self, src: str, dst: str, to_remote: bool) -> list[str]:
        ssh = ["ssh", "-o", "BatchMode=yes"]
        if self.o.get("port"):
            ssh += ["-p", str(self.o["port"])]
        if self.o.get("identity_file"):
            ssh += ["-i", str(self.o["identity_file"])]
        host = self.o["host"]
        a, b = (src, f"{host}:{dst}") if to_remote else (f"{host}:{src}", dst)
        return ["rsync", "-az", "-e", " ".join(shlex.quote(x) for x in ssh), a, b]

    def remote_launch_script(self, workdir: str, job: JobSpec) -> str:
        py = self.o.get("python", "python3")
        resume = " --resume" if job.resume else ""
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in job.env.items())
        inner = (
            f"cd {shlex.quote(workdir)}/bundle && "
            f"{env} {py} -m pip install -q -r requirements.txt && "
            f"PYTHONPATH=$PWD/src {py} -m bitstraining.runner --bundle .{resume}; "
            f"echo $? > {shlex.quote(workdir)}/exit_code"
        )
        return (
            f"nohup bash -c {shlex.quote(inner)} > {shlex.quote(workdir)}/job.log 2>&1 & echo $!"
        )

    # ---- execution ---------------------------------------------------------------------
    @staticmethod
    def _subprocess_run(cmd: list[str], timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)

    def _ok(self, cmd: list[str], what: str) -> str:
        r = self._run(cmd)
        if r.returncode != 0:
            raise ComputeError(f"{what} failed: {r.stderr.strip()[:300]}", fix="check ssh access with `ssh <host> true`")
        return r.stdout.strip()

    def list_offers(self, req: ResourceRequest) -> list[Offer]:
        gpu, n = self.o.get("gpu"), int(self.o.get("n_gpus", 1))
        hw = synthetic_profile(gpu, n, self.o.get("price_per_hour")) if gpu else None
        if hw:
            hw.provider = self.name
        return [Offer(
            provider=self.name, name=str(self.o.get("host", "ssh")), gpu=gpu or "unknown", n_gpus=n,
            vram_gb=hw.vram_gb if hw else 0.0, price_per_hour=self.o.get("price_per_hour"), hardware=hw,
        )]

    def provision(self, offer: Offer) -> Instance:
        root = self.o.get("remote_root", "~/bitstraining")
        wd = f"{root}/job-{secrets.token_hex(3)}"
        self._ok(self.ssh_command(f"mkdir -p {shlex.quote(wd)}"), "creating remote workdir")
        return Instance(id=wd.rsplit("/", 1)[-1], provider=self.name, offer=offer, workdir=wd,
                        host=self.o["host"], started=time.time())

    def sync_workspace(self, inst: Instance, bundle_dir: Path) -> None:
        self._ok(self.rsync_command(f"{bundle_dir}/", f"{inst.workdir}/bundle/", True), "uploading bundle")

    def launch(self, inst: Instance, job: JobSpec) -> JobHandle:
        pid = self._ok(self.ssh_command(self.remote_launch_script(inst.workdir, job)), "starting job")
        return JobHandle(id=f"{inst.id}-{pid}", instance=inst, log_path=f"{inst.workdir}/job.log",
                         started=time.time(), meta={"pid": pid})

    def status(self, job: JobHandle) -> str:
        wd = job.instance.workdir
        r = self._run(self.ssh_command(f"cat {shlex.quote(wd)}/exit_code 2>/dev/null || echo RUNNING"))
        if r.returncode != 0:
            return "preempted"  # host unreachable: spot reclaim or network loss
        out = r.stdout.strip()
        if out == "RUNNING":
            return RUNNING
        return SUCCEEDED if out == "0" else FAILED

    def stream_logs(self, job: JobHandle) -> Iterator[str]:
        seen = 0
        while True:
            r = self._run(self.ssh_command(f"tail -n +{seen + 1} {shlex.quote(job.log_path)} 2>/dev/null"))
            if r.returncode == 0 and r.stdout:
                lines = r.stdout.splitlines()
                seen += len(lines)
                yield from lines
            if self.status(job) != RUNNING:
                return
            time.sleep(3)

    def fetch_artifacts(self, job: JobHandle, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        self._ok(self.rsync_command(f"{job.instance.workdir}/bundle/output/", f"{dest}/", False), "downloading artifacts")

    def cancel(self, job: JobHandle) -> None:
        pid = job.meta.get("pid")
        if pid:
            self._run(self.ssh_command(f"kill -TERM -{int(pid)} 2>/dev/null || kill -TERM {int(pid)}"))

    def teardown(self, inst: Instance) -> None:
        pass  # the machine is yours; we leave it running
