import base64
import io
import json
import zipfile
from pathlib import Path

import pytest

from bitstraining.compute import make_bundle
from bitstraining.compute.base import (
    FAILED,
    PREEMPTED,
    RUNNING,
    SUCCEEDED,
    ComputeProvider,
    Instance,
    JobHandle,
    JobSpec,
    Offer,
    ResourceRequest,
)
from bitstraining.compute.catalog import CatalogProvider
from bitstraining.compute.docker import DockerProvider
from bitstraining.compute.local import LocalProvider
from bitstraining.compute.notebook import NotebookProvider, build_notebook
from bitstraining.compute.orchestrator import run_job
from bitstraining.compute.ssh import SSHProvider
from bitstraining.core.errors import BudgetExceeded, ComputeError
from bitstraining.core.schemas import ExecutionPlan


def tiny_bundle(sim_spec, tmp_path):
    plan = ExecutionPlan(engine="simulate", method="lora", total_steps=12, micro_batch=2, grad_accum=4)
    return make_bundle(sim_spec, plan, tmp_path / "bundle")


# ------------------------------------------------------------------ bundle


def test_bundle_is_self_contained_and_rewrites_paths(sim_spec, tmp_path):
    b = tiny_bundle(sim_spec, tmp_path)
    for f in ("spec.yaml", "plan.json", "requirements.txt", "bootstrap.sh", "src/bitstraining/__init__.py"):
        assert (b / f).exists(), f
    assert "data/train.jsonl" in (b / "spec.yaml").read_text()
    assert str(tmp_path) not in (b / "spec.yaml").read_text()  # no absolute local paths leak
    assert (b / "data" / "train.jsonl").exists()
    assert "bitsandbytes" not in (b / "requirements.txt").read_text()


# ------------------------------------------------------------------ local provider runs a real subprocess


def test_local_provider_runs_the_bundle_as_a_real_job(sim_spec, tmp_path):
    b = tiny_bundle(sim_spec, tmp_path)
    prov = LocalProvider()
    offer = prov.list_offers(ResourceRequest())[0]
    lines = []
    out = run_job(prov, offer, b, JobSpec("t", str(b)), tmp_path / "fetched", log=lines.append)
    assert out.status == "succeeded", "\n".join(lines[-15:])
    assert any(line.startswith("BT_METRIC ") for line in lines)
    res = json.loads((tmp_path / "fetched" / "result.json").read_text())
    assert res["status"] == "completed" and res["simulated"] is True and res["steps"] == 25
    assert out.cost_usd == 0.0 and out.resumes == 0


def test_failed_job_is_reported_failed(sim_spec, tmp_path):
    b = tiny_bundle(sim_spec, tmp_path)
    (b / "plan.json").write_text("{not json")
    prov = LocalProvider()
    out = run_job(prov, prov.list_offers(ResourceRequest())[0], b, JobSpec("t", str(b)), tmp_path / "o",
                  log=lambda s: None)
    assert out.status == "failed"


# ------------------------------------------------------------------ orchestrator semantics with a fake provider


class FakeProvider(ComputeProvider):
    name = "fake"

    def __init__(self, statuses, n_lines=100):
        self.statuses, self.n_lines = list(statuses), n_lines
        self.launches, self.cancelled, self.torn_down, self.fetched = [], False, False, False

    def list_offers(self, req):
        return []

    def provision(self, offer):
        return Instance("i", "fake", offer)

    def sync_workspace(self, inst, bundle_dir):
        pass

    def launch(self, inst, job):
        self.launches.append(job)
        return JobHandle(f"j{len(self.launches)}", inst)

    def status(self, job):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def stream_logs(self, job):
        for i in range(self.n_lines):
            yield f"line {i}"

    def fetch_artifacts(self, job, dest):
        self.fetched = True

    def cancel(self, job):
        self.cancelled = True

    def teardown(self, inst):
        self.torn_down = True


def offer(price):
    return Offer("fake", "x", "t4", 1, 16, price)


def test_budget_cap_cancels_the_job_and_still_tears_down(monkeypatch, tmp_path):
    clock = {"t": 1000.0}

    class T:
        @staticmethod
        def time():
            clock["t"] += 36.0  # every call = 36 simulated seconds
            return clock["t"]

        sleep = staticmethod(lambda s: None)

    monkeypatch.setattr("bitstraining.compute.orchestrator.time", T)
    p = FakeProvider([RUNNING])
    seen = []
    out = run_job(p, offer(10.0), tmp_path, JobSpec("t", str(tmp_path)), tmp_path / "o",
                  budget_usd=1.0, log=seen.append)
    assert out.status == "budget_exceeded" and p.cancelled and p.torn_down and p.fetched
    assert len(seen) < 30  # stopped long before the 100 lines finished
    assert out.cost_usd >= 1.0 - 1e-6


def test_job_that_cannot_afford_five_minutes_is_refused_up_front(tmp_path):
    p = FakeProvider([SUCCEEDED])
    with pytest.raises(BudgetExceeded) as e:
        run_job(p, offer(60.0), tmp_path, JobSpec("t", str(tmp_path)), tmp_path / "o", budget_usd=1.0)
    assert e.value.fix and not p.launches  # nothing was provisioned or started


def test_preemption_triggers_resume_then_success(tmp_path):
    p = FakeProvider([PREEMPTED, SUCCEEDED], n_lines=2)
    out = run_job(p, offer(1.0), tmp_path, JobSpec("t", str(tmp_path)), tmp_path / "o", log=lambda s: None)
    assert out.status == "succeeded" and out.resumes == 1
    assert [j.resume for j in p.launches] == [False, True]


def test_repeated_preemption_gives_up_after_max_resumes(tmp_path):
    p = FakeProvider([PREEMPTED], n_lines=1)
    out = run_job(p, offer(1.0), tmp_path, JobSpec("t", str(tmp_path)), tmp_path / "o", max_resumes=2,
                  log=lambda s: None)
    assert out.status == "failed" and out.resumes == 2 and p.torn_down


def test_instance_is_torn_down_even_if_launch_raises(tmp_path):
    class Boom(FakeProvider):
        def launch(self, inst, job):
            raise ComputeError("no capacity")

    p = Boom([SUCCEEDED])
    with pytest.raises(ComputeError):
        run_job(p, offer(1.0), tmp_path, JobSpec("t", str(tmp_path)), tmp_path / "o")
    assert p.torn_down
    assert FAILED  # status constant exported


# ------------------------------------------------------------------ ssh / docker command construction


class Stub:
    def __init__(self, stdout="", rc=0):
        self.stdout, self.returncode, self.stderr = stdout, rc, ""
        self.cmds = []

    def __call__(self, cmd, *a, **k):
        self.cmds.append(cmd)
        return self


def test_ssh_commands_quote_and_carry_options():
    s = SSHProvider({"host": "ubuntu@10.0.0.5", "port": 2222, "identity_file": "~/.ssh/k"})
    cmd = s.ssh_command("echo hi")
    assert cmd[:2] == ["ssh", "-o"] and "-p" in cmd and "2222" in cmd and "-i" in cmd and cmd[-2] == "ubuntu@10.0.0.5"
    r = s.rsync_command("/local/b/", "~/bt/job/bundle/", True)
    assert r[0] == "rsync" and r[-1] == "ubuntu@10.0.0.5:~/bt/job/bundle/" and "-p 2222" in r[3]
    script = s.remote_launch_script("~/bt/job-1", JobSpec("t", "b", env={"X": "a b; rm -rf /"}, resume=True))
    assert "--resume" in script and "'a b; rm -rf /'" in script  # env values are shell-quoted
    assert "exit_code" in script and script.startswith("nohup bash -c")


def test_ssh_requires_a_host():
    with pytest.raises(ComputeError):
        SSHProvider({}).ssh_command("true")


def test_ssh_status_mapping_including_lost_host():
    h = JobHandle("j", Instance("i", "ssh", workdir="~/w"))
    assert SSHProvider({"host": "h"}, runner=Stub("RUNNING")).status(h) == RUNNING
    assert SSHProvider({"host": "h"}, runner=Stub("0")).status(h) == SUCCEEDED
    assert SSHProvider({"host": "h"}, runner=Stub("1")).status(h) == FAILED
    assert SSHProvider({"host": "h"}, runner=Stub("", rc=255)).status(h) == PREEMPTED


def test_ssh_offer_uses_declared_gpu_and_price():
    o = SSHProvider({"host": "h", "gpu": "a100-80gb", "n_gpus": 2, "price_per_hour": 5.0}).list_offers(ResourceRequest())[0]
    assert o.vram_gb == 80 and o.n_gpus == 2 and o.price_per_hour == 5.0 and o.hardware.n_gpus == 2


def test_docker_command():
    d = DockerProvider({"image": "my/img", "gpus": "all"})
    cmd = d.run_command("bt-1", "/w", JobSpec("t", "b", env={"A": "1"}, resume=True))
    assert cmd[:3] == ["docker", "run", "-d"] and "--gpus" in cmd and "-e" in cmd and "my/img" in cmd
    assert cmd[-1].endswith("--resume")
    assert "--gpus" not in DockerProvider({"gpus": ""}).run_command("n", "/w", JobSpec("t", "b"))


# ------------------------------------------------------------------ notebook / catalog


def test_notebook_embeds_the_bundle_and_is_valid_nbformat(sim_spec, tmp_path):
    nb = build_notebook(tiny_bundle(sim_spec, tmp_path))
    json.dumps(nb)
    assert nb["nbformat"] == 4 and all("cell_type" in c for c in nb["cells"])
    src = "".join(nb["cells"][1]["source"])
    payload = src.split('PAYLOAD = "')[1].split('"')[0]
    names = zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))).namelist()
    assert {"spec.yaml", "plan.json", "src/bitstraining/__init__.py"} <= set(names)


def test_notebook_provider_is_manual_and_fetch_rejects_path_traversal(tmp_path):
    from bitstraining.compute.base import MANUAL

    p = NotebookProvider({"target": "kaggle"})
    assert p.status(JobHandle("j", Instance("i", "notebook"))) == MANUAL
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../../escape.txt", "x")
    h = JobHandle("j", Instance("i", "notebook"), meta={"result_zip": str(evil)})
    with pytest.raises(ComputeError):
        p.fetch_artifacts(h, tmp_path / "dest")
    assert not (tmp_path / "escape.txt").exists()
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("adapter.bin", "x")
    p.fetch_artifacts(JobHandle("j", Instance("i", "notebook"), meta={"result_zip": str(good)}), tmp_path / "ok")
    assert (tmp_path / "ok" / "adapter.bin").exists()


def test_catalog_is_labeled_illustrative_filters_and_cannot_provision():
    c = CatalogProvider()
    offers = c.list_offers(ResourceRequest(min_vram_gb=40, n_gpus=2))
    assert offers and all(o.meta["illustrative"] and o.vram_gb >= 40 for o in offers)
    assert [o.price_per_hour for o in offers] == sorted(o.price_per_hour for o in offers)
    assert all(o.price_per_hour <= 4.0 for o in c.list_offers(ResourceRequest(max_price_per_hour=4.0, n_gpus=2)))
    with pytest.raises(ComputeError) as e:
        c.provision(offers[0])
    assert "ssh" in e.value.fix
    assert Path  # noqa: B018
