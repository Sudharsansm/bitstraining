import random

import pytest

from bitstraining.autotune import autotune, find_max_microbatch
from bitstraining.core.schemas import ExecutionPlan
from bitstraining.engines.base import ProbeContext, ProbeResult
from bitstraining.engines.simulate import SimulateEngine
from bitstraining.eval import GateCriteria, audit_rows, evaluate_gate, exact_match, token_f1
from bitstraining.eval.metrics import contains
from bitstraining.hardware.database import synthetic_profile
from bitstraining.models.introspect import load_model_info
from bitstraining.planner import calibration as calib
from bitstraining.planner.search import make_plans
from bitstraining.tracking import Registry, find_loss_spikes

# ------------------------------------------------------------------ micro-batch search


def fake_probe(limit):
    calls = []

    def probe(mb):
        calls.append(mb)
        return ProbeResult(fits=mb <= limit, peak_gb=float(mb))

    probe.calls = calls
    return probe


@pytest.mark.parametrize("limit,expected", [(0, None), (1, 1), (7, 4), (8, 8), (100, 32)])
def test_find_max_microbatch_finds_boundary(limit, expected):
    best, _ = find_max_microbatch(fake_probe(limit), [1, 2, 4, 8, 16, 32])
    assert best == expected


def test_good_hint_needs_few_probes_and_probing_is_logarithmic():
    p = fake_probe(8)
    find_max_microbatch(p, [1, 2, 4, 8, 16, 32, 64], hint=8)
    assert len(p.calls) <= 4
    p = fake_probe(32)
    find_max_microbatch(p, list(range(1, 65)))
    assert len(p.calls) <= 8  # log2(64) + slack, not 64


# ------------------------------------------------------------------ autotune


def _setup(spec, stats, gpu="rtx 3090"):
    model = load_model_info(spec.model.base)
    hw = synthetic_profile(gpu, 1, 1.0)
    caps = SimulateEngine().capabilities(hw)
    plan = make_plans(spec, model, stats, hw, [caps], calibration={})[0]
    return plan, ProbeContext(spec, model, hw, stats, 700), caps


def test_autotune_picks_a_fitting_batch_learns_calibration_and_caches(spec, stats, tmp_path):
    plan, ctx, caps = _setup(spec, stats)
    kw = dict(cache_path=tmp_path / "cache.json", calibration_path=tmp_path / "cal.json")
    r = autotune(plan, ctx, SimulateEngine(), caps, **kw)
    vram = ctx.hw.gpus[0].vram_gb
    assert max(p.peak_gb for p in r.probes.values() if p.fits) <= vram * 1.0
    assert r.plan.micro_batch in r.probes and r.probes[r.plan.micro_batch].fits
    assert r.calibration and r.calibration["n"] == 1
    assert calib.get(calib.key(ctx.hw.key, plan.engine, plan.method), tmp_path / "cal.json")
    assert r.plan.micro_batch * r.plan.grad_accum >= spec.goal.global_batch_size
    again = autotune(plan, ctx, SimulateEngine(), caps, **kw)
    assert again.from_cache and again.plan.micro_batch == r.plan.micro_batch
    assert len(again.probes) == 1  # cached tuning is verified with a single probe


def test_autotune_reports_infeasible_plans_instead_of_crashing(spec, stats, tmp_path):
    plan, ctx, caps = _setup(spec, stats)

    class Nothing(SimulateEngine):
        def probe(self, plan, ctx, micro_batch=None):
            return ProbeResult(fits=False, note="oom")

    r = autotune(plan, ctx, Nothing(), caps, cache_path=tmp_path / "c.json", calibration_path=tmp_path / "k.json")
    assert not r.plan.estimate.fits and r.notes


# ------------------------------------------------------------------ tracking


def noisy_losses(n=80, spike_at=40, seed=1):
    rnd = random.Random(seed)
    out = [1.0 + 2.0 * (0.97 ** i) + rnd.gauss(0, 0.03) for i in range(n)]
    if spike_at:
        out[spike_at - 1] += 1.5
    return out


def test_spike_detector_finds_the_spike_and_nothing_else():
    steps = list(range(1, 81))
    spikes = find_loss_spikes(steps, noisy_losses())
    assert [s for s, _, _ in spikes] == [40]
    assert find_loss_spikes(steps, noisy_losses(spike_at=0)) == []


def test_registry_end_to_end_with_row_tracing(tmp_path):
    with Registry(tmp_path / "r.sqlite") as reg:
        rid = reg.create_run("n", {"model": {"base": "m"}}, {"p": 1}, {"h": 1}, "datahash", "simulate", True)
        for step, loss in enumerate(noisy_losses(), 1):
            reg.log_metrics(rid, step, {"loss": loss})
            reg.log_batch(rid, step, [step * 2 % 17, step * 3 % 17])
        spikes = reg.loss_spikes(rid)
        assert len(spikes) == 1 and spikes[0]["step"] == 40 and len(spikes[0]["rows"]) == 2
        reg.add_eval(rid, "task", {"delta": 0.1})
        (tmp_path / "adapter").mkdir()
        (tmp_path / "adapter" / "w.bin").write_bytes(b"123")
        sha = reg.add_artifact(rid, "adapter", tmp_path / "adapter")
        reg.finish_run(rid, "completed", {"steps": 80})
        other = reg.create_run("n2", {}, {}, {}, "datahash", "simulate", True)
        lin = reg.lineage(rid)
        assert lin["data_hash"] == "datahash" and lin["simulated"] is True
        assert lin["artifacts"][0]["sha256"] == sha and lin["evals"][0]["result"] == {"delta": 0.1}
        assert lin["other_runs_on_same_data"] == [other]
        assert reg.get_run(rid)["status"] == "completed" and reg.latest_run_id() == other


# ------------------------------------------------------------------ eval


def test_metrics():
    assert exact_match("The Cat!", "the cat") == 1.0 and exact_match("a b", "a c") == 0.0
    assert contains("answer is 144 indeed", "144") == 1.0 and contains("x", "") == 0.0
    assert token_f1("x y z", "x y z") == 1.0 and token_f1("x", "y") == 0.0
    assert token_f1("x y", "x y z w") == pytest.approx(2 * 1.0 * 0.5 / 1.5)  # precision 1, recall 1/2
    assert token_f1("the cat", "cat") == 1.0  # articles are ignored (SQuAD-style normalization)


def test_audit_classifies_each_row_and_summarizes():
    refs = ["a", "b", "c", "d"]
    base = ["a", "x", "c", "x"]
    tuned = ["a", "b", "x", "x"]
    rep = audit_rows(refs, base, tuned, "exact_match")
    assert (rep.improved, rep.worse, rep.same) == (1, 1, 2)
    assert rep.base_mean == 0.5 and rep.tuned_mean == 0.5 and rep.delta == 0
    assert [r["verdict"] for r in rep.rows] == ["same", "improved", "worse", "same"]
    with pytest.raises(ValueError):
        audit_rows(["a"], [], [])


def test_audit_writes_jsonl(tmp_path):
    rep = audit_rows(["a"], ["x"], ["a"], "exact_match")
    rep.write(tmp_path / "audit.jsonl")
    assert '"verdict": "improved"' in (tmp_path / "audit.jsonl").read_text()


def test_gate_requires_both_task_gain_and_no_forgetting():
    win = audit_rows(["a", "b"], ["x", "x"], ["a", "b"], "exact_match")          # +1.0
    flat = audit_rows(["a", "b"], ["a", "b"], ["a", "b"], "exact_match")          # 0
    forgot = audit_rows(["a", "b"], ["a", "b"], ["x", "x"], "exact_match")        # -1.0
    assert evaluate_gate(win, flat, GateCriteria(min_task_gain=0.1)).passed
    r = evaluate_gate(win, forgot, GateCriteria())
    assert not r.passed and "dropped" in r.render()
    assert not evaluate_gate(flat, flat, GateCriteria(min_task_gain=0.1)).passed
    assert not evaluate_gate(win, None).passed  # forgetting unchecked -> blocked by default
    assert evaluate_gate(win, None, GateCriteria(require_forgetting_check=False)).passed
    assert not evaluate_gate(audit_rows([], [], []), flat).passed


def test_gate_can_cap_regressed_rows():
    mixed = audit_rows(["a", "b", "c", "d"], ["a", "b", "x", "x"], ["x", "x", "c", "d"], "exact_match")
    assert mixed.delta == 0
    assert not evaluate_gate(mixed, mixed, GateCriteria(max_worse_fraction=0.25)).passed
    assert isinstance(ExecutionPlan(), ExecutionPlan)
