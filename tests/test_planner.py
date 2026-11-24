import dataclasses

import pytest

from bitstraining.core.errors import PlanningError
from bitstraining.engines.simulate import SimulateEngine
from bitstraining.hardware.database import lookup_gpu, synthetic_profile
from bitstraining.models.introspect import load_model_info
from bitstraining.planner import calibration as calib
from bitstraining.planner.doctor import run_doctor
from bitstraining.planner.search import estimate_plan, make_plans, suggest_gpus
from bitstraining.planner.throughput import STREAM_OVERLAP_LOSS, estimate_throughput

MODEL = load_model_info("meta-llama/Llama-3.1-8B-Instruct")
CAPS = [SimulateEngine().capabilities(None)]


def plans_for(spec, stats, gpu, n=1, price=2.0, **kw):
    return make_plans(spec, MODEL, stats, synthetic_profile(gpu, n, price), CAPS, calibration={}, **kw)


def test_gpu_lookup_prefers_most_specific_match():
    assert lookup_gpu("NVIDIA GeForce RTX 3050 Laptop GPU", 4.0).vram_gb == 4.0
    assert lookup_gpu("NVIDIA A100-SXM4-80GB").vram_gb == 80
    assert lookup_gpu("NVIDIA H100 80GB HBM3").peak_tflops_bf16 == 990
    assert lookup_gpu("Totally Unknown GPU").peak_tflops_bf16 == 0


def test_plan_picks_quantization_when_memory_is_tight_and_full_precision_when_roomy(spec, stats):
    small = plans_for(spec, stats, "rtx 3090")[0]
    big = plans_for(spec, stats, "a100-80gb")[0]
    assert small.estimate.peak_gb_per_gpu <= 24 * 0.92
    assert big.method == "lora" and big.quant == "none"
    assert big.estimate.useful_tokens_per_s > 2 * small.estimate.useful_tokens_per_s


def test_faster_gpu_gives_shorter_time_and_lower_cost_per_run(spec, stats):
    a = plans_for(spec, stats, "a100-80gb", price=2.5)[0].estimate
    h = plans_for(spec, stats, "h100", price=2.5)[0].estimate
    assert h.total_hours < a.total_hours and h.cost_usd < a.cost_usd


def test_every_returned_plan_fits_and_is_ranked_by_time(spec, stats):
    plans = plans_for(spec, stats, "a100-80gb", top_k=5)
    assert all(p.estimate.fits for p in plans)
    hours = [p.estimate.total_hours for p in plans]
    assert hours == sorted(hours)


def test_returned_plans_are_distinct_strategies(spec, stats):
    plans = plans_for(spec, stats, "a100-80gb", top_k=5)
    keys = [(p.engine, p.method, p.layer_stream, p.parallelism) for p in plans]
    assert len(keys) == len(set(keys))


def test_memory_objective_prefers_smallest_footprint(spec, stats):
    spec = dataclasses.replace(spec, goal=dataclasses.replace(spec.goal, objective="memory"))
    best = plans_for(spec, stats, "a100-80gb")[0]
    fast = plans_for(dataclasses.replace(spec, goal=dataclasses.replace(spec.goal, objective="time")), stats, "a100-80gb")[0]
    assert best.estimate.peak_gb_per_gpu <= fast.estimate.peak_gb_per_gpu


def test_restricting_methods_is_respected(spec, stats):
    spec = dataclasses.replace(spec, goal=dataclasses.replace(spec.goal, methods=["qlora"]))
    assert {p.method for p in plans_for(spec, stats, "a100-80gb")} == {"qlora"}
    spec = dataclasses.replace(spec, goal=dataclasses.replace(spec.goal, allow_quantization=False))
    with pytest.raises(PlanningError):
        plans_for(spec, stats, "a100-80gb")


def test_infeasible_plan_raises_with_actionable_fix_and_gpu_suggestions(spec, stats):
    spec = dataclasses.replace(spec, goal=dataclasses.replace(spec.goal, methods=["full"]))
    with pytest.raises(PlanningError) as e:
        plans_for(spec, stats, "rtx 3090")
    assert "needs" not in str(e.value) or e.value.fix
    assert "--gpu" in e.value.fix and "closest plan" in e.value.fix


def test_full_finetune_is_planned_on_enough_gpus_via_fsdp(spec, stats):
    spec = dataclasses.replace(spec, goal=dataclasses.replace(spec.goal, methods=["full"]))
    plans = plans_for(spec, stats, "a100-80gb", n=4)
    assert all(p.method == "full" for p in plans)
    assert plans[0].parallelism in ("fsdp", "ddp") and plans[0].n_gpus == 4


def test_budget_violations_sort_last_and_are_flagged(spec, stats):
    spec = dataclasses.replace(spec, goal=dataclasses.replace(spec.goal, time_budget_hours=1e-6))
    best = plans_for(spec, stats, "a100-80gb")[0]
    assert any("budget" in n for n in best.estimate.notes)


def test_global_batch_is_reached_through_grad_accumulation(spec, stats):
    for p in plans_for(spec, stats, "rtx 3090", top_k=5):
        assert p.micro_batch * p.grad_accum * p.n_gpus >= spec.goal.global_batch_size
        assert p.micro_batch * p.grad_accum * p.n_gpus < spec.goal.global_batch_size + p.micro_batch * p.n_gpus


def test_pre_ampere_gpu_gets_fp16_and_no_flash(spec, stats):
    p = plans_for(spec, stats, "t4")[0]
    assert p.precision == "fp16" and p.attention != "flash"


def test_packing_raises_useful_throughput_on_short_examples(spec, stats):
    hw = synthetic_profile("a100-80gb", 1, 2.0)
    base = plans_for(spec, stats, "a100-80gb")[0]
    unpacked = dataclasses.replace(base, packing=False)
    packed = dataclasses.replace(base, packing=True)
    eu = estimate_plan(unpacked, spec, MODEL, stats, hw, CAPS[0], calibration={})
    ep = estimate_plan(packed, spec, MODEL, stats, hw, CAPS[0], calibration={})
    assert ep.useful_tokens_per_s > eu.useful_tokens_per_s


def test_attention_kernel_and_precision_affect_speed(spec, stats):
    hw = synthetic_profile("a100-80gb", 1, 2.0)
    base = dataclasses.replace(plans_for(spec, stats, "a100-80gb")[0], packing=False, micro_batch=4)

    def tps(**kw):
        return estimate_throughput(MODEL, dataclasses.replace(base, **kw), hw, CAPS[0], 380, 0.9).raw_tokens_per_s

    assert tps(attention="eager") < tps(attention="sdpa")
    assert tps(fused_ce=True) > tps(fused_ce=False)
    assert tps(precision="fp32") < 0.5 * tps(precision="bf16")


def test_layer_streaming_is_slower_than_resident_when_transfer_bound(spec, stats):
    hw = synthetic_profile("rtx 3090", 1, 1.0)
    base = dataclasses.replace(plans_for(spec, stats, "a100-80gb")[0], micro_batch=1, grad_accum=16,
                               packing=False, method="lora")
    res = estimate_throughput(MODEL, base, hw, CAPS[0], 380, 0.9)
    stream = estimate_throughput(MODEL, dataclasses.replace(base, layer_stream=True), hw, CAPS[0], 380, 0.9)
    assert stream.micro_batch_time_s > res.micro_batch_time_s
    assert stream.bottleneck == "transfer"
    assert 0 < STREAM_OVERLAP_LOSS < 1


def test_confidence_is_low_until_calibrated(spec, stats):
    assert plans_for(spec, stats, "a100-80gb")[0].estimate.confidence == "low"
    hw = synthetic_profile("a100-80gb", 1, 2.0)
    key = calib.key(hw.key, "simulate", "lora")
    cal = {key: {"mem_ratio": 1.0, "speed_ratio": 1.0, "n": 1}}
    plans = make_plans(spec, MODEL, stats, hw, CAPS, calibration=cal)
    assert next(p for p in plans if p.method == "lora").estimate.confidence == "medium"


def test_calibration_scales_estimates(spec, stats):
    hw = synthetic_profile("a100-80gb", 1, 2.0)
    base = plans_for(spec, stats, "a100-80gb")[0]
    key = calib.key(hw.key, base.engine, base.method)
    est = estimate_plan(base, spec, MODEL, stats, hw, CAPS[0],
                        calibration={key: {"mem_ratio": 1.5, "speed_ratio": 0.5}})
    assert est.peak_gb_per_gpu == pytest.approx(base.estimate.peak_gb_per_gpu * 1.5, rel=0.01)
    assert est.raw_tokens_per_s == pytest.approx(base.estimate.raw_tokens_per_s * 0.5, rel=0.01)


def test_calibration_store_smooths_and_clamps(tmp_path):
    f = tmp_path / "c.json"
    calib.update("k", mem_ratio=1.2, path=f)
    assert calib.get("k", f)["mem_ratio"] == pytest.approx(1.2)
    calib.update("k", mem_ratio=1.0, path=f)
    assert calib.get("k", f)["mem_ratio"] == pytest.approx(1.1)
    calib.update("k", mem_ratio=100.0, path=f)  # a crazy probe cannot poison the store
    assert calib.get("k", f)["mem_ratio"] <= 2.5


def test_suggest_gpus_returns_cards_with_enough_memory():
    s = suggest_gpus(40)
    assert s and all(float(x.split("(")[1].split()[0]) * 0.92 >= 40 for x in s)


def test_doctor_flags_problems_with_fixes(spec, stats):
    spec = dataclasses.replace(spec, goal=dataclasses.replace(spec.goal, learning_rate=0.05, task="dpo"))
    st = dataclasses.replace(stats, truncation_rate=0.7, duplicate_rate=0.2, num_rows=5)
    rep = run_doctor(spec, synthetic_profile("a100-80gb", 1), MODEL, st, None, None, check_imports=False)
    codes = {f.code for f in rep.findings}
    assert {"DR010", "DR017", "DR007", "DR006", "DR003", "DR019"} <= codes
    assert not rep.ok and all(f.fix for f in rep.findings if f.level == "error")
    assert "fix:" in rep.render()


def test_doctor_detects_train_eval_leakage(spec, stats):
    rows = [{"text": f"row {i}"} for i in range(100)]
    rep = run_doctor(spec, None, None, stats, None, None, rows, rows[:50], check_imports=False)
    assert any(f.code == "DR008" and f.level == "error" for f in rep.findings)


def test_doctor_clean_run_has_no_errors(spec, stats, tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text("{}")
    spec = dataclasses.replace(spec, data=dataclasses.replace(spec.data, train=str(f)), output=str(tmp_path / "o"))
    plans = plans_for(spec, stats, "a100-80gb")
    rep = run_doctor(spec, synthetic_profile("a100-80gb", 1), MODEL, stats, plans, None, check_imports=False)
    assert rep.ok
