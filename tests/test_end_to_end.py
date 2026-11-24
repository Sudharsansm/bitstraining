import json
from dataclasses import replace

import pytest

import bitstraining as bt
from bitstraining.core import registry
from bitstraining.core.errors import ConfigError, PlanningError
from bitstraining.interfaces.cli import main
from bitstraining.tracking import Registry


def test_train_end_to_end_tracks_metrics_batches_and_lineage(sim_spec):
    out = bt.train(sim_spec, log=lambda s: None)
    r = out["result"]
    assert r["status"] == "completed" and r["simulated"] and r["steps"] == 25
    with Registry() as reg:
        run = reg.get_run(out["run_id"])
        assert run["simulated"] == 1 and run["status"] == "completed" and run["data_hash"]
        assert len(reg.metric(out["run_id"], "loss")) == 25
        assert reg.rows_for_steps(out["run_id"], [1])[1]
    assert (sim_spec_out(sim_spec) / "train_normalized.jsonl").exists()


def sim_spec_out(spec):
    from pathlib import Path

    return Path(spec.output)


def test_injected_loss_spike_is_traced_back_to_training_rows(sim_spec):
    spec = replace(sim_spec, extras={"max_steps": 60, "inject_spike_step": 40})
    out = bt.train(spec, log=lambda s: None)
    with Registry() as reg:
        spikes = reg.loss_spikes(out["run_id"])
    assert [s["step"] for s in spikes] == [40]
    n_rows = len((sim_spec_out(spec) / "train_normalized.jsonl").read_text().splitlines())
    assert spikes[0]["rows"] and all(0 <= i < n_rows for i in spikes[0]["rows"])


def test_dry_run_plans_but_runs_nothing(sim_spec):
    out = bt.train(sim_spec, dry_run=True, log=lambda s: None)
    assert "run_id" not in out and out["plan"]["engine"] == "simulate"
    with Registry() as reg:
        assert reg.list_runs() == []


def test_doctor_errors_block_training_unless_forced(sim_spec, tmp_path):
    bad = replace(sim_spec, goal=replace(sim_spec.goal, global_batch_size=10_000))
    with pytest.raises(ConfigError) as e:
        bt.train(bad, log=lambda s: None)
    assert "pre-flight" in str(e.value)
    assert not bt.doctor(bad).ok


def test_doctor_on_missing_data_reports_instead_of_crashing(sim_spec):
    rep = bt.doctor(replace(sim_spec, data=replace(sim_spec.data, train="/nope/missing.jsonl")))
    assert not rep.ok and rep.errors[0].fix


def test_plan_for_named_hardware_works_without_that_hardware(sim_spec):
    spec = replace(sim_spec, engine="auto", model=replace(sim_spec.model, base="meta-llama/Llama-3.1-8B"))
    plans = bt.plan(spec, gpu="a100-80gb", price=2.0)
    assert plans[0].estimate.cost_usd is not None and plans[0].engine == "hf_peft"
    assert plans[0].estimate.fits


def test_local_machine_without_gpu_or_torch_gets_a_helpful_error(sim_spec):
    import importlib.util

    if importlib.util.find_spec("torch"):
        pytest.skip("torch installed here")
    spec = replace(sim_spec, engine="auto", model=replace(sim_spec.model, base="meta-llama/Llama-3.1-8B"))
    with pytest.raises((PlanningError, ConfigError)) as e:
        bt.plan(spec)
    assert e.value.fix


def test_gpu_flag_with_local_provider_is_refused_for_training(sim_spec):
    with pytest.raises(ConfigError) as e:
        bt.train(sim_spec, gpu="t4", log=lambda s: None)
    assert "provider" in e.value.fix


def test_autotune_through_the_api_with_the_simulator(sim_spec):
    r = bt.tune(sim_spec, gpu="rtx 3090")
    assert r.plan.estimate.fits and r.calibration and r.calibration["n"] == 1


def test_remote_path_runs_the_job_in_a_subprocess_and_relays_metrics(sim_spec, monkeypatch):
    from bitstraining.compute.local import LocalProvider

    registry.register("providers", "subproc", LocalProvider)
    spec = replace(sim_spec, compute=replace(sim_spec.compute, provider="subproc"))
    lines = []
    out = bt.train(spec, log=lines.append)
    assert out["result"]["status"] == "completed" and out["result"]["steps"] == 25
    assert any("BT_METRIC" in x for x in lines)
    with Registry() as reg:  # metrics from the subprocess made it into the local registry
        assert len(reg.metric(out["run_id"], "loss")) == 25
    assert (sim_spec_out(spec) / "result.json").exists()


# ------------------------------------------------------------------ CLI


def test_cli_init_plan_doctor_flow(tmp_path, capsys, data_file):
    cfg = tmp_path / "bt.yaml"
    assert main(["init", "--template", "chat", "-o", str(cfg)]) == 0
    assert main(["init", "-o", str(cfg)]) == 2  # refuses to overwrite
    text = cfg.read_text().replace("./data/train.jsonl", str(data_file))
    cfg.write_text(text)
    capsys.readouterr()
    assert main(["plan", "-c", str(cfg), "--gpu", "a100-80gb", "--price", "2", "--top", "2"]) == 0
    out = capsys.readouterr().out
    assert "GB/GPU" in out and "model-based estimates" in out
    assert main(["doctor", "-c", str(cfg), "--gpu", "a100-80gb"]) in (0, 1)


def test_cli_reports_errors_with_fix_and_nonzero_exit(tmp_path, capsys):
    assert main(["plan", "-c", str(tmp_path / "none.yaml")]) == 2
    err = capsys.readouterr().err
    assert "error:" in err and "fix:" in err


def test_cli_data_tools(tmp_path, capsys):
    p = tmp_path / "d.jsonl"
    assert main(["data", "demo", "-o", str(p), "--rows", "40"]) == 0
    assert main(["data", "inspect", str(p)]) == 0
    assert "Schema:          alpaca" in capsys.readouterr().out
    assert main(["data", "clean", str(p), "-o", str(tmp_path / "c.jsonl"), "--near-dup", "0.9"]) == 0
    assert main(["data", "split", str(p), "--prefix", str(tmp_path / "s"), "--eval-fraction", "0.2"]) == 0
    assert (tmp_path / "s.train.jsonl").exists() and (tmp_path / "s.eval.jsonl").exists()


def test_cli_quickstart_and_run_history(capsys):
    assert main(["quickstart"]) == 0
    assert "simulated" in capsys.readouterr().out.lower()
    assert main(["runs", "list"]) == 0
    out = capsys.readouterr().out
    assert "[simulated]" in out
    assert main(["runs", "lineage"]) == 0
    assert main(["runs", "spikes"]) == 0
    assert main(["card", "-o", "CARD_TEST.md"]) == 0
    import os

    assert "SIMULATED RUN" in open("CARD_TEST.md").read()
    os.remove("CARD_TEST.md")


def test_cli_eval_gate_exit_codes(tmp_path, capsys):
    def write(name, rows):
        p = tmp_path / name
        p.write_text("\n".join(json.dumps(r) for r in rows))
        return str(p)

    task = write("task.jsonl", [{"reference": "a", "base": "x", "tuned": "a"}, {"reference": "b", "base": "x", "tuned": "b"}])
    ok = write("gen_ok.jsonl", [{"reference": "c", "base": "c", "tuned": "c"}])
    bad = write("gen_bad.jsonl", [{"reference": "c", "base": "c", "tuned": "z"}])
    assert main(["eval", "--task", task, "--general", ok, "--metric", "exact_match"]) == 0
    assert main(["eval", "--task", task, "--general", bad, "--metric", "exact_match"]) == 1
    assert main(["eval", "--task", task, "--metric", "exact_match"]) == 1  # forgetting unchecked
    assert "GATE" in capsys.readouterr().out


def test_cli_compute_offers_and_plugins(capsys):
    assert main(["compute", "offers", "--min-vram", "40"]) == 0
    assert "illustrative price" in capsys.readouterr().out
    assert main(["plugins"]) == 0
    assert "simulate" in capsys.readouterr().out


def test_cli_notebook_command_writes_a_valid_notebook(tmp_path, sim_spec):
    from bitstraining.core.config import dump_spec

    cfg = tmp_path / "s.yaml"
    dump_spec(replace(sim_spec, engine="auto", model=replace(sim_spec.model, base="Qwen/Qwen2.5-0.5B-Instruct")), cfg)
    nb = tmp_path / "job.ipynb"
    assert main(["notebook", "-c", str(cfg), "-o", str(nb), "--target", "colab"]) == 0
    assert json.loads(nb.read_text())["nbformat"] == 4
