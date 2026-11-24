import pytest

from bitstraining.core.errors import EngineUnavailable
from bitstraining.core.schemas import ExecutionPlan
from bitstraining.deploy import serve_command, write_model_card
from bitstraining.deploy.modelcard import render_model_card
from bitstraining.engines.base import Callbacks, available_engines
from bitstraining.engines.deepspeed import DeepSpeedEngine
from bitstraining.engines.hf_peft import HFPeftEngine
from bitstraining.engines.mlx import MLXEngine
from bitstraining.engines.simulate import SimulateEngine
from bitstraining.engines.unsloth import UnslothEngine
from bitstraining.hardware.database import synthetic_profile
from bitstraining.optimize.quantize import gguf_command


class CharTok:
    """Minimal tokenizer stand-in: one token per character; no chat template."""

    chat_template = None
    eos_token = "!"

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = [ord(c) for c in text]
        return {"input_ids": ids[:max_length] if truncation and max_length else ids}


def test_hf_encode_masks_the_prompt_and_trains_only_on_the_answer():
    row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    enc = HFPeftEngine._encode(CharTok(), row, 512)
    n_answer = len("yo!")
    assert len(enc["input_ids"]) == len(enc["labels"]) == len(enc["attention_mask"])
    assert enc["labels"][-n_answer:] == enc["input_ids"][-n_answer:]
    assert all(x == -100 for x in enc["labels"][:-n_answer])
    assert enc["labels"].count(-100) == len(enc["input_ids"]) - n_answer


def test_hf_encode_truncates_and_skips_rows_without_an_assistant_reply():
    long_row = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a" * 500}]}
    assert len(HFPeftEngine._encode(CharTok(), long_row, 64)["input_ids"]) == 64
    assert HFPeftEngine._encode(CharTok(), {"messages": [{"role": "user", "content": "q"}]}, 64) is None
    # prompt alone fills the window -> no learnable tokens -> row skipped rather than training on nothing
    big_prompt = {"messages": [{"role": "user", "content": "q" * 500}, {"role": "assistant", "content": "a"}]}
    assert HFPeftEngine._encode(CharTok(), big_prompt, 64) is None
    assert HFPeftEngine._encode(CharTok(), {"text": "abc"}, 64)["labels"] == [97, 98, 99]


def test_engine_capabilities_reflect_hardware_and_installation():
    cuda, cpu = synthetic_profile("a100-80gb"), synthetic_profile("a100-80gb")
    cpu.gpus, cpu.gpu_vendor = [], "none"
    planned = HFPeftEngine().capabilities(cuda, assume_installed=True)
    assert planned.available and "qlora" in planned.methods and "paged_adamw8bit" in planned.optimizers
    assert planned.supports_packing and "flash" in planned.attention
    assert "qlora" not in HFPeftEngine().capabilities(cpu, assume_installed=True).methods
    assert not UnslothEngine().capabilities(cpu).available
    assert UnslothEngine().capabilities(cuda, assume_installed=True).available
    assert not MLXEngine().capabilities(cuda, assume_installed=True).available  # not Apple
    d = DeepSpeedEngine().capabilities(cuda, assume_installed=True)
    assert not d.available and "not implemented" in d.reason


def test_unavailable_engines_fail_loudly_with_a_fix():
    with pytest.raises(EngineUnavailable) as e:
        DeepSpeedEngine().train(ExecutionPlan(), None, [], None, Callbacks(), None)
    assert e.value.fix
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        with pytest.raises(EngineUnavailable) as e:
            HFPeftEngine().train(ExecutionPlan(), None, [], None, Callbacks(), None)
        assert "bitstraining[train]" in e.value.fix


def test_auto_engine_selection_never_includes_the_simulator():
    names = {c.name for c in available_engines(synthetic_profile("a100-80gb"), "auto", True)}
    assert "simulate" not in names and "hf_peft" in names
    assert [c.name for c in available_engines(synthetic_profile("t4"), "simulate")] == ["simulate"]


def test_simulate_engine_is_deterministic_labeled_and_resumable(sim_spec, tmp_path):
    plan = ExecutionPlan(engine="simulate", total_steps=20, seed=3)
    rows = [{"text": f"r{i}"} for i in range(50)]
    losses, batches = [], []
    cb = Callbacks(on_step=lambda s, m: losses.append(m["loss"]), on_batch=lambda s, b: batches.append(b))
    res = SimulateEngine().train(plan, sim_spec, rows, None, cb, tmp_path / "a")
    assert res.simulated and res.status == "completed" and res.steps == 25  # extras.max_steps wins
    assert losses[0] > losses[-1]
    assert all(0 <= i < 50 for b in batches for i in b)
    assert "No model was trained" in (tmp_path / "a" / "run_summary.json").read_text()
    losses2 = []
    SimulateEngine().train(plan, sim_spec, rows, None, Callbacks(on_step=lambda s, m: losses2.append(m["loss"])),
                           tmp_path / "b")
    assert losses == losses2  # same seed, same curve
    res3 = SimulateEngine().train(plan, sim_spec, rows, None, Callbacks(), tmp_path / "a",
                                  resume_from=str(tmp_path / "a" / "checkpoint-20"))
    assert res3.steps == 25


def test_simulate_engine_honors_stop_requests(sim_spec, tmp_path):
    count = {"n": 0}

    def stop():
        count["n"] += 1
        return count["n"] > 5

    res = SimulateEngine().train(ExecutionPlan(engine="simulate"), sim_spec, [{"text": "x"}], None,
                                 Callbacks(should_stop=stop), tmp_path)
    assert res.status == "stopped" and res.steps == 5


def test_model_card_flags_simulated_runs_and_includes_lineage(tmp_path):
    run = {"id": "run-1", "simulated": True, "engine": "simulate", "status": "completed",
           "spec": {"name": "demo", "model": {"base": "gpt2"}}, "plan": {"method": "lora", "precision": "bf16"},
           "result": {"steps": 10, "final_loss": 0.5, "peak_gb": 7.5, "tokens_per_s": 900}}
    lin = {"data_hash": "deadbeef", "evals": [{"name": "task", "result": {"delta": 0.1}}],
           "artifacts": [{"kind": "adapter", "path": "out", "sha256": "abc"}]}
    text = render_model_card(run, lin)
    assert "SIMULATED RUN" in text and "deadbeef" in text and "sha256 abc" in text and "7.5 GB" in text
    run["simulated"] = False
    assert "SIMULATED" not in render_model_card(run, lin)
    assert write_model_card(run, lin, tmp_path / "x" / "CARD.md").exists()


def test_serve_and_gguf_commands():
    assert serve_command("m", "vllm", 9000, max_len=4096) == [
        "vllm", "serve", "m", "--host", "127.0.0.1", "--port", "9000", "--max-model-len", "4096"]
    assert serve_command("m.gguf", "llamacpp")[0] == "llama-server"
    with pytest.raises(ValueError):
        serve_command("m", "nope")
    two = gguf_command("merged", "out/m.gguf", "q4_k_m", "/opt/llama.cpp")
    assert len(two) == 2 and two[1][-1] == "Q4_K_M" and two[0][-1] == "f16"
    assert len(gguf_command("merged", "out/m.gguf", "f16", "/opt/llama.cpp")) == 1
    with pytest.raises(ValueError):
        gguf_command("merged", "o.gguf", "q9", "/x")
