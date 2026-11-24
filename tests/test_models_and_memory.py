import pytest

from bitstraining.core.errors import ConfigError
from bitstraining.core.schemas import ExecutionPlan
from bitstraining.models.introspect import from_hf_config, load_model_info, lora_param_count
from bitstraining.planner.memory import estimate_memory

# Published parameter counts (billions). Our counts come from config shapes alone.
PUBLISHED = [
    ("meta-llama/Llama-3.1-8B-Instruct", 8.030),
    ("mistralai/Mistral-7B-v0.1", 7.242),
    ("Qwen/Qwen2.5-7B-Instruct", 7.615),
    ("Qwen/Qwen2.5-0.5B-Instruct", 0.494),
    ("meta-llama/Llama-3.2-1B", 1.236),
    ("meta-llama/Llama-3.1-70B", 70.554),
]


@pytest.mark.parametrize("name,billions", PUBLISHED)
def test_param_count_matches_published(name, billions):
    assert load_model_info(name).params / 1e9 == pytest.approx(billions, rel=0.01)


def test_gpt2_within_one_percent():
    assert load_model_info("gpt2").params == pytest.approx(124.4e6, rel=0.01)


def test_hf_config_path_and_local_dir(tmp_path):
    cfg = {"model_type": "llama", "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8,
           "num_hidden_layers": 32, "intermediate_size": 14336, "vocab_size": 128256}
    (tmp_path / "config.json").write_text(__import__("json").dumps(cfg))
    assert load_model_info(str(tmp_path)).params == from_hf_config(cfg).params
    assert load_model_info(str(tmp_path)).source == "config"


def test_unknown_model_without_size_fails_with_fix():
    with pytest.raises(ConfigError) as e:
        load_model_info("acme/mystery-model")
    assert e.value.fix


def test_name_heuristic_trusts_advertised_size_and_is_flagged():
    m = load_model_info("acme/custom-3b-chat")
    assert m.source == "name-heuristic" and m.params == 3_000_000_000


def test_lora_param_count_llama8b_r16_is_42m():
    assert lora_param_count(load_model_info("meta-llama/Llama-3.1-8B"), 16) == pytest.approx(41.9e6, rel=0.01)


def test_lora_scales_linearly_with_rank_and_target_subset_is_smaller():
    m = load_model_info("meta-llama/Llama-3.1-8B")
    assert lora_param_count(m, 32) == 2 * lora_param_count(m, 16)
    assert lora_param_count(m, 16, "qv") < lora_param_count(m, 16, "attention") < lora_param_count(m, 16)


def plan(**k):
    base = dict(method="lora", precision="bf16", quant="none", activation="none", optimizer="adamw",
                micro_batch=1, attention="flash", lora_r=16)
    base.update(k)
    return ExecutionPlan(**base)


M8B = load_model_info("meta-llama/Llama-3.1-8B")


def total(**k):
    seq = k.pop("seq", 512)
    return estimate_memory(M8B, plan(**k), seq)[0]["total"]


def test_method_memory_ordering():
    full = total(method="full")
    lora = total()
    qlora = total(method="qlora", quant="nf4")
    assert full > lora > qlora


def test_full_finetune_8b_is_around_16_bytes_per_param_plus_activations():
    # bf16 weights 2 + fp32 master 4 + bf16 grads 2 + AdamW fp32 m,v 8 = 16 B/param
    assert total(method="full") == pytest.approx(8.03e9 * 16 / 1024**3, rel=0.12)
    assert total(method="full") > 100  # cannot fit on any 80 GB card


def test_qlora_8b_fits_a_consumer_card_but_lora_bf16_needs_more_than_16gb():
    assert total(method="qlora", quant="nf4", activation="full") < 8.0
    assert total() > 16.0


def test_checkpointing_reduces_activations_and_8bit_adam_reduces_optimizer():
    a = estimate_memory(M8B, plan(micro_batch=4), 2048)[0]
    b = estimate_memory(M8B, plan(micro_batch=4, activation="full"), 2048)[0]
    assert b["activations"] < 0.2 * a["activations"]
    f1 = estimate_memory(M8B, plan(method="full"), 512)[0]
    f2 = estimate_memory(M8B, plan(method="full", optimizer="adamw8bit"), 512)[0]
    assert f2["optimizer"] == pytest.approx(f1["optimizer"] / 4, rel=0.01)


def test_memory_grows_with_batch_and_sequence():
    assert total(micro_batch=2) > total(micro_batch=1)
    assert total(seq=2048) > total(seq=512)


def test_fused_loss_and_flash_attention_reduce_memory():
    assert total(fused_ce=True, seq=2048) < total(seq=2048)
    eager = estimate_memory(M8B, plan(attention="eager"), 2048)[0]["activations"]
    flash = estimate_memory(M8B, plan(attention="flash"), 2048)[0]["activations"]
    assert eager > flash


def test_fsdp_shards_weights_across_gpus():
    one = estimate_memory(M8B, plan(method="full"), 512)[0]
    four = estimate_memory(M8B, plan(method="full", n_gpus=4, parallelism="fsdp"), 512)[0]
    assert four["optimizer"] == pytest.approx(one["optimizer"] / 4, rel=0.01)
    assert four["total"] < one["total"] / 2


def test_breakdown_components_are_individually_correct():
    b = estimate_memory(M8B, plan(), 512)[0]
    n = lora_param_count(M8B, 16)
    assert b["context"] == pytest.approx(0.8)  # CUDA context + kernel workspace
    assert estimate_memory(M8B, plan(), 512, device_kind="mps")[0]["context"] == pytest.approx(0.3)
    assert b["adapters"] == pytest.approx(n * 4 / 1024**3, rel=1e-6)   # fp32 adapter weights
    assert b["gradients"] == pytest.approx(n * 4 / 1024**3, rel=1e-6)  # fp32 adapter gradients
    assert b["optimizer"] == pytest.approx(n * 8 / 1024**3, rel=1e-6)  # AdamW m + v in fp32
    assert b["weights"] == pytest.approx(M8B.params * 2 / 1024**3, rel=1e-6)
    parts = sum(v for k, v in b.items() if k != "total")
    assert b["total"] == pytest.approx(parts, rel=1e-6)  # nothing counted twice or dropped


def test_layer_stream_keeps_vram_small_and_moves_the_stack_to_host_ram():
    resident, host_r = estimate_memory(M8B, plan(), 512)
    streamed, host_s = estimate_memory(M8B, plan(layer_stream=True), 512)
    assert streamed["weights"] < 0.4 * resident["weights"]
    assert host_s >= 14  # the bf16 stack lives in host RAM
