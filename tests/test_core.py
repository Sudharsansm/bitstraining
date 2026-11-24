import pytest

from bitstraining.core import registry
from bitstraining.core.config import load_spec, spec_from_dict
from bitstraining.core.errors import BitstrainingError, ConfigError
from bitstraining.core.schemas import ExecutionPlan, RunSpec


def base(**over):
    d = {"model": {"base": "gpt2"}, "data": {"train": "x.jsonl"}}
    d.update(over)
    return d


def test_unknown_field_suggests_close_match():
    with pytest.raises(ConfigError) as e:
        spec_from_dict(base(goal={"epoch": 3}))
    assert "did you mean 'epochs'" in str(e.value)


@pytest.mark.parametrize("bad", [True, "3", float("nan"), float("inf")])
def test_numeric_fields_reject_bool_string_nan(bad):
    with pytest.raises(ConfigError):
        spec_from_dict(base(goal={"learning_rate": bad}))


@pytest.mark.parametrize("goal", [{"epochs": True}, {"global_batch_size": True}, {"lora_r": False}, {"seed": True}])
def test_booleans_are_not_numbers_even_when_they_would_pass_range_checks(goal):
    # True == 1 would be a *valid* epoch count / batch size, so only a type check can catch it.
    with pytest.raises(ConfigError) as e:
        spec_from_dict(base(goal=goal))
    assert "must be" in str(e.value)


def test_integer_field_rejects_fraction_but_accepts_whole_float():
    with pytest.raises(ConfigError):
        spec_from_dict(base(goal={"global_batch_size": 8.5}))
    assert spec_from_dict(base(goal={"global_batch_size": 8.0})).goal.global_batch_size == 8


@pytest.mark.parametrize("goal", [{"task": "rl"}, {"objective": "vibes"}, {"epochs": 0},
                                  {"learning_rate": 5.0}, {"methods": ["qlora", "bogus"]}])
def test_semantic_validation(goal):
    with pytest.raises(ConfigError):
        spec_from_dict(base(goal=goal))


def test_required_fields():
    with pytest.raises(ConfigError):
        spec_from_dict({"data": {"train": "x"}})
    with pytest.raises(ConfigError):
        spec_from_dict({"model": {"base": "gpt2"}})


def test_yaml_roundtrip(tmp_path):
    from bitstraining.core.config import dump_spec

    s = spec_from_dict(base(goal={"epochs": 2, "methods": ["lora"]}, compute={"gpu": "t4"}))
    dump_spec(s, tmp_path / "s.yaml")
    assert load_spec(tmp_path / "s.yaml") == s


def test_missing_config_file_has_fix():
    with pytest.raises(ConfigError) as e:
        load_spec("nope.yaml")
    assert e.value.fix


def test_plan_json_roundtrip_and_hash_is_stable():
    p = ExecutionPlan(method="qlora", micro_batch=4)
    assert ExecutionPlan.from_dict(p.to_dict()) == p
    assert p.content_hash() == ExecutionPlan(method="qlora", micro_batch=4).content_hash()
    assert p.content_hash() != ExecutionPlan(method="qlora", micro_batch=8).content_hash()


def test_plugin_registry_builtin_runtime_and_unknown():
    assert {"simulate", "hf_peft", "unsloth", "mlx", "deepspeed"} <= set(registry.names("engines"))
    assert {"local", "ssh", "docker", "notebook", "catalog"} <= set(registry.names("providers"))

    class Mine:
        pass

    registry.register("engines", "mine", Mine)
    assert isinstance(registry.create("engines", "mine"), Mine)
    with pytest.raises(BitstrainingError) as e:
        registry.get("engines", "nope")
    assert "available:" in str(e.value)


def test_listing_plugins_does_not_import_heavy_libraries():
    import subprocess
    import sys

    code = ("import sys; from bitstraining.core import registry; registry.names('engines');"
            "sys.exit(1 if any(m in sys.modules for m in ('torch','transformers','peft')) else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
    assert isinstance(RunSpec(), RunSpec)
