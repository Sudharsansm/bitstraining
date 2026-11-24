import json
import random

import pytest

from bitstraining.core.schemas import (
    ComputeSpec,
    DatasetStats,
    DataSpec,
    GoalSpec,
    ModelSpec,
    RunSpec,
)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own ~/.bitstraining and never touches the network."""
    monkeypatch.setenv("BITSTRAINING_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BITSTRAINING_OFFLINE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)


@pytest.fixture
def lengths():
    rnd = random.Random(0)
    return [rnd.randint(60, 700) for _ in range(2000)]


@pytest.fixture
def stats(lengths):
    return DatasetStats(
        num_rows=len(lengths), total_tokens=sum(lengths), mean_len=sum(lengths) / len(lengths),
        p50_len=380, p95_len=660, max_len=700, lengths_sample=lengths, schema="alpaca", data_hash="abc",
    )


@pytest.fixture
def spec():
    return RunSpec(
        name="t", model=ModelSpec(base="meta-llama/Llama-3.1-8B-Instruct"),
        data=DataSpec(train="x.jsonl", max_length=1024),
        goal=GoalSpec(epochs=3, global_batch_size=16), compute=ComputeSpec(),
    )


@pytest.fixture
def data_file(tmp_path):
    p = tmp_path / "train.jsonl"
    with p.open("w") as f:
        for i in range(120):
            f.write(json.dumps({"instruction": f"Question number {i} about topic {i % 7}", "input": "",
                                "output": f"Answer {i} " * (1 + i % 5)}) + "\n")
    return p


@pytest.fixture
def sim_spec(tmp_path, data_file):
    from bitstraining.core.config import spec_from_dict

    return spec_from_dict({
        "name": "simtest", "engine": "simulate",
        "model": {"base": "Qwen/Qwen2.5-0.5B-Instruct"},
        "data": {"train": str(data_file), "max_length": 512, "eval_split": 0.1},
        "goal": {"task": "sft", "epochs": 2, "global_batch_size": 8},
        "output": str(tmp_path / "out"), "extras": {"max_steps": 25},
    })
