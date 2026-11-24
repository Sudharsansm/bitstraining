import json

import pytest

from bitstraining.core.errors import DataError
from bitstraining.data import (
    clean_rows,
    detect_schema,
    ffd_pack,
    leakage,
    near_duplicate_groups,
    normalize_row,
    packing_efficiency,
    padding_efficiency,
    read_rows,
    split_rows,
)
from bitstraining.data.pack import pack_sequences, percentile
from bitstraining.data.stats import compute_stats, inspect_report


def test_loaders_all_formats(tmp_path):
    rows = [{"instruction": "a", "output": "b"}, {"instruction": "c", "output": "d"}]
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    (tmp_path / "a.json").write_text(json.dumps({"data": rows}))
    (tmp_path / "a.csv").write_text("instruction,output\na,b\nc,d\n")
    (tmp_path / "a.txt").write_text("para one\n\npara two\n")
    for name in ("a.jsonl", "a.json", "a.csv"):
        assert [r["instruction"] for r in read_rows(tmp_path / name)] == ["a", "c"]
    assert [r["text"] for r in read_rows(tmp_path / "a.txt")] == ["para one", "para two"]
    assert len(read_rows(tmp_path / "a.jsonl", limit=1)) == 1


def test_loader_errors_are_located_and_actionable(tmp_path):
    (tmp_path / "bad.jsonl").write_text('{"a": 1}\nnot json\n')
    with pytest.raises(DataError) as e:
        read_rows(tmp_path / "bad.jsonl")
    assert "bad.jsonl:2" in str(e.value)
    with pytest.raises(DataError):
        read_rows(tmp_path / "missing.jsonl")
    (tmp_path / "x.xyz").write_text("x")
    with pytest.raises(DataError):
        read_rows(tmp_path / "x.xyz")


@pytest.mark.parametrize("row,schema", [
    ({"messages": [{"role": "user", "content": "q"}]}, "messages"),
    ({"conversations": [{"from": "human", "value": "q"}]}, "sharegpt"),
    ({"instruction": "q", "output": "a"}, "alpaca"),
    ({"prompt": "q", "completion": "a"}, "prompt_completion"),
    ({"question": "q", "answer": "a"}, "prompt_completion"),
    ({"prompt": "q", "chosen": "a", "rejected": "b"}, "preference"),
    ({"text": "hello"}, "text"),
    ({"foo": 1}, "unknown"),
])
def test_schema_detection(row, schema):
    assert detect_schema([row]) == schema


def test_normalization_to_internal_shapes():
    n = normalize_row({"instruction": "Do X", "input": "ctx", "output": "done"}, "alpaca")
    assert n["messages"][0]["content"] == "Do X\n\nctx" and n["messages"][1]["role"] == "assistant"
    n = normalize_row({"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]}, "sharegpt")
    assert [m["role"] for m in n["messages"]] == ["user", "assistant"]
    with pytest.raises(DataError):
        normalize_row({"instruction": "x"}, "alpaca")  # missing output


def test_clean_drops_malformed_empty_and_duplicates_and_reports_indices():
    raw = [
        {"instruction": "q1", "output": "a1"},
        {"instruction": "Q1 ", "output": "A1"},           # duplicate after normalization
        {"instruction": "q2", "output": "   "},           # empty answer
        {"instruction": "q3"},                            # malformed
        {"instruction": "q4", "output": "a4"},
    ]
    kept, rep = clean_rows(raw, "alpaca")
    assert (rep.kept, rep.exact_duplicates, rep.empty, rep.malformed) == (2, 1, 1, 1)
    assert rep.examples["exact_duplicate"] == [1] and rep.examples["malformed"] == [3]
    assert len(raw) == 5  # input untouched


def test_near_duplicates_are_found_and_distinct_text_is_not():
    base = ("the quick brown fox jumps over the lazy dog while the sun sets slowly behind the distant "
            "mountains and the river keeps flowing quietly through the valley below")
    variant = base.replace("quietly", "gently")
    other = ("completely unrelated sentence about training large language models on limited hardware "
             "using low rank adapters quantization and careful memory planning for stability")
    groups = near_duplicate_groups([base, variant, other], threshold=0.7)
    assert groups == [[0, 1]]
    kept, rep = clean_rows([{"text": t} for t in (base, variant, other)], "text", near_dup_threshold=0.7)
    assert rep.near_duplicates == 1 and len(kept) == 2


def test_split_is_deterministic_disjoint_and_never_separates_duplicates():
    rows = [{"text": f"row {i % 40}"} for i in range(200)]  # each text appears 5 times
    tr, ev = split_rows(rows, 0.2, seed=1)
    assert tr and ev and len(tr) + len(ev) == 200
    assert leakage(tr, ev) == []
    assert split_rows(rows, 0.2, seed=1) == (tr, ev)
    assert split_rows(rows, 0.2, seed=2) != (tr, ev)
    assert leakage(rows[:10], rows[5:15]) == [0, 1, 2, 3, 4]


def test_ffd_pack_is_valid_partition_within_capacity(lengths):
    bins = ffd_pack(lengths, 1024)
    flat = sorted(i for b in bins for i in b)
    assert flat == list(range(len(lengths)))  # every item exactly once
    assert all(sum(min(lengths[i], 1024) for i in b) <= 1024 for b in bins)


def test_oversize_items_are_truncated_into_their_own_bin():
    bins = ffd_pack([5000, 10, 10], 100)
    assert any(b == [0] for b in bins)


def test_packing_beats_padding_and_grouping_beats_naive(lengths):
    naive = padding_efficiency(lengths, 8, grouped=False, max_len=1024)
    grouped = padding_efficiency(lengths, 8, grouped=True, max_len=1024)
    packed = packing_efficiency(lengths, 1024)
    assert naive < grouped < packed <= 1.0
    assert packed > 0.95


def test_pack_sequences_position_ids_restart_per_member():
    out = pack_sequences([[1, 2, 3], [4, 5], [6]], 5)
    for ids, pos in out:
        assert len(ids) == len(pos) <= 5
        assert pos[0] == 0
    assert sum(len(i) for i, _ in out) == 6


def test_percentile_nearest_rank():
    assert percentile(list(range(101)), 0.5) == 50
    assert percentile(list(range(101)), 0.95) == 95
    assert percentile([], 0.5) == 0


def test_stats_truncation_and_totals():
    rows = [normalize_row({"text": "x" * 400}, "text") for _ in range(10)]
    st = compute_stats(rows, max_length=50, schema="text")
    assert st.num_rows == 10 and st.truncation_rate == 1.0
    assert st.total_tokens == 10 * 50
    assert st.token_counter.startswith("chars")


def test_inspect_report_mentions_padding_waste(data_file):
    r = inspect_report(str(data_file))
    assert "Schema:          alpaca" in r and "Padding waste with sequence packing" in r
