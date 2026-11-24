"""Read training data from JSONL, JSON, CSV/TSV, TXT, Parquet, folders of text, or `hf://` datasets."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bitstraining.core.errors import DataError

_TEXT_EXT = {".txt", ".md"}


def _read_jsonl(p: Path) -> Iterator[dict[str, Any]]:
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise DataError(
                    f"{p}:{i} is not valid JSON ({e.msg})",
                    fix="each line must be one JSON object; run `bitstraining data inspect` to locate bad rows",
                ) from e
            if not isinstance(obj, dict):
                raise DataError(f"{p}:{i} must be a JSON object, got {type(obj).__name__}")
            yield obj


def _read_json(p: Path) -> Iterator[dict[str, Any]]:
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("data", "rows", "examples", "train"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            raise DataError(f"{p} is a JSON object; expected a list of rows")
    for obj in data:
        if isinstance(obj, dict):
            yield obj
        else:
            raise DataError(f"{p} contains a non-object row: {str(obj)[:60]!r}")


def _read_csv(p: Path, delimiter: str) -> Iterator[dict[str, Any]]:
    with p.open(encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter=delimiter)


def _read_txt(p: Path) -> Iterator[dict[str, Any]]:
    """Blank-line separated paragraphs become rows; one-document files become one row."""
    text = p.read_text(encoding="utf-8")
    parts = [t.strip() for t in text.split("\n\n") if t.strip()]
    for part in parts or [text.strip()]:
        if part:
            yield {"text": part}


def _read_parquet(p: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise DataError(
            "reading Parquet needs pyarrow", fix="pip install 'bitstraining[parquet]'"
        ) from e
    yield from pq.read_table(p).to_pylist()


def _read_hf(spec: str) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise DataError(
            "hf:// datasets need the `datasets` package", fix="pip install 'bitstraining[train]'"
        ) from e
    name, _, split = spec.partition(":")
    ds = load_dataset(name, split=split or "train")
    yield from (dict(r) for r in ds)


def read_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load all rows (optionally the first `limit`). Raises DataError with a fix on failure."""
    s = str(path)
    if s.startswith("hf://"):
        it: Iterator[dict[str, Any]] = _read_hf(s[5:])
    else:
        p = Path(s)
        if not p.exists():
            raise DataError(f"data file not found: {p}", fix="check data.train in your config")
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.suffix.lower() in _TEXT_EXT | {".jsonl", ".json", ".csv", ".parquet"})
            if not files:
                raise DataError(f"no readable files in {p}")
            return _concat(files, limit)
        it = _dispatch(p)
    rows: list[dict[str, Any]] = []
    for r in it:
        rows.append(r)
        if limit and len(rows) >= limit:
            break
    return rows


def _dispatch(p: Path) -> Iterator[dict[str, Any]]:
    ext = p.suffix.lower()
    if ext == ".jsonl" or ext == ".ndjson":
        return _read_jsonl(p)
    if ext == ".json":
        return _read_json(p)
    if ext == ".csv":
        return _read_csv(p, ",")
    if ext == ".tsv":
        return _read_csv(p, "\t")
    if ext in _TEXT_EXT:
        return _read_txt(p)
    if ext == ".parquet":
        return _read_parquet(p)
    raise DataError(
        f"unsupported data format '{ext}'", fix="use .jsonl, .json, .csv, .tsv, .txt or .parquet"
    )


def _concat(files: list[Path], limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in files:
        for r in _dispatch(f):
            rows.append(r)
            if limit and len(rows) >= limit:
                return rows
    return rows
