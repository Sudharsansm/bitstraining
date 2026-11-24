"""Local SQLite registry of runs, plans, metrics, artifacts and lineage.

Lineage chain:  data content hash -> run -> artifacts (adapter, exports) -> eval results.
Optional per-step batch row ids let a loss spike be traced back to the rows that caused it.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import statistics
import time
from pathlib import Path
from typing import Any

from bitstraining.core import paths

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, name TEXT, created REAL, finished REAL, status TEXT,
  engine TEXT, simulated INTEGER, data_hash TEXT,
  spec_json TEXT, plan_json TEXT, hardware_json TEXT, result_json TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT, step INTEGER, key TEXT, value REAL
);
CREATE INDEX IF NOT EXISTS idx_metrics ON metrics (run_id, key, step);
CREATE TABLE IF NOT EXISTS batches (run_id TEXT, step INTEGER, rows_json TEXT);
CREATE INDEX IF NOT EXISTS idx_batches ON batches (run_id, step);
CREATE TABLE IF NOT EXISTS artifacts (
  run_id TEXT, kind TEXT, path TEXT, sha256 TEXT, created REAL
);
CREATE TABLE IF NOT EXISTS evals (
  run_id TEXT, name TEXT, created REAL, result_json TEXT
);
"""


def _sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    if p.is_file():
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    elif p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(str(f.relative_to(p)).encode())
                h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:16]


def find_loss_spikes(
    steps: list[int], losses: list[float], window: int = 20, z: float = 4.0
) -> list[tuple[int, float, float]]:
    """Robust spike detector: a loss is a spike if it exceeds the trailing median by `z` robust
    standard deviations (1.4826 * MAD). Returns (step, loss, score) tuples."""
    out: list[tuple[int, float, float]] = []
    for i in range(len(losses)):
        prev = losses[max(0, i - window) : i]
        if len(prev) < 5:
            continue
        med = statistics.median(prev)
        mad = statistics.median([abs(x - med) for x in prev])
        scale = max(1.4826 * mad, 0.02 * max(abs(med), 1e-6), 1e-6)
        score = (losses[i] - med) / scale
        if score > z:
            out.append((steps[i], losses[i], round(score, 2)))
    return out


class Registry:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else paths.registry_db()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Registry:
        return self

    def __exit__(self, *a: object) -> None:
        self.close()

    # ---------------------------------------------------------------- runs
    def create_run(
        self,
        name: str,
        spec: dict[str, Any],
        plan: dict[str, Any],
        hardware: dict[str, Any],
        data_hash: str,
        engine: str,
        simulated: bool = False,
    ) -> str:
        run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
        with self._db:
            self._db.execute(
                "INSERT INTO runs (id,name,created,status,engine,simulated,data_hash,spec_json,plan_json,hardware_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run_id, name, time.time(), "running", engine, int(simulated), data_hash,
                 json.dumps(spec), json.dumps(plan), json.dumps(hardware)),
            )
        return run_id

    def finish_run(self, run_id: str, status: str, result: dict[str, Any] | None = None) -> None:
        with self._db:
            self._db.execute(
                "UPDATE runs SET status=?, finished=?, result_json=? WHERE id=?",
                (status, time.time(), json.dumps(result or {}), run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        for k in ("spec_json", "plan_json", "hardware_json", "result_json"):
            d[k[:-5]] = json.loads(d.pop(k) or "null")
        return d

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT id,name,created,status,engine,simulated,data_hash FROM runs ORDER BY created DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def latest_run_id(self) -> str | None:
        r = self._db.execute("SELECT id FROM runs ORDER BY created DESC LIMIT 1").fetchone()
        return r["id"] if r else None

    # ---------------------------------------------------------------- metrics
    def log_metrics(self, run_id: str, step: int, metrics: dict[str, float]) -> None:
        with self._db:
            self._db.executemany(
                "INSERT INTO metrics (run_id,step,key,value) VALUES (?,?,?,?)",
                [(run_id, step, k, float(v)) for k, v in metrics.items()],
            )

    def metric(self, run_id: str, key: str) -> list[tuple[int, float]]:
        rows = self._db.execute(
            "SELECT step,value FROM metrics WHERE run_id=? AND key=? ORDER BY step", (run_id, key)
        ).fetchall()
        return [(r["step"], r["value"]) for r in rows]

    def log_batch(self, run_id: str, step: int, row_ids: list[int]) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO batches (run_id,step,rows_json) VALUES (?,?,?)",
                (run_id, step, json.dumps(row_ids)),
            )

    def rows_for_steps(self, run_id: str, steps: list[int]) -> dict[int, list[int]]:
        if not steps:
            return {}
        q = ",".join("?" * len(steps))
        rows = self._db.execute(
            f"SELECT step,rows_json FROM batches WHERE run_id=? AND step IN ({q})", [run_id, *steps]
        ).fetchall()
        return {r["step"]: json.loads(r["rows_json"]) for r in rows}

    def loss_spikes(self, run_id: str, window: int = 20, z: float = 4.0) -> list[dict[str, Any]]:
        series = self.metric(run_id, "loss")
        spikes = find_loss_spikes([s for s, _ in series], [v for _, v in series], window, z)
        rows = self.rows_for_steps(run_id, [s for s, _, _ in spikes])
        return [{"step": s, "loss": v, "score": sc, "rows": rows.get(s, [])} for s, v, sc in spikes]

    # ---------------------------------------------------------------- artifacts / evals / lineage
    def add_artifact(self, run_id: str, kind: str, path: str | Path) -> str:
        p = Path(path)
        digest = _sha256_path(p)
        with self._db:
            self._db.execute(
                "INSERT INTO artifacts (run_id,kind,path,sha256,created) VALUES (?,?,?,?,?)",
                (run_id, kind, str(p), digest, time.time()),
            )
        return digest

    def add_eval(self, run_id: str, name: str, result: dict[str, Any]) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO evals (run_id,name,created,result_json) VALUES (?,?,?,?)",
                (run_id, name, time.time(), json.dumps(result)),
            )

    def lineage(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if not run:
            raise KeyError(run_id)
        arts = [dict(r) for r in self._db.execute(
            "SELECT kind,path,sha256 FROM artifacts WHERE run_id=? ORDER BY created", (run_id,))]
        evals = [{"name": r["name"], "result": json.loads(r["result_json"])} for r in self._db.execute(
            "SELECT name,result_json FROM evals WHERE run_id=? ORDER BY created", (run_id,))]
        siblings = [r["id"] for r in self._db.execute(
            "SELECT id FROM runs WHERE data_hash=? AND id<>? ORDER BY created", (run["data_hash"], run_id))]
        return {
            "data_hash": run["data_hash"], "run": run_id, "base": (run["spec"] or {}).get("model", {}).get("base"),
            "engine": run["engine"], "simulated": bool(run["simulated"]), "artifacts": arts,
            "evals": evals, "other_runs_on_same_data": siblings,
        }
