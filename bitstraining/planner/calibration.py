"""Calibration store: measured/predicted ratios learned from real probe runs.

Keyed by "<hardware key>|<engine>|<method>". The planner multiplies its memory estimate and
throughput estimate by these ratios, so predictions improve with use on a given machine.
Ratios are smoothed with an exponential moving average and clamped to a sane range so a single
bad probe cannot poison later plans.
"""

from __future__ import annotations

import json
from pathlib import Path

from bitstraining.core import paths

_LO, _HI = 0.4, 2.5
_ALPHA = 0.5


def _file(path: Path | None) -> Path:
    return path or paths.calibration_file()


def load(path: Path | None = None) -> dict[str, dict[str, float]]:
    f = _file(path)
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return {}


def key(hw_key: str, engine: str, method: str) -> str:
    return f"{hw_key}|{engine}|{method}"


def get(k: str, path: Path | None = None) -> dict[str, float] | None:
    return load(path).get(k)


def update(
    k: str,
    mem_ratio: float | None = None,
    speed_ratio: float | None = None,
    path: Path | None = None,
) -> dict[str, float]:
    """Fold a new observation (measured / predicted) into the stored ratios."""
    data = load(path)
    cur = data.get(k, {"mem_ratio": 1.0, "speed_ratio": 1.0, "n": 0})
    n = cur.get("n", 0)
    for name, val in (("mem_ratio", mem_ratio), ("speed_ratio", speed_ratio)):
        if val is None or val <= 0:
            continue
        val = min(max(val, _LO), _HI)
        cur[name] = val if n == 0 else (1 - _ALPHA) * cur.get(name, 1.0) + _ALPHA * val
    cur["n"] = n + 1
    data[k] = cur
    f = _file(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data, indent=2, sort_keys=True))
    return cur
