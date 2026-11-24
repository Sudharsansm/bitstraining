"""Filesystem locations. Override the root with BITSTRAINING_HOME."""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    root = Path(os.environ.get("BITSTRAINING_HOME", Path.home() / ".bitstraining"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def registry_db() -> Path:
    return home() / "registry.sqlite"


def tuning_cache() -> Path:
    return home() / "tuning_cache.json"


def calibration_file() -> Path:
    return home() / "calibration.json"


def hardware_cache() -> Path:
    return home() / "hardware_profile.json"
