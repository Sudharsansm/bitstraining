"""Autotuner: measure, don't guess. Finds the micro-batch that fits and is fastest, learns
calibration ratios from real probes, and caches results per (hardware, model, strategy)."""

from bitstraining.autotune.microbatch import find_max_microbatch
from bitstraining.autotune.tuner import TuneResult, autotune, bake_off

__all__ = ["TuneResult", "autotune", "bake_off", "find_max_microbatch"]
