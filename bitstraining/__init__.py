"""Bitstraining: Train. Fine-tune. Optimize. Deploy.

A hardware-aware orchestration and intelligence layer for model training.
The core imports no heavy ML library; engines load them lazily.
"""

__version__ = "0.1.0"

from bitstraining.interfaces.api import autotune as tune  # noqa: E402  (not `autotune`: that is a subpackage)
from bitstraining.interfaces.api import (  # noqa: E402
    build_context,
    doctor,
    inspect_data,
    plan,
    profile_hardware,
    train,
)

__all__ = [
    "__version__",
    "build_context",
    "doctor",
    "inspect_data",
    "plan",
    "profile_hardware",
    "train",
    "tune",
]
