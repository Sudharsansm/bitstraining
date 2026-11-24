"""Training strategies: method definitions shared by every engine (portable across backends)."""

from bitstraining.strategies.catalog import (
    METHODS,
    OPTIMIZER_STATE_BYTES,
    Method,
    weight_bytes_per_param,
)

__all__ = ["METHODS", "OPTIMIZER_STATE_BYTES", "Method", "weight_bytes_per_param"]
