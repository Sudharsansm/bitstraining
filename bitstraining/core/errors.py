"""Exception hierarchy. Every error can carry a human-readable fix suggestion."""

from __future__ import annotations


class BitstrainingError(Exception):
    """Base class. `fix` is shown to the user after the message."""

    def __init__(self, message: str, fix: str | None = None):
        super().__init__(message)
        self.message = message
        self.fix = fix

    def __str__(self) -> str:
        return self.message if not self.fix else f"{self.message}\n  fix: {self.fix}"


class ConfigError(BitstrainingError):
    """The run configuration is invalid."""


class DataError(BitstrainingError):
    """A dataset could not be read or is unusable."""


class PlanningError(BitstrainingError):
    """No feasible execution plan exists for the request."""


class EngineUnavailable(BitstrainingError):
    """A training engine's library is not installed or the engine is not implemented."""


class ComputeError(BitstrainingError):
    """A compute provider failed."""


class BudgetExceeded(ComputeError):
    """A run would exceed (or has exceeded) its hard budget cap."""


class OutOfMemory(BitstrainingError):
    """Raised by probes when a configuration does not fit in device memory."""
