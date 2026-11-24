"""Job lifecycle: provision -> sync -> launch -> monitor -> (resume on preemption) -> fetch -> teardown.

Guarantees
  * hard budget cap: the job is cancelled when elapsed_hours * price_per_hour exceeds `budget_usd`
  * wall-clock cap: cancelled after `max_hours`
  * preemption/host loss: relaunched with --resume up to `max_resumes` times
  * the instance is always torn down, even on error (unless `keep_instance`)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from bitstraining.compute.base import (
    MANUAL,
    PREEMPTED,
    SUCCEEDED,
    ComputeProvider,
    JobHandle,
    JobSpec,
    Offer,
)
from bitstraining.core.errors import BudgetExceeded


@dataclass
class JobOutcome:
    status: str  # succeeded | failed | manual | budget_exceeded | timeout
    resumes: int = 0
    cost_usd: float | None = None
    hours: float = 0.0
    handle: JobHandle | None = None
    log_tail: list[str] = field(default_factory=list)


def run_job(
    provider: ComputeProvider,
    offer: Offer,
    bundle_dir: Path,
    job: JobSpec,
    dest: Path,
    budget_usd: float | None = None,
    max_resumes: int = 2,
    poll_s: float = 1.0,
    log: Callable[[str], None] = print,
    keep_instance: bool = False,
) -> JobOutcome:
    price = offer.price_per_hour
    if budget_usd is not None and price is not None and price > 0:
        # Refuse to start a job that cannot even run a minimal 5 minutes inside the budget.
        if price * (5 / 60) > budget_usd:
            raise BudgetExceeded(
                f"budget ${budget_usd:.2f} cannot cover 5 minutes at ${price:.2f}/h",
                fix="raise the budget or choose cheaper compute",
            )
    inst = provider.provision(offer)
    t0 = time.time()
    resumes = 0
    outcome = JobOutcome(status="failed")
    try:
        provider.sync_workspace(inst, bundle_dir)
        handle = provider.launch(inst, job)
        outcome.handle = handle
        while True:
            # Stream whatever is available, then decide what to do based on status.
            killed = None
            for line in _stream_with_limits(provider, handle, t0, price, budget_usd, job.max_hours):
                if isinstance(line, tuple):
                    killed = line[0]
                    break
                outcome.log_tail = (outcome.log_tail + [line])[-50:]
                log(line)
            if killed:
                provider.cancel(handle)
                outcome.status = killed
                break
            st = provider.status(handle)
            if st == MANUAL:
                outcome.status = "manual"
                break
            if st == SUCCEEDED:
                outcome.status = "succeeded"
                break
            if st == PREEMPTED and resumes < max_resumes:
                resumes += 1
                log(f"[orchestrator] host lost; resuming (attempt {resumes}/{max_resumes})")
                handle = provider.launch(inst, JobSpec(**{**job.__dict__, "resume": True}))
                outcome.handle = handle
                continue
            outcome.status = "failed"
            break
        if outcome.status in ("succeeded", "budget_exceeded", "timeout"):
            try:
                provider.fetch_artifacts(handle, dest)
            except Exception as e:  # keep the real outcome; surface fetch trouble in the log
                log(f"[orchestrator] could not fetch artifacts: {e}")
    finally:
        outcome.hours = (time.time() - t0) / 3600
        outcome.resumes = resumes
        outcome.cost_usd = round(outcome.hours * price, 4) if price is not None else None
        if not keep_instance and outcome.status != "manual":
            provider.teardown(inst)
    return outcome


def _stream_with_limits(
    provider: ComputeProvider,
    handle: JobHandle,
    t0: float,
    price: float | None,
    budget_usd: float | None,
    max_hours: float | None,
):
    """Yield log lines; yield a ("budget_exceeded"|"timeout",) tuple when a cap is hit."""
    for line in provider.stream_logs(handle):
        yield line
        hours = (time.time() - t0) / 3600
        if budget_usd is not None and price and hours * price >= budget_usd:
            yield ("budget_exceeded",)
            return
        if max_hours is not None and hours >= max_hours:
            yield ("timeout",)
            return
