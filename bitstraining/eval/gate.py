"""Pre-deploy gate: block export unless the tuned model wins on the task AND did not forget.

Two legs, as in the plan:
  leg 1  task quality: tuned must beat base by at least `min_task_gain` on the task set
  leg 2  forgetting:   on a general-capability set the tuned score may drop at most `max_forgetting`
Optionally a `max_worse_fraction` caps how many individual task rows regressed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bitstraining.core.schemas import Serializable
from bitstraining.eval.audit import AuditReport


@dataclass
class GateCriteria(Serializable):
    min_task_gain: float = 0.0
    max_forgetting: float = 0.05
    max_worse_fraction: float | None = None
    require_forgetting_check: bool = True


@dataclass
class GateResult(Serializable):
    passed: bool = False
    reasons: list[str] = field(default_factory=list)
    task_delta: float | None = None
    forgetting_delta: float | None = None

    def render(self) -> str:
        head = "GATE PASSED" if self.passed else "GATE FAILED"
        return head + "".join(f"\n  - {r}" for r in self.reasons)


def evaluate_gate(
    task: AuditReport,
    general: AuditReport | None = None,
    criteria: GateCriteria | None = None,
) -> GateResult:
    c = criteria or GateCriteria()
    res = GateResult(passed=True, task_delta=task.delta)
    if task.n == 0:
        res.passed = False
        res.reasons.append("task set is empty; nothing was measured")
        return res
    if task.delta < c.min_task_gain:
        res.passed = False
        res.reasons.append(f"task gain {task.delta:+.3f} is below the required {c.min_task_gain:+.3f}")
    else:
        res.reasons.append(f"task gain {task.delta:+.3f} meets the required {c.min_task_gain:+.3f}")
    if c.max_worse_fraction is not None:
        frac = task.worse / task.n
        if frac > c.max_worse_fraction:
            res.passed = False
            res.reasons.append(f"{frac:.1%} of task rows got worse (limit {c.max_worse_fraction:.1%})")
    if general is None:
        if c.require_forgetting_check:
            res.passed = False
            res.reasons.append("no general-capability set supplied, so forgetting was not checked")
    else:
        res.forgetting_delta = general.delta
        if general.delta < -c.max_forgetting:
            res.passed = False
            res.reasons.append(f"general capability dropped {general.delta:+.3f} (limit -{c.max_forgetting:.3f})")
        else:
            res.reasons.append(f"general capability change {general.delta:+.3f} is within tolerance")
    return res
