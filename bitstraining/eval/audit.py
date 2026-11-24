"""Per-row audit: which examples got better, worse, or stayed the same between two models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from bitstraining.core.schemas import Serializable
from bitstraining.eval.metrics import METRICS, Metric


@dataclass
class AuditReport(Serializable):
    n: int = 0
    base_mean: float = 0.0
    tuned_mean: float = 0.0
    improved: int = 0
    worse: int = 0
    same: int = 0
    rows: list[dict[str, object]] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return self.tuned_mean - self.base_mean

    def summary(self) -> str:
        return (f"{self.n} rows | base {self.base_mean:.3f} -> tuned {self.tuned_mean:.3f} "
                f"({self.delta:+.3f}) | better {self.improved}, worse {self.worse}, same {self.same}")

    def write(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            for r in self.rows:
                f.write(json.dumps(r) + "\n")


def audit_rows(
    references: list[str],
    base_preds: list[str],
    tuned_preds: list[str],
    metric: str | Metric = "token_f1",
    prompts: list[str] | None = None,
    tol: float = 1e-9,
) -> AuditReport:
    """Score both models on every row and classify each as improved / worse / same."""
    if not (len(references) == len(base_preds) == len(tuned_preds)):
        raise ValueError("references, base_preds and tuned_preds must have equal length")
    fn: Metric = METRICS[metric] if isinstance(metric, str) else metric
    rep = AuditReport(n=len(references))
    if not references:
        return rep
    for i, (ref, b, t) in enumerate(zip(references, base_preds, tuned_preds, strict=True)):
        sb, st = fn(b, ref), fn(t, ref)
        verdict = "improved" if st > sb + tol else "worse" if st < sb - tol else "same"
        setattr(rep, {"improved": "improved", "worse": "worse", "same": "same"}[verdict],
                getattr(rep, verdict) + 1)
        rep.rows.append({"row": i, "prompt": prompts[i] if prompts else None, "base": sb, "tuned": st,
                         "verdict": verdict})
        rep.base_mean += sb
        rep.tuned_mean += st
    rep.base_mean /= rep.n
    rep.tuned_mean /= rep.n
    return rep
