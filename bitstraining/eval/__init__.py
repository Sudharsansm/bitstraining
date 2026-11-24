"""Evaluation: metrics, per-row audit, forgetting check and a pre-deploy gate."""

from bitstraining.eval.audit import AuditReport, audit_rows
from bitstraining.eval.gate import GateCriteria, GateResult, evaluate_gate
from bitstraining.eval.metrics import METRICS, contains, exact_match, token_f1

__all__ = [
    "METRICS",
    "AuditReport",
    "GateCriteria",
    "GateResult",
    "audit_rows",
    "contains",
    "evaluate_gate",
    "exact_match",
    "token_f1",
]
