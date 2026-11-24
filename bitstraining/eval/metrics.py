"""Reference-based metrics. Each maps (prediction, reference) -> score in [0, 1].

For open-ended quality, plug an LLM judge in as any callable with the same signature
(see `bitstraining.eval.audit.audit_rows(metric=...)`). Judges are not bundled, because a judge model is
a deployment choice; treat their scores as noisy and audit per row.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable

Metric = Callable[[str, str], float]

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize(pred) == normalize(ref) else 0.0


def contains(pred: str, ref: str) -> float:
    r = normalize(ref)
    return 1.0 if r and r in normalize(pred) else 0.0


def token_f1(pred: str, ref: str) -> float:
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p or not r:
        return 1.0 if p == r else 0.0
    common = Counter(p) & Counter(r)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(r)
    return 2 * prec * rec / (prec + rec)


METRICS: dict[str, Metric] = {"exact_match": exact_match, "contains": contains, "token_f1": token_f1}
