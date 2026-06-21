"""Per-request latency-budget decomposition.

Turns a run's per-stage timings (WorkflowState.stage_timings) into the "total is
X ms, here is how it breaks down" view the video highlights — the kind of answer
that shows you think about performance engineering, not just correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class StageBudget:
    stage: str
    duration_ms: float
    pct_of_total: float


def decompose_latency(stage_timings: Dict[str, float]) -> Dict[str, Any]:
    """Decompose stage timings into a budget sorted slowest-first.

    Returns total, per-stage breakdown (duration + % of total), and the single
    slowest stage — the thing you would optimize first.
    """
    total = round(sum(stage_timings.values()), 2)
    budget: List[StageBudget] = []
    for stage, duration in stage_timings.items():
        pct = round(100 * duration / total, 1) if total > 0 else 0.0
        budget.append(StageBudget(stage=stage, duration_ms=round(duration, 2), pct_of_total=pct))

    budget.sort(key=lambda b: b.duration_ms, reverse=True)
    return {
        "total_ms": total,
        "stages": [vars(b) for b in budget],
        "slowest_stage": budget[0].stage if budget else None,
    }
