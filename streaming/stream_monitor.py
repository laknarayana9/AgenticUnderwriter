"""Streaming anomaly monitor over the live run-event feed.

Maintains a rolling window of recent runs and raises anomalies when health
signals breach thresholds — the SRE-for-AI loop: detect a degradation, say what
broke, in real time. Decoupled from the metrics source (fed MetricSnapshots), so
it is unit-testable without a server.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional

_ADVERSE = {"REFER", "DECLINE"}


@dataclass
class MetricSnapshot:
    run_id: str
    latency_ms: float
    succeeded: bool
    decision: Optional[str] = None
    has_citations: bool = False
    timestamp: float = field(default_factory=time.time)

    @property
    def is_adverse(self) -> bool:
        return (self.decision or "").upper() in _ADVERSE


@dataclass
class AnomalyThresholds:
    min_samples: int = 5
    failure_rate_max: float = 0.2
    latency_p95_max_ms: float = 5000.0
    citation_coverage_min: float = 1.0  # adverse decisions must be fully grounded
    min_adverse_samples: int = 3


@dataclass
class Anomaly:
    kind: str
    severity: str
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]


class StreamMonitor:
    """Rolling-window health monitor fed one MetricSnapshot per completed run."""

    def __init__(self, window: int = 200, thresholds: Optional[AnomalyThresholds] = None):
        self._window: Deque[MetricSnapshot] = deque(maxlen=window)
        self._thresholds = thresholds or AnomalyThresholds()
        self._lock = threading.Lock()

    def observe(self, snapshot: MetricSnapshot) -> List[Anomaly]:
        """Record a snapshot and return any anomalies currently active."""
        with self._lock:
            self._window.append(snapshot)
        return self.detect()

    def observe_metric(self, metric: Any) -> List[Anomaly]:
        """Adapter for an observability.RequestMetric (duck-typed)."""
        return self.observe(MetricSnapshot(
            run_id=getattr(metric, "run_id", ""),
            latency_ms=float(getattr(metric, "latency_ms", 0.0)),
            succeeded=bool(getattr(metric, "succeeded", True)),
            decision=getattr(metric, "decision", None),
            has_citations=bool(getattr(metric, "has_citations", False)),
        ))

    def detect(self) -> List[Anomaly]:
        with self._lock:
            window = list(self._window)
        t = self._thresholds
        if len(window) < t.min_samples:
            return []

        anomalies: List[Anomaly] = []
        n = len(window)

        failure_rate = sum(1 for s in window if not s.succeeded) / n
        if failure_rate > t.failure_rate_max:
            anomalies.append(Anomaly("failure_rate", "high", "Workflow failure rate is elevated.",
                                     round(failure_rate, 4), t.failure_rate_max))

        p95 = _p95([s.latency_ms for s in window])
        if p95 > t.latency_p95_max_ms:
            anomalies.append(Anomaly("latency_p95", "medium", "p95 request latency exceeded budget.",
                                     round(p95, 1), t.latency_p95_max_ms))

        adverse = [s for s in window if s.is_adverse]
        if len(adverse) >= t.min_adverse_samples:
            coverage = sum(1 for s in adverse if s.has_citations) / len(adverse)
            if coverage < t.citation_coverage_min:
                anomalies.append(Anomaly("citation_coverage", "high",
                                         "An adverse decision shipped without guideline citations.",
                                         round(coverage, 4), t.citation_coverage_min))
        return anomalies

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            window = list(self._window)
        if not window:
            return {"samples": 0}
        n = len(window)
        adverse = [s for s in window if s.is_adverse]
        return {
            "samples": n,
            "failure_rate": round(sum(1 for s in window if not s.succeeded) / n, 4),
            "latency_p95_ms": round(_p95([s.latency_ms for s in window]), 1),
            "citation_coverage": (
                round(sum(1 for s in adverse if s.has_citations) / len(adverse), 4) if adverse else None
            ),
            "anomalies": [asdict(a) for a in self.detect()],
        }

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            window = list(self._window)
        return [asdict(s) for s in window[-limit:]]

    def clear(self) -> None:
        with self._lock:
            self._window.clear()
