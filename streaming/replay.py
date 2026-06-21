"""Replay mode — feed recorded run events back through a fresh monitor.

Lets you reproduce a past degradation deterministically for debugging: take a
sequence of recorded snapshots (e.g. reconstructed from stored run records) and
replay them through a StreamMonitor, returning the anomaly timeline. This is the
"feed recorded input back through the pipeline" capability the video calls out.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from streaming.stream_monitor import AnomalyThresholds, MetricSnapshot, StreamMonitor


def snapshot_from_run_record(record: Any) -> MetricSnapshot:
    """Reconstruct a MetricSnapshot from a stored RunRecord-like object."""
    state = record.workflow_state
    packet = state.decision_packet
    total_ms = round(sum((state.stage_timings or {}).values()), 2)
    return MetricSnapshot(
        run_id=state.run_id or "",
        latency_ms=total_ms,
        succeeded=record.status != "failed",
        decision=packet.decision.value if packet else None,
        has_citations=bool(packet.citations) if packet else False,
    )


def replay_snapshots(
    snapshots: Iterable[MetricSnapshot],
    thresholds: AnomalyThresholds | None = None,
) -> Dict[str, Any]:
    """Replay snapshots through a fresh monitor, returning the anomaly timeline."""
    monitor = StreamMonitor(thresholds=thresholds)
    timeline: List[Dict[str, Any]] = []
    for snap in snapshots:
        anomalies = monitor.observe(snap)
        timeline.append({
            "run_id": snap.run_id,
            "anomaly_kinds": [a.kind for a in anomalies],
        })
    return {
        "replayed": len(timeline),
        "final_summary": monitor.summary(),
        "timeline": timeline,
    }
