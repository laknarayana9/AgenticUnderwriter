"""Read-only query layer for the MCP server.

Pure functions over the persisted DB plus metrics reconstructed from stored runs.
No mutation, no LLM, no network. Metrics/anomalies are rebuilt from persisted run
records (not the API's in-memory ring buffer) so they are correct from a separate
MCP process. Unit-testable without an MCP client.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from app.pii_masker import PIIMasker
from app.underwriting_rules import RULESET_VERSION
from storage.database import db
from streaming.latency_budget import decompose_latency
from streaming.replay import snapshot_from_run_record
from streaming.stream_monitor import StreamMonitor

_NOT_FOUND = "run_not_found"


def list_runs(status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    runs = db.list_runs(limit=limit, status=status)
    return {"count": len(runs), "runs": runs}


def _mask_submission(submission: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not submission:
        return submission
    masked, _ = PIIMasker().mask_submission_context(submission)
    return masked


def get_run(run_id: str, mask_pii: bool = True) -> Dict[str, Any]:
    record = db.get_run_record(run_id)
    if record is None:
        return {"error": _NOT_FOUND, "run_id": run_id}
    state = record.workflow_state
    submission = _mask_submission(state.submission_raw) if mask_pii else state.submission_raw
    packet = state.decision_packet
    return {
        "run_id": record.run_id,
        "status": record.status,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        "submission": submission,
        "decision": packet.decision.value if packet else None,
        "needs_human_review": packet.needs_human_review if packet else None,
        "pii_masked": mask_pii,
    }


def get_decision(run_id: str) -> Dict[str, Any]:
    record = db.get_run_record(run_id)
    if record is None:
        return {"error": _NOT_FOUND, "run_id": run_id}
    packet = record.workflow_state.decision_packet
    if packet is None:
        return {"error": "no_decision_packet", "run_id": run_id}
    rationale = packet.producer_rationale
    return {
        "run_id": run_id,
        "decision": packet.decision.value,
        "confidence": packet.decision_confidence,
        "reason_summary": packet.reason_summary,
        "review_reason_codes": packet.review_reason_codes,
        "citations": packet.citations,
        "next_steps": packet.next_steps,
        "rationale_source": rationale.source if rationale else None,
        "needs_human_review": packet.needs_human_review,
    }


def get_audit_trail(run_id: str) -> Dict[str, Any]:
    record = db.get_run_record(run_id)
    if record is None:
        return {"error": _NOT_FOUND, "run_id": run_id}
    state = record.workflow_state
    return {
        "run_id": run_id,
        "ruleset_version": RULESET_VERSION,
        "events": state.events,
        "critic_verdicts": state.critic_verdicts,
        "stage_timings_ms": state.stage_timings,
    }


def list_pending_reviews() -> Dict[str, Any]:
    runs = db.list_runs(limit=50, status="pending_review")
    reviews = []
    for run in runs:
        record = db.get_run_record(run["run_id"])
        packet = record.workflow_state.decision_packet if record else None
        if packet:
            reviews.append({
                "run_id": record.run_id,
                "decision": packet.decision.value,
                "review_reason_codes": packet.review_reason_codes,
                "confidence": packet.decision_confidence,
            })
    return {"count": len(reviews), "pending_reviews": reviews}


def get_latency_budget(run_id: str) -> Dict[str, Any]:
    record = db.get_run_record(run_id)
    if record is None:
        return {"error": _NOT_FOUND, "run_id": run_id}
    timings = record.workflow_state.stage_timings or {}
    if not timings:
        return {"error": "no_stage_timings", "run_id": run_id}
    return {"run_id": run_id, **decompose_latency(timings)}


def _reconstruct_monitor(limit: int = 200) -> StreamMonitor:
    """Rebuild a monitor window from persisted run records (cross-process safe)."""
    monitor = StreamMonitor()
    for run in db.list_runs(limit=limit):
        record = db.get_run_record(run["run_id"])
        if record:
            monitor.observe(snapshot_from_run_record(record))
    return monitor


def get_metrics() -> Dict[str, Any]:
    summary = _reconstruct_monitor().summary()
    summary["source"] = "reconstructed_from_persisted_runs"
    return summary


def get_anomalies() -> Dict[str, Any]:
    anomalies = _reconstruct_monitor().detect()
    return {
        "count": len(anomalies),
        "anomalies": [asdict(a) for a in anomalies],
        "source": "reconstructed_from_persisted_runs",
    }
