"""Tracing abstraction for workflow spans and latency metrics."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class SpanRecord:
    trace_id: str
    span_id: str
    name: str
    tracer_name: str
    start_time_unix: float
    end_time_unix: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    status: Optional[str] = None
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time_unix is None:
            return None
        return (self.end_time_unix - self.start_time_unix) * 1000


class SpanExporter(Protocol):
    def export(self, span: SpanRecord) -> None:
        """Persist or emit a completed span."""


class LoggingSpanExporter:
    """Exporter that emits structured span records to application logs."""

    def export(self, span: SpanRecord) -> None:
        logger.info(
            "trace span=%s trace_id=%s span_id=%s status=%s duration_ms=%.2f attributes=%s",
            span.name,
            span.trace_id,
            span.span_id,
            span.status or "OK",
            span.duration_ms or 0.0,
            span.attributes,
        )


class InMemorySpanExporter:
    """Exporter for tests and local audit inspection."""

    def __init__(self):
        self.spans: List[SpanRecord] = []

    def export(self, span: SpanRecord) -> None:
        self.spans.append(span)

    def clear(self) -> None:
        self.spans.clear()


class CompositeSpanExporter:
    def __init__(self, exporters: List[SpanExporter]):
        self.exporters = exporters

    def export(self, span: SpanRecord) -> None:
        for exporter in self.exporters:
            exporter.export(span)


class WorkflowSpan:
    """Context-managed span with an OpenTelemetry-like surface."""

    def __init__(self, tracer_name: str, operation_name: str, exporter: SpanExporter):
        self.record = SpanRecord(
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
            name=operation_name,
            tracer_name=tracer_name,
            start_time_unix=time.time(),
        )
        self.exporter = exporter

    def __enter__(self) -> "WorkflowSpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_val is not None:
            self.record.status = "ERROR"
            self.record.error = str(exc_val)
            self.record.attributes.setdefault("exception.type", getattr(exc_type, "__name__", str(exc_type)))
            self.record.attributes.setdefault("exception.message", str(exc_val))
        else:
            self.record.status = self.record.status or "OK"
        self.record.end_time_unix = time.time()
        self.exporter.export(self.record)

    def set_attribute(self, key: str, value: Any) -> None:
        self.record.attributes[key] = value

    def set_status(self, status: str) -> None:
        self.record.status = status


class WorkflowTracer:
    """Tracer returned to workflow code."""

    def __init__(self, name: str, exporter: SpanExporter):
        self.name = name
        self.exporter = exporter

    def start_as_current_span(self, operation_name: str) -> WorkflowSpan:
        return WorkflowSpan(self.name, operation_name, self.exporter)


class OpenTelemetrySpanExporter:
    """Adapter boundary for OpenTelemetry when the SDK is installed."""

    def __init__(self):
        try:
            from opentelemetry import trace
        except ImportError as exc:
            raise RuntimeError("opentelemetry-api is not installed") from exc
        self.trace = trace

    def export(self, span: SpanRecord) -> None:
        tracer = self.trace.get_tracer(span.tracer_name)
        with tracer.start_as_current_span(span.name) as otel_span:
            for key, value in span.attributes.items():
                otel_span.set_attribute(key, value)
            otel_span.set_attribute("underwriter.trace_id", span.trace_id)
            otel_span.set_attribute("underwriter.span_id", span.span_id)
            if span.duration_ms is not None:
                otel_span.set_attribute("underwriter.duration_ms", span.duration_ms)
            if span.error:
                otel_span.record_exception(Exception(span.error))


_memory_exporter = InMemorySpanExporter()
_provider_exporter: Optional[SpanExporter] = None


def _build_exporter() -> SpanExporter:
    backend = os.getenv("TRACE_BACKEND", "memory").strip().lower()
    exporters: List[SpanExporter] = [_memory_exporter]

    if backend in {"logging", "log"}:
        exporters.append(LoggingSpanExporter())
    elif backend in {"otel", "opentelemetry"}:
        try:
            exporters.append(OpenTelemetrySpanExporter())
        except RuntimeError as exc:
            logger.warning("OpenTelemetry trace backend unavailable; using in-memory tracing: %s", exc)
    elif backend not in {"memory", "in_memory", "none", ""}:
        logger.warning("Unsupported TRACE_BACKEND=%s; using in-memory tracing", backend)

    return CompositeSpanExporter(exporters)


def get_tracer(name: Optional[str] = None) -> WorkflowTracer:
    """Return a tracer with the same surface as common tracing SDKs."""
    global _provider_exporter
    if _provider_exporter is None:
        _provider_exporter = _build_exporter()
    return WorkflowTracer(name or "agentic_underwriter", _provider_exporter)


def get_recorded_spans() -> List[SpanRecord]:
    """Return completed in-memory spans for tests and audit tooling."""
    return list(_memory_exporter.spans)


def clear_recorded_spans() -> None:
    """Clear completed in-memory spans."""
    _memory_exporter.clear()


def record_workflow_latency(workflow_name: str, duration_ms: float) -> None:
    """Record workflow latency as a completed span-compatible metric."""
    tracer = get_tracer("workflow_metrics")
    with tracer.start_as_current_span("workflow_latency") as span:
        span.set_attribute("workflow.name", workflow_name)
        span.set_attribute("workflow.duration_ms", duration_ms)


# ---------------------------------------------------------------------------
# Request-level quality metrics (Tier 2.6)
#
# SRE-style production signals on top of tracing: latency percentiles,
# cost/request, citation coverage (grounding), and failure rate. Kept in-process
# (bounded ring buffer) so it works with zero infrastructure; an optional Langfuse
# sink mirrors each metric when configured.
# ---------------------------------------------------------------------------

_ADVERSE_DECISIONS = {"REFER", "DECLINE"}


@dataclass
class RequestMetric:
    """One processed quote run."""
    run_id: str
    latency_ms: float
    status: str
    decision: Optional[str] = None
    has_citations: bool = False
    llm_used: bool = False
    estimated_cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def succeeded(self) -> bool:
        return self.status != "failed"

    @property
    def is_adverse(self) -> bool:
        return (self.decision or "").upper() in _ADVERSE_DECISIONS


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


class MetricsCollector:
    """Thread-safe bounded collector of request metrics with summary aggregation."""

    def __init__(self, maxlen: int = 1000):
        self._metrics: Deque[RequestMetric] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._langfuse = _LangfuseSink()

    def record(self, metric: RequestMetric) -> None:
        with self._lock:
            self._metrics.append(metric)
        self._langfuse.emit(metric)

    def clear(self) -> None:
        with self._lock:
            self._metrics.clear()

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            metrics = list(self._metrics)
        if not metrics:
            return {"requests": 0}

        latencies = [m.latency_ms for m in metrics]
        failures = [m for m in metrics if not m.succeeded]
        adverse = [m for m in metrics if m.is_adverse]
        adverse_cited = [m for m in adverse if m.has_citations]
        costs = [m.estimated_cost_usd for m in metrics]

        return {
            "requests": len(metrics),
            "failure_rate": round(len(failures) / len(metrics), 4),
            "latency_p50_ms": round(_percentile(latencies, 0.50), 1),
            "latency_p95_ms": round(_percentile(latencies, 0.95), 1),
            # Grounding: adverse (REFER/DECLINE) decisions are the ones that must
            # cite guideline evidence. null when there are no adverse decisions yet.
            "citation_coverage": round(len(adverse_cited) / len(adverse), 4) if adverse else None,
            "adverse_decisions": len(adverse),
            "llm_usage_rate": round(sum(1 for m in metrics if m.llm_used) / len(metrics), 4),
            "total_cost_usd": round(sum(costs), 6),
            "avg_cost_per_request_usd": round(sum(costs) / len(metrics), 6),
            "decisions": _decision_counts(metrics),
        }


def _decision_counts(metrics: List[RequestMetric]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for metric in metrics:
        if metric.decision:
            counts[metric.decision] = counts.get(metric.decision, 0) + 1
    return counts


class _LangfuseSink:
    """Optional Langfuse mirror. A graceful no-op unless the package is installed
    and LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set — so it never blocks the
    request path or CI."""

    def __init__(self):
        self._client = None
        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            return
        try:
            from langfuse import Langfuse
            self._client = Langfuse()
            logger.info("Langfuse metrics sink enabled")
        except Exception as exc:  # import or auth failure → stay a no-op
            logger.info("Langfuse sink unavailable, metrics stay in-process: %s", exc)
            self._client = None

    def emit(self, metric: RequestMetric) -> None:
        if self._client is None:
            return
        try:
            self._client.trace(
                name="underwriting_request",
                metadata={
                    "run_id": metric.run_id,
                    "status": metric.status,
                    "decision": metric.decision,
                    "latency_ms": metric.latency_ms,
                    "has_citations": metric.has_citations,
                    "llm_used": metric.llm_used,
                    "estimated_cost_usd": metric.estimated_cost_usd,
                },
            )
        except Exception as exc:
            logger.debug("Langfuse emit failed (ignored): %s", exc)


_metrics_collector = MetricsCollector()


def record_request_metric(metric: RequestMetric) -> None:
    """Record one processed request into the global metrics collector."""
    _metrics_collector.record(metric)


def get_metrics_summary() -> Dict[str, Any]:
    """Aggregate request metrics: latency p50/p95, failure rate, citation coverage, cost."""
    return _metrics_collector.summary()


def clear_request_metrics() -> None:
    """Reset the metrics collector (tests)."""
    _metrics_collector.clear()
