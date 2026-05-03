"""Tracing abstraction for workflow spans and latency metrics."""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

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
