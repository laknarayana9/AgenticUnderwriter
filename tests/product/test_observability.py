from observability import (
    RequestMetric,
    clear_recorded_spans,
    clear_request_metrics,
    get_metrics_summary,
    get_recorded_spans,
    get_tracer,
    record_request_metric,
    record_workflow_latency,
)


def test_tracer_records_completed_span_with_attributes():
    clear_recorded_spans()
    tracer = get_tracer("test_workflow")

    with tracer.start_as_current_span("unit_of_work") as span:
        span.set_attribute("run_id", "run_123")

    spans = get_recorded_spans()
    assert len(spans) == 1
    assert spans[0].name == "unit_of_work"
    assert spans[0].attributes["run_id"] == "run_123"
    assert spans[0].status == "OK"
    assert spans[0].duration_ms is not None


def test_workflow_latency_is_recorded_as_trace_metric():
    clear_recorded_spans()

    record_workflow_latency("phase_a", 42.5)

    span = get_recorded_spans()[0]
    assert span.name == "workflow_latency"
    assert span.attributes["workflow.name"] == "phase_a"
    assert span.attributes["workflow.duration_ms"] == 42.5


def test_metrics_summary_empty_when_no_requests():
    clear_request_metrics()
    assert get_metrics_summary() == {"requests": 0}


def test_metrics_summary_aggregates_latency_failure_and_grounding():
    clear_request_metrics()
    # Two grounded adverse decisions, one ungrounded adverse, one accept, one failure.
    record_request_metric(RequestMetric("r1", latency_ms=100, status="pending_review",
                                        decision="REFER", has_citations=True, llm_used=True,
                                        estimated_cost_usd=0.002))
    record_request_metric(RequestMetric("r2", latency_ms=200, status="pending_review",
                                        decision="DECLINE", has_citations=True))
    record_request_metric(RequestMetric("r3", latency_ms=300, status="pending_review",
                                        decision="REFER", has_citations=False))
    record_request_metric(RequestMetric("r4", latency_ms=50, status="completed",
                                        decision="ACCEPT", has_citations=False))
    record_request_metric(RequestMetric("r5", latency_ms=400, status="failed"))

    s = get_metrics_summary()
    assert s["requests"] == 5
    assert s["failure_rate"] == round(1 / 5, 4)
    assert s["latency_p50_ms"] == 200  # median of 50/100/200/300/400
    # 3 adverse (REFER/DECLINE/REFER), 2 of them cited
    assert s["adverse_decisions"] == 3
    assert s["citation_coverage"] == round(2 / 3, 4)
    assert s["llm_usage_rate"] == round(1 / 5, 4)
    assert s["total_cost_usd"] == 0.002
    assert s["decisions"]["REFER"] == 2


def test_citation_coverage_is_null_without_adverse_decisions():
    clear_request_metrics()
    record_request_metric(RequestMetric("r1", latency_ms=10, status="completed", decision="ACCEPT"))
    s = get_metrics_summary()
    assert s["citation_coverage"] is None
