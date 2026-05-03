from observability import clear_recorded_spans, get_recorded_spans, get_tracer, record_workflow_latency


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
