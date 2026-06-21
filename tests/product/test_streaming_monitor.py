"""Tests for the Tier 4 streaming monitor: anomaly detection, latency budget,
explanation fallback, replay, and the API/WebSocket surface. All hermetic."""

from fastapi.testclient import TestClient

from streaming.latency_budget import decompose_latency
from streaming.stream_monitor import AnomalyThresholds, MetricSnapshot, StreamMonitor
from streaming.explain import explain_anomaly
from streaming.replay import replay_snapshots


def _ok(run_id, latency=100.0, decision="ACCEPT", cited=False):
    return MetricSnapshot(run_id=run_id, latency_ms=latency, succeeded=True,
                          decision=decision, has_citations=cited)


def test_monitor_silent_below_min_samples():
    mon = StreamMonitor()
    for i in range(3):
        assert mon.observe(_ok(f"r{i}")) == []


def test_monitor_detects_failure_rate_spike():
    mon = StreamMonitor(thresholds=AnomalyThresholds(min_samples=5, failure_rate_max=0.2))
    for i in range(4):
        mon.observe(_ok(f"ok{i}"))
    # 2 failures out of 6 = 0.33 > 0.2
    mon.observe(MetricSnapshot("f1", 100, succeeded=False))
    anomalies = mon.observe(MetricSnapshot("f2", 100, succeeded=False))
    kinds = {a.kind for a in anomalies}
    assert "failure_rate" in kinds


def test_monitor_detects_latency_breach():
    mon = StreamMonitor(thresholds=AnomalyThresholds(min_samples=5, latency_p95_max_ms=500))
    anomalies = []
    for i in range(6):
        anomalies = mon.observe(_ok(f"r{i}", latency=900.0))
    assert any(a.kind == "latency_p95" for a in anomalies)


def test_monitor_flags_uncited_adverse_decision():
    mon = StreamMonitor(thresholds=AnomalyThresholds(min_samples=3, min_adverse_samples=3,
                                                     citation_coverage_min=1.0))
    mon.observe(_ok("a1", decision="REFER", cited=True))
    mon.observe(_ok("a2", decision="DECLINE", cited=True))
    anomalies = mon.observe(_ok("a3", decision="REFER", cited=False))  # uncited adverse
    assert any(a.kind == "citation_coverage" for a in anomalies)


def test_latency_budget_decomposition():
    budget = decompose_latency({"retrieval": 60.0, "assessment": 20.0, "rating": 20.0})
    assert budget["total_ms"] == 100.0
    assert budget["slowest_stage"] == "retrieval"
    assert budget["stages"][0]["pct_of_total"] == 60.0


def test_explanation_falls_back_to_deterministic_without_llm():
    mon = StreamMonitor(thresholds=AnomalyThresholds(min_samples=5, latency_p95_max_ms=10))
    anomalies = []
    for i in range(6):
        anomalies = mon.observe(_ok(f"r{i}", latency=900.0))
    result = explain_anomaly(anomalies[0], llm_service=None)
    assert result["source"] == "deterministic"
    assert "latency" in result["explanation"].lower()


def test_replay_reproduces_anomaly_timeline():
    snaps = [MetricSnapshot(f"f{i}", 100, succeeded=False) for i in range(6)]
    result = replay_snapshots(snaps, thresholds=AnomalyThresholds(min_samples=5, failure_rate_max=0.2))
    assert result["replayed"] == 6
    assert any("failure_rate" in entry["anomaly_kinds"] for entry in result["timeline"])


def test_latency_budget_endpoint_and_ws_after_real_run():
    from app.main import app, stream_monitor
    from observability import clear_request_metrics
    clear_request_metrics()
    stream_monitor.clear()
    client = TestClient(app)

    payload = {"submission": {"applicant": {"full_name": "Jane Doe"},
               "risk": {"property_address": "123 Oak St, Palo Alto", "occupancy": "owner_occupied_primary",
                        "dwelling_type": "single_family", "year_built": 1990, "roof_age_years": 25,
                        "construction_type": "frame", "stories": 1},
               "coverage_request": {"coverage_a": 500000, "deductible": 1000}}}
    resp = client.post("/quote/ho3", json=payload)
    run_id = resp.json()["run_id"]

    # Latency budget is populated from the instrumented stages.
    budget = client.get(f"/runs/{run_id}/latency-budget").json()
    assert budget["total_ms"] >= 0
    assert budget["stages"]
    assert budget["slowest_stage"]

    # The monitor saw the run via the metric listener.
    assert client.get("/monitor/summary").json()["samples"] >= 1

    # WebSocket streams a snapshot on connect and on poll.
    with client.websocket_connect("/ws/monitor") as ws:
        first = ws.receive_json()
        assert first["type"] == "snapshot"
        ws.send_text("poll")
        second = ws.receive_json()
        assert second["type"] == "snapshot"
