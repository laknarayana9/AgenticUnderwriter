# Real-time run-event monitoring (Tier 4)

The in-domain version of the video's real-time track. Underwriting isn't a voice
or vision pipeline, so instead of streaming audio/video this streams the
workflow's **own run events** and applies the same engineering skills the video
asks for: a live streaming pipeline, latency-budget decomposition, resilience
(timeouts + graceful degradation), and a replay mode for debugging.

## Components (`streaming/`)

| file | role |
|---|---|
| `stream_monitor.py` | rolling-window anomaly detector fed one snapshot per run |
| `latency_budget.py` | decompose a run's per-stage timings into a budget |
| `explain.py` | human-readable anomaly explanation, LLM-with-timeout → deterministic fallback |
| `replay.py` | replay recorded snapshots through a fresh monitor |

## 1. Streaming pipeline

Every completed run already emits an `observability.RequestMetric`. The monitor
subscribes to that stream (`add_metric_listener`), so it sees each run in real
time with **zero changes to the workflow** and no external infrastructure.

Endpoints:
- `GET /monitor/summary` — rolling-window health (failure rate, p95, citation coverage, active anomalies)
- `GET /monitor/anomalies` — current anomalies, each with an explanation
- `WS /ws/monitor` — streams a snapshot on connect and on each client poll

## 2. Anomaly detection

Over a rolling window (default 200 runs), the monitor raises:

| anomaly | condition | why it matters |
|---|---|---|
| `failure_rate` | failures / N > 0.2 | runs are erroring before producing a decision |
| `latency_p95` | p95 latency > budget | tail latency regressed |
| `citation_coverage` | an adverse (REFER/DECLINE) decision shipped uncited | governance alarm — the verifier guardrail is the safety net and this means it leaked |

Thresholds are configurable (`AnomalyThresholds`). Citation coverage defaults to
1.0 because in this system every adverse decision **must** be grounded.

## 3. Latency-budget decomposition

The workflow now times each stage (`WorkflowState.stage_timings`, via the
`timed_stage` context manager that also keeps the trace span). Decompose any run:

```bash
GET /runs/{run_id}/latency-budget
```

```json
{
  "run_id": "...",
  "total_ms": 12.4,
  "stages": [
    {"stage": "retrieval", "duration_ms": 7.1, "pct_of_total": 57.3},
    {"stage": "decision_packaging", "duration_ms": 3.0, "pct_of_total": 24.2}
  ],
  "slowest_stage": "retrieval"
}
```

The budget reflects exactly the stages that ran — a run that pauses at intake
shows only the stages up to the pause. That's the "total is X ms, here's where it
went, and here's what I'd optimize first" answer.

## 4. Resilience

- **Timeouts + graceful degradation.** Anomaly explanation tries an LLM under a
  hard timeout (`explain.py`); on timeout, error, or no provider it falls back to
  a deterministic template. The monitoring path never hangs or fails because an
  optional LLM is slow.
- **Isolated listeners.** A failing metric listener can't break the request path
  (exceptions are swallowed and logged).
- **Replay mode.** `replay.py` feeds recorded snapshots (reconstructable from
  stored run records via `snapshot_from_run_record`) through a fresh monitor and
  returns the anomaly timeline — reproduce a past degradation deterministically.

## Safety / CI

In-process and dependency-free; nothing here touches the deterministic decision
path (stage instrumentation is additive timing only). All logic — anomaly
detection, budget math, explanation fallback, replay, and the WebSocket — is
covered by hermetic tests with no external services.
