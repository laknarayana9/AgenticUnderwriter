# Observability — request quality metrics (Tier 2.6)

On top of per-node tracing (`observability.py`, OpenTelemetry-compatible spans),
the system records SRE-style **request-level quality metrics** and exposes them
live. This is the difference between "it runs" and "I can tell you whether it's
healthy and why it degraded."

## Metrics

Recorded once per processed quote run (`record_request_metric`), aggregated by an
in-process bounded ring buffer (last 1000 runs), and served at `GET /metrics`
(also embedded in `GET /stats` under `quality_metrics`):

| metric | meaning |
|---|---|
| `latency_p50_ms` / `latency_p95_ms` | request latency percentiles (p95 surfaces tail latency that averages hide) |
| `failure_rate` | fraction of runs that errored (`status == failed`) |
| `citation_coverage` | fraction of **adverse** (REFER/DECLINE) decisions that carry ≥1 guideline citation — the grounding signal; `null` until an adverse decision occurs |
| `llm_usage_rate` | fraction of runs whose rationale came from an LLM (vs. deterministic fallback) |
| `total_cost_usd` / `avg_cost_per_request_usd` | estimated LLM cost (see below) |
| `decisions` | count by decision type |

### Why these choices

- **p50/p95, not average** — averages hide worst-case latency; p95 is what a user
  actually feels.
- **Citation coverage is scoped to REFER/DECLINE**, because those are the
  decisions the system *requires* to be grounded (the verifier guardrail enforces
  it). ACCEPT with no citation is legitimately grounded in deterministic rules, so
  including it would dilute the signal. A coverage below 1.0 means an adverse
  decision shipped without evidence — a real alarm.
- **Cost is honest, not invented.** It is 0 unless an LLM produced the rationale,
  and even then uses a configurable blended rate `LLM_COST_PER_1K_USD` (default 0)
  so no provider pricing is hard-coded into the decision path. Set it to your
  model's rate to get real cost/request.

## Example

```bash
curl localhost:8000/metrics
```

```json
{
  "requests": 1,
  "failure_rate": 0.0,
  "latency_p50_ms": 2.6,
  "latency_p95_ms": 2.6,
  "citation_coverage": 1.0,
  "adverse_decisions": 1,
  "llm_usage_rate": 0.0,
  "total_cost_usd": 0.0,
  "decisions": {"REFER": 1}
}
```

## Optional Langfuse sink

If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set and the `langfuse`
package is installed, each request metric is mirrored to Langfuse as a trace.
Otherwise the sink is a **graceful no-op** — metrics stay in-process and nothing
on the request path depends on an external service. This keeps CI hermetic and
the app self-hostable with zero observability infrastructure.

```bash
pip install langfuse
export LANGFUSE_PUBLIC_KEY=pk-... LANGFUSE_SECRET_KEY=sk-...
```

## Tracing vs. metrics

- **Tracing** (`get_tracer`): per-node spans (intake → … → decision packaging),
  each with attributes and timing, exportable to OpenTelemetry (`TRACE_BACKEND=otel`).
- **Metrics** (`get_metrics_summary`): aggregated request-level quality signals.

Both are in-process by default and require no external dependencies.
