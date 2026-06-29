# MCP Server — read-only observation surface

A standards-compliant [Model Context Protocol](https://modelcontextprotocol.io)
server that lets an external agent (Claude Desktop, Claude Code, any MCP client)
**observe** the governed underwriting system — runs, decisions, audit trails,
pending reviews, and reconstructed metrics.

It is **read-only by design**. There are deliberately no tools that submit a
quote, approve a review, or alter a decision. The agent-facing surface inherits
the system's core invariant: the LLM/agent never drives the governed decision. A
guardrail test asserts the registered tool set contains no mutating verb.

## Run it

```bash
python -m mcp_server          # stdio transport
```

Wire it into an MCP client (e.g. Claude Desktop `claude_desktop_config.json` or a
project `.mcp.json`):

```json
{
  "mcpServers": {
    "agentic-underwriter": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "/absolute/path/to/AgenticUnderwriter"
    }
  }
}
```

Then ask the agent things like *"what was the decision and audit trail for run X?"*
or *"are there any open anomalies?"*.

## Tools (all read-only)

| Tool | Returns |
|---|---|
| `list_runs(status?, limit?)` | recent runs + status |
| `get_run(run_id, mask_pii=true)` | status + decision + submission (PII masked by default) |
| `get_decision(run_id)` | decision, confidence, reason codes, citations, rationale source |
| `get_audit_trail(run_id)` | event log, critic verdicts, per-stage timings, ruleset version |
| `list_pending_reviews()` | REFER/DECLINE runs awaiting a human |
| `get_latency_budget(run_id)` | per-stage latency decomposition |
| `get_metrics()` | latency percentiles, failure rate, citation coverage, decision mix |
| `get_anomalies()` | current health anomalies |

`get_metrics` / `get_anomalies` are **reconstructed from persisted run records**
(not the API's in-memory ring buffer), so they are correct from a separate MCP
process. Responses tag `source: reconstructed_from_persisted_runs`.

## Resources

| URI | Content |
|---|---|
| `ruleset://current` | active underwriting ruleset version |
| `guideline://index` | index of guideline documents |
| `guideline://{doc_id}` | full text of a guideline document |

## Data governance

`get_run` masks applicant PII (name, email, phone, address) by default via the
same `PIIMasker` used at the LLM boundary; pass `mask_pii=false` to retrieve raw
values. No tool writes, and no tool reaches the decision path.

## Why read-only

MCP is usually about letting an agent *act*. Here the whole architecture keeps
the LLM/agent out of the decision and control path, so the MCP surface is scoped
to **observation**: an external agent can explain, audit, and monitor the
governed system, but cannot drive it. That consistency is the point — the
guardrail is enforced in code and asserted in CI, not just documented.
