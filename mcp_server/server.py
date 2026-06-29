"""Read-only MCP server (FastMCP) over the governed underwriting system.

Every tool is read-only. There are deliberately NO tools that submit quotes,
approve reviews, or alter a decision — the agent-facing surface inherits the
same invariant the workflow enforces (the LLM/agent never drives the decision).
The guardrail test asserts this registry contains no mutating verbs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from app.underwriting_rules import RULESET_VERSION
from mcp_server import queries

mcp = FastMCP("agentic-underwriter")

# Explicit registry of the (read-only) tools this server exposes. Asserted in
# tests to (a) match what FastMCP actually registered and (b) contain no
# mutating verb — the structural guarantee that this surface cannot drive a run.
READ_ONLY_TOOL_NAMES = [
    "list_runs",
    "get_run",
    "get_decision",
    "get_audit_trail",
    "list_pending_reviews",
    "get_latency_budget",
    "get_metrics",
    "get_anomalies",
]

_DOCS_DIR = Path(__file__).resolve().parent.parent / "app" / "externaldata" / "docs"


# --------------------------------------------------------------------------
# Tools (read-only)
# --------------------------------------------------------------------------
@mcp.tool()
def list_runs(status: Optional[str] = None, limit: int = 50) -> dict:
    """List recent underwriting runs, optionally filtered by status
    (processing, waiting_for_info, pending_review, completed, failed)."""
    return queries.list_runs(status=status, limit=limit)


@mcp.tool()
def get_run(run_id: str, mask_pii: bool = True) -> dict:
    """Get a run's status, decision, and submission. Applicant PII is masked
    unless mask_pii is set to false."""
    return queries.get_run(run_id, mask_pii=mask_pii)


@mcp.tool()
def get_decision(run_id: str) -> dict:
    """Get the decision packet for a run: ACCEPT/REFER/DECLINE, confidence,
    reason codes, citations, rationale source, and next steps."""
    return queries.get_decision(run_id)


@mcp.tool()
def get_audit_trail(run_id: str) -> dict:
    """Get the immutable audit trail for a run: event log, critic verdicts,
    per-stage timings, and ruleset version."""
    return queries.get_audit_trail(run_id)


@mcp.tool()
def list_pending_reviews() -> dict:
    """List REFER/DECLINE runs awaiting human review."""
    return queries.list_pending_reviews()


@mcp.tool()
def get_latency_budget(run_id: str) -> dict:
    """Get the per-stage latency-budget decomposition for a run."""
    return queries.get_latency_budget(run_id)


@mcp.tool()
def get_metrics() -> dict:
    """Get request-quality metrics reconstructed from persisted runs:
    latency percentiles, failure rate, citation coverage, decision mix."""
    return queries.get_metrics()


@mcp.tool()
def get_anomalies() -> dict:
    """Get current health anomalies (failure spikes, latency breaches, uncited
    adverse decisions) reconstructed from persisted runs."""
    return queries.get_anomalies()


# --------------------------------------------------------------------------
# Resources (read-only reference data)
# --------------------------------------------------------------------------
@mcp.resource("ruleset://current")
def ruleset_resource() -> str:
    """The active underwriting ruleset version (the governed decision contract)."""
    return f"Active underwriting ruleset version: {RULESET_VERSION}"


@mcp.resource("guideline://index")
def guideline_index() -> str:
    """Index of available underwriting-guideline documents."""
    docs = sorted(p.stem for p in _DOCS_DIR.glob("*.md"))
    body = "\n".join(f"- guideline://{doc}" for doc in docs)
    return f"Available guideline documents:\n{body}"


@mcp.resource("guideline://{doc_id}")
def guideline_doc(doc_id: str) -> str:
    """The full text of a named underwriting-guideline document."""
    path = _DOCS_DIR / f"{doc_id}.md"
    if not path.exists():
        return f"Guideline '{doc_id}' not found. See guideline://index."
    return path.read_text(encoding="utf-8")
