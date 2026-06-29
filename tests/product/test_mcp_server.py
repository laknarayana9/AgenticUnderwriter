"""Tests for the read-only MCP server.

Hermetic: seeds runs via the FastAPI test client (same SQLite the MCP queries
read), then exercises the query layer and the resources directly — no live MCP
client needed. Includes the governance guardrail test that locks the surface to
read-only.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from mcp_server import queries
from mcp_server.server import READ_ONLY_TOOL_NAMES, guideline_doc, guideline_index, mcp, ruleset_resource

_MUTATING_VERBS = ("submit", "approve", "override", "create", "delete", "update", "set", "post", "write", "run_quote")


@pytest.fixture(scope="module")
def seeded_run_id():
    client = TestClient(app)
    payload = {"submission": {"applicant": {"full_name": "Avery Chen"},
               "risk": {"property_address": "742 Evergreen Terrace, Sacramento fire zone",
                        "occupancy": "owner_occupied_primary", "dwelling_type": "single_family",
                        "year_built": 1990, "roof_age_years": 25, "construction_type": "frame", "stories": 1},
               "coverage_request": {"coverage_a": 500000, "deductible": 1000}}}
    resp = client.post("/quote/ho3", json=payload)
    assert resp.status_code == 200
    return resp.json()["run_id"]


# --- Governance guardrail -------------------------------------------------
def test_registered_tools_match_declared_readonly_set():
    registered = set(mcp._tool_manager._tools.keys())
    assert registered == set(READ_ONLY_TOOL_NAMES)


def test_no_tool_exposes_a_mutating_verb():
    for name in mcp._tool_manager._tools.keys():
        assert not any(verb in name.lower() for verb in _MUTATING_VERBS), f"mutating tool exposed: {name}"


# --- Tools ----------------------------------------------------------------
def test_list_and_get_run(seeded_run_id):
    listing = queries.list_runs(limit=10)
    assert listing["count"] >= 1
    run = queries.get_run(seeded_run_id)
    assert run["run_id"] == seeded_run_id
    assert run["decision"] in {"ACCEPT", "REFER", "DECLINE"}


def test_get_decision_and_audit(seeded_run_id):
    decision = queries.get_decision(seeded_run_id)
    assert decision["decision"] in {"ACCEPT", "REFER", "DECLINE"}
    audit = queries.get_audit_trail(seeded_run_id)
    assert audit["ruleset_version"]
    assert isinstance(audit["events"], list) and audit["events"]


def test_pii_masked_by_default(seeded_run_id):
    masked = queries.get_run(seeded_run_id, mask_pii=True)
    assert masked["pii_masked"] is True
    assert "Avery Chen" not in json.dumps(masked["submission"])
    unmasked = queries.get_run(seeded_run_id, mask_pii=False)
    assert "Avery Chen" in json.dumps(unmasked["submission"])


def test_latency_budget_and_metrics(seeded_run_id):
    budget = queries.get_latency_budget(seeded_run_id)
    assert budget["stages"] and budget["slowest_stage"]
    metrics = queries.get_metrics()
    assert metrics["samples"] >= 1
    assert metrics["source"] == "reconstructed_from_persisted_runs"
    anomalies = queries.get_anomalies()
    assert "count" in anomalies


def test_missing_run_returns_error():
    assert queries.get_run("does-not-exist")["error"] == "run_not_found"
    assert queries.get_decision("does-not-exist")["error"] == "run_not_found"


# --- Resources ------------------------------------------------------------
def test_resources_expose_ruleset_and_guidelines():
    assert "ruleset" in ruleset_resource().lower()
    index = guideline_index()
    assert "guideline://" in index
    # Pull one real guideline doc that the index advertises.
    doc = guideline_doc("uw_guidelines_homeowners")
    assert len(doc) > 100
    assert guideline_doc("nonexistent").lower().startswith("guideline 'nonexistent' not found")
