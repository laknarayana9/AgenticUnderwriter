#!/usr/bin/env python3
"""
One-command product walkthrough for the local Agentic Underwriter demo.

Run from the repo root:
    python scripts/demo_walkthrough.py
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.main import app
from tests.demo_scenarios import get_scenario


client = TestClient(app)


def main() -> None:
    print("\nAgentic Underwriter: one-command demo")
    print("=" * 44)

    missing_info_run = run_missing_roof_age_flow()
    wildfire_run = run_wildfire_review_flow()

    print("\nDone")
    print("-" * 44)
    print(f"Missing-info audit: /runs/{missing_info_run}/audit")
    print(f"Review packet:       /reviews/{wildfire_run}")
    print("\nTo run the API server afterward:")
    print("  uvicorn app.main:app --reload")


def run_missing_roof_age_flow() -> str:
    print("\n1. Missing-info loop: missing roof age")

    response = post("/quote/ho3", {"submission": get_scenario(3)["submission"]})
    assert_status(response, "waiting_for_info")

    run_id = response["run_id"]
    question = response["required_questions"][0]
    print(f"   run_id:   {run_id}")
    print(f"   status:   {response['status']}")
    print(f"   question: {question['question_text']}")

    resumed = post(
        f"/runs/{run_id}/answers",
        {
            "answered_by": "underwriter",
            "answers": {"roof_age_years": 7},
        },
    )
    assert_status(resumed, "completed")
    print(f"   resumed:  {resumed['status']} -> {resumed['decision']['decision']}")
    print(f"   premium:  ${resumed['premium']['annual_premium']:,}")

    audit = get(f"/runs/{run_id}/audit")
    events = [event["event"] for event in audit["workflow_state"]["events"]]
    print(f"   audit:    {' -> '.join(events)}")
    return run_id


def run_wildfire_review_flow() -> str:
    print("\n2. Wildfire evidence and human review")

    submission = deepcopy(get_scenario(2)["submission"])
    response = post("/quote/ho3", {"submission": submission})
    assert_status(response, "waiting_for_info")

    run_id = response["run_id"]
    question = response["required_questions"][0]
    print(f"   run_id:   {run_id}")
    print(f"   status:   {response['status']}")
    print(f"   question: {question['question_text']}")

    resumed = post(
        f"/runs/{run_id}/answers",
        {
            "answered_by": "agent",
            "answers": {"wildfire_mitigation_evidence": True},
        },
    )
    assert_status(resumed, "pending_review")
    print(f"   resumed:  {resumed['status']} -> {resumed['decision']['decision']}")
    print(f"   reasons:  {', '.join(resumed['decision']['review_reason_codes'])}")

    packet = get(f"/reviews/{run_id}")
    print(f"   packet:   AI recommends {packet['ai_recommendation']}")

    approved = post(
        f"/reviews/{run_id}/actions",
        {
            "action": "approve",
            "reviewer": "senior_uw",
            "note": "Demo review: citations and referral rationale reviewed.",
        },
    )
    assert_status(approved, "completed")
    print(f"   review:   final decision {approved['final_decision']}")
    return run_id


def get(path: str) -> Dict[str, Any]:
    with redirect_stdout(StringIO()):
        response = client.get(path)
    response.raise_for_status()
    return response.json()


def post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    with redirect_stdout(StringIO()):
        response = client.post(path, json=payload)
    response.raise_for_status()
    return response.json()


def assert_status(payload: Dict[str, Any], expected: str) -> None:
    actual = payload.get("status")
    if actual != expected:
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise AssertionError(f"Expected status {expected!r}, got {actual!r}")


if __name__ == "__main__":
    main()
