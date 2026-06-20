from copy import deepcopy

from fastapi.testclient import TestClient

from app.main import app
from tests.demo_scenarios import get_scenario


client = TestClient(app)


def test_ho3_accept_returns_cited_decision_and_premium():
    response = client.post("/quote/ho3", json={"submission": get_scenario(1)["submission"]})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["decision"]["decision"] == "ACCEPT"
    assert body["decision"]["confidence"] >= 0.8
    assert body["requires_human_review"] is False
    assert body["premium"]["annual_premium"] > 0
    assert body["citations"]


def test_ho3_referral_returns_reason_codes_and_citations():
    response = client.post("/quote/ho3", json={"submission": get_scenario(2)["submission"]})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "waiting_for_info"
    assert body["decision"]["decision"] == "REFER"
    assert body["requires_human_review"] is True
    assert body["required_questions"][0]["question_id"] == "wildfire_mitigation_evidence"

    resume_response = client.post(
        f"/runs/{body['run_id']}/answers",
        json={
            "answers": {"wildfire_mitigation_evidence": True},
            "answered_by": "agent",
        },
    )

    assert resume_response.status_code == 200
    resumed = resume_response.json()
    assert resumed["run_id"] == body["run_id"]
    assert resumed["status"] == "pending_review"
    assert resumed["decision"]["decision"] == "REFER"
    assert "ROOF_AGE" in resumed["decision"]["review_reason_codes"]
    assert resumed["citations"]


def test_legacy_quote_payload_remains_supported():
    response = client.post(
        "/quote/run",
        json={
            "submission": {
                "applicant_name": "Legacy User",
                "address": "456 Legacy Ln, Oakland, CA 94601",
                "property_type": "single_family",
                "coverage_amount": 300000,
                "construction_year": 2005,
                "roof_type": "composite",
            },
            "use_agentic": False
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["decision"] == "ACCEPT"
    assert body["premium"]["annual_premium"] > 0


def test_missing_roof_age_can_resume_same_run_to_completion():
    response = client.post("/quote/ho3", json={"submission": get_scenario(3)["submission"]})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "waiting_for_info"
    assert body["required_questions"][0]["question_id"] == "roof_age_years"

    resume_response = client.post(
        f"/runs/{body['run_id']}/answers",
        json={
            "answers": {"roof_age_years": 7},
            "answered_by": "underwriter",
        },
    )

    assert resume_response.status_code == 200
    resumed = resume_response.json()
    assert resumed["run_id"] == body["run_id"]
    assert resumed["status"] == "completed"
    assert resumed["decision"]["decision"] == "ACCEPT"

    audit_response = client.get(f"/runs/{body['run_id']}/audit")
    events = audit_response.json()["workflow_state"]["events"]
    event_names = {event["event"] for event in events}
    # Core workflow events must be present; pii_masked / critic_verdict are additive
    assert {"workflow_paused_for_missing_info", "missing_info_answers_received", "workflow_completed"}.issubset(event_names)


def test_missing_occupancy_asks_targeted_question_and_resumes():
    submission = deepcopy(get_scenario(1)["submission"])
    del submission["risk"]["occupancy"]

    response = client.post("/quote/ho3", json={"submission": submission})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "waiting_for_info"
    assert body["required_questions"][0]["question_id"] == "occupancy"
    assert body["required_questions"][0]["options"]

    resume_response = client.post(
        f"/runs/{body['run_id']}/answers",
        json={"answers": {"occupancy": "owner_occupied_primary"}},
    )

    assert resume_response.status_code == 200
    resumed = resume_response.json()
    assert resumed["run_id"] == body["run_id"]
    assert resumed["status"] == "completed"
