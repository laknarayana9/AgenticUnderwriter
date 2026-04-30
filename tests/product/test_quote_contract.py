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
    assert body["status"] == "pending_review"
    assert body["decision"]["decision"] == "REFER"
    assert body["requires_human_review"] is True
    assert "WILDFIRE_HIGH" in body["decision"]["review_reason_codes"]
    assert "WILDFIRE_HIGH" in body["referral_triggers"]
    assert body["citations"]


def test_legacy_quote_payload_remains_supported():
    response = client.post(
        "/quote/run",
        json={
            "applicant_name": "Legacy User",
            "address": "456 Legacy Ln, Oakland, CA 94601",
            "property_type": "single_family",
            "coverage_amount": 300000,
            "construction_year": 2005,
            "roof_type": "composite",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["decision"] == "ACCEPT"
    assert body["premium"]["annual_premium"] > 0
