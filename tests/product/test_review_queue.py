from fastapi.testclient import TestClient

from app.main import app
from tests.demo_scenarios import get_scenario


client = TestClient(app)


def _create_pending_review():
    response = client.post("/quote/ho3", json={"submission": get_scenario(4)["submission"]})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending_review"
    return body


def test_pending_reviews_list_and_decision_packet_view():
    pending = _create_pending_review()

    queue_response = client.get("/reviews/pending")
    assert queue_response.status_code == 200
    queue = queue_response.json()
    queued_ids = {review["run_id"] for review in queue["reviews"]}
    assert pending["run_id"] in queued_ids

    packet_response = client.get(f"/reviews/{pending['run_id']}")
    assert packet_response.status_code == 200
    packet = packet_response.json()
    assert packet["decision_packet"]["decision"] == "REFER"
    assert packet["ai_recommendation"] == "REFER"
    assert packet["final_review_decision"]["final_decision"] is None


def test_approve_stores_final_decision_without_overwriting_ai_recommendation():
    pending = _create_pending_review()

    response = client.post(
        f"/reviews/{pending['run_id']}/actions",
        json={
            "action": "approve",
            "reviewer": "senior_uw",
            "note": "Citations and referral rationale reviewed.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["ai_recommendation"] == "REFER"
    assert body["final_decision"] == "REFER"
    assert body["decision_packet"]["decision"] == "REFER"

    packet = client.get(f"/reviews/{pending['run_id']}").json()
    assert packet["decision_packet"]["decision"] == "REFER"
    assert packet["final_review_decision"]["final_decision"] == "REFER"


def test_override_requires_reviewer_note_and_records_separate_final_decision():
    pending = _create_pending_review()

    missing_note = client.post(
        f"/reviews/{pending['run_id']}/actions",
        json={
            "action": "override",
            "reviewer": "senior_uw",
            "final_decision": "ACCEPT",
        },
    )
    assert missing_note.status_code == 400
    assert "note" in missing_note.json()["detail"].lower()

    response = client.post(
        f"/reviews/{pending['run_id']}/actions",
        json={
            "action": "override",
            "reviewer": "senior_uw",
            "final_decision": "ACCEPT",
            "note": "Updated inspection documentation supports acceptance.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ai_recommendation"] == "REFER"
    assert body["final_decision"] == "ACCEPT"
    assert body["decision_packet"]["decision"] == "REFER"
    assert body["review"]["reviewer_notes"] == "Updated inspection documentation supports acceptance."


def test_reviewer_can_request_more_info_from_pending_review():
    pending = _create_pending_review()

    response = client.post(
        f"/reviews/{pending['run_id']}/actions",
        json={
            "action": "request_more_info",
            "reviewer": "senior_uw",
            "requested_info": ["Provide electrical update documentation."],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "waiting_for_info"
    assert body["final_decision"] is None

    run_response = client.get(f"/runs/{pending['run_id']}")
    run_body = run_response.json()
    assert run_body["status"] == "waiting_for_info"
    assert run_body["workflow_state"]["required_questions"][0]["question_text"] == "Provide electrical update documentation."
