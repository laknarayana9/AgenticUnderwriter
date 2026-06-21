"""Locks in the fine-tune governance-consequence narrative against the REAL
workflow (mock extractions, no model/network): a hallucinated roof age
underwrites on a fabricated fact, while correct abstention makes the workflow
pause to ask. This is the claim the portfolio demo makes; the test proves it
holds end to end."""

from scripts.extraction_workflow_demo import _MOCK_BASE, _MOCK_TUNED, fields_to_submission, run_case
from workflows.agent_workflow import UnderwritingWorkflow


def test_field_mapping_passes_roof_age_through_including_null():
    assert fields_to_submission(_MOCK_BASE)["risk"]["roof_age_years"] == 10
    assert fields_to_submission(_MOCK_TUNED)["risk"]["roof_age_years"] is None


def test_hallucinated_roof_age_underwrites_without_pausing():
    workflow = UnderwritingWorkflow()
    outcome = run_case(workflow, "base", _MOCK_BASE)
    assert outcome["roof_age_hallucinated"] is True
    assert outcome["workflow_status"] == "completed"
    assert outcome["asked_for_roof_age"] is False


def test_correct_abstention_makes_workflow_pause_for_roof_age():
    workflow = UnderwritingWorkflow()
    outcome = run_case(workflow, "tuned", _MOCK_TUNED)
    assert outcome["extracted_roof_age_years"] is None
    assert outcome["workflow_status"] == "waiting_for_info"
    assert outcome["asked_for_roof_age"] is True
