from tests.demo_scenarios import create_submission_from_scenario, get_all_scenarios, get_scenario
from workflows.agent_workflow import run_agent_workflow


def test_demo_scenarios_produce_expected_decisions():
    for scenario in get_all_scenarios():
        submission = create_submission_from_scenario(scenario)

        workflow_state = run_agent_workflow(submission.model_dump())

        assert workflow_state.decision_packet is not None
        assert workflow_state.decision_packet.decision.value == scenario["expected_decision"]
        if scenario.get("expected_status"):
            assert workflow_state.status == scenario["expected_status"]


def test_high_value_low_risk_quote_has_sane_premium():
    scenario = get_scenario(9)
    submission = create_submission_from_scenario(scenario)

    workflow_state = run_agent_workflow(submission.model_dump())

    premium = workflow_state.decision_packet.premium_indication
    assert workflow_state.status == "completed"
    assert premium["annual_premium"] > 0
    assert premium["annual_premium"] < submission.coverage_request.coverage_a * 0.01
