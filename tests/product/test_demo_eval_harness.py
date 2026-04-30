from evals.demo_harness import build_report, render_markdown


def test_demo_eval_harness_reports_all_guardrails_pass():
    report = build_report()

    assert report["scenario_count"] == 10
    assert report["all_passed"] is True
    assert report["checks"]["ran_all_10_demo_scenarios"] is True
    assert report["checks"]["expected_vs_actual_decision"] is True
    assert report["checks"]["refer_decline_citations_exist"] is True
    assert report["checks"]["no_silent_accept_on_missing_critical_info"] is True

    markdown = render_markdown(report)
    assert "# Demo Evaluation Report" in markdown
    assert "Overall: PASS" in markdown
