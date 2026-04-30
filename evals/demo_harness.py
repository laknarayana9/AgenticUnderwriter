"""
Demo scenario evaluation harness.

Runs the 10 curated demo scenarios and emits a compact Markdown or JSON report
covering expected-vs-actual decisions, citation guardrails, and missing-info
handling.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tests.demo_scenarios import create_submission_from_scenario, get_all_scenarios


REFER_OR_DECLINE = {"REFER", "DECLINE"}
CRITICAL_FIELD_PATHS = (
    "applicant.full_name",
    "risk.property_address",
    "risk.occupancy",
    "risk.roof_age_years",
)


def build_report() -> Dict[str, Any]:
    """Run every demo scenario and return a serializable report."""
    with contextlib.redirect_stdout(sys.stderr):
        from workflows.agent_workflow import PhaseAWorkflow

        workflow = PhaseAWorkflow()
        scenario_results = [
            _evaluate_scenario(workflow, index, scenario)
            for index, scenario in enumerate(get_all_scenarios(), start=1)
        ]

    failed = [result for result in scenario_results if not result["passed"]]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(scenario_results),
        "expected_scenario_count": 10,
        "passed": len(scenario_results) - len(failed),
        "failed": len(failed),
        "all_passed": not failed and len(scenario_results) == 10,
        "checks": {
            "ran_all_10_demo_scenarios": len(scenario_results) == 10,
            "expected_vs_actual_decision": all(
                result["checks"]["decision_matches_expected"] for result in scenario_results
            ),
            "refer_decline_citations_exist": all(
                result["checks"]["citations_present_when_required"]
                for result in scenario_results
            ),
            "no_silent_accept_on_missing_critical_info": all(
                result["checks"]["no_silent_accept_on_missing_critical_info"]
                for result in scenario_results
            ),
        },
        "scenarios": scenario_results,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    """Render the report as simple Markdown."""
    lines = [
        "# Demo Evaluation Report",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Summary",
        "",
        f"- Scenarios: {report['scenario_count']} / {report['expected_scenario_count']}",
        f"- Passed: {report['passed']}",
        f"- Failed: {report['failed']}",
        f"- Overall: {'PASS' if report['all_passed'] else 'FAIL'}",
        "",
        "## Guardrails",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for check_name, passed in report["checks"].items():
        lines.append(f"| {_label(check_name)} | {'PASS' if passed else 'FAIL'} |")

    lines.extend([
        "",
        "## Scenarios",
        "",
        "| # | Scenario | Expected | Actual | Status | Citations | Missing-Info Guard | Result |",
        "| ---: | --- | --- | --- | --- | ---: | --- | --- |",
    ])
    for result in report["scenarios"]:
        lines.append(
            "| {index} | {name} | {expected} | {actual} | {status} | {citations} | {missing_guard} | {passed} |".format(
                index=result["index"],
                name=_escape_table(result["name"]),
                expected=result["expected_decision"],
                actual=result["actual_decision"],
                status=result["actual_status"],
                citations=result["citation_count"],
                missing_guard="PASS" if result["checks"]["no_silent_accept_on_missing_critical_info"] else "FAIL",
                passed="PASS" if result["passed"] else "FAIL",
            )
        )

    failures = [result for result in report["scenarios"] if not result["passed"]]
    if failures:
        lines.extend(["", "## Failures", ""])
        for result in failures:
            lines.append(f"- Scenario {result['index']}: {_escape_table(result['name'])}")
            for failure in result["failures"]:
                lines.append(f"  - {failure}")

    return "\n".join(lines) + "\n"


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 10 demo underwriting scenarios.")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Report format to emit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path for the report. Defaults to stdout.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = build_report()
    rendered = (
        json.dumps(report, indent=2, sort_keys=True)
        if args.format == "json"
        else render_markdown(report)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if report["all_passed"] else 1


def _evaluate_scenario(workflow: Any, index: int, scenario: Dict[str, Any]) -> Dict[str, Any]:
    submission = create_submission_from_scenario(scenario)
    workflow_state = workflow.run(submission.model_dump())
    packet = workflow_state.decision_packet
    actual_decision = packet.decision.value if packet else None
    expected_status = scenario.get("expected_status")
    citations = packet.citations if packet else []
    required_questions = workflow_state.required_questions
    critical_missing_fields = _missing_critical_fields(scenario["submission"])
    missing_info_is_active = bool(critical_missing_fields or required_questions)

    checks = {
        "decision_matches_expected": actual_decision == scenario["expected_decision"],
        "status_matches_expected": expected_status is None or workflow_state.status == expected_status,
        "citations_present_when_required": (
            actual_decision not in REFER_OR_DECLINE or bool(citations)
        ),
        "no_silent_accept_on_missing_critical_info": (
            not missing_info_is_active
            or (
                actual_decision != "ACCEPT"
                and workflow_state.status == "waiting_for_info"
                and bool(required_questions)
            )
        ),
    }
    failures = _failures(checks, scenario, actual_decision, workflow_state.status)

    return {
        "index": index,
        "name": scenario["name"],
        "description": scenario["description"],
        "expected_decision": scenario["expected_decision"],
        "actual_decision": actual_decision,
        "expected_status": expected_status,
        "actual_status": workflow_state.status,
        "citation_count": len(citations),
        "citations_required": actual_decision in REFER_OR_DECLINE,
        "review_reason_codes": packet.review_reason_codes if packet else [],
        "required_questions": required_questions,
        "critical_missing_fields": critical_missing_fields,
        "checks": checks,
        "passed": all(checks.values()),
        "failures": failures,
        "run_id": workflow_state.run_id,
        "quote_id": workflow_state.quote_id,
    }


def _missing_critical_fields(submission: Dict[str, Any]) -> List[str]:
    return [
        field_path
        for field_path in CRITICAL_FIELD_PATHS
        if _is_missing(_get_path(submission, field_path))
    ]


def _get_path(payload: Dict[str, Any], field_path: str) -> Any:
    current: Any = payload
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "unknown", "unk", "tbd", "n/a", "na", "unsure", "uncertain"}
    return False


def _failures(
    checks: Dict[str, bool],
    scenario: Dict[str, Any],
    actual_decision: Optional[str],
    actual_status: str,
) -> List[str]:
    messages = []
    if not checks["decision_matches_expected"]:
        messages.append(
            f"expected decision {scenario['expected_decision']}, got {actual_decision}"
        )
    if not checks["status_matches_expected"]:
        messages.append(
            f"expected status {scenario.get('expected_status')}, got {actual_status}"
        )
    if not checks["citations_present_when_required"]:
        messages.append("REFER/DECLINE decision did not include citations")
    if not checks["no_silent_accept_on_missing_critical_info"]:
        messages.append("missing critical info was silently accepted")
    return messages


def _label(value: str) -> str:
    return value.replace("_", " ").capitalize()


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
