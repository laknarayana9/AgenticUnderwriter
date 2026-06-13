import json
from pathlib import Path

from evals.run import evaluate_cases, load_dataset, main


DATASET = Path("evals/datasets/ho3_labeled.jsonl")


def test_eval_dataset_has_portfolio_scale_labels():
    cases = load_dataset(DATASET)
    strata = {case.stratum for case in cases}

    assert len(cases) >= 150
    assert len(strata) >= 10
    assert all(case.expected.decision in {"ACCEPT", "REFER", "DECLINE"} for case in cases)
    assert {"completed", "pending_review", "waiting_for_info"}.issubset({case.expected.status for case in cases})
    assert any(case.expected.gold_citations for case in cases)


def test_eval_runner_reports_perfect_metrics_for_labeled_subset(tmp_path):
    cases = load_dataset(DATASET)[:6]
    subset_path = tmp_path / "subset.jsonl"
    subset_path.write_text(
        "\n".join(json.dumps(case.model_dump()) for case in cases),
        encoding="utf-8",
    )

    exit_code = main(["--dataset", str(subset_path), "--json"])

    assert exit_code == 0


def test_eval_report_surfaces_label_failures():
    case = load_dataset(DATASET)[0]
    bad_case = case.model_copy(deep=True)
    bad_case.expected.decision = "DECLINE"

    report = evaluate_cases([bad_case])

    assert report.decision_accuracy == 0
    assert report.failures[0].case_id == case.id


def test_faithfulness_is_perfect_for_grounded_subset():
    cases = load_dataset(DATASET)[:12]
    report = evaluate_cases(cases)

    # The governed packet only cites retrieved chunks and facts it actually
    # produced, so deterministic runs are fully grounded.
    assert report.faithfulness == 1.0
    assert all(result.faithfulness == 1.0 for result in report.results)
