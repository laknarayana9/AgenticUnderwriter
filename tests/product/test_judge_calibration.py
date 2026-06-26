"""Tests for the LLM-as-judge calibration harness.

Two layers:
  1. Fixture integrity  — the hand-labeled calibration set is well-formed and
     class-balanced (no live model needed).
  2. Metric correctness — the agreement / confusion / kappa math is verified
     against a deterministic fake judge with known outputs, so CI proves the
     numbers are computed correctly with no API key and no snapshot dependency.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.judge_calibration import (
    FAITHFUL,
    UNFAITHFUL,
    CalibrationRecord,
    Confusion,
    SimulatedJudge,
    Verdict,
    cohens_kappa,
    compute_metrics,
    confusion_matrix,
    load_calibration_set,
    load_snapshot,
    run_judge,
)

DATASET = Path("evals/datasets/judge_calibration.jsonl")
SNAPSHOT = Path("evals/datasets/judge_calibration_snapshot.json")


# --------------------------------------------------------------------------- #
# Fixture integrity
# --------------------------------------------------------------------------- #
def test_calibration_set_is_balanced_and_well_formed():
    records = load_calibration_set(DATASET)

    assert len(records) >= 20
    labels = [r.human_label for r in records]
    assert set(labels) == {FAITHFUL, UNFAITHFUL}

    n_faithful = labels.count(FAITHFUL)
    n_unfaithful = labels.count(UNFAITHFUL)
    # Roughly balanced so accuracy is not trivially inflated by a skewed prior.
    assert abs(n_faithful - n_unfaithful) <= 2

    ids = [r.id for r in records]
    assert len(ids) == len(set(ids)), "calibration record ids must be unique"
    for record in records:
        assert record.human_note, f"{record.id} is missing a human_note"
        assert record.summary, f"{record.id} has an empty rationale summary"


def test_calibration_set_covers_each_defect_category():
    records = load_calibration_set(DATASET)
    categories = {r.category for r in records if r.human_label == UNFAITHFUL}
    assert {
        "unsupported_claim",
        "fabricated_citation",
        "overstated_severity",
        "wrong_conclusion",
    }.issubset(categories)


# --------------------------------------------------------------------------- #
# Metric correctness (deterministic fake judge with known outputs)
# --------------------------------------------------------------------------- #
def _record(rid: str, human_label: str, category: str = "x") -> CalibrationRecord:
    return CalibrationRecord(
        id=rid,
        category=category,
        rationale={"summary": "s", "supporting_facts": [], "citation_chunk_ids": []},
        evidence=[],
        human_label=human_label,
        human_note="note",
    )


def test_confusion_matrix_counts_positive_class_as_unfaithful():
    pairs = [
        (UNFAITHFUL, UNFAITHFUL),  # TP
        (UNFAITHFUL, FAITHFUL),    # FN
        (FAITHFUL, UNFAITHFUL),    # FP
        (FAITHFUL, FAITHFUL),      # TN
    ]
    c = confusion_matrix(pairs)
    assert (c.tp, c.fn, c.fp, c.tn) == (1, 1, 1, 1)
    assert c.n == 4


def test_metrics_match_hand_computed_values():
    # 10 records: 6 unfaithful, 4 faithful. Judge gets a known set right/wrong.
    records = (
        [_record(f"U{i}", UNFAITHFUL) for i in range(6)]
        + [_record(f"F{i}", FAITHFUL) for i in range(4)]
    )
    # Judge calls 4 of 6 unfaithful correctly (2 missed) and 1 faithful wrong.
    predictions = {
        "U0": False, "U1": False, "U2": False, "U3": False,  # caught (passed=False)
        "U4": True, "U5": True,                              # missed (passed=True)
        "F0": True, "F1": True, "F2": True,                  # correct
        "F3": False,                                          # false alarm
    }
    verdicts = {rid: Verdict(record_id=rid, passed=passed) for rid, passed in predictions.items()}

    m = compute_metrics(records, verdicts)
    assert (m.confusion.tp, m.confusion.fn, m.confusion.fp, m.confusion.tn) == (4, 2, 1, 3)
    assert m.n == 10
    assert m.agreement == pytest.approx(7 / 10)
    assert m.recall == pytest.approx(4 / 6)
    assert m.precision == pytest.approx(4 / 5)
    assert m.false_negative_rate == pytest.approx(2 / 6)
    assert m.false_positive_rate == pytest.approx(1 / 4)
    assert m.f1 == pytest.approx(2 * (4 / 5) * (4 / 6) / ((4 / 5) + (4 / 6)))


def test_cohens_kappa_perfect_and_chance():
    # Perfect agreement -> kappa 1.0
    assert cohens_kappa(Confusion(tp=5, fp=0, fn=0, tn=5)) == pytest.approx(1.0)
    # Hand-computed: po=0.75, pe=0.5 -> kappa 0.5
    assert cohens_kappa(Confusion(tp=6, fp=0, fn=6, tn=12)) == pytest.approx(0.5)
    # Empty -> 0.0 (no division error)
    assert cohens_kappa(Confusion()) == 0.0


# --------------------------------------------------------------------------- #
# Simulated judge: structural guarantees (not tied to exact numbers)
# --------------------------------------------------------------------------- #
def test_simulated_judge_is_deterministic():
    records = load_calibration_set(DATASET)
    judge = SimulatedJudge()
    first = {rid: v.passed for rid, v in run_judge(records, judge).items()}
    second = {rid: v.passed for rid, v in run_judge(records, judge).items()}
    assert first == second


def test_simulated_judge_always_catches_structural_defects():
    """Fabricated citations and blatantly unsupported claims must be caught by
    the deterministic signals regardless of threshold tuning."""
    records = load_calibration_set(DATASET)
    verdicts = run_judge(records, SimulatedJudge())
    for record in records:
        if record.category in {"fabricated_citation", "unsupported_claim"}:
            assert verdicts[record.id].passed is False, (
                f"{record.id} ({record.category}) should have been caught"
            )


def test_committed_snapshot_matches_simulated_judge():
    """The committed snapshot must reflect the current simulated judge + fixtures
    so `--backend snapshot` reproduces the live `--backend simulated` run."""
    if not SNAPSHOT.exists():
        pytest.skip("snapshot not recorded")
    records = load_calibration_set(DATASET)
    live = {rid: v.passed for rid, v in run_judge(records, SimulatedJudge()).items()}
    _, snap = load_snapshot(SNAPSHOT)
    snap_passed = {rid: v.passed for rid, v in snap.items()}
    assert snap_passed == live, "snapshot is stale; re-run `python -m evals.judge_calibration --record`"
