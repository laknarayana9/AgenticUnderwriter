"""CI-friendly evaluation runner for labeled HO3 underwriting submissions."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from pydantic import BaseModel, Field, ValidationError

# Allow running as `python evals/run.py` from the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflows.agent_workflow import UnderwritingWorkflow


class ExpectedLabel(BaseModel):
    decision: str
    reason_codes: List[str] = Field(default_factory=list)
    gold_citations: List[str] = Field(default_factory=list)
    status: Optional[str] = None
    # Trust-oriented fields
    expected_refusal: Optional[bool] = Field(
        None, description="True if the case should trigger human review (needs_human_review=True)"
    )
    expected_pii_free_rationale: bool = Field(
        False, description="True to assert that no PII appears in the producer rationale text"
    )


class EvalCase(BaseModel):
    id: str
    title: str
    stratum: str = "unspecified"
    submission: Dict[str, Any]
    expected: ExpectedLabel


@dataclass
class EvalCaseResult:
    case_id: str
    title: str
    expected_decision: str
    actual_decision: Optional[str]
    expected_reason_codes: List[str]
    actual_reason_codes: List[str]
    gold_citations: List[str]
    actual_citations: List[str]
    expected_status: Optional[str]
    status: Optional[str]
    decision_match: bool
    reason_code_match: bool
    retrieval_recall_at_k: Optional[float]
    faithfulness: float = 1.0
    rationale_quality: Optional[float] = None
    error: Optional[str] = None
    # Trust-oriented metrics
    citation_accuracy: Optional[float] = None   # % of cited IDs found in retrieved_chunks
    refusal_match: Optional[bool] = None        # expected_refusal matches actual needs_human_review
    pii_in_rationale: Optional[bool] = None     # True if PII detected in rationale text


@dataclass
class EvalReport:
    case_count: int
    decision_accuracy: float
    reason_code_match: float
    retrieval_recall_at_k: float
    faithfulness: float
    status_match: Optional[float]
    rationale_quality: Optional[float]
    failures: List[EvalCaseResult] = field(default_factory=list)
    results: List[EvalCaseResult] = field(default_factory=list)
    # Trust-oriented aggregate metrics
    citation_accuracy: Optional[float] = None   # mean across cases with cited IDs
    refusal_accuracy: Optional[float] = None    # mean across cases with expected_refusal set
    pii_leak_rate: Optional[float] = None       # fraction of expected_pii_free cases where PII leaked


def load_dataset(path: Path) -> List[EvalCase]:
    cases: List[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
                cases.append(EvalCase.model_validate(payload))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"Invalid dataset row {line_number}: {exc}") from exc
    if not cases:
        raise ValueError("Dataset did not contain any evaluation cases")
    return cases


def evaluate_cases(
    cases: Sequence[EvalCase],
    *,
    k: int = 5,
    include_rationale_quality: bool = False,
    workflow: Optional[UnderwritingWorkflow] = None,
) -> EvalReport:
    if workflow is None:
        with contextlib.redirect_stdout(io.StringIO()):
            workflow = UnderwritingWorkflow()
    results = [
        _evaluate_case(
            workflow,
            case,
            k=k,
            include_rationale_quality=include_rationale_quality,
        )
        for case in cases
    ]
    return _build_report(results)


def _evaluate_case(
    workflow: UnderwritingWorkflow,
    case: EvalCase,
    *,
    k: int,
    include_rationale_quality: bool,
) -> EvalCaseResult:
    try:
        state = workflow.run(case.submission)
        packet = state.decision_packet
        actual_decision = packet.decision.value if packet else None
        actual_reason_codes = packet.review_reason_codes if packet else []
        actual_citations = [
            citation.get("chunk_id", "")
            for citation in (packet.citations if packet else [])
            if isinstance(citation, dict) and citation.get("chunk_id")
        ][:k]
        rationale_quality = (
            _score_rationale_quality(packet, case.expected.reason_codes)
            if include_rationale_quality
            else None
        )
        status_match = case.expected.status is None or state.status == case.expected.status
        decision_match = actual_decision == case.expected.decision
        reason_code_match = _as_set(actual_reason_codes) == _as_set(case.expected.reason_codes)
        retrieval_recall = _recall_at_k(case.expected.gold_citations, actual_citations)
        faithfulness = _score_faithfulness(packet, state)

        # Trust metrics
        citation_accuracy = _score_citation_accuracy(packet, state)
        refusal_match = _score_refusal_match(case.expected, packet)
        pii_in_rationale = (
            _score_pii_in_rationale(packet, case.submission)
            if case.expected.expected_pii_free_rationale
            else None
        )

        return EvalCaseResult(
            case_id=case.id,
            title=case.title,
            expected_decision=case.expected.decision,
            actual_decision=actual_decision,
            expected_reason_codes=case.expected.reason_codes,
            actual_reason_codes=actual_reason_codes,
            gold_citations=case.expected.gold_citations,
            actual_citations=actual_citations,
            expected_status=case.expected.status,
            status=state.status,
            decision_match=decision_match and status_match,
            reason_code_match=reason_code_match,
            retrieval_recall_at_k=retrieval_recall,
            faithfulness=faithfulness,
            rationale_quality=rationale_quality,
            citation_accuracy=citation_accuracy,
            refusal_match=refusal_match,
            pii_in_rationale=pii_in_rationale,
        )
    except Exception as exc:
        return EvalCaseResult(
            case_id=case.id,
            title=case.title,
            expected_decision=case.expected.decision,
            actual_decision=None,
            expected_reason_codes=case.expected.reason_codes,
            actual_reason_codes=[],
            gold_citations=case.expected.gold_citations,
            actual_citations=[],
            expected_status=case.expected.status,
            status=None,
            decision_match=False,
            reason_code_match=False,
            retrieval_recall_at_k=0.0 if case.expected.gold_citations else None,
            faithfulness=0.0,
            rationale_quality=0.0 if include_rationale_quality else None,
            citation_accuracy=None,
            refusal_match=False if case.expected.expected_refusal is not None else None,
            pii_in_rationale=None,
            error=str(exc),
        )


def _build_report(results: Sequence[EvalCaseResult]) -> EvalReport:
    case_count = len(results)
    citation_results = [
        result.retrieval_recall_at_k
        for result in results
        if result.retrieval_recall_at_k is not None
    ]
    status_labeled = [
        result
        for result in results
        if result.expected_status is not None
    ]
    rationale_scores = [
        result.rationale_quality
        for result in results
        if result.rationale_quality is not None
    ]

    # Trust aggregates
    citation_accuracy_scores = [r.citation_accuracy for r in results if r.citation_accuracy is not None]
    refusal_labeled = [r for r in results if r.refusal_match is not None]
    pii_check_cases = [r for r in results if r.pii_in_rationale is not None]

    failures = [
        result
        for result in results
        if (
            not result.decision_match
            or not result.reason_code_match
            or (result.retrieval_recall_at_k is not None and result.retrieval_recall_at_k < 1.0)
            or result.faithfulness < 1.0
            or result.refusal_match is False
            or result.pii_in_rationale is True
            or result.error
        )
    ]

    return EvalReport(
        case_count=case_count,
        decision_accuracy=_mean(1.0 if result.decision_match else 0.0 for result in results),
        reason_code_match=_mean(1.0 if result.reason_code_match else 0.0 for result in results),
        retrieval_recall_at_k=_mean(citation_results),
        faithfulness=_mean(result.faithfulness for result in results),
        status_match=_mean(
            1.0 if result.status == result.expected_status else 0.0
            for result in status_labeled
        ) if status_labeled else None,
        rationale_quality=_mean(rationale_scores) if rationale_scores else None,
        failures=failures,
        results=list(results),
        citation_accuracy=_mean(citation_accuracy_scores) if citation_accuracy_scores else None,
        refusal_accuracy=_mean(1.0 if r.refusal_match else 0.0 for r in refusal_labeled) if refusal_labeled else None,
        pii_leak_rate=_mean(1.0 if r.pii_in_rationale else 0.0 for r in pii_check_cases) if pii_check_cases else None,
    )


def _recall_at_k(gold_citations: Sequence[str], actual_citations: Sequence[str]) -> Optional[float]:
    if not gold_citations:
        return None
    gold = set(gold_citations)
    actual = set(actual_citations)
    return len(gold & actual) / len(gold)


def _score_rationale_quality(packet: Any, expected_reason_codes: Sequence[str]) -> float:
    if not packet or not packet.producer_rationale:
        return 0.0
    rationale = packet.producer_rationale
    summary = rationale.summary.lower()
    score = 0.4 if summary else 0.0
    if rationale.supporting_facts:
        score += 0.2
    if rationale.citation_chunk_ids:
        score += 0.2
    if not expected_reason_codes:
        score += 0.2 if "eligible" in summary or "no referral" in summary else 0.0
    else:
        tokens = [token.lower() for code in expected_reason_codes for token in code.split("_")]
        matched = sum(1 for token in tokens if token and token in summary)
        score += 0.2 * (matched / max(1, len(tokens)))
    return round(min(1.0, score), 4)


def _score_faithfulness(packet: Any, state: Any) -> float:
    """
    Deterministic groundedness check for the produced rationale.

    Faithfulness here means the decision packet only references evidence and
    facts that actually exist in the run — it does not invent citations or
    supporting facts. We average two sub-checks where each applies:

    - citation grounding: every cited chunk_id was actually retrieved this run.
    - fact grounding: every supporting fact's key appears in facts_used.

    Cases with no citations or no supporting facts (e.g. accepts and missing-info
    pauses, which are grounded in deterministic rules rather than retrieval) are
    vacuously faithful for the check that does not apply. A packet that cites a
    chunk that was never retrieved, or asserts a fact the rules never produced,
    scores below 1.0.
    """
    if packet is None:
        return 0.0

    retrieved_ids = {
        chunk.get("chunk_id")
        for chunk in (state.retrieval or {}).get("retrieved_chunks", [])
        if isinstance(chunk, dict) and chunk.get("chunk_id")
    }

    sub_scores: List[float] = []

    cited_ids = [cid for cid in (packet.evidence_cited or []) if cid]
    if cited_ids:
        grounded = sum(1 for cid in cited_ids if cid in retrieved_ids)
        sub_scores.append(grounded / len(cited_ids))

    rationale = packet.producer_rationale
    if rationale and rationale.supporting_facts:
        facts_used = packet.facts_used or {}
        fact_keys = {str(key).lower() for key in facts_used}
        grounded_facts = sum(
            1 for fact in rationale.supporting_facts
            if str(fact).split(":", 1)[0].strip().lower() in fact_keys
        )
        sub_scores.append(grounded_facts / len(rationale.supporting_facts))

    if not sub_scores:
        return 1.0
    return round(sum(sub_scores) / len(sub_scores), 4)


def _score_citation_accuracy(packet: Any, state: Any) -> Optional[float]:
    """Fraction of cited chunk IDs that exist in the retrieved_chunks for this run."""
    if packet is None:
        return None
    cited_ids = [cid for cid in (packet.evidence_cited or []) if cid]
    if not cited_ids:
        return None
    retrieved_ids = {
        chunk.get("chunk_id")
        for chunk in (state.retrieval or {}).get("retrieved_chunks", [])
        if isinstance(chunk, dict) and chunk.get("chunk_id")
    }
    grounded = sum(1 for cid in cited_ids if cid in retrieved_ids)
    return round(grounded / len(cited_ids), 4)


def _score_refusal_match(expected: ExpectedLabel, packet: Any) -> Optional[bool]:
    """True if the actual needs_human_review matches expected_refusal (when set)."""
    if expected.expected_refusal is None:
        return None
    actual_refusal = bool(packet.needs_human_review) if packet else True
    return actual_refusal == expected.expected_refusal


# PII patterns: basic regex for common formats that should never appear in a rationale
_PII_PATTERNS = [
    re.compile(r"\b\d{3}[-.\s]\d{2}[-.\s]\d{4}\b"),           # SSN
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
    re.compile(r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}"),  # phone
]


def _score_pii_in_rationale(packet: Any, submission: Dict[str, Any]) -> bool:
    """
    Return True if PII appears anywhere in the shipped producer rationale.

    Scans both summary and supporting_facts — the full output the API returns
    to callers — not just the summary field.
    """
    if not packet or not packet.producer_rationale:
        return False
    rationale = packet.producer_rationale

    # Concatenate every text field the API ships to callers
    texts: List[str] = []
    if rationale.summary:
        texts.append(rationale.summary)
    if rationale.supporting_facts:
        texts.extend(str(f) for f in rationale.supporting_facts)
    combined = " ".join(texts)
    if not combined:
        return False

    # Literal value check
    applicant = submission.get("applicant", {})
    risk = submission.get("risk", {})
    pii_values = [
        applicant.get("full_name", ""),
        applicant.get("email", ""),
        applicant.get("phone", ""),
        risk.get("property_address", ""),
    ]
    for value in pii_values:
        if value and len(value) > 3 and value in combined:
            return True

    # Pattern check (SSN / email / phone formats)
    return any(pattern.search(combined) for pattern in _PII_PATTERNS)


def _print_report(report: EvalReport, *, k: int, include_rationale_quality: bool) -> None:
    print("HO3 Evaluation Report")
    print("=====================")
    print(f"cases: {report.case_count}")
    print(f"decision_accuracy: {report.decision_accuracy:.3f}")
    print(f"reason_code_match: {report.reason_code_match:.3f}")
    print(f"retrieval_recall@{k}: {report.retrieval_recall_at_k:.3f}")
    print(f"faithfulness: {report.faithfulness:.3f}")
    if include_rationale_quality and report.rationale_quality is not None:
        print(f"rationale_quality: {report.rationale_quality:.3f}")
    # Trust metrics
    if report.citation_accuracy is not None:
        print(f"citation_accuracy: {report.citation_accuracy:.3f}")
    if report.refusal_accuracy is not None:
        print(f"refusal_accuracy: {report.refusal_accuracy:.3f}")
    if report.pii_leak_rate is not None:
        print(f"pii_leak_rate: {report.pii_leak_rate:.3f}")
    print("")

    if not report.failures:
        print("No eval failures.")
        return

    print("Failures")
    print("--------")
    for result in report.failures[:20]:
        print(
            f"{result.case_id}: expected decision={result.expected_decision} "
            f"actual={result.actual_decision} status={result.status}"
        )
        if result.expected_reason_codes != result.actual_reason_codes:
            print(f"  reason_codes expected={result.expected_reason_codes} actual={result.actual_reason_codes}")
        if result.retrieval_recall_at_k is not None and result.retrieval_recall_at_k < 1.0:
            print(
                f"  citations recall={result.retrieval_recall_at_k:.3f} "
                f"gold={result.gold_citations} actual={result.actual_citations}"
            )
        if result.error:
            print(f"  error={result.error}")
    if len(report.failures) > 20:
        print(f"... {len(report.failures) - 20} additional failures omitted")


def _as_set(items: Iterable[str]) -> Set[str]:
    return {item for item in items if item}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run labeled HO3 workflow evaluations.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL eval dataset.")
    parser.add_argument("--k", type=int, default=5, help="Retrieval cutoff for recall@k.")
    parser.add_argument("--min-decision-accuracy", type=float, default=1.0)
    parser.add_argument("--min-reason-code-match", type=float, default=0.95)
    parser.add_argument("--min-retrieval-recall", type=float, default=0.75)
    parser.add_argument("--min-faithfulness", type=float, default=None)
    parser.add_argument("--min-rationale-quality", type=float, default=None)
    parser.add_argument("--include-llm-rationale-quality", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary JSON.")
    args = parser.parse_args(argv)

    try:
        cases = load_dataset(Path(args.dataset))
        report = evaluate_cases(
            cases,
            k=args.k,
            include_rationale_quality=args.include_llm_rationale_quality,
        )
    except Exception as exc:
        print(f"eval_error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({
            "cases": report.case_count,
            "decision_accuracy": report.decision_accuracy,
            "reason_code_match": report.reason_code_match,
            f"retrieval_recall@{args.k}": report.retrieval_recall_at_k,
            "faithfulness": report.faithfulness,
            "rationale_quality": report.rationale_quality,
            "citation_accuracy": report.citation_accuracy,
            "refusal_accuracy": report.refusal_accuracy,
            "pii_leak_rate": report.pii_leak_rate,
            "failure_count": len(report.failures),
        }, indent=2, sort_keys=True))
    else:
        _print_report(
            report,
            k=args.k,
            include_rationale_quality=args.include_llm_rationale_quality,
        )

    thresholds = [
        ("decision_accuracy", report.decision_accuracy, args.min_decision_accuracy),
        ("reason_code_match", report.reason_code_match, args.min_reason_code_match),
        ("retrieval_recall", report.retrieval_recall_at_k, args.min_retrieval_recall),
    ]
    if args.min_faithfulness is not None:
        thresholds.append(("faithfulness", report.faithfulness, args.min_faithfulness))
    if args.min_rationale_quality is not None:
        thresholds.append(("rationale_quality", report.rationale_quality or 0.0, args.min_rationale_quality))

    failed = [
        name
        for name, actual, expected in thresholds
        if actual < expected
    ]
    if failed:
        print(f"threshold_failure: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
