"""Calibrate the rationale-faithfulness LLM judge against human labels.

The shipped gatekeeper is ``CriticAgent`` in ``workflows/critic.py``: an LLM
judge that decides whether a producer rationale is faithful to the retrieved
underwriting evidence. It *fails open* (no critic configured, or any error =>
``passed=True``), and nothing measures how often a real judge actually agrees
with a human. This module quantifies that.

It runs the judge over a hand-labeled calibration set
(``evals/datasets/judge_calibration.jsonl``) and reports judge<->human
agreement plus the judge's failure modes -- most importantly the
**false-negative rate** (unfaithful rationales the judge passed), which is the
fail-open risk made measurable.

Backends
--------
- ``llm``      : the real shipped judge. Reuses the *exact* critic system prompt,
                 user template, and JSON schema from ``workflows/critic.py`` plus
                 the same deterministic citation pre-check. Requires an API key
                 and ``LLM_STRUCTURED_OUTPUT_ENABLED=true``.
- ``simulated``: a deterministic stand-in used when no API key is available
                 (CI, and the environment this was authored in). It runs the
                 *real* citation pre-check, then a lexical-overlap groundedness
                 heuristic. It is a genuine but weak judge: it catches fabricated
                 citations and blatantly unsupported claims, but fails open on
                 semantic defects (overstated severity, flipped conclusions)
                 where the wording still overlaps the evidence. Every number it
                 produces is a real measurement of this stand-in.
- ``snapshot`` : recompute the report from a previously recorded snapshot file,
                 with no judge calls at all (fully offline + deterministic).

Positive class for all metrics = "unfaithful" (the judge SHOULD catch it).

CLI examples
------------
    # Deterministic, no key (default): score with the simulated stand-in
    python -m evals.judge_calibration

    # Record a snapshot + report from the simulated stand-in
    python -m evals.judge_calibration --record

    # Calibrate the REAL LLM judge (needs API key) and record it
    LLM_STRUCTURED_OUTPUT_ENABLED=true LLM_PROVIDER=openai \\
        python -m evals.judge_calibration --backend llm --record

    # Recompute the report offline from the committed snapshot
    python -m evals.judge_calibration --backend snapshot
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Allow running as `python evals/judge_calibration.py` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_DATASET = _REPO_ROOT / "evals" / "datasets" / "judge_calibration.jsonl"
_DEFAULT_SNAPSHOT = _REPO_ROOT / "evals" / "datasets" / "judge_calibration_snapshot.json"
_DEFAULT_REPORT = _REPO_ROOT / "evals" / "reports" / "judge_calibration.md"

FAITHFUL = "faithful"
UNFAITHFUL = "unfaithful"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationRecord:
    id: str
    category: str
    rationale: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    human_label: str
    human_note: str

    @property
    def summary(self) -> str:
        return str(self.rationale.get("summary", ""))

    @property
    def supporting_facts(self) -> List[str]:
        return [str(f) for f in (self.rationale.get("supporting_facts") or [])]

    @property
    def citation_chunk_ids(self) -> List[str]:
        return [str(c) for c in (self.rationale.get("citation_chunk_ids") or [])]

    @property
    def evidence_ids(self) -> set:
        return {str(c.get("chunk_id", "")) for c in self.evidence if c.get("chunk_id")}

    @property
    def evidence_text(self) -> str:
        return " ".join(str(c.get("text", "")) for c in self.evidence)


@dataclass
class Verdict:
    """A judge's call on one record. ``passed`` True => predicted faithful."""
    record_id: str
    passed: bool
    unsupported_facts: List[str] = field(default_factory=list)
    invalid_citation_ids: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def predicted_label(self) -> str:
        return FAITHFUL if self.passed else UNFAITHFUL


def load_calibration_set(path: Path = _DEFAULT_DATASET) -> List[CalibrationRecord]:
    """Load JSONL fixtures. Lines that are blank or start with '#' are ignored."""
    records: List[CalibrationRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
                records.append(
                    CalibrationRecord(
                        id=payload["id"],
                        category=payload.get("category", "unspecified"),
                        rationale=payload.get("rationale", {}),
                        evidence=payload.get("evidence", []),
                        human_label=payload["human_label"],
                        human_note=payload.get("human_note", ""),
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid calibration row {line_number}: {exc}") from exc
    if not records:
        raise ValueError("Calibration set contained no records")
    return records


# --------------------------------------------------------------------------- #
# Judge backends
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in",
    "on", "and", "or", "for", "with", "no", "not", "this", "that", "it", "its",
    "as", "at", "by", "from", "within", "per", "also", "will", "must", "shall",
    "should", "may", "risk", "home", "applicant", "submission",
}
_TOKEN_RE = re.compile(r"[a-z]+")
# Directive phrases that mean "this is NOT a clean accept" when present in the
# evidence the rationale claims to rest on.
_REFER_DIRECTIVE_RE = re.compile(r"shall be referred|shall refer|must be declined|must decline", re.IGNORECASE)
# Accept-language a rationale summary uses when it concludes the risk is clean.
_ACCEPT_LANGUAGE_RE = re.compile(
    r"\baccept(ed)?\b|issued without|without referral|within tolerance|no referral|acceptable as-is",
    re.IGNORECASE,
)


def _stem(token: str) -> str:
    """Crude suffix stripper so morphological variants align (referred/referral
    -> referr, years -> year, exceeds -> exceed). Good enough for a lexical
    grounding signal; it is not a real stemmer."""
    for suffix in ("ements", "ities", "ing", "ed", "es", "ly", "al", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _content_tokens(text: str) -> List[str]:
    """Lowercase, stemmed alphabetic tokens with stopwords removed."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]


def _citation_precheck(record: CalibrationRecord) -> List[str]:
    """The exact deterministic check the production critic runs first:
    any cited chunk id absent from the evidence is invalid."""
    evidence_ids = record.evidence_ids
    return [cid for cid in record.citation_chunk_ids if cid and cid not in evidence_ids]


class SimulatedJudge:
    """Deterministic lexical stand-in for the live LLM critic.

    Two real signals, both mirroring how the production critic behaves:
      1. Citation pre-check (identical to ``CriticAgent``): fabricated citation
         ids -> unfaithful.
      2. Lexical groundedness: for each supporting fact, the fraction of content
         tokens that also appear in the evidence text. If any fact falls below
         ``overlap_threshold`` it is flagged unsupported -> unfaithful.

    This catches structural and vocabulary-level fabrications but, by design,
    cannot see semantic defects (overstated severity, flipped conclusions) that
    reuse the evidence's own words. That blind spot is the point: it is exactly
    the fail-open behavior the calibration is meant to expose.
    """

    backend_name = "simulated"

    def __init__(self, overlap_threshold: float = 0.34) -> None:
        self.overlap_threshold = overlap_threshold

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.backend_name, "overlap_threshold": self.overlap_threshold}

    def judge(self, record: CalibrationRecord) -> Verdict:
        # Signal 1: citation pre-check (identical to the production critic).
        invalid = _citation_precheck(record)
        if invalid:
            return Verdict(
                record_id=record.id,
                passed=False,
                invalid_citation_ids=invalid,
                reason=f"citation id(s) not in evidence: {invalid}",
            )

        # Signal 2: lexical grounding of each supporting fact in the evidence.
        evidence_tokens = set(_content_tokens(record.evidence_text))
        unsupported: List[str] = []
        for fact in record.supporting_facts:
            # Drop a leading "key:" prefix so the heuristic scores the claim text.
            claim = fact.split(":", 1)[1] if ":" in fact else fact
            tokens = _content_tokens(claim)
            if not tokens:
                continue
            overlap = sum(1 for t in tokens if t in evidence_tokens) / len(tokens)
            if overlap < self.overlap_threshold:
                unsupported.append(fact)
        if unsupported:
            return Verdict(
                record_id=record.id,
                passed=False,
                unsupported_facts=unsupported,
                reason="supporting fact(s) lack lexical grounding in evidence",
            )

        # Signal 3: decision inversion. If the evidence carries a refer/decline
        # directive but the summary concludes the risk is clean (accept), the
        # rationale has drawn the opposite conclusion from its own evidence.
        if _REFER_DIRECTIVE_RE.search(record.evidence_text) and _ACCEPT_LANGUAGE_RE.search(record.summary):
            return Verdict(
                record_id=record.id,
                passed=False,
                reason="summary concludes accept while evidence carries a refer/decline directive",
            )

        # NOTE: this stand-in has no signal for *overstated severity* (evidence
        # supports a milder action than the rationale states while reusing the
        # same vocabulary). That blind spot is intentional and documented; it is
        # the residual fail-open mode the calibration is meant to surface.
        return Verdict(record_id=record.id, passed=True, reason="grounded")


class LLMJudge:
    """The real shipped judge. Reuses the production critic prompt + schema so
    the calibration characterizes the gatekeeper that actually runs."""

    backend_name = "llm"

    def __init__(self) -> None:
        # Imported lazily so the simulated path has no LLM dependency.
        from app.llm_service import LLMServiceConfig, StructuredLLMService
        from workflows.critic import (
            _CRITIC_SYSTEM_PROMPT,
            _CRITIC_USER_TEMPLATE,
            _CriticResponseModel,
            CriticAgent,
        )

        self._system_prompt = _CRITIC_SYSTEM_PROMPT
        self._user_template = _CRITIC_USER_TEMPLATE
        self._schema = _CriticResponseModel.schema

        provider = os.getenv("CRITIC_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "openai")).strip().lower()
        model = os.getenv("CRITIC_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini")).strip()
        config = LLMServiceConfig(
            enabled=os.getenv("LLM_STRUCTURED_OUTPUT_ENABLED", "false").lower() in {"1", "true", "yes"},
            provider=provider,
            model=model,
            api_key=CriticAgent._api_key_for(provider),
            base_url=os.getenv("OLLAMA_BASE_URL") if provider == "ollama" else None,
        )
        self._service = StructuredLLMService(config=config)
        self._provider = provider
        self._model = model
        if not self._service.provider:
            raise RuntimeError(
                "LLM judge backend requested but no provider is configured. "
                "Set LLM_STRUCTURED_OUTPUT_ENABLED=true and the relevant API key, "
                "or use --backend simulated."
            )

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.backend_name, "provider": self._provider, "model": self._model}

    def judge(self, record: CalibrationRecord) -> Verdict:
        invalid = _citation_precheck(record)
        if invalid:
            return Verdict(
                record_id=record.id,
                passed=False,
                invalid_citation_ids=invalid,
                reason=f"citation id(s) not in evidence: {invalid}",
            )
        evidence_text = "\n\n".join(
            f"[{c.get('chunk_id', '?')}] {str(c.get('text', ''))[:500]}" for c in record.evidence
        ) or "(no retrieved evidence)"
        user_prompt = self._user_template.format(
            summary=record.summary[:800],
            supporting_facts=json.dumps(record.supporting_facts),
            cited_ids=json.dumps(record.citation_chunk_ids),
            evidence=evidence_text,
        )
        raw = self._service.provider.generate_json(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
            schema=self._schema,
        )
        passed = bool(raw.get("passed", True))
        unsupported = raw.get("unsupported_facts", [])
        return Verdict(
            record_id=record.id,
            passed=passed,
            unsupported_facts=unsupported if isinstance(unsupported, list) else [],
            reason=str(raw.get("feedback_for_generator", "")),
        )


def build_judge(backend: str):
    if backend == "simulated":
        return SimulatedJudge()
    if backend == "llm":
        return LLMJudge()
    raise ValueError(f"Unknown judge backend: {backend!r}")


# --------------------------------------------------------------------------- #
# Metrics (pure functions over (human_label, predicted_label) pairs)
# --------------------------------------------------------------------------- #
@dataclass
class Confusion:
    """Positive class = UNFAITHFUL (the judge should catch it)."""
    tp: int = 0  # human unfaithful, judge unfaithful  (correctly caught)
    fp: int = 0  # human faithful,   judge unfaithful  (false alarm)
    fn: int = 0  # human unfaithful, judge faithful    (MISSED -> fail-open)
    tn: int = 0  # human faithful,   judge faithful

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


def confusion_matrix(pairs: Sequence[Tuple[str, str]]) -> Confusion:
    """pairs = [(human_label, predicted_label), ...] using FAITHFUL/UNFAITHFUL."""
    c = Confusion()
    for human, predicted in pairs:
        human_pos = human == UNFAITHFUL
        pred_pos = predicted == UNFAITHFUL
        if human_pos and pred_pos:
            c.tp += 1
        elif (not human_pos) and pred_pos:
            c.fp += 1
        elif human_pos and (not pred_pos):
            c.fn += 1
        else:
            c.tn += 1
    return c


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def cohens_kappa(c: Confusion) -> float:
    """Cohen's kappa for the binary judge-vs-human agreement."""
    n = c.n
    if n == 0:
        return 0.0
    po = _safe_div(c.tp + c.tn, n)
    p_judge_pos = _safe_div(c.tp + c.fp, n)
    p_human_pos = _safe_div(c.tp + c.fn, n)
    pe = p_judge_pos * p_human_pos + (1 - p_judge_pos) * (1 - p_human_pos)
    if math.isclose(pe, 1.0):
        return 1.0 if math.isclose(po, 1.0) else 0.0
    return (po - pe) / (1 - pe)


@dataclass
class CalibrationMetrics:
    n: int
    human_unfaithful: int
    human_faithful: int
    confusion: Confusion
    agreement: float
    precision: float
    recall: float
    f1: float
    false_negative_rate: float
    false_positive_rate: float
    kappa: float
    per_category: Dict[str, Dict[str, int]]
    disagreements: List[Dict[str, str]]


def compute_metrics(
    records: Sequence[CalibrationRecord],
    verdicts: Dict[str, Verdict],
) -> CalibrationMetrics:
    pairs: List[Tuple[str, str]] = []
    per_category: Dict[str, Dict[str, int]] = {}
    disagreements: List[Dict[str, str]] = []

    for record in records:
        verdict = verdicts[record.id]
        human = record.human_label
        predicted = verdict.predicted_label
        pairs.append((human, predicted))

        cat = per_category.setdefault(record.category, {"total": 0, "caught": 0, "missed": 0, "false_alarm": 0})
        cat["total"] += 1
        if human == UNFAITHFUL:
            cat["caught" if predicted == UNFAITHFUL else "missed"] += 1
        elif predicted == UNFAITHFUL:
            cat["false_alarm"] += 1

        if human != predicted:
            kind = "false_negative (missed)" if human == UNFAITHFUL else "false_positive (false alarm)"
            disagreements.append({
                "id": record.id,
                "category": record.category,
                "kind": kind,
                "human_label": human,
                "judge_label": predicted,
                "human_note": record.human_note,
                "judge_reason": verdict.reason,
            })

    c = confusion_matrix(pairs)
    precision = _safe_div(c.tp, c.tp + c.fp)
    recall = _safe_div(c.tp, c.tp + c.fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return CalibrationMetrics(
        n=c.n,
        human_unfaithful=c.tp + c.fn,
        human_faithful=c.tn + c.fp,
        confusion=c,
        agreement=_safe_div(c.tp + c.tn, c.n),
        precision=precision,
        recall=recall,
        f1=f1,
        false_negative_rate=_safe_div(c.fn, c.tp + c.fn),
        false_positive_rate=_safe_div(c.fp, c.fp + c.tn),
        kappa=cohens_kappa(c),
        per_category=per_category,
        disagreements=disagreements,
    )


# --------------------------------------------------------------------------- #
# Snapshot + report I/O
# --------------------------------------------------------------------------- #
def run_judge(records: Sequence[CalibrationRecord], judge) -> Dict[str, Verdict]:
    return {record.id: judge.judge(record) for record in records}


def write_snapshot(path: Path, judge_meta: Dict[str, Any], verdicts: Dict[str, Verdict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "judge": judge_meta,
        "verdicts": {
            rid: {
                "passed": v.passed,
                "unsupported_facts": v.unsupported_facts,
                "invalid_citation_ids": v.invalid_citation_ids,
                "reason": v.reason,
            }
            for rid, v in verdicts.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_snapshot(path: Path) -> Tuple[Dict[str, Any], Dict[str, Verdict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdicts = {
        rid: Verdict(
            record_id=rid,
            passed=bool(v["passed"]),
            unsupported_facts=v.get("unsupported_facts", []),
            invalid_citation_ids=v.get("invalid_citation_ids", []),
            reason=v.get("reason", ""),
        )
        for rid, v in payload.get("verdicts", {}).items()
    }
    return payload.get("judge", {}), verdicts


def render_report(metrics: CalibrationMetrics, judge_meta: Dict[str, Any], generated_at: str) -> str:
    c = metrics.confusion
    backend = judge_meta.get("backend", "unknown")
    lines: List[str] = []
    lines.append("# LLM-as-Judge Calibration Report")
    lines.append("")
    lines.append(f"- **Generated:** {generated_at}")
    lines.append(f"- **Judge backend:** `{backend}`"
                 + (f" (provider=`{judge_meta.get('provider')}`, model=`{judge_meta.get('model')}`)"
                    if backend == "llm" else ""))
    lines.append(f"- **Cases:** {metrics.n} "
                 f"({metrics.human_unfaithful} unfaithful / {metrics.human_faithful} faithful, human-labeled)")
    if backend == "simulated":
        lines.append("- **Note:** numbers are from the deterministic *simulated* stand-in judge, "
                     "not a live model. See `docs/judge_calibration_provenance.md`.")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"| --- | --- |")
    lines.append(f"| Agreement (accuracy) | {metrics.agreement:.1%} |")
    lines.append(f"| Cohen's kappa | {metrics.kappa:.3f} |")
    lines.append(f"| Precision (unfaithful) | {metrics.precision:.1%} |")
    lines.append(f"| Recall / catch rate (unfaithful) | {metrics.recall:.1%} |")
    lines.append(f"| F1 (unfaithful) | {metrics.f1:.3f} |")
    lines.append(f"| **False-negative rate (fail-open)** | **{metrics.false_negative_rate:.1%}** |")
    lines.append(f"| False-positive rate (false alarms) | {metrics.false_positive_rate:.1%} |")
    lines.append("")
    lines.append("## Confusion matrix (positive = unfaithful)")
    lines.append("")
    lines.append("| | judge: unfaithful | judge: faithful |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| **human: unfaithful** | TP = {c.tp} | FN = {c.fn} (missed) |")
    lines.append(f"| **human: faithful** | FP = {c.fp} | TN = {c.tn} |")
    lines.append("")
    lines.append("## Catch rate by defect category")
    lines.append("")
    lines.append("| Category | Total | Caught | Missed | False alarms |")
    lines.append("| --- | --- | --- | --- | --- |")
    for cat in sorted(metrics.per_category):
        stats = metrics.per_category[cat]
        lines.append(f"| {cat} | {stats['total']} | {stats['caught']} | {stats['missed']} | {stats['false_alarm']} |")
    lines.append("")
    lines.append("## Disagreements (judge vs human)")
    lines.append("")
    if not metrics.disagreements:
        lines.append("None — judge matched every human label.")
    else:
        for d in metrics.disagreements:
            lines.append(f"- **{d['id']}** ({d['category']}, {d['kind']}): "
                         f"human=`{d['human_label']}`, judge=`{d['judge_label']}`. "
                         f"Human note: {d['human_note']}")
    lines.append("")
    lines.append("## How to interpret")
    lines.append("")
    lines.append("- **False-negative rate** is the share of genuinely unfaithful rationales the judge "
                 "passed. This is the fail-open risk: hallucinated justifications that would ship.")
    lines.append("- **Cohen's kappa** discounts agreement that could happen by chance, so it is harder "
                 "to game than raw accuracy on an imbalanced set.")
    lines.append("- The **by-category** table shows *which* defects the judge misses, which is the "
                 "actionable output: it points at the prompt/rule changes that would close the gap.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the rationale-faithfulness LLM judge against human labels.")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--backend", choices=["simulated", "llm", "snapshot"], default="simulated",
                        help="simulated (default, deterministic), llm (real judge, needs key), "
                             "or snapshot (recompute from committed snapshot).")
    parser.add_argument("--snapshot", type=Path, default=_DEFAULT_SNAPSHOT)
    parser.add_argument("--report-path", type=Path, default=_DEFAULT_REPORT)
    parser.add_argument("--record", action="store_true",
                        help="Write the snapshot and the markdown report to disk.")
    parser.add_argument("--min-agreement", type=float, default=None,
                        help="Soft threshold: return non-zero if agreement falls below this.")
    parser.add_argument("--min-kappa", type=float, default=None,
                        help="Soft threshold: return non-zero if Cohen's kappa falls below this.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable metrics JSON.")
    args = parser.parse_args(argv)

    records = load_calibration_set(args.dataset)

    if args.backend == "snapshot":
        judge_meta, verdicts = load_snapshot(args.snapshot)
        missing = [r.id for r in records if r.id not in verdicts]
        if missing:
            print(f"snapshot_error: snapshot missing verdicts for {missing}", file=sys.stderr)
            return 2
    else:
        try:
            judge = build_judge(args.backend)
        except RuntimeError as exc:
            print(f"backend_error: {exc}", file=sys.stderr)
            return 2
        judge_meta = judge.describe()
        verdicts = run_judge(records, judge)

    metrics = compute_metrics(records, verdicts)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if args.record:
        write_snapshot(args.snapshot, judge_meta, verdicts)
        report = render_report(metrics, judge_meta, generated_at)
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(report + "\n", encoding="utf-8")
        print(f"wrote snapshot -> {args.snapshot}")
        print(f"wrote report   -> {args.report_path}")

    if args.json:
        print(json.dumps({
            "backend": judge_meta.get("backend"),
            "n": metrics.n,
            "agreement": round(metrics.agreement, 4),
            "kappa": round(metrics.kappa, 4),
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "f1": round(metrics.f1, 4),
            "false_negative_rate": round(metrics.false_negative_rate, 4),
            "false_positive_rate": round(metrics.false_positive_rate, 4),
            "confusion": {"tp": metrics.confusion.tp, "fp": metrics.confusion.fp,
                          "fn": metrics.confusion.fn, "tn": metrics.confusion.tn},
        }, indent=2, sort_keys=True))
    else:
        print(render_report(metrics, judge_meta, generated_at))

    exit_code = 0
    if args.min_agreement is not None and metrics.agreement < args.min_agreement:
        print(f"threshold_failure: agreement {metrics.agreement:.3f} < {args.min_agreement}", file=sys.stderr)
        exit_code = 1
    if args.min_kappa is not None and metrics.kappa < args.min_kappa:
        print(f"threshold_failure: kappa {metrics.kappa:.3f} < {args.min_kappa}", file=sys.stderr)
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
