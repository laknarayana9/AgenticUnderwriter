"""
Versioned deterministic underwriting rules for the demo HO3 workflow.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from models.schemas import DecisionType, HO3Submission


RULESET_VERSION = "UW-RULESET-2026.04"


@dataclass(frozen=True)
class RuleFinding:
    rule_id: str
    outcome: DecisionType
    severity: str
    reason_code: str
    message: str
    citation_query: str


@dataclass(frozen=True)
class RuleEvaluation:
    ruleset_version: str
    decision: DecisionType
    confidence: float
    eligibility_score: float
    findings: List[RuleFinding]
    facts_used: Dict[str, Any]


def evaluate_underwriting_rules(
    submission: HO3Submission,
    hazard_profile: Optional[Dict[str, Any]] = None,
    property_profile: Optional[Dict[str, Any]] = None,
) -> RuleEvaluation:
    """Evaluate deterministic demo underwriting rules."""
    hazard_profile = hazard_profile or {}
    property_profile = property_profile or {}
    findings: List[RuleFinding] = []

    if submission.risk.occupancy in {"tenant_occupied", "vacant"}:
        findings.append(RuleFinding(
            rule_id="UW-OCC-001",
            outcome=DecisionType.REFER,
            severity="medium",
            reason_code="OCCUPANCY_REVIEW",
            message=f"Occupancy is {submission.risk.occupancy}.",
            citation_query="occupancy tenant vacant homeowners eligibility referred",
        ))

    if submission.risk.dwelling_type == "commercial":
        findings.append(RuleFinding(
            rule_id="UW-ELIG-001",
            outcome=DecisionType.DECLINE,
            severity="high",
            reason_code="INELIGIBLE_DWELLING",
            message="Commercial property is not eligible for an HO3 homeowners quote.",
            citation_query="eligible risk types commercial operations must not homeowners",
        ))

    if submission.risk.year_built < 1940:
        findings.append(RuleFinding(
            rule_id="UW-CONST-001",
            outcome=DecisionType.REFER,
            severity="medium",
            reason_code="OLD_CONSTRUCTION",
            message="Pre-1940 construction requires manual review.",
            citation_query="construction maintenance standards older dwelling referred",
        ))

    roof_age = submission.risk.roof_age_years
    if roof_age is None:
        findings.append(RuleFinding(
            rule_id="UW-ROOF-001",
            outcome=DecisionType.REFER,
            severity="medium",
            reason_code="MISSING_ROOF_AGE",
            message="Roof age must be provided for all submissions.",
            citation_query="roof age must be provided all submissions",
        ))
    elif roof_age >= 20:
        findings.append(RuleFinding(
            rule_id="UW-ROOF-002",
            outcome=DecisionType.REFER,
            severity="medium",
            reason_code="ROOF_AGE",
            message="Roof age is at or above the referral threshold.",
            citation_query="roof age greater than 20 years shall be referred",
        ))

    wildfire_band = hazard_profile.get("wildfire_band")
    if wildfire_band == "High":
        findings.append(RuleFinding(
            rule_id="UW-WF-001",
            outcome=DecisionType.REFER,
            severity="high",
            reason_code="WILDFIRE_HIGH",
            message="High wildfire band requires defensible-space evidence.",
            citation_query="wildfire high defensible space shall refer must request evidence",
        ))
    elif wildfire_band == "Severe":
        findings.append(RuleFinding(
            rule_id="UW-WF-002",
            outcome=DecisionType.DECLINE,
            severity="high",
            reason_code="WILDFIRE_SEVERE",
            message="Severe wildfire band is a hard decline unless mitigation is documented.",
            citation_query="wildfire severe must decline mitigation documented",
        ))

    if hazard_profile.get("flood_sfha"):
        findings.append(RuleFinding(
            rule_id="UW-FLD-001",
            outcome=DecisionType.REFER,
            severity="high",
            reason_code="FLOOD_SFHA",
            message="Special Flood Hazard Area requires flood review.",
            citation_query="special flood hazard area sfha shall refer flood insurance status",
        ))

    decision = _highest_severity_decision(findings)
    eligibility_score = _eligibility_score(findings)

    return RuleEvaluation(
        ruleset_version=RULESET_VERSION,
        decision=decision,
        confidence=_confidence(decision, findings),
        eligibility_score=eligibility_score,
        findings=findings,
        facts_used={
            "occupancy": submission.risk.occupancy,
            "dwelling_type": submission.risk.dwelling_type,
            "year_built": submission.risk.year_built,
            "roof_age_years": submission.risk.roof_age_years,
            "wildfire_band": wildfire_band,
            "flood_sfha": hazard_profile.get("flood_sfha", False),
            "territory": property_profile.get("territory"),
        },
    )


def build_rule_query(evaluation: RuleEvaluation) -> str:
    """Build a retrieval query from fired rules."""
    if not evaluation.findings:
        return "homeowners eligibility accept no referral decline triggers required fields"
    return " ".join(finding.citation_query for finding in evaluation.findings)


def findings_as_risk_factors(findings: List[RuleFinding]) -> List[Dict[str, Any]]:
    return [
        {
            "rule_id": finding.rule_id,
            "code": finding.reason_code,
            "severity": finding.severity,
            "because": finding.message,
        }
        for finding in findings
    ]


def _highest_severity_decision(findings: List[RuleFinding]) -> DecisionType:
    if any(finding.outcome == DecisionType.DECLINE for finding in findings):
        return DecisionType.DECLINE
    if any(finding.outcome == DecisionType.REFER for finding in findings):
        return DecisionType.REFER
    return DecisionType.ACCEPT


def _eligibility_score(findings: List[RuleFinding]) -> float:
    score = 0.92
    score -= 0.18 * len([finding for finding in findings if finding.severity == "medium"])
    score -= 0.28 * len([finding for finding in findings if finding.severity == "high"])
    return max(0.05, min(0.99, score))


def _confidence(decision: DecisionType, findings: List[RuleFinding]) -> float:
    if decision == DecisionType.DECLINE:
        return 0.93
    if findings:
        return 0.9 if any(finding.severity == "high" for finding in findings) else 0.86
    return 0.82
