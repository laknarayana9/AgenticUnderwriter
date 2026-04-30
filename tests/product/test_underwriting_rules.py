from app.underwriting_rules import evaluate_underwriting_rules
from models.schemas import HO3Submission


def _submission(**risk_overrides):
    risk = {
        "property_address": "123 Main St, Oakland, CA 94601",
        "occupancy": "owner_occupied_primary",
        "dwelling_type": "single_family",
        "year_built": 2010,
        "roof_age_years": 8,
        "construction_type": "frame",
        "stories": 1,
    }
    risk.update(risk_overrides)
    return HO3Submission(
        applicant={"full_name": "Rule Test"},
        risk=risk,
        coverage_request={"coverage_a": 500000, "deductible": 1000},
    )


def test_clean_risk_accepts_with_no_rule_findings():
    evaluation = evaluate_underwriting_rules(_submission(), hazard_profile={"wildfire_band": "Low"})

    assert evaluation.decision.value == "ACCEPT"
    assert evaluation.findings == []
    assert evaluation.ruleset_version


def test_missing_roof_age_fires_explicit_rule():
    evaluation = evaluate_underwriting_rules(_submission(roof_age_years=None))

    assert evaluation.decision.value == "REFER"
    assert [finding.rule_id for finding in evaluation.findings] == ["UW-ROOF-001"]
    assert evaluation.findings[0].reason_code == "MISSING_ROOF_AGE"


def test_old_roof_fires_referral_rule():
    evaluation = evaluate_underwriting_rules(_submission(roof_age_years=22))

    assert evaluation.decision.value == "REFER"
    assert "UW-ROOF-002" in [finding.rule_id for finding in evaluation.findings]


def test_high_wildfire_fires_referral_rule():
    evaluation = evaluate_underwriting_rules(_submission(), hazard_profile={"wildfire_band": "High"})

    assert evaluation.decision.value == "REFER"
    assert "UW-WF-001" in [finding.rule_id for finding in evaluation.findings]


def test_commercial_property_declines():
    evaluation = evaluate_underwriting_rules(_submission(dwelling_type="commercial"))

    assert evaluation.decision.value == "DECLINE"
    assert "UW-ELIG-001" in [finding.rule_id for finding in evaluation.findings]
