"""Generate the stratified HO3 evaluation dataset.

The labels are derived from the governed deterministic rule contract plus the
workflow's documented missing-info gates. This keeps the dataset reproducible
while still exercising varied combinations of accept, refer, decline, and
waiting-for-info outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.underwriting_rules import evaluate_underwriting_rules
from models.schemas import HO3Submission


DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "ho3_labeled.jsonl"

ACCEPT_CITATION = "uw_guidelines_homeowners_5__decision_outcomes_5_1_accept_11"
CITATION_BY_REASON = {
    "OCCUPANCY_REVIEW": "uw_guidelines_homeowners_1__eligibility_overview_1_4_home_business___short_term_rental_4",
    "INELIGIBLE_DWELLING": "uw_guidelines_homeowners_1__eligibility_overview_1_1_eligible_risk_types_1",
    "OLD_CONSTRUCTION": "uw_guidelines_homeowners_2__construction___maintenance_standards_2_2_electrical___plumbing_6",
    "ROOF_AGE": "uw_guidelines_homeowners_2__construction___maintenance_standards_2_1_roof_condition_and_roof_age_5",
    "WILDFIRE_HIGH": "uw_guidelines_homeowners_4__catastrophe___hazard_signals_4_1_wildfire_hazard_8",
    "WILDFIRE_SEVERE": "uw_guidelines_homeowners_4__catastrophe___hazard_signals_4_1_wildfire_hazard_8",
    "WILDFIRE_SEVERE_MITIGATED": "uw_guidelines_homeowners_4__catastrophe___hazard_signals_4_1_wildfire_hazard_8",
    "FLOOD_SFHA": "uw_guidelines_homeowners_4__catastrophe___hazard_signals_4_2_flood_hazard_9",
}

NAMES = [
    "Avery Chen", "Blake Rivera", "Casey Patel", "Dana Morgan", "Elliot Stone",
    "Finley Brooks", "Gray Harper", "Hayden Lee", "Indigo Watts", "Jordan Kim",
    "Kai Bennett", "Logan Price", "Morgan Fox", "Noa Sanchez", "Oakley Green",
    "Parker Young", "Quinn Adams", "Reese Nelson", "Sawyer Cruz", "Tatum Hayes",
    "Uma Ross", "Vale Morris", "Wren Carter", "Xen Taylor", "Yael Martin",
    "Zion Edwards", "Amari Scott", "Briar Walker", "Camden Flores", "Devon Hughes",
    "Emery Long", "Frankie Nguyen", "Gale Moreno", "Harper Singh", "Ira Collins",
    "Jules Park", "Kendall Ortiz", "Lane Carter", "Mika Shah", "Nico Brown",
]


def main() -> None:
    cases = build_cases()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.write_text(
        "\n".join(json.dumps(case, separators=(",", ":")) for case in cases) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} eval cases to {DATASET_PATH}")


def build_cases() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    rows.extend(_series("accept_low_risk", 24, _accept_low_risk))
    rows.extend(_series("accept_high_wildfire_mitigated", 12, _accept_high_wildfire_mitigated))
    rows.extend(_series("missing_intake", 24, _missing_intake))
    rows.extend(_series("missing_contextual_wildfire", 12, _missing_contextual_wildfire))
    rows.extend(_series("roof_referral", 16, _roof_referral))
    rows.extend(_series("occupancy_referral", 14, _occupancy_referral))
    rows.extend(_series("old_construction_referral", 14, _old_construction_referral))
    rows.extend(_series("wildfire_referral", 14, _wildfire_referral))
    rows.extend(_series("flood_referral", 14, _flood_referral))
    rows.extend(_series("commercial_decline", 14, _commercial_decline))
    rows.extend(_series("severe_wildfire", 14, _severe_wildfire))
    rows.extend(_series("multi_trigger", 24, _multi_trigger))

    for idx, row in enumerate(rows, start=1):
        row["id"] = f"HO3-EVAL-{idx:03d}"
    return rows


def _series(stratum: str, count: int, factory) -> Iterable[Dict[str, Any]]:
    for idx in range(count):
        submission, title = factory(idx)
        yield _case(stratum, idx, title, submission)


def _case(stratum: str, idx: int, title: str, submission: Dict[str, Any]) -> Dict[str, Any]:
    expected = _expected_for(submission)
    return {
        "id": "HO3-EVAL-000",
        "title": title,
        "stratum": stratum,
        "submission": submission,
        "expected": expected,
    }


def _expected_for(submission: Dict[str, Any]) -> Dict[str, Any]:
    missing_questions = _intake_missing_questions(submission)
    if missing_questions:
        return {
            "decision": "REFER",
            "reason_codes": [question.upper() for question in missing_questions],
            "gold_citations": [],
            "status": "waiting_for_info",
        }

    hazard_profile = _hazard_profile(submission["risk"]["property_address"])
    if (
        hazard_profile["wildfire_band"] in {"High", "Severe"}
        and submission["risk"].get("wildfire_mitigation_evidence") is None
    ):
        return {
            "decision": "REFER",
            "reason_codes": ["WILDFIRE_MITIGATION_EVIDENCE"],
            "gold_citations": [],
            "status": "waiting_for_info",
        }

    evaluation = evaluate_underwriting_rules(
        HO3Submission(**submission),
        hazard_profile=hazard_profile,
        property_profile={"territory": _territory(submission["risk"]["property_address"])},
    )
    reason_codes = [finding.reason_code for finding in evaluation.findings]
    return {
        "decision": evaluation.decision.value,
        "reason_codes": reason_codes,
        "gold_citations": _gold_citations(reason_codes),
        "status": "completed" if evaluation.decision.value == "ACCEPT" else "pending_review",
    }


def _intake_missing_questions(submission: Dict[str, Any]) -> List[str]:
    risk = submission.get("risk", {})
    applicant = submission.get("applicant", {})
    questions = []
    if _missing(applicant.get("full_name")):
        questions.append("applicant_name")
    if _missing(risk.get("property_address")):
        questions.append("property_address")
    if risk.get("occupancy") not in {"owner_occupied_primary", "owner_occupied_secondary", "tenant_occupied", "vacant"}:
        questions.append("occupancy")
    if _missing(risk.get("roof_age_years")):
        questions.append("roof_age_years")
    return questions


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "unknown", "unk", "tbd", "n/a", "na", "unsure", "uncertain"}
    return False


def _gold_citations(reason_codes: List[str]) -> List[str]:
    if not reason_codes:
        return [ACCEPT_CITATION]
    if "OCCUPANCY_REVIEW" in reason_codes and len(reason_codes) > 1:
        reason_codes = [code for code in reason_codes if code != "OCCUPANCY_REVIEW"]
    citations = []
    for code in reason_codes:
        citation = CITATION_BY_REASON.get(code)
        if citation and citation not in citations:
            citations.append(citation)
    return citations[:5]


def _hazard_profile(address: str) -> Dict[str, Any]:
    address_lower = address.lower()
    wildfire_band = "Low"
    if any(token in address_lower for token in ["severe fire zone", "severe wildfire"]):
        wildfire_band = "Severe"
    elif any(token in address_lower for token in ["sacramento", "santa rosa", "paradise", "fire zone"]):
        wildfire_band = "High"
    flood_sfha = any(token in address_lower for token in ["river", "fresno"])
    return {
        "wildfire_band": wildfire_band,
        "wildfire_risk_score": {"Low": 0.15, "High": 0.78, "Severe": 0.95}[wildfire_band],
        "flood_sfha": flood_sfha,
        "flood_risk_score": 0.82 if flood_sfha else 0.2,
    }


def _territory(address: str) -> str:
    address_lower = address.lower()
    if any(token in address_lower for token in ["sacramento", "santa rosa", "paradise", "fire zone", "river", "fresno"]):
        return "HighRiskCounty"
    if any(token in address_lower for token in ["palo alto", "san diego", "oakland"]):
        return "LowRiskCounty"
    return "MediumRiskCounty"


def _base(idx: int, *, address: str, name: Optional[str] = None) -> Dict[str, Any]:
    dwelling_types = ["single_family", "condo", "townhouse", "row_house"]
    construction_types = ["frame", "masonry", "superior_masonry"]
    return {
        "applicant": {
            "full_name": name or NAMES[idx % len(NAMES)],
            "email": f"applicant{idx + 1}@example.com",
            "phone": f"+1-555-010-{idx % 100:02d}",
        },
        "risk": {
            "property_address": address,
            "occupancy": "owner_occupied_primary" if idx % 5 else "owner_occupied_secondary",
            "dwelling_type": dwelling_types[idx % len(dwelling_types)],
            "year_built": 1985 + (idx % 36),
            "roof_age_years": 3 + (idx % 16),
            "construction_type": construction_types[idx % len(construction_types)],
            "stories": 1 + (idx % 3),
        },
        "coverage_request": {
            "coverage_a": 300000 + (idx % 18) * 25000,
            "coverage_b_pct": 10,
            "coverage_c_pct": 50,
            "coverage_d_pct": 20,
            "coverage_e": 300000,
            "coverage_f": 5000,
            "deductible": [1000, 1500, 2500][idx % 3],
        },
    }


def _accept_low_risk(idx: int):
    cities = ["Palo Alto", "San Diego", "Oakland", "Long Beach", "Pasadena", "Monterey"]
    submission = _base(idx, address=f"{100 + idx} Market St, {cities[idx % len(cities)]}, CA 94{idx % 100:03d}")
    submission["risk"]["year_built"] = 1990 + (idx % 31)
    submission["risk"]["roof_age_years"] = 2 + (idx % 16)
    return submission, "Clean low-risk homeowner submission"


def _accept_high_wildfire_mitigated(idx: int):
    cities = ["Sacramento", "Santa Rosa", "Paradise"]
    submission = _base(idx, address=f"{200 + idx} Fire Zone Rd, {cities[idx % len(cities)]}, CA 95{idx % 100:03d}")
    submission["risk"]["year_built"] = 2000 + (idx % 21)
    submission["risk"]["roof_age_years"] = 4 + (idx % 12)
    submission["risk"]["wildfire_mitigation_evidence"] = True
    submission["risk"]["mitigation_notes"] = "Defensible-space documentation provided."
    return submission, "High wildfire band with documented mitigation"


def _missing_intake(idx: int):
    submission = _base(idx, address=f"{300 + idx} Hill St, Los Angeles, CA 900{idx % 10}")
    missing_type = idx % 4
    if missing_type == 0:
        submission["applicant"]["full_name"] = "unknown"
    elif missing_type == 1:
        submission["risk"]["property_address"] = ""
    elif missing_type == 2:
        submission["risk"]["occupancy"] = "unsure"
    else:
        submission["risk"]["roof_age_years"] = None
    return submission, "Intake missing-information pause"


def _missing_contextual_wildfire(idx: int):
    submission = _base(idx, address=f"{400 + idx} Ridge Rd, Sacramento, CA 958{idx % 100:02d}")
    submission["risk"]["year_built"] = 1990 + (idx % 25)
    submission["risk"]["roof_age_years"] = 5 + (idx % 10)
    return submission, "High wildfire contextual mitigation pause"


def _roof_referral(idx: int):
    submission = _base(idx, address=f"{500 + idx} Cedar Ave, Long Beach, CA 908{idx % 100:02d}")
    submission["risk"]["roof_age_years"] = 20 + (idx % 18)
    return submission, "Roof age referral"


def _occupancy_referral(idx: int):
    submission = _base(idx, address=f"{600 + idx} Coastal Hwy, Monterey, CA 939{idx % 100:02d}")
    submission["risk"]["occupancy"] = "tenant_occupied" if idx % 2 == 0 else "vacant"
    submission["risk"]["roof_age_years"] = 5 + (idx % 10)
    return submission, "Occupancy referral"


def _old_construction_referral(idx: int):
    submission = _base(idx, address=f"{700 + idx} Grove Ave, Oakland, CA 946{idx % 100:02d}")
    submission["risk"]["year_built"] = 1900 + (idx % 40)
    submission["risk"]["roof_age_years"] = 5 + (idx % 12)
    return submission, "Pre-1940 construction referral"


def _wildfire_referral(idx: int):
    cities = ["Sacramento", "Santa Rosa", "Paradise"]
    submission = _base(idx, address=f"{800 + idx} Oak Ave, {cities[idx % len(cities)]}, CA 95{idx % 100:03d}")
    submission["risk"]["roof_age_years"] = 5 + (idx % 12)
    submission["risk"]["wildfire_mitigation_evidence"] = False
    return submission, "High wildfire referral without mitigation"


def _flood_referral(idx: int):
    suffix = "River Rd" if idx % 2 == 0 else "Maple Ave"
    city = "Fresno" if idx % 3 else "Bakersfield"
    submission = _base(idx, address=f"{900 + idx} {suffix}, {city}, CA 937{idx % 100:02d}")
    submission["risk"]["roof_age_years"] = 5 + (idx % 12)
    return submission, "Flood SFHA referral"


def _commercial_decline(idx: int):
    submission = _base(idx, address=f"{1000 + idx} Industrial Way, Los Angeles, CA 900{idx % 100:02d}")
    submission["risk"]["dwelling_type"] = "commercial"
    submission["risk"]["construction_type"] = "fire_resistive"
    submission["risk"]["roof_age_years"] = 5 + (idx % 12)
    return submission, "Commercial dwelling decline"


def _severe_wildfire(idx: int):
    submission = _base(idx, address=f"{1100 + idx} Severe Fire Zone Vista, Napa, CA 945{idx % 100:02d}")
    submission["risk"]["roof_age_years"] = 5 + (idx % 12)
    submission["risk"]["wildfire_mitigation_evidence"] = idx % 3 == 0
    if submission["risk"]["wildfire_mitigation_evidence"]:
        submission["risk"]["mitigation_notes"] = "Brush clearance and ember-resistant vent evidence attached."
    return submission, "Severe wildfire risk"


def _multi_trigger(idx: int):
    patterns = [
        {"address": "River Bend, Fresno", "occupancy": "tenant_occupied", "roof": 23},
        {"address": "Severe Wildfire Industrial Rd, Napa", "dwelling": "commercial", "mitigation": False},
        {"address": "Fire Zone River Rd, Sacramento", "roof": 25, "mitigation": False},
        {"address": "River Walk, Fresno", "year": 1932, "roof": 22},
        {"address": "Severe Fire Zone River Rd, Napa", "year": 1935, "mitigation": True},
        {"address": "Warehouse Row, Oakland", "dwelling": "commercial", "year": 1925},
    ]
    pattern = patterns[idx % len(patterns)]
    submission = _base(idx, address=f"{1200 + idx} {pattern['address']}, CA 95{idx % 100:03d}")
    submission["risk"]["roof_age_years"] = pattern.get("roof", 8 + (idx % 10))
    submission["risk"]["year_built"] = pattern.get("year", 1980 + (idx % 30))
    submission["risk"]["occupancy"] = pattern.get("occupancy", "owner_occupied_primary")
    submission["risk"]["dwelling_type"] = pattern.get("dwelling", "single_family")
    if "mitigation" in pattern:
        submission["risk"]["wildfire_mitigation_evidence"] = pattern["mitigation"]
    return submission, "Multiple underwriting triggers"


if __name__ == "__main__":
    main()
