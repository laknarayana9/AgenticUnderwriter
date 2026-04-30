"""
Phase A Demo Scenarios for End-to-End Testing
10 curated HO3 scenarios to test the complete workflow with correct status transitions and decision packets.
"""

from models.schemas import HO3Submission, Applicant, RiskProfile, CoverageRequest

DEMO_SCENARIOS = [
    {
        "name": "Scenario 1: Standard Quote - Low Risk",
        "description": "Typical single-family home with no risk factors, should quote eligible",
        "expected_decision": "ACCEPT",
        "submission": {
            "applicant": {
                "full_name": "John Smith",
                "email": "john.smith@example.com",
                "phone": "+1-415-555-0123"
            },
            "risk": {
                "property_address": "123 Main St, San Francisco, CA 94102",
                "occupancy": "owner_occupied_primary",
                "dwelling_type": "single_family",
                "year_built": 2000,
                "roof_age_years": 10,
                "construction_type": "frame",
                "stories": 2
            },
            "coverage_request": {
                "coverage_a": 500000,
                "coverage_b_pct": 10,
                "coverage_c_pct": 50,
                "coverage_d_pct": 20,
                "coverage_e": 300000,
                "coverage_f": 5000,
                "deductible": 1000
            }
        }
    },
    {
        "name": "Scenario 2: Wildfire High Risk - Refer",
        "description": "Property in high wildfire zone should trigger referral",
        "expected_decision": "REFER",
        "submission": {
            "applicant": {
                "full_name": "Jane Doe",
                "email": "jane.doe@example.com",
                "phone": "+1-916-555-0456"
            },
            "risk": {
                "property_address": "456 Oak Ave, Sacramento, CA 95814",
                "occupancy": "owner_occupied_primary",
                "dwelling_type": "single_family",
                "year_built": 1985,
                "roof_age_years": 20,
                "construction_type": "frame",
                "stories": 1
            },
            "coverage_request": {
                "coverage_a": 400000,
                "coverage_b_pct": 10,
                "coverage_c_pct": 50,
                "coverage_d_pct": 20,
                "coverage_e": 300000,
                "coverage_f": 5000,
                "deductible": 1500
            }
        }
    },
    {
        "name": "Scenario 3: Missing Roof Age - Need Info",
        "description": "Missing roof age should trigger need more info workflow",
        "expected_decision": "REFER",
        "expected_status": "waiting_for_info",
        "submission": {
            "applicant": {
                "full_name": "Robert Johnson",
                "email": "robert.j@example.com",
                "phone": "+1-213-555-0789"
            },
            "risk": {
                "property_address": "789 Pine St, Los Angeles, CA 90001",
                "occupancy": "owner_occupied_primary",
                "dwelling_type": "single_family",
                "year_built": 1995,
                "roof_age_years": None,  # Missing
                "construction_type": "frame",
                "stories": 1
            },
            "coverage_request": {
                "coverage_a": 350000,
                "coverage_b_pct": 10,
                "coverage_c_pct": 50,
                "coverage_d_pct": 20,
                "coverage_e": 300000,
                "coverage_f": 5000,
                "deductible": 1000
            }
        }
    },
    {
        "name": "Scenario 4: Old Construction - Refer",
        "description": "Property built before 1940 should trigger referral for additional review",
        "expected_decision": "REFER",
        "submission": {
            "applicant": {
                "full_name": "Mary Williams",
                "email": "mary.w@example.com",
                "phone": "+1-415-555-0234"
            },
            "risk": {
                "property_address": "321 Elm St, San Francisco, CA 94103",
                "occupancy": "owner_occupied_primary",
                "dwelling_type": "single_family",
                "year_built": 1920,
                "roof_age_years": 15,
                "construction_type": "frame",
                "stories": 2
            },
            "coverage_request": {
                "coverage_a": 600000,
                "coverage_b_pct": 10,
                "coverage_c_pct": 50,
                "coverage_d_pct": 20,
                "coverage_e": 300000,
                "coverage_f": 5000,
                "deductible": 2000
            }
        }
    },
    {
        "name": "Scenario 5: Condo - Quote Eligible",
        "description": "Condominium should be eligible with appropriate coverage",
        "expected_decision": "ACCEPT",
        "submission": {
            "applicant": {
                "full_name": "David Brown",
                "email": "david.b@example.com",
                "phone": "+1-619-555-0567"
            },
            "risk": {
                "property_address": "555 Beach Blvd, San Diego, CA 92101",
                "occupancy": "owner_occupied_primary",
                "dwelling_type": "condo",
                "year_built": 2010,
                "roof_age_years": 5,
                "construction_type": "frame",
                "stories": 1
            },
            "coverage_request": {
                "coverage_a": 300000,
                "coverage_b_pct": 10,
                "coverage_c_pct": 50,
                "coverage_d_pct": 20,
                "coverage_e": 300000,
                "coverage_f": 5000,
                "deductible": 1000
            }
        }
    },
    {
        "name": "Scenario 6: Flood Risk - Refer",
        "description": "High flood risk should trigger referral",
        "expected_decision": "REFER",
        "submission": {
            "applicant": {
                "full_name": "Sarah Miller",
                "email": "sarah.m@example.com",
                "phone": "+1-559-555-0890"
            },
            "risk": {
                "property_address": "888 River Rd, Fresno, CA 93701",
                "occupancy": "owner_occupied_primary",
                "dwelling_type": "single_family",
                "year_built": 1990,
                "roof_age_years": 12,
                "construction_type": "frame",
                "stories": 1
            },
            "coverage_request": {
                "coverage_a": 450000,
                "coverage_b_pct": 10,
                "coverage_c_pct": 50,
                "coverage_d_pct": 20,
                "coverage_e": 300000,
                "coverage_f": 5000,
                "deductible": 1000
            }
        }
    },
    {
        "name": "Scenario 7: Townhouse - Quote Eligible",
        "description": "Townhouse should be eligible with standard coverage",
        "expected_decision": "ACCEPT",
        "submission": {
            "applicant": {
                "full_name": "Michael Davis",
                "email": "michael.d@example.com",
                "phone": "+1-510-555-0123"
            },
            "risk": {
                "property_address": "1000 Civic Dr, Oakland, CA 94601",
                "occupancy": "owner_occupied_primary",
                "dwelling_type": "townhouse",
                "year_built": 2015,
                "roof_age_years": 8,
                "construction_type": "frame",
                "stories": 3
            },
            "coverage_request": {
                "coverage_a": 550000,
                "coverage_b_pct": 10,
                "coverage_c_pct": 50,
                "coverage_d_pct": 20,
                "coverage_e": 300000,
                "coverage_f": 5000,
                "deductible": 1500
            }
        }
    },
    {
        "name": "Scenario 8: Claims History - Refer",
        "description": "Multiple claims in past 5 years should trigger referral",
        "expected_decision": "REFER",
        "submission": {
            "applicant": {
                "full_name": "Emily Wilson",
                "email": "emily.w@example.com",
                "phone": "+1-707-555-0456"
            },
            "risk": {
                "property_address": "2000 Valley Rd, Santa Rosa, CA 95401",
                "occupancy": "owner_occupied_primary",
                "dwelling_type": "single_family",
                "year_built": 1980,
                "roof_age_years": 18,
                "construction_type": "frame",
                "stories": 1
            },
            "coverage_request": {
                "coverage_a": 380000,
                "coverage_b_pct": 10,
                "coverage_c_pct": 50,
                "coverage_d_pct": 20,
                "coverage_e": 300000,
                "coverage_f": 5000,
                "deductible": 1000
            }
        }
    },
    {
        "name": "Scenario 9: High Coverage - Quote Eligible",
        "description": "High value home with no risk factors should be eligible",
        "expected_decision": "ACCEPT",
        "submission": {
            "applicant": {
                "full_name": "James Taylor",
                "email": "james.t@example.com",
                "phone": "+1-650-555-0789"
            },
            "risk": {
                "property_address": "3000 Hillside Ave, Palo Alto, CA 94301",
                "occupancy": "owner_occupied_primary",
                "dwelling_type": "single_family",
                "year_built": 2018,
                "roof_age_years": 3,
                "construction_type": "superior_masonry",
                "stories": 2
            },
            "coverage_request": {
                "coverage_a": 1500000,
                "coverage_b_pct": 10,
                "coverage_c_pct": 50,
                "coverage_d_pct": 20,
                "coverage_e": 500000,
                "coverage_f": 10000,
                "deductible": 2500
            }
        }
    },
    {
        "name": "Scenario 10: Tenant Occupied - Refer",
        "description": "Tenant-occupied property should trigger referral",
        "expected_decision": "REFER",
        "submission": {
            "applicant": {
                "full_name": "Lisa Anderson",
                "email": "lisa.a@example.com",
                "phone": "+1-831-555-0234"
            },
            "risk": {
                "property_address": "4000 Coastal Hwy, Monterey, CA 93940",
                "occupancy": "tenant_occupied",
                "dwelling_type": "single_family",
                "year_built": 2005,
                "roof_age_years": 12,
                "construction_type": "frame",
                "stories": 2
            },
            "coverage_request": {
                "coverage_a": 425000,
                "coverage_b_pct": 10,
                "coverage_c_pct": 50,
                "coverage_d_pct": 20,
                "coverage_e": 300000,
                "coverage_f": 5000,
                "deductible": 1000
            }
        }
    }
]


def get_scenario(index: int) -> dict:
    """Get a demo scenario by index (1-based)."""
    if 1 <= index <= len(DEMO_SCENARIOS):
        return DEMO_SCENARIOS[index - 1]
    raise IndexError(f"Scenario index {index} out of range (1-{len(DEMO_SCENARIOS)})")


def get_all_scenarios() -> list:
    """Get all demo scenarios."""
    return DEMO_SCENARIOS


def create_submission_from_scenario(scenario: dict) -> HO3Submission:
    """Create an HO3Submission object from a scenario."""
    submission_data = scenario["submission"]
    return HO3Submission(
        applicant=Applicant(**submission_data["applicant"]),
        risk=RiskProfile(**submission_data["risk"]),
        coverage_request=CoverageRequest(**submission_data["coverage_request"])
    )


if __name__ == "__main__":
    # Print all scenarios for review
    print("Phase A Demo Scenarios")
    print("=" * 80)
    for i, scenario in enumerate(DEMO_SCENARIOS, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Expected Decision: {scenario['expected_decision']}")
        if "expected_status" in scenario:
            print(f"   Expected Status: {scenario['expected_status']}")
