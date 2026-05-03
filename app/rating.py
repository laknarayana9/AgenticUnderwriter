"""Transparent deterministic premium indication for the HO3 workflow."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class RatingTool:
    """Calculate a simple, explainable premium indication."""

    def calculate_premium(self, coverage_amount: float, risk_factors: Dict[str, Any]) -> Dict[str, Any]:
        base_premium = coverage_amount * 0.002
        hazard_scores = risk_factors.get("hazard_scores", {})

        territory_factor = {
            "LowRiskCounty": 0.95,
            "MediumRiskCounty": 1.0,
            "HighRiskCounty": 1.1,
        }.get(risk_factors.get("territory"), 1.0)

        construction_year = risk_factors.get("construction_year") or 2000
        if construction_year >= 2000:
            age_factor = 0.95
        elif construction_year >= 1980:
            age_factor = 1.0
        else:
            age_factor = 1.1

        wildfire_score = hazard_scores.get("wildfire_risk", 0)
        flood_score = hazard_scores.get("flood_risk", 0)
        hazard_factor = 1.0
        if wildfire_score >= 0.9:
            hazard_factor += 0.20
        elif wildfire_score >= 0.7:
            hazard_factor += 0.12
        elif wildfire_score >= 0.4:
            hazard_factor += 0.05
        if flood_score >= 0.7:
            hazard_factor += 0.10

        annual_premium = round(base_premium * territory_factor * age_factor * hazard_factor, 2)
        logger.info("Calculated premium: $%.2f", annual_premium)
        return {
            "annual_premium": annual_premium,
            "currency": "USD",
            "base_premium": round(base_premium, 2),
            "factors": {
                "territory": territory_factor,
                "construction_age": age_factor,
                "hazard": round(hazard_factor, 3),
            },
        }
