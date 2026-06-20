"""Extraction target schema and prompt for the fine-tuning task.

The model reads a free-text producer note and emits a fixed JSON object. Fields
not stated in the note must be `null` — this is the abstention/refusal behavior
the eval scores (a model that guesses a roof age that was never mentioned is
worse than one that returns null).
"""

from __future__ import annotations

from typing import Any, Dict, List

# Ordered list of fields the model must always emit (null when not stated).
EXTRACTION_FIELDS: List[str] = [
    "applicant_name",
    "property_address",
    "occupancy",
    "dwelling_type",
    "year_built",
    "roof_age_years",
    "coverage_a",
    "deductible",
]

OCCUPANCY_VALUES = [
    "owner_occupied_primary",
    "owner_occupied_secondary",
    "tenant_occupied",
    "vacant",
]
DWELLING_VALUES = ["single_family", "condo", "townhouse", "row_house", "commercial"]

SYSTEM_PROMPT = (
    "You extract structured homeowner-insurance intake data from a free-text "
    "producer note. Return ONLY a JSON object with exactly these keys: "
    + ", ".join(EXTRACTION_FIELDS)
    + ". Rules: use the canonical occupancy values "
    + "(owner_occupied_primary, owner_occupied_secondary, tenant_occupied, vacant) "
    + "and dwelling_type values (single_family, condo, townhouse, row_house, commercial). "
    + "year_built, roof_age_years, coverage_a, and deductible are integers. "
    + "If a field is not stated in the note, set it to null. Do not guess. "
    + "Output JSON only, no prose."
)


def empty_target() -> Dict[str, Any]:
    """A target object with every field absent (all null)."""
    return {field: None for field in EXTRACTION_FIELDS}


def normalize_target(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a (possibly partial) dict into the canonical field set/order."""
    target = empty_target()
    for field in EXTRACTION_FIELDS:
        if field in obj and obj[field] is not None:
            target[field] = obj[field]
    return target
