"""Hermetic tests for the vision evidence service (Phase 1).

No network, no real model, no image files: a fake provider supplies extraction
results, and the deterministic stub covers the no-provider path. Verifies the
fenced boundary, abstention, confidence gating, and the mapping into the
submission that the deterministic rules consume.
"""

import hashlib

from app.vision_service import (
    VISION_ATTRIBUTES,
    VisionEvidenceService,
    VisionServiceConfig,
    fold_vision_into_submission,
)
from models.schemas import VisionEvidence


class _FakeProvider:
    model = "fake-vision-1"

    def __init__(self, payload):
        self._payload = payload

    def extract(self, image_bytes, system_prompt):
        return self._payload


def _submission(**risk_overrides):
    risk = {
        "property_address": "123 Pine St, Sacramento fire zone, CA",
        "occupancy": "owner_occupied_primary",
        "dwelling_type": "single_family",
        "year_built": 2005,
        "roof_age_years": 8,
        "construction_type": "frame",
        "stories": 1,
    }
    risk.update(risk_overrides)
    return {"applicant": {"full_name": "Jane Doe"}, "risk": risk,
            "coverage_request": {"coverage_a": 500000, "deductible": 1000}}


def test_no_provider_returns_fully_abstained_evidence():
    svc = VisionEvidenceService(config=VisionServiceConfig(), provider=None)
    ev = svc.extract_evidence(b"fake-image-bytes")
    assert isinstance(ev, VisionEvidence)
    assert ev.source == "stub"
    assert ev.image_sha256 == hashlib.sha256(b"fake-image-bytes").hexdigest()
    for key in VISION_ATTRIBUTES:
        attr = getattr(ev, key)
        assert attr.visible is False and attr.value is None


def test_provider_result_is_parsed_into_schema():
    payload = {
        "defensible_space_present": {"value": True, "confidence": 0.9, "visible": True},
        "roof_condition": {"value": "worn", "confidence": 0.7, "visible": True},
        "hazards": {"value": ["trampoline"], "confidence": 0.8, "visible": True},
    }
    svc = VisionEvidenceService(provider=_FakeProvider(payload))
    ev = svc.extract_evidence(b"img")
    assert ev.source == "llm"
    assert ev.model == "fake-vision-1"
    assert ev.defensible_space_present.value is True
    assert ev.roof_condition.value == "worn"
    # Unmentioned attributes default to abstained.
    assert ev.tarp_present.visible is False


def test_provider_error_degrades_to_abstained():
    class _Boom:
        model = "x"
        def extract(self, image_bytes, system_prompt):
            raise RuntimeError("vision API down")
    svc = VisionEvidenceService(provider=_Boom())
    ev = svc.extract_evidence(b"img")
    assert ev.source == "stub"
    assert ev.defensible_space_present.visible is False


def test_slow_provider_times_out_to_abstained():
    """A provider slower than the timeout degrades to abstained, fast."""
    import time

    class _SlowProvider:
        model = "slow"
        def extract(self, image_bytes, system_prompt):
            time.sleep(1.0)
            return {"defensible_space_present": {"value": True, "confidence": 0.9, "visible": True}}

    svc = VisionEvidenceService(
        config=VisionServiceConfig(timeout_s=0.2),
        provider=_SlowProvider(),
    )
    start = time.monotonic()
    ev = svc.extract_evidence(b"img")
    elapsed = time.monotonic() - start
    assert ev.source == "stub"               # degraded
    assert elapsed < 0.9                       # returned without waiting for the slow call


def test_confident_defensible_space_folds_into_wildfire_field():
    payload = {"defensible_space_present": {"value": True, "confidence": 0.92, "visible": True}}
    ev = VisionEvidenceService(provider=_FakeProvider(payload)).extract_evidence(b"img")
    updated, applied = fold_vision_into_submission(_submission(), ev, min_confidence=0.6)
    assert updated["risk"]["wildfire_mitigation_evidence"] is True
    assert applied["fields"]["risk.wildfire_mitigation_evidence"] is True
    assert applied["image_sha256"] == ev.image_sha256
    assert "photo" in updated["risk"]["mitigation_notes"].lower()


def test_low_confidence_does_not_fold_so_gate_can_ask():
    payload = {"defensible_space_present": {"value": True, "confidence": 0.4, "visible": True}}
    ev = VisionEvidenceService(provider=_FakeProvider(payload)).extract_evidence(b"img")
    updated, applied = fold_vision_into_submission(_submission(), ev, min_confidence=0.6)
    assert updated["risk"].get("wildfire_mitigation_evidence") is None
    assert applied["fields"] == {}


def test_not_visible_does_not_fold():
    payload = {"defensible_space_present": {"value": None, "confidence": 0.0, "visible": False}}
    ev = VisionEvidenceService(provider=_FakeProvider(payload)).extract_evidence(b"img")
    updated, _ = fold_vision_into_submission(_submission(), ev, min_confidence=0.6)
    assert updated["risk"].get("wildfire_mitigation_evidence") is None


def test_producer_provided_value_is_not_overwritten():
    payload = {"defensible_space_present": {"value": False, "confidence": 0.95, "visible": True}}
    ev = VisionEvidenceService(provider=_FakeProvider(payload)).extract_evidence(b"img")
    # Producer already asserted True; vision must not override it.
    updated, applied = fold_vision_into_submission(_submission(wildfire_mitigation_evidence=True), ev)
    assert updated["risk"]["wildfire_mitigation_evidence"] is True
    assert applied["fields"] == {}


def test_other_attributes_recorded_as_provenance_only():
    payload = {
        "roof_condition": {"value": "tarped", "confidence": 0.85, "visible": True},
        "hazards": {"value": ["pool"], "confidence": 0.9, "visible": True},
    }
    ev = VisionEvidenceService(provider=_FakeProvider(payload)).extract_evidence(b"img")
    updated, applied = fold_vision_into_submission(_submission(), ev)
    # Recorded, but not folded into any rule-consumed field.
    assert applied["observed"]["roof_condition"] == "tarped"
    assert "wildfire_mitigation_evidence" not in applied["fields"]
