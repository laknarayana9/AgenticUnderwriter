"""Phase 2/3 tests: vision wired into the live workflow, the API endpoint, the
provider build path, and the vision eval metrics. Hermetic — a fake vision
provider drives extraction; no network, no real model, no image files."""

import json

from fastapi.testclient import TestClient

from app.vision_service import VisionEvidenceService, VisionServiceConfig
from evals.vision_eval import score_predictions
from models.schemas import VisionAttribute, VisionEvidence
from workflows.agent_workflow import UnderwritingWorkflow


class _FakeVisionProvider:
    model = "fake-vision-1"

    def __init__(self, payload):
        self._payload = payload

    def extract(self, image_bytes, system_prompt):
        return self._payload


# High-wildfire address, mitigation status unknown — the contrast hinges on vision.
_SUBMISSION = {
    "applicant": {"full_name": "Avery Chen"},
    "risk": {
        "property_address": "742 Fire Zone Rd, Sacramento, CA 95818",
        "occupancy": "owner_occupied_primary", "dwelling_type": "single_family",
        "year_built": 2008, "roof_age_years": 7, "construction_type": "frame", "stories": 1,
    },
    "coverage_request": {"coverage_a": 500000, "deductible": 1000},
}


def _workflow_with_vision(payload):
    wf = UnderwritingWorkflow()
    wf.vision_service.provider = _FakeVisionProvider(payload)
    return wf


def test_vision_confirming_defensible_space_lets_run_proceed():
    wf = _workflow_with_vision({"defensible_space_present": {"value": True, "confidence": 0.93, "visible": True}})
    state = wf.run(_SUBMISSION, image_bytes=b"img")
    assert state.submission_raw["risk"]["wildfire_mitigation_evidence"] is True
    assert state.status != "waiting_for_info"
    assert any(e["event"] == "vision_evidence_applied" for e in state.events)
    assert "vision_intake" in state.stage_timings


def test_vision_abstention_makes_workflow_pause():
    wf = _workflow_with_vision({"defensible_space_present": {"value": None, "confidence": 0.0, "visible": False}})
    state = wf.run(_SUBMISSION, image_bytes=b"img")
    assert state.submission_raw["risk"].get("wildfire_mitigation_evidence") is None
    assert state.status == "waiting_for_info"


def test_no_image_skips_vision_stage():
    state = UnderwritingWorkflow().run(_SUBMISSION)  # no image
    assert "vision_intake" not in state.stage_timings
    assert not any(e["event"] == "vision_evidence_applied" for e in state.events)


def test_with_photo_endpoint_runs_and_records_vision_event():
    from app.main import app, workflow
    # Inject a confirming fake provider into the app's workflow vision service.
    workflow.vision_service.provider = _FakeVisionProvider(
        {"defensible_space_present": {"value": True, "confidence": 0.95, "visible": True}}
    )
    client = TestClient(app)
    resp = client.post(
        "/quote/ho3/with-photo",
        data={"submission": json.dumps(_SUBMISSION)},
        files={"photo": ("roof.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    audit = client.get(f"/runs/{run_id}/audit").json()
    events = [e["event"] for e in audit["workflow_state"]["events"]]
    assert "vision_evidence_applied" in events
    workflow.vision_service.provider = None  # reset


def test_provider_build_returns_none_without_key():
    # enabled but no key -> graceful None (abstained), no crash.
    svc = VisionEvidenceService(config=VisionServiceConfig(enabled=True, provider="openai", api_key=""))
    assert svc.provider is None


def test_ollama_vision_config_and_build(monkeypatch):
    monkeypatch.setenv("VISION_ENABLED", "true")
    monkeypatch.setenv("VISION_PROVIDER", "ollama")
    monkeypatch.delenv("VISION_MODEL", raising=False)
    cfg = VisionServiceConfig.from_env()
    assert cfg.provider == "ollama"
    assert cfg.model == "llama3.2-vision"
    assert cfg.base_url
    from app.providers.ollama_vision_provider import OllamaVisionProvider
    svc = VisionEvidenceService(config=cfg)
    assert isinstance(svc.provider, OllamaVisionProvider)


def test_ollama_vision_extract_parses_response(monkeypatch):
    import urllib.request
    from app.providers.ollama_vision_provider import OllamaVisionProvider

    class _Resp:
        def __init__(self, data): self._d = data
        def read(self): return self._d
        def __enter__(self): return self
        def __exit__(self, *a): return False

    content = json.dumps({"defensible_space_present": {"value": True, "confidence": 0.9, "visible": True}})
    body = json.dumps({"message": {"content": content}}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(body))

    out = OllamaVisionProvider(model="llama3.2-vision").extract(b"img", "sys")
    assert out["defensible_space_present"]["value"] is True


def test_ollama_vision_failure_degrades_to_abstained(monkeypatch):
    import urllib.request
    from app.providers.ollama_vision_provider import OllamaVisionProvider

    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    svc = VisionEvidenceService(provider=OllamaVisionProvider(model="x"))
    ev = svc.extract_evidence(b"img")
    assert ev.source == "stub"  # any provider error degrades to abstained


def test_vision_eval_metrics():
    pred = VisionEvidence(
        image_sha256="x", model="m", source="llm",
        defensible_space_present=VisionAttribute(value=True, confidence=0.9, visible=True),
        roof_condition=VisionAttribute(value="worn", confidence=0.8, visible=True),
        tarp_present=VisionAttribute(value=None, confidence=0.0, visible=False),
    )
    gold = {"defensible_space_present": True, "roof_condition": "good", "tarp_present": None}
    report = score_predictions([pred], [gold])
    # 2 known attrs (defensible ✓, roof_condition ✗) -> 0.5; 1 abstain attr, correctly abstained -> 1.0
    assert report["attribute_accuracy"] == 0.5
    assert report["abstention_correctness"] == 1.0
