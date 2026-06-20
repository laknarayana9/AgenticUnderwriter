from app.llm_service import LLMServiceConfig, StructuredLLMService
from workflows.agents import DecisionPackagerAgent


class FakeJSONProvider:
    provider_name = "fake"
    model = "fake-structured-model"

    def __init__(self, payload):
        self.payload = payload

    def generate_json(self, system_prompt, user_prompt, schema):
        return self.payload


def test_missing_info_wording_falls_back_when_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = StructuredLLMService(config=LLMServiceConfig.from_env())

    questions = [{
        "question_id": "roof_age_years",
        "field_path": "risk.roof_age_years",
        "answer_key": "risk.roof_age_years",
        "question": "What is the roof age in years?",
        "question_text": "What is the roof age in years?",
        "question_type": "numeric",
        "required": True,
    }]

    result = service.word_missing_info_questions(questions)

    assert result[0]["question_id"] == "roof_age_years"
    assert result[0]["question_text"] == "What is the roof age in years?"
    assert result[0]["wording_source"] == "fallback"


def test_missing_info_wording_uses_validated_provider_output():
    provider = FakeJSONProvider({
        "questions": [{
            "question_id": "wildfire_mitigation_evidence",
            "field_path": "risk.wildfire_mitigation_evidence",
            "answer_key": "risk.wildfire_mitigation_evidence",
            "question_text": "Has defensible-space or wildfire mitigation documentation been provided for this property?",
            "question_type": "boolean",
            "required": True,
            "options": None,
            "context": {},
        }]
    })
    service = StructuredLLMService(
        config=LLMServiceConfig(provider="disabled"),
        provider=provider,
    )

    result = service.word_missing_info_questions([{
        "question_id": "wildfire_mitigation_evidence",
        "field_path": "risk.wildfire_mitigation_evidence",
        "answer_key": "risk.wildfire_mitigation_evidence",
        "question": "Is defensible-space or wildfire mitigation evidence documented for this property?",
        "question_text": "Is defensible-space or wildfire mitigation evidence documented for this property?",
        "question_type": "boolean",
        "required": True,
    }])

    assert result[0]["question_text"].startswith("Has defensible-space")
    assert result[0]["wording_source"] == "llm"


def test_provider_cannot_change_question_contract():
    provider = FakeJSONProvider({
        "questions": [{
            "question_id": "roof_age_years",
            "field_path": "risk.occupancy",
            "question_text": "What is the occupancy?",
            "question_type": "text",
            "required": True,
        }]
    })
    service = StructuredLLMService(
        config=LLMServiceConfig(provider="disabled"),
        provider=provider,
    )

    result = service.word_missing_info_questions([{
        "question_id": "roof_age_years",
        "field_path": "risk.roof_age_years",
        "question": "What is the roof age in years?",
        "question_text": "What is the roof age in years?",
        "question_type": "numeric",
        "required": True,
    }])

    assert result[0]["field_path"] == "risk.roof_age_years"
    assert result[0]["question_text"] == "What is the roof age in years?"
    assert result[0]["wording_source"] == "fallback"


def test_decision_packager_validates_producer_rationale_without_changing_decision():
    provider = FakeJSONProvider({
        "summary": "This risk is referred because the governed rules found an older roof and elevated wildfire exposure.",
        "supporting_facts": ["roof_age_years: 20", "wildfire_band: High"],
        "citation_chunk_ids": ["uw-1"],
    })
    service = StructuredLLMService(
        config=LLMServiceConfig(provider="disabled"),
        provider=provider,
    )
    packager = DecisionPackagerAgent(llm_service=service)

    packet = packager.package(
        {
            "decision": "REFER",
            "confidence": 0.84,
            "reasoning": "REFER based on underwriting triggers: roof age and wildfire risk.",
            "risk_factors": [{"code": "ROOF_AGE", "because": "Roof age requires review."}],
            "facts_used": {"roof_age_years": 20, "wildfire_band": "High"},
            "citations": [{"chunk_id": "uw-1", "doc_id": "uw", "section": "roof"}],
        },
        {"annual_premium": 1250, "currency": "USD"},
        [],
    )

    assert packet.decision.value == "REFER"
    assert packet.reason_summary.startswith("This risk is referred")
    assert packet.producer_rationale.source == "llm"
    assert "ROOF_AGE" in packet.review_reason_codes


# ---------------------------------------------------------------------------
# mask_map wiring: PII must not reach the LLM prompt
# ---------------------------------------------------------------------------

class CapturingProvider:
    """Records every user_prompt it receives so tests can inspect them."""

    provider_name = "capturing"
    model = "capturing-model"

    def __init__(self, payload):
        self.payload = payload
        self.calls: list = []

    def generate_json(self, system_prompt, user_prompt, schema):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.payload


def test_pii_masked_before_rationale_llm_call():
    """mask_map must scrub name and address out of the user_prompt sent to the LLM."""
    provider = CapturingProvider({
        "summary": "Property accepted under HO3 guidelines.",
        "supporting_facts": ["roof_age_years: 5"],
        "citation_chunk_ids": [],
    })
    service = StructuredLLMService(
        config=LLMServiceConfig(provider="disabled"),
        provider=provider,
    )

    mask_map = {
        "[APPLICANT_NAME]": "Jane Smith",
        "[PROPERTY_ADDRESS]": "742 Evergreen Terrace, Springfield, IL 62701",
    }
    decision_data = {
        "decision": "ACCEPT",
        "confidence": 0.95,
        "risk_factors": [],
        "facts_used": {
            "applicant_name": "Jane Smith",
            "property_address": "742 Evergreen Terrace, Springfield, IL 62701",
            "roof_age_years": 5,
        },
        "reasoning": "Clean risk for Jane Smith at 742 Evergreen Terrace, Springfield, IL 62701.",
    }

    service.generate_producer_rationale(
        decision_data=decision_data,
        citations=[],
        fallback_summary="Accepted.",
        mask_map=mask_map,
    )

    assert provider.calls, "expected at least one LLM call"
    prompt = provider.calls[0]["user_prompt"]
    assert "Jane Smith" not in prompt
    assert "742 Evergreen Terrace" not in prompt
    assert "[APPLICANT_NAME]" in prompt or "[PROPERTY_ADDRESS]" in prompt


def test_no_mask_map_still_calls_llm():
    """Omitting mask_map must not break the call — just no scrubbing."""
    provider = CapturingProvider({
        "summary": "Referred due to elevated wildfire exposure.",
        "supporting_facts": ["wildfire_band: High"],
        "citation_chunk_ids": [],
    })
    service = StructuredLLMService(
        config=LLMServiceConfig(provider="disabled"),
        provider=provider,
    )

    result = service.generate_producer_rationale(
        decision_data={"decision": "REFER", "confidence": 0.7, "risk_factors": [], "facts_used": {}},
        citations=[],
        fallback_summary="Referred.",
    )

    assert result.source == "llm"
    assert provider.calls


def test_packager_threads_mask_map_to_llm_service():
    """DecisionPackagerAgent.package(mask_map=...) must forward the map to generate_producer_rationale."""
    provider = CapturingProvider({
        "summary": "REFER — elevated hazard.",
        "supporting_facts": ["wildfire_band: Severe"],
        "citation_chunk_ids": [],
    })
    service = StructuredLLMService(
        config=LLMServiceConfig(provider="disabled"),
        provider=provider,
    )
    packager = DecisionPackagerAgent(llm_service=service)

    mask_map = {"[APPLICANT_NAME]": "Bob Tester", "[PROPERTY_ADDRESS]": "1 Fire Lane, CA 95000"}
    decision_data = {
        "decision": "REFER",
        "confidence": 0.75,
        "reasoning": "Bob Tester at 1 Fire Lane, CA 95000 is in a severe wildfire zone.",
        "risk_factors": [{"code": "WILDFIRE_HIGH", "because": "Severe zone."}],
        "facts_used": {"wildfire_band": "Severe"},
        "citations": [],
    }

    packager.package(decision_data, {"annual_premium": 0}, [], mask_map=mask_map)

    assert provider.calls
    prompt = provider.calls[0]["user_prompt"]
    assert "Bob Tester" not in prompt
    assert "1 Fire Lane" not in prompt


# ---------------------------------------------------------------------------
# compare_models.py: fallback must not count as JSON valid
# ---------------------------------------------------------------------------

def test_fallback_output_not_counted_as_json_valid(monkeypatch):
    """When the provider is unavailable and service returns source='fallback',
    run_provider must mark json_valid=False and populate error."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from scripts.compare_models import run_provider

    cases = [
        {
            "id": "test-fallback-case",
            "submission": {
                "applicant": {"full_name": "Test User"},
                "risk": {
                    "property_address": "1 Main St, CA",
                    "occupancy": "owner_occupied_primary",
                    "dwelling_type": "single_family",
                    "year_built": 2010,
                    "roof_age_years": 5,
                    "construction_type": "frame",
                    "stories": 1,
                },
                "coverage_request": {"coverage_a": 400000},
            },
            "expected": {"decision": "ACCEPT", "reason_codes": [], "gold_citations": []},
        }
    ]

    # Use a provider that has no API key set so StructuredLLMService returns fallback
    results = run_provider("openai", "gpt-4o-mini", cases)

    assert len(results) == 1
    r = results[0]
    # When the provider is unavailable the service silently returns source="fallback";
    # that must be reported as not-valid so the comparison table is honest.
    assert not r.json_valid, "fallback output must not be counted as json_valid"
    assert not r.schema_valid
    assert r.error is not None
