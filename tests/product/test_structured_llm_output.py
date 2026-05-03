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
