"""Nebius Token Factory provider boundary tests (no network calls)."""

from app.llm_service import (
    NEBIUS_DEFAULT_BASE_URL,
    LLMServiceConfig,
    NebiusJSONProvider,
    StructuredLLMService,
)
from app.rag_engine import NebiusEmbeddingProvider, RAGEngine, RAGRetrievalConfig


def test_config_routes_nebius_provider_and_base_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nebius")
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    config = LLMServiceConfig.from_env()

    assert config.provider == "nebius"
    assert config.api_key == "test-key"
    assert config.base_url == NEBIUS_DEFAULT_BASE_URL
    assert config.model  # provider-appropriate default applied


def test_nebius_provider_requires_api_key():
    service = StructuredLLMService(
        config=LLMServiceConfig(enabled=True, provider="nebius", api_key=None),
    )
    # Missing key -> provider unavailable -> deterministic fallback wording.
    assert service.provider is None

    result = service.word_missing_info_questions([{
        "question_id": "roof_age_years",
        "field_path": "risk.roof_age_years",
        "question": "What is the roof age in years?",
        "question_text": "What is the roof age in years?",
        "question_type": "numeric",
        "required": True,
    }])
    assert result[0]["wording_source"] == "fallback"


def test_nebius_json_provider_targets_nebius_base_url(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import app.llm_service as llm_module
    monkeypatch.setitem(__import__("sys").modules, "openai", type("M", (), {"OpenAI": FakeOpenAI}))

    provider = NebiusJSONProvider(api_key="k", model="meta-llama/Llama-3.3-70B-Instruct")

    assert provider.provider_name == "nebius"
    assert captured["base_url"] == NEBIUS_DEFAULT_BASE_URL
    assert captured["api_key"] == "k"


def test_nebius_embedding_falls_back_to_hashing_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    rag = RAGEngine(
        chroma_path=str(tmp_path / "chroma"),
        config=RAGRetrievalConfig(
            retrieval_mode="semantic",
            embeddings_enabled=True,
            embedding_model="nebius:BAAI/bge-en-icl",
        ),
    )

    # Provider is unavailable offline, so we fall back to deterministic hashing
    # embeddings rather than dropping straight to lexical.
    assert rag.embeddings_available
    assert rag.embedding_provider.model_name == "hashing-underwriting-v1"


def test_nebius_embedding_provider_raises_without_key(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        NebiusEmbeddingProvider("BAAI/bge-en-icl")
    except RuntimeError as exc:
        assert "NEBIUS_API_KEY" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected RuntimeError when API key is missing")
