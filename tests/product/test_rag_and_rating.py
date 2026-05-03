from app.rag_engine import (
    RAGEngine,
    RAGRetrievalConfig,
    RETRIEVAL_MODE_LEXICAL,
    RETRIEVAL_MODE_SEMANTIC,
)
from app.rating import RatingTool


def test_rag_lexical_fallback_retrieves_citable_guidelines(tmp_path):
    rag = RAGEngine(chroma_path=str(tmp_path / "chroma"))
    rag.ingest_documents()

    chunks = rag.retrieve("roof age shall be referred wildfire high defensible space", n_results=3)

    assert chunks
    assert all(chunk.chunk_id for chunk in chunks)
    assert any("roof" in chunk.text.lower() or "wildfire" in chunk.text.lower() for chunk in chunks)


def test_semantic_mode_falls_back_to_lexical_when_embeddings_disabled(tmp_path):
    rag = RAGEngine(
        chroma_path=str(tmp_path / "chroma"),
        config=RAGRetrievalConfig(
            retrieval_mode=RETRIEVAL_MODE_SEMANTIC,
            embeddings_enabled=False,
        ),
    )
    summary = rag.ingest_documents()

    chunks = rag.retrieve("high wildfire risk roof age referral", n_results=3)

    assert summary["configured_retrieval_mode"] == RETRIEVAL_MODE_SEMANTIC
    assert summary["effective_retrieval_mode"] == RETRIEVAL_MODE_LEXICAL
    assert chunks
    assert all(chunk.metadata["retrieval_mode"] == RETRIEVAL_MODE_LEXICAL for chunk in chunks)


def test_semantic_mode_returns_structured_citation_results(tmp_path):
    rag = RAGEngine(
        chroma_path=str(tmp_path / "chroma"),
        config=RAGRetrievalConfig(
            retrieval_mode=RETRIEVAL_MODE_SEMANTIC,
            embeddings_enabled=True,
            embedding_model="hashing-underwriting-v1",
        ),
    )
    summary = rag.ingest_documents()

    chunks = rag.retrieve("high wildfire risk roof age referral", n_results=3)

    assert summary["effective_retrieval_mode"] == RETRIEVAL_MODE_SEMANTIC
    assert summary["embedding_model"] == "hashing-underwriting-v1"
    assert chunks
    for chunk in chunks:
        assert chunk.doc_id
        assert chunk.doc_version
        assert chunk.section
        assert chunk.chunk_id
        assert chunk.text
        assert chunk.relevance_score is not None
        assert chunk.metadata["retrieval_mode"] == RETRIEVAL_MODE_SEMANTIC
        assert chunk.metadata["embedding_model"] == "hashing-underwriting-v1"


def test_rag_config_reads_environment(monkeypatch):
    monkeypatch.setenv("RAG_RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("RAG_EMBEDDINGS_ENABLED", "true")
    monkeypatch.setenv("EMBEDDING_MODEL", "hashing-underwriting-v1")

    config = RAGRetrievalConfig.from_env()

    assert config.retrieval_mode == "hybrid"
    assert config.embeddings_enabled is True
    assert config.embedding_model == "hashing-underwriting-v1"


def test_rating_tool_returns_transparent_sane_premium():
    premium = RatingTool().calculate_premium(
        500000,
        {
            "territory": "HighRiskCounty",
            "construction_year": 1985,
            "hazard_scores": {
                "wildfire_risk": 0.78,
                "flood_risk": 0.2,
            },
        },
    )

    assert premium["annual_premium"] > 0
    assert premium["annual_premium"] < 10000
    assert premium["base_premium"] == 1000
    assert premium["factors"]["territory"] == 1.1
    assert premium["factors"]["hazard"] > 1
