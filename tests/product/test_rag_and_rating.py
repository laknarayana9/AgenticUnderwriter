import importlib.util

import pytest

from app.rag_engine import (
    RAGEngine,
    RAGRetrievalConfig,
    RETRIEVAL_MODE_BM25,
    RETRIEVAL_MODE_HYBRID,
    RETRIEVAL_MODE_LEXICAL,
    RETRIEVAL_MODE_SEMANTIC,
)
from app.rating import RatingTool

_BM25_AVAILABLE = importlib.util.find_spec("rank_bm25") is not None
_CROSS_ENCODER_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None


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
    monkeypatch.setenv("RAG_HYBRID_ALPHA", "0.7")
    monkeypatch.setenv("RAG_RERANK_ENABLED", "true")
    monkeypatch.setenv("RAG_RERANK_TOP_N", "15")

    config = RAGRetrievalConfig.from_env()

    assert config.retrieval_mode == "hybrid"
    assert config.embeddings_enabled is True
    assert config.embedding_model == "hashing-underwriting-v1"
    assert config.hybrid_alpha == 0.7
    assert config.rerank_enabled is True
    assert config.rerank_top_n == 15


def test_hybrid_alpha_is_clamped_to_unit_interval(monkeypatch):
    monkeypatch.setenv("RAG_HYBRID_ALPHA", "5.0")
    assert RAGRetrievalConfig.from_env().hybrid_alpha == 1.0
    monkeypatch.setenv("RAG_HYBRID_ALPHA", "-2")
    assert RAGRetrievalConfig.from_env().hybrid_alpha == 0.0


@pytest.mark.skipif(not _BM25_AVAILABLE, reason="rank_bm25 not installed")
def test_bm25_mode_retrieves_citable_guidelines_without_embeddings(tmp_path):
    rag = RAGEngine(
        chroma_path=str(tmp_path / "chroma"),
        config=RAGRetrievalConfig(
            retrieval_mode=RETRIEVAL_MODE_BM25,
            embeddings_enabled=False,
        ),
    )
    rag.ingest_documents()

    chunks = rag.retrieve("roof age shall be referred wildfire high defensible space", n_results=3)

    assert chunks
    assert all(chunk.chunk_id for chunk in chunks)
    assert all(chunk.metadata["retrieval_mode"] == RETRIEVAL_MODE_BM25 for chunk in chunks)
    assert any("roof" in chunk.text.lower() or "wildfire" in chunk.text.lower() for chunk in chunks)


def test_hybrid_rrf_merges_bm25_and_semantic(tmp_path):
    rag = RAGEngine(
        chroma_path=str(tmp_path / "chroma"),
        config=RAGRetrievalConfig(
            retrieval_mode=RETRIEVAL_MODE_HYBRID,
            embeddings_enabled=True,
            embedding_model="hashing-underwriting-v1",
        ),
    )
    summary = rag.ingest_documents()

    chunks = rag.retrieve("high wildfire risk roof age referral", n_results=3)

    assert summary["effective_retrieval_mode"] == RETRIEVAL_MODE_HYBRID
    assert chunks
    for chunk in chunks:
        assert chunk.relevance_score is not None
        assert chunk.metadata["retrieval_mode"] == RETRIEVAL_MODE_HYBRID
        # When rank_bm25 is present, hybrid fuses via RRF; otherwise it falls back
        # through bm25 -> lexical but still returns citable hybrid-tagged chunks.
        if _BM25_AVAILABLE:
            assert chunk.metadata.get("fusion") == "rrf"


def test_rerank_disabled_by_default_keeps_base_order(tmp_path):
    rag = RAGEngine(chroma_path=str(tmp_path / "chroma"))
    rag.ingest_documents()
    assert rag.rerank_active is False
    chunks = rag.retrieve("roof age wildfire referral", n_results=3)
    assert chunks
    assert all(not chunk.metadata.get("reranked") for chunk in chunks)


def test_rerank_unavailable_falls_back_gracefully(tmp_path, monkeypatch):
    """rerank_enabled but the model/package missing must not break retrieval."""
    rag = RAGEngine(
        chroma_path=str(tmp_path / "chroma"),
        config=RAGRetrievalConfig(
            retrieval_mode=RETRIEVAL_MODE_BM25,
            rerank_enabled=True,
        ),
    )
    # Force the reranker to be unavailable regardless of environment.
    rag._reranker = None
    rag.ingest_documents()

    chunks = rag.retrieve("roof age wildfire referral", n_results=3)

    assert chunks
    assert rag.rerank_active is False
    assert all(not chunk.metadata.get("reranked") for chunk in chunks)


@pytest.mark.skipif(not _CROSS_ENCODER_AVAILABLE, reason="sentence-transformers not installed")
def test_reranker_reorders_and_annotates_when_available(tmp_path):
    rag = RAGEngine(
        chroma_path=str(tmp_path / "chroma"),
        config=RAGRetrievalConfig(
            retrieval_mode=RETRIEVAL_MODE_BM25,
            rerank_enabled=True,
            rerank_top_n=10,
        ),
    )
    rag.ingest_documents()
    if not rag.rerank_active:
        pytest.skip("cross-encoder model could not be loaded in this environment")

    chunks = rag.retrieve("when must a roof be referred for its age", n_results=3)

    assert chunks
    for chunk in chunks:
        assert chunk.metadata.get("reranked") is True
        assert "pre_rerank_rank" in chunk.metadata
        assert "rerank_score" in chunk.metadata


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
