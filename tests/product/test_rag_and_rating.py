import importlib.util
import os

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
# Tests that actually load a sentence-transformers/cross-encoder model hit the
# HuggingFace hub on first run, so they are NOT hermetic. Keep them opt-in
# (RUN_MODEL_TESTS=1) so the default suite — and CI — never reaches the network.
_RUN_MODEL_TESTS = _CROSS_ENCODER_AVAILABLE and os.getenv("RUN_MODEL_TESTS") == "1"
_MODEL_TEST_REASON = "model-download test; set RUN_MODEL_TESTS=1 (and install sentence-transformers) to run"


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


@pytest.mark.skipif(not _RUN_MODEL_TESTS, reason=_MODEL_TEST_REASON)
def test_sentence_transformer_embeddings_produce_semantic_results(tmp_path):
    """The real embedding provider (not the hashing default) drives semantic mode."""
    rag = RAGEngine(
        chroma_path=str(tmp_path / "chroma"),
        config=RAGRetrievalConfig(
            retrieval_mode=RETRIEVAL_MODE_SEMANTIC,
            embeddings_enabled=True,
            embedding_model="sentence-transformers:all-MiniLM-L6-v2",
        ),
    )
    summary = rag.ingest_documents()

    assert summary["effective_retrieval_mode"] == RETRIEVAL_MODE_SEMANTIC
    assert summary["embedding_model"] == "all-MiniLM-L6-v2"

    chunks = rag.retrieve("how old can a roof be before referral", n_results=3)
    assert chunks
    for chunk in chunks:
        assert chunk.relevance_score is not None
        assert chunk.metadata["retrieval_mode"] == RETRIEVAL_MODE_SEMANTIC
        assert chunk.metadata["embedding_model"] == "all-MiniLM-L6-v2"


def test_unknown_embedding_model_falls_back_to_hashing(tmp_path):
    """A non-ST, non-nebius model name uses deterministic hashing embeddings."""
    rag = RAGEngine(
        chroma_path=str(tmp_path / "chroma"),
        config=RAGRetrievalConfig(
            retrieval_mode=RETRIEVAL_MODE_SEMANTIC,
            embeddings_enabled=True,
            embedding_model="hashing-underwriting-v1",
        ),
    )
    rag.ingest_documents()
    assert rag.embeddings_available
    assert rag.embedding_provider.model_name == "hashing-underwriting-v1"


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


@pytest.mark.skipif(not _RUN_MODEL_TESTS, reason=_MODEL_TEST_REASON)
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
