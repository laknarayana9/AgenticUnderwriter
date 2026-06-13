from app.rag_engine import (
    CHUNK_STRATEGY_FIXED,
    CHUNK_STRATEGY_HEADER,
    RAGEngine,
    RAGRetrievalConfig,
)


def _ingest(strategy: str, tmp_path) -> RAGEngine:
    rag = RAGEngine(
        chroma_path=str(tmp_path / "chroma"),
        config=RAGRetrievalConfig(chunk_strategy=strategy),
    )
    rag.ingest_documents()
    return rag


def test_chunk_strategy_reads_environment(monkeypatch):
    monkeypatch.setenv("RAG_CHUNK_STRATEGY", "fixed")
    assert RAGRetrievalConfig.from_env().chunk_strategy == CHUNK_STRATEGY_FIXED

    monkeypatch.setenv("RAG_CHUNK_STRATEGY", "not-a-strategy")
    assert RAGRetrievalConfig.from_env().chunk_strategy == CHUNK_STRATEGY_HEADER


def test_fixed_and_header_strategies_produce_distinct_chunking(tmp_path):
    header_rag = _ingest(CHUNK_STRATEGY_HEADER, tmp_path / "h")
    fixed_rag = _ingest(CHUNK_STRATEGY_FIXED, tmp_path / "f")

    header_lengths = [len(c.text) for c in header_rag.chunks]
    fixed_lengths = [len(c.text) for c in fixed_rag.chunks]

    assert header_rag.chunks and fixed_rag.chunks
    # Fixed windows are larger and fewer than header subsections.
    assert sum(fixed_lengths) / len(fixed_lengths) > sum(header_lengths) / len(header_lengths)
    # Fixed-window chunk ids carry the strategy marker.
    assert all("fixed_window" in c.metadata.get("subsection", "") or "fixed_window" in c.chunk_id
               for c in fixed_rag.chunks)


def test_fixed_strategy_still_retrieves_citable_chunks(tmp_path):
    fixed_rag = _ingest(CHUNK_STRATEGY_FIXED, tmp_path)
    chunks = fixed_rag.retrieve("roof age wildfire defensible space referral", n_results=3)

    assert chunks
    assert all(chunk.chunk_id and chunk.text for chunk in chunks)
