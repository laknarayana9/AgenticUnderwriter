from app.rag_engine import RAGEngine
from tools import RatingTool


def test_rag_lexical_fallback_retrieves_citable_guidelines(tmp_path):
    rag = RAGEngine(chroma_path=str(tmp_path / "chroma"))
    rag.ingest_documents()

    chunks = rag.retrieve("roof age shall be referred wildfire high defensible space", n_results=3)

    assert chunks
    assert all(chunk.chunk_id for chunk in chunks)
    assert any("roof" in chunk.text.lower() or "wildfire" in chunk.text.lower() for chunk in chunks)


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
