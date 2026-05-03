#!/usr/bin/env python3
"""
Compare lexical, semantic, and hybrid RAG retrieval for a query.

Example:
    python scripts/compare_retrieval.py --query "high wildfire risk roof age referral"
"""

from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from textwrap import shorten
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag_engine import (  # noqa: E402
    RAGEngine,
    RAGRetrievalConfig,
    RETRIEVAL_MODE_HYBRID,
    RETRIEVAL_MODE_LEXICAL,
    RETRIEVAL_MODE_SEMANTIC,
)
from models.schemas import RetrievalChunk  # noqa: E402


def main() -> None:
    args = parse_args()
    config = RAGRetrievalConfig(
        retrieval_mode=RETRIEVAL_MODE_HYBRID,
        embeddings_enabled=not args.no_embeddings,
        embedding_model=args.embedding_model,
    )
    rag = RAGEngine(config=config)
    with redirect_stdout(StringIO()):
        summary = rag.ingest_documents()

    print("RAG retrieval comparison")
    print("=" * 80)
    print(f"query: {args.query}")
    print(f"configured_mode: {summary['configured_retrieval_mode']}")
    print(f"effective_mode: {summary['effective_retrieval_mode']}")
    print(f"embedding_model: {summary['embedding_model'] or 'none'}")

    comparisons = rag.compare_retrieval(args.query, n_results=args.limit)
    for mode in [RETRIEVAL_MODE_LEXICAL, RETRIEVAL_MODE_SEMANTIC, RETRIEVAL_MODE_HYBRID]:
        print_results(mode, comparisons[mode])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare lexical, semantic, and hybrid retrieval.")
    parser.add_argument("--query", required=True, help="Retrieval query to compare.")
    parser.add_argument("--limit", type=int, default=5, help="Number of results per retrieval mode.")
    parser.add_argument(
        "--embedding-model",
        default="hashing-underwriting-v1",
        help="Embedding model/provider name. Use hashing-underwriting-v1 for built-in deterministic embeddings.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable embeddings to show semantic/hybrid fallback to lexical retrieval.",
    )
    return parser.parse_args()


def print_results(mode: str, chunks: Iterable[RetrievalChunk]) -> None:
    print(f"\n{mode.upper()} RESULTS")
    print("-" * 80)
    for idx, chunk in enumerate(chunks, start=1):
        score = chunk.relevance_score if chunk.relevance_score is not None else 0.0
        source = f"{chunk.doc_id}:{chunk.section}"
        snippet = shorten(" ".join(chunk.text.split()), width=110, placeholder=" ...")
        print(f"{idx}. score={score:.3f} source={source}")
        print(f"   chunk={chunk.chunk_id}")
        print(f"   {snippet}")


if __name__ == "__main__":
    main()
