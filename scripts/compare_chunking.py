#!/usr/bin/env python3
"""
Compare chunking strategies (header-based vs fixed-size) on the underwriting
corpus, across all three retrieval modes (lexical, semantic, hybrid).

For each (strategy, mode) pair we run a fixed probe set and measure whether the
gold phrase for each probe shows up in the top-k retrieved chunks
(hit@k) and how highly it ranks (mean reciprocal rank). This is the chunking
comparison report the project's evaluation deliverable calls for, and the
lexical-vs-hybrid columns double as the reranking-impact view.

Example:
    python scripts/compare_chunking.py
    python scripts/compare_chunking.py --k 5 --out docs/chunking_comparison.md
"""

from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag_engine import (  # noqa: E402
    CHUNK_STRATEGY_FIXED,
    CHUNK_STRATEGY_HEADER,
    RETRIEVAL_MODE_HYBRID,
    RETRIEVAL_MODE_LEXICAL,
    RETRIEVAL_MODE_SEMANTIC,
    RAGEngine,
    RAGRetrievalConfig,
)

STRATEGIES = [CHUNK_STRATEGY_HEADER, CHUNK_STRATEGY_FIXED]
MODES = [RETRIEVAL_MODE_LEXICAL, RETRIEVAL_MODE_SEMANTIC, RETRIEVAL_MODE_HYBRID]


@dataclass(frozen=True)
class Probe:
    """A query plus a gold phrase that should appear in a correctly retrieved chunk."""
    query: str
    gold_phrase: str


# Probes target distinct underwriting rules so a "hit" means the strategy
# surfaced the chunk that actually answers the query.
PROBES: List[Probe] = [
    Probe("what roof age triggers a referral", "roof"),
    Probe("high wildfire hazard band requirements", "wildfire"),
    Probe("defensible space mitigation evidence", "defensible"),
    Probe("flood zone special flood hazard area eligibility", "flood"),
    Probe("tenant occupied rental property eligibility", "occup"),
    Probe("old construction electrical and plumbing standards", "electrical"),
    Probe("ineligible dwelling types", "eligible"),
    Probe("accept decision outcome low risk", "accept"),
]


@dataclass
class StrategyModeResult:
    strategy: str
    mode: str
    effective_mode: str
    chunk_count: int
    mean_chunk_chars: int
    hit_at_k: float
    mrr: float


def evaluate(strategy: str, mode: str, k: int) -> StrategyModeResult:
    config = RAGRetrievalConfig(
        retrieval_mode=mode,
        embeddings_enabled=mode in {RETRIEVAL_MODE_SEMANTIC, RETRIEVAL_MODE_HYBRID},
        embedding_model="hashing-underwriting-v1",
        chunk_strategy=strategy,
    )
    rag = RAGEngine(config=config)
    with redirect_stdout(StringIO()):
        summary = rag.ingest_documents()

    lengths = [len(chunk.text) for chunk in rag.chunks] or [0]
    hits = 0
    reciprocal_ranks = 0.0
    for probe in PROBES:
        chunks = rag.retrieve(probe.query, n_results=k, mode=mode)
        rank = _first_hit_rank(chunks, probe.gold_phrase)
        if rank is not None:
            hits += 1
            reciprocal_ranks += 1.0 / rank

    n = len(PROBES)
    return StrategyModeResult(
        strategy=strategy,
        mode=mode,
        effective_mode=summary["effective_retrieval_mode"],
        chunk_count=summary["total_chunks"],
        mean_chunk_chars=sum(lengths) // len(lengths),
        hit_at_k=hits / n,
        mrr=reciprocal_ranks / n,
    )


def _first_hit_rank(chunks, gold_phrase: str) -> Optional[int]:
    needle = gold_phrase.lower()
    for rank, chunk in enumerate(chunks, start=1):
        if needle in chunk.text.lower():
            return rank
    return None


def render_report(results: List[StrategyModeResult], k: int) -> str:
    lines: List[str] = []
    lines.append("# Chunking Strategy Comparison")
    lines.append("")
    lines.append(
        f"Corpus retrieval quality for header-based vs fixed-size chunking, "
        f"measured over {len(PROBES)} probe queries at k={k}."
    )
    lines.append("")
    lines.append("| Strategy | Retrieval mode | Chunks | Mean chars | Hit@k | MRR |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for r in results:
        mode_label = r.mode if r.mode == r.effective_mode else f"{r.mode}→{r.effective_mode}"
        lines.append(
            f"| {r.strategy} | {mode_label} | {r.chunk_count} | {r.mean_chunk_chars} "
            f"| {r.hit_at_k:.3f} | {r.mrr:.3f} |"
        )
    lines.append("")
    lines.extend(_render_takeaways(results))
    lines.append("")
    return "\n".join(lines)


def _render_takeaways(results: List[StrategyModeResult]) -> List[str]:
    by_strategy: Dict[str, List[StrategyModeResult]] = {}
    for r in results:
        by_strategy.setdefault(r.strategy, []).append(r)

    lines = ["## Takeaways", ""]
    for strategy, rows in by_strategy.items():
        best = max(rows, key=lambda r: (r.hit_at_k, r.mrr))
        lines.append(
            f"- **{strategy}**: best hit@k={best.hit_at_k:.3f} (mrr={best.mrr:.3f}) "
            f"in {best.mode} mode; {rows[0].chunk_count} chunks @ ~{rows[0].mean_chunk_chars} chars each."
        )

    # Reranking view: hybrid vs lexical for each strategy.
    for strategy, rows in by_strategy.items():
        modes = {r.mode: r for r in rows}
        if RETRIEVAL_MODE_LEXICAL in modes and RETRIEVAL_MODE_HYBRID in modes:
            delta = modes[RETRIEVAL_MODE_HYBRID].mrr - modes[RETRIEVAL_MODE_LEXICAL].mrr
            lines.append(
                f"- **{strategy}** reranking impact (hybrid vs lexical MRR): {delta:+.3f}."
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare header vs fixed-size chunking.")
    parser.add_argument("--k", type=int, default=5, help="Top-k cutoff for hit@k / MRR.")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write the markdown report.")
    args = parser.parse_args()

    results = [
        evaluate(strategy, mode, args.k)
        for strategy in STRATEGIES
        for mode in MODES
    ]
    report = render_report(results, args.k)
    print(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()
