"""Model comparison runner for InsuranceRag.

Runs the same producer-rationale task across all configured LLM providers
and outputs a cost/quality/latency comparison table.

Usage:
    # All providers with keys set:
    python scripts/compare_models.py

    # Specific providers only:
    python scripts/compare_models.py --providers openai claude

    # Custom dataset:
    python scripts/compare_models.py --dataset evals/datasets/ho3_labeled.jsonl --max-cases 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as `python scripts/compare_models.py` from the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Cost table (USD per 1k tokens) — update when pricing changes
# ---------------------------------------------------------------------------
_COST_PER_1K: Dict[str, Dict[str, float]] = {
    "openai/gpt-4o-mini":    {"input": 0.000150, "output": 0.000600},
    "openai/gpt-4o":         {"input": 0.002500, "output": 0.010000},
    "claude/claude-sonnet-4-6": {"input": 0.003000, "output": 0.015000},
    "claude/claude-haiku-4-5-20251001": {"input": 0.000250, "output": 0.001250},
    "gemini/gemini-1.5-flash": {"input": 0.000075, "output": 0.000300},
    "gemini/gemini-1.5-pro":   {"input": 0.001250, "output": 0.005000},
    "ollama/*":              {"input": 0.0, "output": 0.0},
    "nebius/*":              {"input": 0.000100, "output": 0.000400},
}

_PROVIDER_ENVS = {
    "openai":  "OPENAI_API_KEY",
    "claude":  "ANTHROPIC_API_KEY",
    "gemini":  "GOOGLE_API_KEY",
    "nebius":  "NEBIUS_API_KEY",
    "ollama":  None,  # no key required
}


@dataclass
class CaseResult:
    case_id: str
    provider: str
    model: str
    latency_ms: float
    json_valid: bool
    schema_valid: bool
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class ProviderSummary:
    provider: str
    model: str
    total_cases: int
    json_valid_pct: float
    schema_valid_pct: float
    avg_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    avg_cost_per_case_usd: float
    errors: int
    notes: str = ""


def _estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    key = f"{provider}/{model}"
    rates = _COST_PER_1K.get(key) or _COST_PER_1K.get(f"{provider}/*") or {"input": 0.0, "output": 0.0}
    return (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]


def _load_cases(path: Path, max_cases: int) -> List[Dict[str, Any]]:
    cases = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            # Skip cases that are expected to pause (waiting_for_info) — those
            # don't produce a rationale to evaluate
            if row.get("expected", {}).get("status") == "waiting_for_info":
                continue
            cases.append(row)
            if len(cases) >= max_cases:
                break
    return cases


def _build_fake_decision(case: Dict[str, Any]) -> Dict[str, Any]:
    """Build a plausible decision_data dict from the eval case for the rationale call."""
    expected = case.get("expected", {})
    decision = expected.get("decision", "ACCEPT")
    reason_codes = expected.get("reason_codes", [])
    return {
        "decision": decision,
        "preliminary_decision": decision,
        "confidence": 0.9 if not reason_codes else 0.82,
        "eligibility_score": 0.9 if decision == "ACCEPT" else 0.5,
        "risk_factors": [
            {"code": code, "severity": "medium", "because": f"Rule triggered: {code}"}
            for code in reason_codes
        ],
        "facts_used": {
            "year_built": case["submission"]["risk"].get("year_built"),
            "roof_age_years": case["submission"]["risk"].get("roof_age_years"),
            "occupancy": case["submission"]["risk"].get("occupancy"),
        },
        "reasoning": f"Deterministic rule evaluation produced {decision}.",
        "citations": [
            {"chunk_id": cid, "section": "guideline", "doc_version": "v1"}
            for cid in expected.get("gold_citations", [])
        ],
    }


def run_provider(
    provider_name: str,
    model: str,
    cases: List[Dict[str, Any]],
) -> List[CaseResult]:
    from app.llm_service import LLMServiceConfig, StructuredLLMService, LLMUnavailable
    from models.schemas import ProducerRationaleOutput

    # Build config for this provider
    api_key_env = _PROVIDER_ENVS.get(provider_name, "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env, "") if api_key_env else ""

    config = LLMServiceConfig(
        enabled=True,
        provider=provider_name,
        model=model,
        api_key=api_key,
        base_url=os.getenv("OLLAMA_BASE_URL") if provider_name == "ollama" else None,
    )
    try:
        service = StructuredLLMService(config=config)
    except Exception as exc:
        return [
            CaseResult(
                case_id=c["id"], provider=provider_name, model=model,
                latency_ms=0, json_valid=False, schema_valid=False, error=str(exc)
            )
            for c in cases
        ]

    results = []
    for case in cases:
        decision_data = _build_fake_decision(case)
        citations = decision_data.pop("citations", [])
        fallback_summary = f"Decision: {decision_data['decision']}"
        t0 = time.monotonic()
        error = None
        json_valid = False
        schema_valid = False
        output = None
        try:
            output = service.generate_producer_rationale(
                decision_data=decision_data,
                citations=citations,
                fallback_summary=fallback_summary,
            )
            schema_valid = isinstance(output, ProducerRationaleOutput) and output.source == "llm"
            # Only count as json_valid when the provider actually responded with JSON;
            # a silent fallback (source="fallback") means the provider was unavailable.
            json_valid = schema_valid
            if not json_valid:
                error = "provider_unavailable_or_fallback"
        except Exception as exc:
            error = str(exc)
        latency_ms = (time.monotonic() - t0) * 1000

        # Token counts not exposed by all providers; estimate from text length
        prompt_chars = len(fallback_summary) + 200
        output_chars = len(getattr(output, 'summary', '') or '') if output is not None else 0
        input_tokens = prompt_chars // 4
        output_tokens = max(output_chars // 4, 1)
        cost = _estimate_cost(provider_name, model, input_tokens, output_tokens)

        results.append(CaseResult(
            case_id=case["id"],
            provider=provider_name,
            model=model,
            latency_ms=latency_ms,
            json_valid=json_valid,
            schema_valid=schema_valid,
            error=error,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        ))
    return results


def summarize(results: List[CaseResult], provider: str, model: str) -> ProviderSummary:
    if not results:
        return ProviderSummary(provider=provider, model=model, total_cases=0,
                               json_valid_pct=0, schema_valid_pct=0, avg_latency_ms=0,
                               p95_latency_ms=0, total_cost_usd=0, avg_cost_per_case_usd=0,
                               errors=0)
    latencies = sorted(r.latency_ms for r in results)
    p95_idx = int(len(latencies) * 0.95)
    total_cost = sum(r.estimated_cost_usd for r in results)
    return ProviderSummary(
        provider=provider,
        model=model,
        total_cases=len(results),
        json_valid_pct=100 * sum(r.json_valid for r in results) / len(results),
        schema_valid_pct=100 * sum(r.schema_valid for r in results) / len(results),
        avg_latency_ms=sum(latencies) / len(latencies),
        p95_latency_ms=latencies[min(p95_idx, len(latencies) - 1)],
        total_cost_usd=total_cost,
        avg_cost_per_case_usd=total_cost / len(results),
        errors=sum(1 for r in results if r.error),
    )


def render_markdown(summaries: List[ProviderSummary]) -> str:
    lines = [
        "# Model Comparison — InsuranceRag (Producer Rationale Task)\n",
        "| Provider | Model | Cases | JSON Valid % | Schema Valid % | Avg Latency (ms) | p95 Latency (ms) | Est. Cost/Case ($) | Total Cost ($) | Errors |",
        "|----------|-------|-------|-------------|----------------|-----------------|------------------|--------------------|----------------|--------|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.provider} | {s.model} | {s.total_cases} "
            f"| {s.json_valid_pct:.1f}% | {s.schema_valid_pct:.1f}% "
            f"| {s.avg_latency_ms:.0f} | {s.p95_latency_ms:.0f} "
            f"| ${s.avg_cost_per_case_usd:.6f} | ${s.total_cost_usd:.4f} | {s.errors} |"
        )
    lines += [
        "",
        "## Notes",
        "- Cost estimates use public pricing; token counts are approximated from character length.",
        "- `Schema Valid %` measures whether the provider returned a valid `ProducerRationaleOutput` (source=llm).",
        "- Ollama cost is $0 (local inference).",
        "- Run with `--providers openai claude gemini ollama` to compare all four.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLM providers on the underwriting rationale task")
    parser.add_argument("--dataset", default="evals/datasets/ho3_labeled.jsonl")
    parser.add_argument("--max-cases", type=int, default=30, help="Max cases per provider")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["openai", "claude", "gemini", "ollama"],
        choices=["openai", "claude", "gemini", "ollama", "nebius"],
    )
    parser.add_argument("--output-dir", default="evals/reports")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Run: python evals/generate_dataset.py first")
        return

    cases = _load_cases(dataset_path, args.max_cases)
    print(f"Loaded {len(cases)} evaluation cases from {dataset_path}\n")

    # Default models per provider
    default_models = {
        "openai": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "claude": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
        "gemini": os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        "ollama": os.getenv("OLLAMA_MODEL", "llama3.2"),
        "nebius": os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
    }

    all_results: Dict[str, List[CaseResult]] = {}
    summaries: List[ProviderSummary] = []

    for provider in args.providers:
        env_var = _PROVIDER_ENVS.get(provider)
        if env_var and not os.getenv(env_var):
            print(f"[{provider}] Skipping — {env_var} not set")
            continue
        model = default_models[provider]
        print(f"[{provider}/{model}] Running {len(cases)} cases...")
        t0 = time.monotonic()
        results = run_provider(provider, model, cases)
        elapsed = time.monotonic() - t0
        all_results[provider] = results
        s = summarize(results, provider, model)
        summaries.append(s)
        print(
            f"  → JSON valid: {s.json_valid_pct:.1f}%  "
            f"Schema valid: {s.schema_valid_pct:.1f}%  "
            f"Avg latency: {s.avg_latency_ms:.0f}ms  "
            f"Total time: {elapsed:.1f}s  "
            f"Errors: {s.errors}"
        )

    if not summaries:
        print("\nNo providers ran. Set at least one API key and retry.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "model_comparison.md"
    md_path.write_text(render_markdown(summaries))
    print(f"\nMarkdown report: {md_path}")

    json_path = output_dir / "model_comparison.json"
    json_path.write_text(json.dumps(
        {
            "summaries": [vars(s) for s in summaries],
            "raw_results": {
                p: [vars(r) for r in rs]
                for p, rs in all_results.items()
            },
        },
        indent=2,
    ))
    print(f"JSON report:     {json_path}")


if __name__ == "__main__":
    main()
