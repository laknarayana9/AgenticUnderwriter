"""Local inference benchmark for small language models (Ollama).

Measures the metrics that actually characterize local inference — the ones a
hiring manager wants to see reasoned about:

  - TTFT   : time to first token (responsiveness)
  - tok/s  : decode throughput, from Ollama's authoritative eval_count/eval_duration
  - total  : end-to-end wall latency
  - memory : resident memory of the Ollama runner process (psutil)
  - json%  : fraction of outputs that parse as valid JSON (a cheap quality proxy)

Why Ollama natively: its streaming API returns exact server-side token counts and
durations, so tok/s and TTFT are measured, not estimated from character counts.

Usage:
    # Single model (defaults to llama3.2:3b):
    python scripts/benchmark_inference.py

    # Quantization comparison (Project 2, phase 3) — pull the tags first:
    #   ollama pull llama3.2:3b-instruct-q4_K_M
    #   ollama pull llama3.2:3b-instruct-q5_K_M
    #   ollama pull llama3.2:3b-instruct-q8_0
    python scripts/benchmark_inference.py \
        --models llama3.2:3b-instruct-q4_K_M llama3.2:3b-instruct-q5_K_M llama3.2:3b-instruct-q8_0

    # Custom temperature and repetitions:
    python scripts/benchmark_inference.py --temperature 0 --runs 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_OLLAMA_URL = "http://localhost:11434"

# A standardized, domain-flavored prompt set. The instruction asks for JSON so
# the json-valid rate is a meaningful (if minimal) quality signal.
DEFAULT_PROMPTS: List[str] = [
    'Extract the roof age in years from this note and reply as JSON {"roof_age_years": <int>}: "Roof was redone about 12 years ago."',
    'Classify occupancy as JSON {"occupancy": "owner_occupied_primary|tenant_occupied|vacant"}: "The owners live there full time."',
    'Reply as JSON {"eligible": true|false} — Is a commercial warehouse eligible for an HO3 homeowners policy?',
    'Summarize as JSON {"summary": "<one sentence>"}: "High wildfire band with no defensible-space evidence; refer to underwriting."',
    'Reply as JSON {"deductible": <int>} given the note: "Standard one thousand dollar deductible applies."',
    'Extract the year built as JSON {"year_built": <int>}: "Built in nineteen eighty five."',
    'Reply as JSON {"flood_zone": true|false}: "Property sits inside a Special Flood Hazard Area."',
    'Classify severity as JSON {"severity": "low|medium|high"}: "Roof is 25 years old, above the referral threshold."',
]


@dataclass
class RunSample:
    ttft_ms: float
    total_ms: float
    tokens_per_sec: float
    prompt_tokens: int
    output_tokens: int
    json_valid: bool
    error: Optional[str] = None


@dataclass
class ModelBenchmark:
    model: str
    temperature: float
    samples: List[RunSample] = field(default_factory=list)
    memory_mb: Optional[float] = None

    def _ok(self) -> List[RunSample]:
        return [s for s in self.samples if s.error is None]

    def summary(self) -> Dict[str, float]:
        ok = self._ok()
        if not ok:
            return {"runs": 0, "errors": len(self.samples)}

        def pct(values: List[float], p: float) -> float:
            ordered = sorted(values)
            idx = min(int(len(ordered) * p), len(ordered) - 1)
            return ordered[idx]

        ttft = [s.ttft_ms for s in ok]
        total = [s.total_ms for s in ok]
        tps = [s.tokens_per_sec for s in ok if s.tokens_per_sec > 0]
        return {
            "runs": len(ok),
            "errors": len(self.samples) - len(ok),
            "ttft_p50_ms": round(statistics.median(ttft), 1),
            "ttft_p95_ms": round(pct(ttft, 0.95), 1),
            "total_p50_ms": round(statistics.median(total), 1),
            "total_p95_ms": round(pct(total, 0.95), 1),
            "tokens_per_sec": round(statistics.mean(tps), 1) if tps else 0.0,
            "json_valid_pct": round(100 * sum(s.json_valid for s in ok) / len(ok), 1),
            "memory_mb": round(self.memory_mb, 1) if self.memory_mb else None,
        }


class OllamaBenchmarkClient:
    """Streaming client over Ollama's native /api/generate for exact timing."""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL):
        import httpx  # already a project dependency
        self._httpx = httpx
        self.base_url = base_url.rstrip("/")

    def available(self) -> bool:
        try:
            resp = self._httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def model_memory_mb(self, model: str) -> Optional[float]:
        """Loaded-model footprint as reported by Ollama's /api/ps (RAM, or VRAM if
        the model is GPU-resident). This is the runner's weights, not the harness."""
        try:
            resp = self._httpx.get(f"{self.base_url}/api/ps", timeout=2.0)
            resp.raise_for_status()
            for entry in resp.json().get("models", []):
                if entry.get("name") == model or entry.get("model") == model:
                    size = entry.get("size_vram") or entry.get("size") or 0
                    return size / (1024 * 1024) if size else None
        except Exception:
            return None
        return None

    def generate(self, model: str, prompt: str, temperature: float) -> RunSample:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        text_parts: List[str] = []
        ttft_ms = 0.0
        prompt_tokens = output_tokens = 0
        eval_duration_ns = 0
        start = time.perf_counter()
        try:
            with self._httpx.stream("POST", f"{self.base_url}/api/generate", json=payload, timeout=120.0) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token and ttft_ms == 0.0:
                        ttft_ms = (time.perf_counter() - start) * 1000
                    text_parts.append(token)
                    if chunk.get("done"):
                        prompt_tokens = chunk.get("prompt_eval_count", 0)
                        output_tokens = chunk.get("eval_count", 0)
                        eval_duration_ns = chunk.get("eval_duration", 0)
        except Exception as exc:  # network / model errors: record, don't crash the run
            return RunSample(0.0, 0.0, 0.0, 0, 0, False, error=str(exc))

        total_ms = (time.perf_counter() - start) * 1000
        # Authoritative decode throughput from server-side counters.
        tps = (output_tokens / (eval_duration_ns / 1e9)) if eval_duration_ns else 0.0
        text = "".join(text_parts).strip()
        return RunSample(
            ttft_ms=round(ttft_ms, 1),
            total_ms=round(total_ms, 1),
            tokens_per_sec=round(tps, 1),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            json_valid=_is_json(text),
        )


def _is_json(text: str) -> bool:
    """Whether the model output contains a parseable JSON object."""
    candidate = text.strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        return False
    try:
        json.loads(candidate[start : end + 1])
        return True
    except json.JSONDecodeError:
        return False


def _ollama_runner_memory_mb() -> Optional[float]:
    """Fallback memory metric: resident set of the Ollama runner process(es) via
    psutil, matching the model-server runner ("llama-server") as well as the
    "ollama" server. Used only when /api/ps does not report a size."""
    try:
        import psutil
    except ImportError:
        return None
    total = 0
    found = False
    for proc in psutil.process_iter(["name", "memory_info"]):
        name = (proc.info.get("name") or "").lower()
        if "llama-server" in name or "ollama" in name:
            mem = proc.info.get("memory_info")
            if mem:
                total += mem.rss
                found = True
    return (total / (1024 * 1024)) if found else None


def benchmark_model(client: OllamaBenchmarkClient, model: str, prompts: List[str],
                    temperature: float, runs: int) -> ModelBenchmark:
    bench = ModelBenchmark(model=model, temperature=temperature)
    # Warm up so the first prompt doesn't pay model-load time in the measured set.
    client.generate(model, "warmup", temperature)
    for prompt in prompts:
        for _ in range(runs):
            bench.samples.append(client.generate(model, prompt, temperature))
    # Prefer Ollama's reported loaded-model size; fall back to the runner RSS.
    bench.memory_mb = client.model_memory_mb(model) or _ollama_runner_memory_mb()
    return bench


def render_markdown(benchmarks: List[ModelBenchmark]) -> str:
    lines = [
        "# Local Inference Benchmark (Ollama)\n",
        "| Model | Temp | Runs | TTFT p50 (ms) | TTFT p95 (ms) | Total p50 (ms) | tok/s | JSON valid % | Mem (MB) | Errors |",
        "|-------|------|------|---------------|---------------|----------------|-------|--------------|----------|--------|",
    ]
    for b in benchmarks:
        s = b.summary()
        if not s.get("runs"):
            lines.append(f"| {b.model} | {b.temperature} | 0 | — | — | — | — | — | — | {s.get('errors', 0)} |")
            continue
        lines.append(
            f"| {b.model} | {b.temperature} | {s['runs']} "
            f"| {s['ttft_p50_ms']} | {s['ttft_p95_ms']} | {s['total_p50_ms']} "
            f"| {s['tokens_per_sec']} | {s['json_valid_pct']}% "
            f"| {s['memory_mb'] if s['memory_mb'] is not None else '—'} | {s['errors']} |"
        )
    lines += [
        "",
        "## Notes",
        "- TTFT and tok/s are measured from Ollama's streaming API; tok/s uses the",
        "  server-side `eval_count / eval_duration`, not character estimates.",
        "- Memory is the loaded-model footprint from Ollama /api/ps (RAM, or VRAM",
        "  if GPU-resident); falls back to the runner process RSS via psutil.",
        "- JSON valid % is a minimal quality proxy: does the output parse as JSON.",
        "- For a quantization study, pass several quant tags to `--models`.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local SLM inference (Ollama).")
    parser.add_argument("--models", nargs="+", default=["llama3.2:3b"], help="Ollama model tags to benchmark.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=1, help="Repetitions per prompt.")
    parser.add_argument("--prompts-file", help="Optional newline-delimited prompts file.")
    parser.add_argument("--base-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--output-dir", default="evals/reports")
    args = parser.parse_args()

    client = OllamaBenchmarkClient(args.base_url)
    if not client.available():
        print(f"Ollama not reachable at {args.base_url}. Start it with `ollama serve`.")
        return

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [p for p in Path(args.prompts_file).read_text().splitlines() if p.strip()]

    print(f"Benchmarking {len(args.models)} model(s) on {len(prompts)} prompts "
          f"x {args.runs} run(s) at temperature {args.temperature}\n")

    benchmarks: List[ModelBenchmark] = []
    for model in args.models:
        print(f"[{model}] running...")
        bench = benchmark_model(client, model, prompts, args.temperature, args.runs)
        benchmarks.append(bench)
        s = bench.summary()
        if s.get("runs"):
            print(f"  → TTFT p50 {s['ttft_p50_ms']}ms  tok/s {s['tokens_per_sec']}  "
                  f"JSON {s['json_valid_pct']}%  mem {s['memory_mb']}MB  errors {s['errors']}")
        else:
            print(f"  → no successful runs (errors {s.get('errors', 0)}) — is the tag pulled?")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inference_benchmark.md").write_text(render_markdown(benchmarks))
    (output_dir / "inference_benchmark.json").write_text(
        json.dumps([{"model": b.model, "temperature": b.temperature, **b.summary()} for b in benchmarks], indent=2)
    )
    print(f"\nReports written to {output_dir}/inference_benchmark.[md|json]")


if __name__ == "__main__":
    main()
