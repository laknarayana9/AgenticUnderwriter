"""Temperature variance study for local inference (Ollama).

Runs the same prompts repeatedly at two temperatures (0.0 and 0.7 by default) and
measures how much the outputs vary. This demonstrates the practical point that
production paths use temperature 0 for reproducibility, and quantifies what you
give up at higher temperatures.

Metrics per temperature (averaged over prompts):
  - consistency : fraction of runs equal to the most common output for a prompt
                  (1.0 = perfectly repeatable; lower = more variance)
  - distinct    : unique outputs / runs (1/runs = identical every time)
  - json_valid  : fraction of outputs that parse as JSON

Usage:
    python scripts/temperature_study.py --runs 5
    python scripts/temperature_study.py --temperatures 0 0.7 1.2 --runs 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmark_inference import (  # noqa: E402
    DEFAULT_PROMPTS,
    OllamaBenchmarkClient,
    _is_json,
)


def study_temperature(client: OllamaBenchmarkClient, model: str, prompts: List[str],
                      temperature: float, runs: int) -> Dict[str, float]:
    consistencies: List[float] = []
    distinct_rates: List[float] = []
    json_flags: List[bool] = []

    client.generate(model, "warmup", temperature)
    for prompt in prompts:
        outputs: List[str] = []
        for _ in range(runs):
            try:
                outputs.append(_capture_text(client, model, prompt, temperature))
            except Exception:
                continue
        if not outputs:
            continue
        counts = Counter(outputs)
        modal_freq = counts.most_common(1)[0][1]
        consistencies.append(modal_freq / len(outputs))
        distinct_rates.append(len(counts) / len(outputs))
        json_flags.extend(_is_json(o) for o in outputs)

    return {
        "temperature": temperature,
        "consistency": round(statistics.mean(consistencies), 3) if consistencies else 0.0,
        "distinct_rate": round(statistics.mean(distinct_rates), 3) if distinct_rates else 0.0,
        "json_valid_pct": round(100 * sum(json_flags) / len(json_flags), 1) if json_flags else 0.0,
        "prompts": len(consistencies),
        "runs_per_prompt": runs,
    }


def _capture_text(client: OllamaBenchmarkClient, model: str, prompt: str, temperature: float) -> str:
    """Collect the raw text output for a single generation (for variance scoring)."""
    payload = {"model": model, "prompt": prompt, "stream": True, "options": {"temperature": temperature}}
    parts: List[str] = []
    with client._httpx.stream("POST", f"{client.base_url}/api/generate", json=payload, timeout=120.0) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            parts.append(chunk.get("response", ""))
    return "".join(parts).strip()


def render_markdown(model: str, rows: List[Dict[str, float]], runs: int) -> str:
    lines = [
        f"# Temperature Variance Study — {model}\n",
        f"Same prompts, {runs} runs each, per temperature.\n",
        "| Temperature | Consistency | Distinct rate | JSON valid % |",
        "|-------------|-------------|---------------|--------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['temperature']} | {r['consistency']} | {r['distinct_rate']} | {r['json_valid_pct']}% |"
        )
    lines += [
        "",
        "## Interpretation",
        "- **consistency** = fraction of runs matching the most common output "
        "(1.0 = perfectly repeatable).",
        "- **distinct rate** = unique outputs / runs (1/runs = identical every time).",
        "- Temperature 0 should be near-deterministic (consistency ~1.0); higher",
        "  temperatures trade repeatability for diversity. The governed workflow",
        "  uses temperature 0 so the same submission yields the same rationale.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure output variance vs temperature (Ollama).")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.7])
    parser.add_argument("--runs", type=int, default=5, help="Runs per prompt per temperature.")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--output-dir", default="evals/reports")
    args = parser.parse_args()

    client = OllamaBenchmarkClient(args.base_url)
    if not client.available():
        print(f"Ollama not reachable at {args.base_url}. Start it with `ollama serve`.")
        return

    print(f"Temperature study: {args.model}, temps {args.temperatures}, "
          f"{len(DEFAULT_PROMPTS)} prompts x {args.runs} runs\n")

    rows: List[Dict[str, float]] = []
    for temp in args.temperatures:
        print(f"[temp={temp}] running...")
        row = study_temperature(client, args.model, DEFAULT_PROMPTS, temp, args.runs)
        rows.append(row)
        print(f"  → consistency {row['consistency']}  distinct {row['distinct_rate']}  "
              f"JSON {row['json_valid_pct']}%")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "temperature_study.md").write_text(render_markdown(args.model, rows, args.runs))
    (output_dir / "temperature_study.json").write_text(json.dumps(rows, indent=2))
    print(f"\nReports written to {output_dir}/temperature_study.[md|json]")


if __name__ == "__main__":
    main()
