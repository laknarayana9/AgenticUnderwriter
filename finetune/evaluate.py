"""Evaluate base vs fine-tuned extraction on the holdout set (Nebius Token Factory).

Runs each holdout note through one or both models via the OpenAI-compatible chat
API and reports JSON validity, exact match, field accuracy, and refusal
correctness — the before/after table for the fine-tune write-up. Requires
NEBIUS_API_KEY (inference spends a small amount of credit).

    # Compare the base model and the fine-tuned model:
    NEBIUS_API_KEY=... python -m finetune.evaluate \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --tuned-model 'ft:meta-llama/Llama-3.1-8B-Instruct-...'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune.metrics import score_predictions  # noqa: E402
from finetune.schema import SYSTEM_PROMPT  # noqa: E402

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"


def load_holdout(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Return (user_notes, gold_targets) from a chat-format JSONL."""
    notes: List[str] = []
    golds: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row["messages"]
            user = next(m["content"] for m in messages if m["role"] == "user")
            gold = json.loads(messages[-1]["content"])
            notes.append(user)
            golds.append(gold)
    return notes, golds


def run_model(client: Any, model: str, notes: List[str]) -> List[str]:
    outputs: List[str] = []
    for note in notes:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": note},
            ],
            temperature=0,
        )
        outputs.append(resp.choices[0].message.content or "")
    return outputs


def _row(name: str, scores: Dict[str, Any]) -> str:
    rc = scores["refusal_correctness"]
    return (
        f"| {name} | {scores['n']} | {scores['json_valid_rate']:.3f} "
        f"| {scores['exact_match_rate']:.3f} | {scores['field_accuracy']:.3f} "
        f"| {rc if rc is not None else '—'} |"
    )


def render_markdown(results: List[Tuple[str, Dict[str, Any]]]) -> str:
    lines = [
        "# Fine-Tune Evaluation — HO3 Extraction\n",
        "| Model | N | JSON valid | Exact match | Field accuracy | Refusal correctness |",
        "|-------|---|-----------|-------------|----------------|---------------------|",
    ]
    lines += [_row(name, scores) for name, scores in results]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base vs fine-tuned extraction.")
    parser.add_argument("--holdout-file", default=str(Path(__file__).resolve().parent / "data" / "holdout.jsonl"))
    parser.add_argument("--base-model", default=None, help="Base model id to evaluate (optional).")
    parser.add_argument("--tuned-model", default=None, help="Fine-tuned model id to evaluate (optional).")
    parser.add_argument("--max-cases", type=int, default=100)
    parser.add_argument("--output-dir", default="evals/reports")
    args = parser.parse_args()

    if not args.base_model and not args.tuned_model:
        print("Provide --base-model and/or --tuned-model.")
        sys.exit(1)

    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("NEBIUS_API_KEY is not set.")
        sys.exit(1)

    notes, golds = load_holdout(Path(args.holdout_file))
    notes, golds = notes[: args.max_cases], golds[: args.max_cases]
    print(f"Evaluating on {len(notes)} holdout cases\n")

    from openai import OpenAI

    client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=api_key)

    results: List[Tuple[str, Dict[str, Any]]] = []
    for label, model in [("base", args.base_model), ("fine-tuned", args.tuned_model)]:
        if not model:
            continue
        print(f"[{label}] {model} ...")
        preds = run_model(client, model, notes)
        scores = score_predictions(preds, golds)
        results.append((f"{label} ({model})", scores))
        print(f"  → JSON {scores['json_valid_rate']:.3f}  exact {scores['exact_match_rate']:.3f} "
              f"  field_acc {scores['field_accuracy']:.3f}  refusal {scores['refusal_correctness']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "finetune_eval.md").write_text(render_markdown(results))
    (output_dir / "finetune_eval.json").write_text(
        json.dumps([{"model": name, **scores} for name, scores in results], indent=2)
    )
    print(f"\nReports written to {output_dir}/finetune_eval.[md|json]")


if __name__ == "__main__":
    main()
