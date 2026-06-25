"""LangSmith tracing, dataset upload, and evaluation for the HO3 underwriting workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as `python evals/langsmith_eval.py` from the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langsmith import Client, traceable

from evals.run import load_dataset
from workflows.agent_workflow import UnderwritingWorkflow


DATASET_NAME = "ho3-golden-206"
DATASET_PATH = _REPO_ROOT / "evals" / "datasets" / "ho3_labeled.jsonl"


@traceable(name="underwriting_workflow")
def run_workflow(submission: dict) -> dict:
    """Run the underwriting workflow and return a serialisable result dict.

    LangSmith traces this automatically when LANGSMITH_TRACING=true.
    When tracing is off the decorator is a no-op.
    """
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        workflow = UnderwritingWorkflow()
    state = workflow.run(submission)
    packet = state.decision_packet
    return {
        "decision": packet.decision.value if packet else None,
        "reason_codes": packet.review_reason_codes if packet else [],
        "citations": [
            c.get("chunk_id")
            for c in (packet.citations or [])
            if isinstance(c, dict) and c.get("chunk_id")
        ],
        "status": state.status,
        "retrieved_chunks": (state.retrieval or {}).get("retrieved_chunks", []),
    }


def upload_dataset(
    dataset_path: Path,
    dataset_name: str,
    client: Client,
) -> str:
    """Upload the golden eval set to LangSmith as a versioned dataset.

    Idempotent: returns the existing dataset URL if the dataset already exists.
    Returns the dataset URL.
    """
    for ds in client.list_datasets():
        if ds.name == dataset_name:
            print(f"Dataset '{dataset_name}' already exists — skipping creation.")
            return ds.url

    cases = load_dataset(dataset_path)
    ds = client.create_dataset(dataset_name=dataset_name, description="HO3 golden eval set — 206 labeled cases")

    inputs = [{"submission": case.submission} for case in cases]
    outputs = [
        {
            "decision": case.expected.decision,
            "reason_codes": case.expected.reason_codes,
            "gold_citations": case.expected.gold_citations,
        }
        for case in cases
    ]
    client.create_examples(inputs=inputs, outputs=outputs, dataset_id=ds.id)
    print(f"Uploaded {len(cases)} examples to '{dataset_name}'.")
    return ds.url


# ---------------------------------------------------------------------------
# Evaluators — each takes (outputs, reference_outputs) and returns a score dict
# ---------------------------------------------------------------------------

def eval_decision_accuracy(outputs: Dict[str, Any], reference_outputs: Dict[str, Any]) -> Dict[str, Any]:
    predicted = (outputs.get("decision") or "").upper()
    expected = (reference_outputs.get("decision") or "").upper()
    return {"key": "decision_accuracy", "score": 1.0 if predicted == expected else 0.0}


def eval_retrieval_recall(outputs: Dict[str, Any], reference_outputs: Dict[str, Any]) -> Dict[str, Any]:
    gold = set(reference_outputs.get("gold_citations") or [])
    if not gold:
        return {"key": "retrieval_recall@5", "score": 1.0}
    actual = set((outputs.get("citations") or [])[:5])
    score = round(len(gold & actual) / len(gold), 4)
    return {"key": "retrieval_recall@5", "score": score}


def eval_faithfulness(outputs: Dict[str, Any], reference_outputs: Dict[str, Any]) -> Dict[str, Any]:
    citations = outputs.get("citations") or []
    if not citations:
        return {"key": "faithfulness", "score": 1.0}
    retrieved_ids = {
        chunk.get("chunk_id")
        for chunk in (outputs.get("retrieved_chunks") or [])
        if isinstance(chunk, dict) and chunk.get("chunk_id")
    }
    grounded = sum(1 for cid in citations if cid in retrieved_ids)
    score = round(grounded / len(citations), 4)
    return {"key": "faithfulness", "score": score}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _require_api_key() -> str:
    key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not key:
        print("error: LANGSMITH_API_KEY is not set. Export it before running.", file=sys.stderr)
        sys.exit(1)
    return key


def cmd_upload_dataset(args: argparse.Namespace) -> None:
    _require_api_key()
    client = Client()
    url = upload_dataset(DATASET_PATH, DATASET_NAME, client)
    print(f"Dataset URL: {url}")


def cmd_run_eval(args: argparse.Namespace) -> None:
    _require_api_key()
    client = Client()
    results = client.evaluate(
        run_workflow,
        data=DATASET_NAME,
        evaluators=[eval_decision_accuracy, eval_retrieval_recall, eval_faithfulness],
        experiment_prefix=args.experiment_prefix,
        max_concurrency=1,
    )
    print(f"Experiment URL: {results.experiment_name}")
    print("Done.")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="LangSmith dataset upload and evaluation for HO3 underwriting.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("upload-dataset", help="Upload ho3_labeled.jsonl as a LangSmith dataset.")

    run_parser = sub.add_parser("run-eval", help="Run client.evaluate against the golden dataset.")
    run_parser.add_argument(
        "--experiment-prefix",
        default="run",
        help="Prefix for the LangSmith experiment name (e.g. 'baseline', 'improved').",
    )

    args = parser.parse_args(argv)
    if args.command == "upload-dataset":
        cmd_upload_dataset(args)
    elif args.command == "run-eval":
        cmd_run_eval(args)


if __name__ == "__main__":
    main()
