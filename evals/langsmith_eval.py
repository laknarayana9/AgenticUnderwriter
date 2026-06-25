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
# Evaluator functions (used by client.evaluate in Task 3)
# ---------------------------------------------------------------------------

def eval_decision_accuracy(run: Any, example: Any) -> dict:
    """Return 1.0 if the predicted decision matches the gold decision."""
    predicted = (run.outputs or {}).get("decision")
    expected = (example.outputs or {}).get("decision")
    score = 1.0 if predicted == expected else 0.0
    return {"key": "decision_accuracy", "score": score}


def eval_faithfulness(run: Any, example: Any) -> dict:
    """Return fraction of reason codes supported by retrieved citations."""
    reason_codes = (run.outputs or {}).get("reason_codes", [])
    citations = (run.outputs or {}).get("citations", [])
    if not reason_codes:
        return {"key": "faithfulness", "score": 1.0}
    supported = sum(1 for rc in reason_codes if rc in citations)
    score = supported / len(reason_codes)
    return {"key": "faithfulness", "score": score}


def eval_retrieval_recall(run: Any, example: Any) -> dict:
    """Return fraction of gold citations that appear in retrieved chunks."""
    gold_citations = (example.outputs or {}).get("gold_citations", [])
    retrieved = (run.outputs or {}).get("retrieved_chunks", [])
    if not gold_citations:
        return {"key": "retrieval_recall", "score": 1.0}
    found = sum(1 for gc in gold_citations if gc in retrieved)
    score = found / len(gold_citations)
    return {"key": "retrieval_recall", "score": score}
