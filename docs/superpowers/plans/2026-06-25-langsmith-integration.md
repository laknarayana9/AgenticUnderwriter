# LangSmith Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LangSmith as an opt-in tracing and evaluation layer — dataset upload, three `client.evaluate` evaluators, and `@traceable` workflow instrumentation, all disabled by default.

**Architecture:** A single new module `evals/langsmith_eval.py` contains the `@traceable` workflow wrapper, a dataset upload command, and an evaluation runner. The existing `evals/run.py` CI harness is untouched. Tracing is activated by the standard `LANGSMITH_TRACING=true` env var; when unset or false, the SDK is a no-op.

**Tech Stack:** `langsmith` SDK (PyPI), `UnderwritingWorkflow` from `workflows/agent_workflow.py`, `EvalCase`/`load_dataset` from `evals/run.py`.

## Global Constraints

- Do not modify `evals/run.py` or `observability.py`
- `langsmith` dependency goes only in `requirements-demo.txt`, not in any core requirements file
- All LangSmith env vars default to disabled/empty in `.env.example`
- Dataset name in LangSmith: `"ho3-golden-206"` (exact string — used as key in upload and eval commands)
- `max_concurrency=1` in `client.evaluate` to respect free-tier rate limits
- Python path: scripts run from repo root as `python evals/langsmith_eval.py <command>`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `evals/langsmith_eval.py` | All LangSmith concerns: traceable wrapper, dataset upload, eval runner, CLI |
| Modify | `requirements-demo.txt` | Add `langsmith` dependency |
| Modify | `.env.example` | Add LangSmith env var stubs |

---

## Task 1: Add `langsmith` dependency and env var stubs

**Files:**
- Modify: `requirements-demo.txt`
- Modify: `.env.example`

**Interfaces:**
- Produces: `langsmith` package available to import; env var names documented

- [ ] **Step 1: Add langsmith to requirements-demo.txt**

Open `requirements-demo.txt`. It currently contains:
```
# Optional interactive portfolio demo
streamlit>=1.37.0
```

Change it to:
```
# Optional interactive portfolio demo
streamlit>=1.37.0
# LangSmith tracing and evaluation (optional, commercial)
langsmith>=0.1.0
```

- [ ] **Step 2: Add env var stubs to .env.example**

Open `.env.example`. Append the following block at the end of the file (after the Modal section):

```
# LangSmith (optional — off by default; set LANGSMITH_TRACING=true to enable)
LANGSMITH_TRACING=false
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=AgenticUnderwriter
```

- [ ] **Step 3: Install the package**

```bash
pip install langsmith>=0.1.0
```

Expected: langsmith installs successfully with no conflicts.

- [ ] **Step 4: Verify import**

```bash
python -c "import langsmith; print(langsmith.__version__)"
```

Expected: prints a version string like `0.1.x`.

- [ ] **Step 5: Commit**

```bash
git add requirements-demo.txt .env.example
git commit -m "chore: add langsmith dependency and env var stubs"
```

---

## Task 2: Create `evals/langsmith_eval.py` — traceable wrapper and dataset upload

**Files:**
- Create: `evals/langsmith_eval.py`

**Interfaces:**
- Consumes: `load_dataset` and `EvalCase` from `evals/run.py`; `UnderwritingWorkflow` from `workflows/agent_workflow.py`
- Produces:
  - `run_workflow(submission: dict) -> dict` — `@traceable` wrapper, returns `{decision, reason_codes, citations, status, retrieved_chunks}`
  - `upload_dataset(dataset_path: Path, dataset_name: str, client: Client) -> str` — uploads golden set, returns dataset URL

- [ ] **Step 1: Write the failing test for dataset upload**

Create `tests/product/test_langsmith_eval.py`:

```python
"""Tests for LangSmith eval module — dataset upload and evaluator logic."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Guard: skip entire module if langsmith not installed
pytest.importorskip("langsmith")

from evals.langsmith_eval import (
    eval_decision_accuracy,
    eval_faithfulness,
    eval_retrieval_recall,
    upload_dataset,
)


DATASET_PATH = Path("evals/datasets/ho3_labeled.jsonl")


def _make_client(existing_datasets=None):
    """Return a mock langsmith Client."""
    client = MagicMock()
    existing = existing_datasets or []
    client.list_datasets.return_value = iter(existing)
    created = MagicMock()
    created.url = "https://smith.langchain.com/datasets/test-id"
    client.create_dataset.return_value = created
    return client


def test_upload_dataset_creates_dataset_and_returns_url():
    client = _make_client(existing_datasets=[])
    url = upload_dataset(DATASET_PATH, "ho3-golden-206", client)

    assert client.create_dataset.called
    call_kwargs = client.create_dataset.call_args
    assert "ho3-golden-206" in str(call_kwargs)
    assert client.create_examples.called
    assert url == "https://smith.langchain.com/datasets/test-id"


def test_upload_dataset_skips_creation_when_already_exists():
    existing = MagicMock()
    existing.name = "ho3-golden-206"
    existing.url = "https://smith.langchain.com/datasets/existing-id"
    client = _make_client(existing_datasets=[existing])

    url = upload_dataset(DATASET_PATH, "ho3-golden-206", client)

    client.create_dataset.assert_not_called()
    assert url == "https://smith.langchain.com/datasets/existing-id"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/product/test_langsmith_eval.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` for `evals.langsmith_eval`.

- [ ] **Step 3: Create `evals/langsmith_eval.py` with the traceable wrapper and upload function**

```python
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
```

- [ ] **Step 4: Run the upload test to verify it passes**

```bash
pytest tests/product/test_langsmith_eval.py::test_upload_dataset_creates_dataset_and_returns_url tests/product/test_langsmith_eval.py::test_upload_dataset_skips_creation_when_already_exists -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/langsmith_eval.py tests/product/test_langsmith_eval.py
git commit -m "feat(langsmith): add traceable workflow wrapper and dataset upload"
```

---

## Task 3: Add three evaluators and wire up the CLI

**Files:**
- Modify: `evals/langsmith_eval.py`

**Interfaces:**
- Consumes: `run_workflow` and `upload_dataset` from Task 2
- Produces:
  - `eval_decision_accuracy(outputs: dict, reference_outputs: dict) -> dict`
  - `eval_retrieval_recall(outputs: dict, reference_outputs: dict) -> dict`
  - `eval_faithfulness(outputs: dict, reference_outputs: dict) -> dict`
  - `main(argv)` — CLI entry point

- [ ] **Step 1: Add evaluator tests to `tests/product/test_langsmith_eval.py`**

Append these tests to the existing file:

```python
# ---------------------------------------------------------------------------
# Evaluator unit tests
# ---------------------------------------------------------------------------

def test_eval_decision_accuracy_match():
    result = eval_decision_accuracy(
        outputs={"decision": "ACCEPT"},
        reference_outputs={"decision": "ACCEPT"},
    )
    assert result == {"key": "decision_accuracy", "score": 1.0}


def test_eval_decision_accuracy_mismatch():
    result = eval_decision_accuracy(
        outputs={"decision": "REFER"},
        reference_outputs={"decision": "ACCEPT"},
    )
    assert result == {"key": "decision_accuracy", "score": 0.0}


def test_eval_retrieval_recall_full():
    result = eval_retrieval_recall(
        outputs={"citations": ["chunk_a", "chunk_b"]},
        reference_outputs={"gold_citations": ["chunk_a", "chunk_b"]},
    )
    assert result == {"key": "retrieval_recall@5", "score": 1.0}


def test_eval_retrieval_recall_partial():
    result = eval_retrieval_recall(
        outputs={"citations": ["chunk_a", "chunk_x"]},
        reference_outputs={"gold_citations": ["chunk_a", "chunk_b"]},
    )
    assert result == {"key": "retrieval_recall@5", "score": 0.5}


def test_eval_retrieval_recall_no_gold():
    result = eval_retrieval_recall(
        outputs={"citations": ["chunk_a"]},
        reference_outputs={"gold_citations": []},
    )
    assert result == {"key": "retrieval_recall@5", "score": 1.0}


def test_eval_faithfulness_all_grounded():
    result = eval_faithfulness(
        outputs={
            "citations": ["chunk_a", "chunk_b"],
            "retrieved_chunks": [{"chunk_id": "chunk_a"}, {"chunk_id": "chunk_b"}],
        },
        reference_outputs={},
    )
    assert result == {"key": "faithfulness", "score": 1.0}


def test_eval_faithfulness_ungrounded_citation():
    result = eval_faithfulness(
        outputs={
            "citations": ["chunk_a", "chunk_missing"],
            "retrieved_chunks": [{"chunk_id": "chunk_a"}],
        },
        reference_outputs={},
    )
    assert result == {"key": "faithfulness", "score": 0.5}


def test_eval_faithfulness_no_citations():
    result = eval_faithfulness(
        outputs={"citations": [], "retrieved_chunks": []},
        reference_outputs={},
    )
    assert result == {"key": "faithfulness", "score": 1.0}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/product/test_langsmith_eval.py -k "eval_decision or eval_retrieval or eval_faithfulness" -v
```

Expected: `ImportError` — the evaluator functions don't exist yet.

- [ ] **Step 3: Add evaluators and CLI to `evals/langsmith_eval.py`**

Append to the bottom of `evals/langsmith_eval.py` (after the `upload_dataset` function):

```python
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
```

- [ ] **Step 4: Run all evaluator tests**

```bash
pytest tests/product/test_langsmith_eval.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Smoke-test the CLI help**

```bash
python evals/langsmith_eval.py --help
python evals/langsmith_eval.py upload-dataset --help
python evals/langsmith_eval.py run-eval --help
```

Expected: help text prints for each command with no import errors.

- [ ] **Step 6: Commit**

```bash
git add evals/langsmith_eval.py tests/product/test_langsmith_eval.py
git commit -m "feat(langsmith): add evaluators and CLI commands"
```

---

## Task 4: Live smoke test — upload dataset and run one experiment

> This task uses real credentials. Ensure `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, and `LANGSMITH_PROJECT=AgenticUnderwriter` are exported in your shell.

**Files:** none (read-only verification)

- [ ] **Step 1: Export credentials**

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_API_KEY=<your-key>
export LANGSMITH_PROJECT=AgenticUnderwriter
```

- [ ] **Step 2: Upload the golden dataset**

```bash
python evals/langsmith_eval.py upload-dataset
```

Expected output:
```
Uploaded 206 examples to 'ho3-golden-206'.
Dataset URL: https://smith.langchain.com/o/.../datasets/...
```

Open the URL and verify 206 examples appear with `submission`, `decision`, `gold_citations` fields.

- [ ] **Step 3: Run a baseline experiment**

```bash
python evals/langsmith_eval.py run-eval --experiment-prefix baseline
```

Expected: the command streams progress (one line per example) and prints an experiment URL on completion. This will take several minutes for 206 cases.

- [ ] **Step 4: Verify in LangSmith UI**

Open the experiment URL. Confirm:
- Three scorer columns appear: `decision_accuracy`, `retrieval_recall@5`, `faithfulness`
- Each row shows per-case scores
- Aggregate scores appear in the header row

- [ ] **Step 5: Run a second experiment for comparison view**

```bash
python evals/langsmith_eval.py run-eval --experiment-prefix improved
```

In the LangSmith UI, select both experiments and use the comparison view. Screenshot or save the URL — this is the portfolio artifact.

- [ ] **Step 6: Turn tracing back off (optional)**

```bash
export LANGSMITH_TRACING=false
```

Or leave it on if you want ongoing traces during development.

- [ ] **Step 7: Final commit**

```bash
git add .
git commit -m "feat(langsmith): complete LangSmith integration — tracing, dataset, evaluators"
```

---

## Self-Review

**Spec coverage:**
- ✅ Configurable tracing via `@traceable` — Task 2
- ✅ Off by default — `LANGSMITH_TRACING=false` in `.env.example`, Task 1
- ✅ Dataset upload — Task 2
- ✅ Three evaluators (`client.evaluate`) — Task 3
- ✅ Shareable experiment link + comparison view — Task 4
- ✅ `langsmith` in `requirements-demo.txt` only — Task 1
- ✅ `evals/run.py` untouched — no task modifies it
- ✅ `observability.py` untouched — no task modifies it
- ✅ Fail-fast on missing `LANGSMITH_API_KEY` — `_require_api_key()` in Task 3

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:**
- `upload_dataset(dataset_path, dataset_name, client)` defined in Task 2, imported in test in Task 2 ✅
- `eval_decision_accuracy`, `eval_retrieval_recall`, `eval_faithfulness` defined in Task 3, imported in test in Task 3 ✅
- `run_workflow(submission: dict) -> dict` defined in Task 2, passed to `client.evaluate` in Task 3 ✅
- `outputs["citations"]` and `outputs["retrieved_chunks"]` keys produced by `run_workflow` and consumed by `eval_faithfulness` ✅
