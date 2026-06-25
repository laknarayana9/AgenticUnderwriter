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
