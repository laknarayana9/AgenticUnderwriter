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
