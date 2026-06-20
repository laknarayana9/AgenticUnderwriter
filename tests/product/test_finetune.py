"""Hermetic tests for the fine-tuning track: dataset shape, metrics, validation.

No model or network: covers the dataset generator format, the extraction metrics
(JSON validity, exact match, refusal correctness), and the submit-time dataset
validation.
"""

import json

import pytest

from finetune.generate_dataset import build_examples, write_jsonl
from finetune.metrics import (
    field_accuracy,
    is_exact_match,
    is_json_valid,
    refusal_correctness,
    score_predictions,
)
from finetune.schema import EXTRACTION_FIELDS, empty_target, normalize_target
from finetune.submit import validate_dataset


def test_generated_examples_are_chat_format_with_full_target():
    examples = build_examples(20, seed=1)
    assert len(examples) == 20
    for ex in examples:
        msgs = ex["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        target = json.loads(msgs[-1]["content"])
        # Every field present (null when not stated), canonical key set.
        assert set(target.keys()) == set(EXTRACTION_FIELDS)


def test_generation_is_deterministic_for_a_seed():
    assert build_examples(10, seed=42) == build_examples(10, seed=42)
    assert build_examples(10, seed=42) != build_examples(10, seed=43)


def test_dropped_fields_are_null_in_gold():
    # Across a sample, at least some fields must be null (abstention is exercised).
    examples = build_examples(50, seed=3)
    targets = [json.loads(ex["messages"][-1]["content"]) for ex in examples]
    assert any(t["roof_age_years"] is None for t in targets)
    assert any(t["deductible"] is None for t in targets)


def test_metrics_field_accuracy_and_exact_match():
    gold = normalize_target({"applicant_name": "Avery Chen", "roof_age_years": 12, "year_built": 1990})
    perfect = dict(gold)
    assert is_exact_match(perfect, gold)
    assert field_accuracy(perfect, gold) == 1.0

    one_wrong = dict(gold)
    one_wrong["roof_age_years"] = 99
    assert not is_exact_match(one_wrong, gold)
    assert field_accuracy(one_wrong, gold) == (len(EXTRACTION_FIELDS) - 1) / len(EXTRACTION_FIELDS)


def test_refusal_correctness_rewards_nulls_on_absent_fields():
    gold = empty_target()
    gold["applicant_name"] = "Avery Chen"  # only this field is stated
    # Model that correctly leaves the rest null:
    abstains = empty_target()
    abstains["applicant_name"] = "Avery Chen"
    assert refusal_correctness(abstains, gold) == 1.0
    # Model that hallucinates a roof age the note never gave:
    hallucinates = dict(abstains)
    hallucinates["roof_age_years"] = 15
    assert refusal_correctness(hallucinates, gold) < 1.0


def test_refusal_correctness_is_none_when_all_fields_present():
    gold = {f: ("x" if "name" in f or "address" in f else 1) for f in EXTRACTION_FIELDS}
    gold = normalize_target(gold)
    assert refusal_correctness(gold, gold) is None


def test_score_predictions_counts_invalid_json():
    golds = [normalize_target({"applicant_name": "A"}), normalize_target({"applicant_name": "B"})]
    preds = [json.dumps(golds[0]), "not json at all"]
    scores = score_predictions(preds, golds)
    assert scores["n"] == 2
    assert scores["json_valid_rate"] == 0.5
    assert is_json_valid(preds[0]) and not is_json_valid(preds[1])


def test_validate_dataset_accepts_generated_file(tmp_path):
    examples = build_examples(5, seed=9)
    path = tmp_path / "train.jsonl"
    write_jsonl(path, examples)
    assert validate_dataset(path) == 5


def test_validate_dataset_rejects_bad_format(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
    with pytest.raises(ValueError):
        validate_dataset(path)
