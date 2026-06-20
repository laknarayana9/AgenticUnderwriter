"""Deterministic metrics for the extraction fine-tune.

All pure functions over (prediction_text, gold_dict) so they are unit-testable
without any model or network. Three signals the video calls out for fine-tune
evals: JSON validity, exact match, and refusal correctness.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from finetune.schema import EXTRACTION_FIELDS, normalize_target


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of a JSON object from model output. None if invalid."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def is_json_valid(text: str) -> bool:
    return parse_json_object(text) is not None


def field_accuracy(pred: Dict[str, Any], gold: Dict[str, Any]) -> float:
    """Fraction of fields whose predicted value equals the gold value."""
    pred = normalize_target(pred)
    gold = normalize_target(gold)
    correct = sum(1 for f in EXTRACTION_FIELDS if pred[f] == gold[f])
    return correct / len(EXTRACTION_FIELDS)


def is_exact_match(pred: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    """True only if every field matches."""
    return field_accuracy(pred, gold) == 1.0


def refusal_correctness(pred: Dict[str, Any], gold: Dict[str, Any]) -> Optional[float]:
    """Of the fields absent in gold (null), the fraction the model also left null.

    None when the gold has no absent fields (metric does not apply to this case).
    A low score means the model hallucinated values for unstated fields.
    """
    pred = normalize_target(pred)
    gold = normalize_target(gold)
    absent = [f for f in EXTRACTION_FIELDS if gold[f] is None]
    if not absent:
        return None
    correct = sum(1 for f in absent if pred[f] is None)
    return correct / len(absent)


def score_predictions(
    predictions: List[str],
    golds: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate metrics over a holdout set of raw prediction strings + gold dicts."""
    assert len(predictions) == len(golds), "predictions and golds must align"
    n = len(predictions)
    json_valid = 0
    exact = 0
    field_acc_sum = 0.0
    refusal_scores: List[float] = []

    for text, gold in zip(predictions, golds):
        parsed = parse_json_object(text)
        if parsed is None:
            # Invalid JSON: counts against json-valid and exact; field acc 0.
            continue
        json_valid += 1
        field_acc_sum += field_accuracy(parsed, gold)
        if is_exact_match(parsed, gold):
            exact += 1
        rc = refusal_correctness(parsed, gold)
        if rc is not None:
            refusal_scores.append(rc)

    return {
        "n": n,
        "json_valid_rate": round(json_valid / n, 4) if n else 0.0,
        "exact_match_rate": round(exact / n, 4) if n else 0.0,
        "field_accuracy": round(field_acc_sum / n, 4) if n else 0.0,
        "refusal_correctness": round(sum(refusal_scores) / len(refusal_scores), 4)
        if refusal_scores
        else None,
    }
