"""Vision extraction eval: scores VisionEvidence against human-labeled images.

Mirrors the text fine-tune eval. Two signals that matter for a fenced extractor:
attribute accuracy (did it read the visible features right) and abstention
correctness (did it leave unassessable attributes `visible=false` instead of
guessing). The metric functions are pure and unit-tested; the CLI runner needs a
real provider (VISION_ENABLED + key) and a manifest of labeled images, run at the
end against actual photos.

Manifest format (JSONL), one record per image:
    {"image_path": "imgs/roof1.jpg",
     "gold": {"defensible_space_present": true, "roof_condition": "worn",
              "tarp_present": null}}   # null = not assessable -> model should abstain
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.schemas import VisionEvidence  # noqa: E402


def score_one(evidence: VisionEvidence, gold: Dict[str, Any]) -> Dict[str, int]:
    """Per-image tallies: correct/total for known attrs, and abstention hits/total."""
    known_total = known_correct = 0
    abstain_total = abstain_correct = 0
    for attr, gold_value in gold.items():
        predicted = getattr(evidence, attr, None)
        if predicted is None:
            continue
        if gold_value is None:
            # Should abstain (not assessable from the image).
            abstain_total += 1
            if not predicted.visible:
                abstain_correct += 1
        else:
            known_total += 1
            if predicted.visible and predicted.value == gold_value:
                known_correct += 1
    return {
        "known_total": known_total,
        "known_correct": known_correct,
        "abstain_total": abstain_total,
        "abstain_correct": abstain_correct,
    }


def score_predictions(predictions: List[VisionEvidence], golds: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert len(predictions) == len(golds), "predictions and golds must align"
    agg = {"known_total": 0, "known_correct": 0, "abstain_total": 0, "abstain_correct": 0}
    for ev, gold in zip(predictions, golds):
        for k, v in score_one(ev, gold).items():
            agg[k] += v
    return {
        "n": len(predictions),
        "attribute_accuracy": round(agg["known_correct"] / agg["known_total"], 4) if agg["known_total"] else None,
        "abstention_correctness": round(agg["abstain_correct"] / agg["abstain_total"], 4) if agg["abstain_total"] else None,
        "known_attributes": agg["known_total"],
        "abstain_attributes": agg["abstain_total"],
    }


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score vision extraction against labeled images.")
    parser.add_argument("--manifest", required=True, help="JSONL of {image_path, gold}.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    from app.vision_service import VisionEvidenceService

    service = VisionEvidenceService()
    if not service.provider:
        print("Vision provider not configured. Set VISION_ENABLED=true and OPENAI_API_KEY.")
        sys.exit(1)

    rows = _load_manifest(Path(args.manifest))
    base = Path(args.manifest).resolve().parent
    predictions, golds = [], []
    for row in rows:
        img = (base / row["image_path"]).read_bytes()
        predictions.append(service.extract_evidence(img))
        golds.append(row["gold"])

    report = score_predictions(predictions, golds)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"images: {report['n']}")
        print(f"attribute_accuracy:     {report['attribute_accuracy']}")
        print(f"abstention_correctness: {report['abstention_correctness']}")


if __name__ == "__main__":
    main()
