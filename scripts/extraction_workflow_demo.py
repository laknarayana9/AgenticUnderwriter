"""Demo: extraction model -> live governed workflow (base vs fine-tuned).

The portfolio money shot for the fine-tune. A producer note that OMITS roof age
is extracted into structured intake, then run through the real underwriting
workflow. The governance consequence of refusal correctness becomes visible:

  - base model      -> hallucinates a roof age the note never gave, so the
                       workflow underwrites on a fabricated fact (no pause).
  - fine-tuned model -> returns null for the unstated field, so the workflow
                       correctly PAUSES and asks for the roof age.

Same model id you'd set in production (LLM_MODEL=ft:...), same governed pipeline.

Usage:
    # Illustrative, deterministic, no model/key required:
    python scripts/extraction_workflow_demo.py --demo-contrast

    # Real models (needs NEBIUS_API_KEY):
    NEBIUS_API_KEY=... python scripts/extraction_workflow_demo.py \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --tuned-model 'ft:meta-llama/Llama-3.1-8B-Instruct-...'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune.metrics import normalize_target, parse_json_object  # noqa: E402
from finetune.schema import SYSTEM_PROMPT  # noqa: E402

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# A clean, low-risk note that deliberately says NOTHING about roof age. Every
# other field is stated, so roof age is the sole differentiator.
DEMO_NOTE = (
    "Avery Chen is applying for a homeowners policy at 742 Evergreen Terrace, "
    "Palo Alto, CA 94301. It's a single-family home, owner-occupied as their "
    "primary residence, built in 2005. They want $500,000 of dwelling coverage "
    "with a $1,000 deductible."
)

# Canned extractions for --demo-contrast (no model/key needed). The base model
# hallucinates a roof age; the fine-tuned model correctly abstains (null).
_MOCK_BASE = {
    "applicant_name": "Avery Chen", "property_address": "742 Evergreen Terrace, Palo Alto, CA 94301",
    "occupancy": "owner_occupied_primary", "dwelling_type": "single_family", "year_built": 2005,
    "roof_age_years": 10,  # <-- HALLUCINATED: the note never mentions roof age
    "coverage_a": 500000, "deductible": 1000,
}
_MOCK_TUNED = {**_MOCK_BASE, "roof_age_years": None}  # correct abstention


def extract_with_model(model: str, note: str) -> Dict[str, Any]:
    """Call a Nebius-hosted model to extract intake fields from the note."""
    from openai import OpenAI

    client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=os.environ["NEBIUS_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": note}],
        temperature=0,
    )
    parsed = parse_json_object(resp.choices[0].message.content or "") or {}
    return normalize_target(parsed)


def fields_to_submission(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Map extracted fields into the workflow's HO3 submission shape.

    roof_age_years is passed through as-is: a null here is what makes the
    intake normalizer pause and ask, which is the whole point of the demo."""
    return {
        "applicant": {"full_name": fields.get("applicant_name") or ""},
        "risk": {
            "property_address": fields.get("property_address") or "",
            "occupancy": fields.get("occupancy") or "owner_occupied_primary",
            "dwelling_type": fields.get("dwelling_type") or "single_family",
            "year_built": fields.get("year_built") or 2000,
            "roof_age_years": fields.get("roof_age_years"),
            "construction_type": "frame",
            "stories": 1,
        },
        "coverage_request": {
            "coverage_a": fields.get("coverage_a") or 300000,
            "deductible": fields.get("deductible") or 1000,
        },
    }


def run_case(workflow, label: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    """Run extracted fields through the live workflow and summarize the outcome."""
    state = workflow.run(fields_to_submission(fields))
    roof = fields.get("roof_age_years")
    asked_roof = any(
        "roof" in (q.get("question", "") + q.get("question_text", "")).lower()
        for q in state.required_questions
    )
    return {
        "label": label,
        "extracted_roof_age_years": roof,
        "roof_age_hallucinated": roof is not None,  # the note never stated it
        "workflow_status": state.status,
        "asked_for_roof_age": asked_roof,
        "decision": state.decision_packet.decision.value if state.decision_packet else None,
    }


def _print_outcome(o: Dict[str, Any]) -> None:
    print(f"\n=== {o['label']} ===")
    print(f"  extracted roof_age_years : {o['extracted_roof_age_years']}")
    if o["roof_age_hallucinated"]:
        print("  -> HALLUCINATED: note never stated roof age; value invented.")
        print(f"  -> workflow status      : {o['workflow_status']}  decision: {o['decision']}")
        print("  -> CONSEQUENCE: underwrote on a fabricated fact (no missing-info pause).")
    else:
        print("  -> correctly abstained (null) on the unstated field.")
        print(f"  -> workflow status      : {o['workflow_status']}  asked for roof age: {o['asked_for_roof_age']}")
        print("  -> CONSEQUENCE: workflow correctly paused to gather the missing fact.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraction model -> governed workflow demo.")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--tuned-model", default=None)
    parser.add_argument("--demo-contrast", action="store_true",
                        help="Use canned base/fine-tuned extractions (no model/key).")
    args = parser.parse_args()

    from workflows.agent_workflow import UnderwritingWorkflow

    workflow = UnderwritingWorkflow()

    print("Producer note (roof age intentionally omitted):")
    print(f"  {DEMO_NOTE}")

    if args.demo_contrast or (not args.base_model and not args.tuned_model):
        _print_outcome(run_case(workflow, "BASE (hallucinates)", _MOCK_BASE))
        _print_outcome(run_case(workflow, "FINE-TUNED (abstains)", _MOCK_TUNED))
        if not (args.base_model or args.tuned_model):
            print("\n(Illustrative contrast. Pass --base-model / --tuned-model with "
                  "NEBIUS_API_KEY to run real extraction.)")
        return

    if not os.getenv("NEBIUS_API_KEY"):
        print("\nNEBIUS_API_KEY not set. Use --demo-contrast for the offline illustration.")
        sys.exit(1)

    for label, model in [("BASE", args.base_model), ("FINE-TUNED", args.tuned_model)]:
        if not model:
            continue
        fields = extract_with_model(model, DEMO_NOTE)
        _print_outcome(run_case(workflow, f"{label} ({model})", fields))


if __name__ == "__main__":
    main()
