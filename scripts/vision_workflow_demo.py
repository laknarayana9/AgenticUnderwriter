"""Demo: property photo -> fenced vision extraction -> live governed workflow.

The multimodal sibling of scripts/extraction_workflow_demo.py. For a high-wildfire
property whose mitigation status is unknown, it contrasts what the workflow does
depending on what the vision model sees:

  - photo shows defensible space (high confidence) -> folded into
    wildfire_mitigation_evidence -> the wildfire mitigation gate is satisfied,
    the run proceeds.
  - photo is unclear / model abstains -> the field stays null -> the deterministic
    missing-info gate correctly PAUSES to request mitigation evidence.

Same governance boundary as the text extractor: vision proposes, rules decide.

Usage:
    # Deterministic illustration (no model/key, fake vision results):
    python scripts/vision_workflow_demo.py --demo-contrast

    # Real model on an actual image (needs VISION_ENABLED=true + OPENAI_API_KEY):
    VISION_ENABLED=true OPENAI_API_KEY=... python scripts/vision_workflow_demo.py --image roof.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# High-wildfire address with mitigation status intentionally unknown.
DEMO_SUBMISSION: Dict[str, Any] = {
    "applicant": {"full_name": "Avery Chen"},
    "risk": {
        "property_address": "742 Fire Zone Rd, Sacramento, CA 95818",
        "occupancy": "owner_occupied_primary",
        "dwelling_type": "single_family",
        "year_built": 2008,
        "roof_age_years": 7,
        "construction_type": "frame",
        "stories": 1,
        # wildfire_mitigation_evidence intentionally omitted
    },
    "coverage_request": {"coverage_a": 500000, "deductible": 1000},
}

_SEES_DEFENSIBLE_SPACE = {"defensible_space_present": {"value": True, "confidence": 0.93, "visible": True}}
_UNCLEAR = {"defensible_space_present": {"value": None, "confidence": 0.0, "visible": False}}


class _FakeVisionProvider:
    model = "fake-vision-demo"

    def __init__(self, payload):
        self._payload = payload

    def extract(self, image_bytes, system_prompt):
        return self._payload


def _run_case(label: str, provider) -> None:
    from workflows.agent_workflow import UnderwritingWorkflow

    workflow = UnderwritingWorkflow()
    if provider is not None:
        workflow.vision_service.provider = provider  # inject for the demo
    state = workflow.run(DEMO_SUBMISSION, image_bytes=b"demo-image")

    asked_mitigation = any(
        "mitigation" in (q.get("question", "") + q.get("question_text", "")).lower()
        or "defensible" in (q.get("question", "") + q.get("question_text", "")).lower()
        for q in state.required_questions
    )
    folded = (state.submission_raw or {}).get("risk", {}).get("wildfire_mitigation_evidence")
    print(f"\n=== {label} ===")
    print(f"  folded wildfire_mitigation_evidence : {folded}")
    print(f"  workflow status                     : {state.status}")
    if state.status == "waiting_for_info":
        print(f"  asked for mitigation evidence       : {asked_mitigation}")
        print("  -> CONSEQUENCE: vision could not confirm mitigation; workflow paused to ask a human.")
    else:
        decision = state.decision_packet.decision.value if state.decision_packet else None
        print(f"  decision                            : {decision}")
        print("  -> CONSEQUENCE: vision confirmed defensible space; the run proceeded without a pause.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Property photo -> governed workflow demo.")
    parser.add_argument("--demo-contrast", action="store_true", help="Deterministic fake-vision contrast (no key).")
    parser.add_argument("--image", help="Path to a real property photo (needs VISION_ENABLED + OPENAI_API_KEY).")
    args = parser.parse_args()

    print("High-wildfire property; mitigation status unknown at intake:")
    print(f"  {DEMO_SUBMISSION['risk']['property_address']}")

    if args.image:
        from workflows.agent_workflow import UnderwritingWorkflow
        workflow = UnderwritingWorkflow()
        if not workflow.vision_service.provider:
            print("\nVision provider not configured. Set VISION_ENABLED=true and OPENAI_API_KEY, "
                  "or use --demo-contrast.")
            sys.exit(1)
        image_bytes = Path(args.image).read_bytes()
        state = workflow.run(DEMO_SUBMISSION, image_bytes=image_bytes)
        folded = (state.submission_raw or {}).get("risk", {}).get("wildfire_mitigation_evidence")
        print(f"\nReal vision run → folded wildfire_mitigation_evidence={folded}, status={state.status}")
        return

    # Default: deterministic contrast.
    _run_case("VISION SEES DEFENSIBLE SPACE", _FakeVisionProvider(_SEES_DEFENSIBLE_SPACE))
    _run_case("VISION UNCLEAR / ABSTAINS", _FakeVisionProvider(_UNCLEAR))
    print("\n(Illustrative contrast. Use --image with VISION_ENABLED + OPENAI_API_KEY for a real photo.)")


if __name__ == "__main__":
    main()
