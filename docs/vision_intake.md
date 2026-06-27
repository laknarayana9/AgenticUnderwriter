# Vision evidence intake (multimodal)

A property photo can be run through a **fenced vision service** that extracts
structured risk attributes and folds the confident ones into the HO3 submission
*before* the deterministic rules run. Vision is **upstream-and-guarded input,
never the decision** — the same boundary ADR-0001 draws for the text LLM.

## Flow

```
photo ──▶ VisionEvidenceService ──▶ VisionEvidence ──▶ fold_vision_into_submission ──▶ rules
            (fenced, abstaining)      (per-attribute        (confidence-gated,
                                       confidence+visible)   never overwrites)
```

- `app/vision_service.py` — `VisionEvidenceService.extract_evidence(image_bytes)`
  returns a `VisionEvidence` with, per attribute, a `value`, a `confidence`, and a
  `visible` flag. **Abstention is first-class:** `visible=false` means "could not
  assess from the image," and the model is instructed to abstain rather than guess.
- `fold_vision_into_submission(...)` maps **only** `defensible_space_present` into
  the rule-consumed `risk.wildfire_mitigation_evidence`, and only when it is
  `visible` and `confidence >= VISION_MIN_CONFIDENCE`. It never overwrites a value
  the producer already supplied. Other attributes (roof condition, hazards, …) are
  recorded as provenance only — no rule consumes them yet.

## The governance payoff

For a high-wildfire property whose mitigation status is unknown:

| vision result | folded field | workflow outcome |
|---|---|---|
| defensible space visible, high confidence | `wildfire_mitigation_evidence = True` | proceeds (mitigation gate satisfied) |
| unclear / model abstains | left `null` | **pauses** — missing-info gate asks a human |

So the confidence/abstention behavior has a direct, auditable consequence:
vision either *proposes* a fact the rules then act on, or it stays silent and the
deterministic gate asks — it can never decide. `scripts/vision_workflow_demo.py
--demo-contrast` shows both ends against the real workflow.

## Configuration

```bash
VISION_ENABLED=true                 # off by default (CI/offline → deterministic stub)
VISION_PROVIDER=openai              # openai (default) or ollama (local, on-device)
VISION_MODEL=gpt-4o                 # vision-capable model
VISION_MIN_CONFIDENCE=0.6           # below this, defer to the missing-info gate
OPENAI_API_KEY=...                  # for VISION_PROVIDER=openai

# Fully local / private alternative (photos never leave the machine):
VISION_PROVIDER=ollama
VISION_MODEL=llama3.2-vision        # or llava
OLLAMA_BASE_URL=http://localhost:11434
```

Disabled or unavailable → the service returns fully abstained evidence, so the
workflow behaves exactly as it would without a photo. Any provider error degrades
the same way (never crashes the request).

## API

```bash
# multipart: submission JSON as a form field + the photo file
curl -X POST localhost:8000/quote/ho3/with-photo \
  -F submission='{"applicant":{"full_name":"Avery Chen"}, "risk":{...}, "coverage_request":{...}}' \
  -F photo=@roof.jpg
```

## Evaluation

`evals/vision_eval.py` scores extraction against human-labeled images on
**attribute accuracy** and **abstention correctness** (the vision analogue of the
text extractor's refusal correctness). It needs a real provider + a manifest of
labeled photos, run against actual images.

## Safety / provenance / PII

- Only the **image SHA-256** is stored (provenance handle in the audit trail);
  raw images are not persisted.
- The vision stage is timed (`stage_timings["vision_intake"]`), so it shows up in
  the per-request latency budget.
- **PII caveat:** property photos can contain faces, plates, or house numbers.
  Hosted providers (OpenAI) send the image out; the **local Ollama provider
  (`VISION_PROVIDER=ollama`) keeps photos on-device** — the privacy-preserving
  option for real deployments. See failure mode #11 in `docs/failure-modes.md`.
