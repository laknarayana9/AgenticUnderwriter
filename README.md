# Agentic Underwriter

Evidence-backed HO3 quote underwriting prototype.

This repo demonstrates a compact quote-to-underwrite workflow for homeowners
submissions. It is built to show the product loop reviewers care about:
normalize an intake, identify missing or uncertain facts, ask targeted follow-up
questions, resume the same quote run, produce a decision packet, and route
referrals to a human review queue.

## What It Shows

- FastAPI quote endpoints for legacy quote payloads and canonical HO3 payloads.
- Seven-step agent workflow orchestration for intake, routing, enrichment,
  retrieval, assessment, verification, rating, and decision packaging.
- Missing-info loop for roof age, occupancy, applicant/address gaps, and
  wildfire mitigation evidence.
- Same-run resume through `/runs/{run_id}/answers` with audit events preserved.
- Human review queue for referred or declined risks.
- Decision packets with AI recommendation, confidence, reason codes, citations,
  next steps, premium indication, facts used, and a local demo trace reference.
- Versioned deterministic underwriting rules backed by synthetic guideline
  retrieval.

## Run

```bash
pip install -r requirements.txt
python -m pytest
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## One-Command Demo

Run the product walkthrough without starting a separate server:

```bash
python scripts/demo_walkthrough.py
```

The script uses FastAPI's in-process test client to exercise the real API
routes. It walks through a missing roof-age pause and same-run resume, then a
wildfire mitigation follow-up that moves into human review and approval.

## Core API Flow

Start a canonical HO3 run:

```bash
curl -X POST http://localhost:8000/quote/ho3 \
  -H "Content-Type: application/json" \
  -d '{
    "submission": {
      "applicant": {"full_name": "Robert Johnson"},
      "risk": {
        "property_address": "789 Pine St, Los Angeles, CA 90001",
        "occupancy": "owner_occupied_primary",
        "dwelling_type": "single_family",
        "year_built": 1995,
        "roof_age_years": null,
        "construction_type": "frame",
        "stories": 1
      },
      "coverage_request": {"coverage_a": 350000, "deductible": 1000}
    }
  }'
```

If required facts are missing, the response returns `status:
"waiting_for_info"` and a `required_questions` list. Each question includes a
`question_id`, `field_path`, `question_text`, `question_type`, and any available
options.

Resume the same run after the agent or underwriter answers:

```bash
curl -X POST http://localhost:8000/runs/{run_id}/answers \
  -H "Content-Type: application/json" \
  -d '{
    "answered_by": "underwriter",
    "answers": {"roof_age_years": 7}
  }'
```

The resumed response keeps the original `run_id`, appends answer events to the
audit trail, and continues through retrieval, assessment, rating, and decision
packaging.

## Human Review Flow

Referral and decline outcomes move to `pending_review` and can be listed:

```bash
curl http://localhost:8000/reviews/pending
```

Inspect the review packet:

```bash
curl http://localhost:8000/reviews/{run_id}
```

Approve the AI recommendation, override it, or request more information:

```bash
curl -X POST http://localhost:8000/reviews/{run_id}/actions \
  -H "Content-Type: application/json" \
  -d '{
    "action": "approve",
    "reviewer": "senior_uw",
    "note": "Citations and referral rationale reviewed."
  }'
```

The workflow stores the AI recommendation separately from the human final
decision so the audit trail does not overwrite model output.

## Decision Packet

Completed and referred runs return a `decision` object sourced from the internal
decision packet:

- `decision`: `ACCEPT`, `REFER`, or `DECLINE`
- `confidence`: decision confidence
- `review_reason_codes`: underwriting triggers such as `ROOF_AGE` or
  `WILDFIRE_HIGH`
- `citations`: retrieved guideline snippets used to support referral or decline
- `next_steps`: producer or underwriter actions
- `premium`: transparent demo premium indication

Use `/runs/{run_id}/audit` for the full workflow state, node outputs, required
questions, answer events, and final completion events.

## Demo Scenarios

The product tests use curated scenarios in `tests/demo_scenarios.py`, including:

- low-risk accepted quote
- high wildfire referral with mitigation-evidence follow-up
- missing roof age waiting for info and same-run resume
- old construction referral
- flood referral
- tenant-occupied referral

## Test Scope

The default pytest suite runs maintained product tests only:

```bash
python -m pytest
```

These tests cover quote contracts, missing-info resume, review queue actions,
explicit rule triggers, RAG fallback retrieval, rating sanity checks, and
end-to-end demo scenarios.

## Observability

`observability.py` is a local demo tracer. It logs span-like attributes and
latency, and decision packets include `local-demo://...` trace references. This
repo does not claim production distributed tracing or persisted trace event
storage.

## Current Limits

This is a local prototype. External hazard, claims, geocoding, RCE, auth,
idempotency, production tracing, and production deployment are intentionally not
claimed here. Hazard enrichment and guideline retrieval are deterministic demo
implementations, not live carrier or third-party data integrations.
