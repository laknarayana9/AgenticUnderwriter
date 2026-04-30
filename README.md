# Agentic Underwriter

Evidence-backed HO3 quote underwriting prototype.

This repo keeps the implementation intentionally small:

- FastAPI quote endpoints
- versioned deterministic underwriting rules
- agent workflow orchestration
- synthetic guideline retrieval with citations
- transparent demo rating output
- maintained product tests

## Run

```bash
pip install -r requirements.txt
python -m pytest
uvicorn app.main:app --reload
```

## Test Scope

The default pytest suite runs maintained product tests only:

```bash
python -m pytest
```

These tests cover quote contracts, explicit rule triggers, RAG fallback retrieval,
rating sanity checks, and end-to-end demo scenarios.

## Demo Evaluation Harness

Run the polished demo harness when you need an auditable proof report for the
10 curated underwriting scenarios:

```bash
python -m evals.demo_harness --format markdown --output reports/demo-eval.md
python -m evals.demo_harness --format json --output reports/demo-eval.json
```

The report shows expected vs. actual decisions, verifies citations on every
REFER/DECLINE outcome, and confirms missing critical information never results
in a silent ACCEPT.

## Current Limits

This is a local prototype. External hazard, claims, geocoding, RCE, auth,
idempotency, and production deployment are intentionally not claimed here.
