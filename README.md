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

## Current Limits

This is a local prototype. External hazard, claims, geocoding, RCE, auth,
idempotency, and production deployment are intentionally not claimed here.
