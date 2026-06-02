# API Testing & SQLite Verification

End-to-end manual test guide for every endpoint in the Agentic Quote-to-Underwrite
API, plus how to verify what was persisted in SQLite.

The endpoint list below is the source of truth; the live, always-current schema is
served at:

- Swagger UI: `${BASE_URL}/docs`
- ReDoc: `${BASE_URL}/redoc`
- Raw spec: `${BASE_URL}/openapi.json`

## Setup

Pick a base URL depending on where you are testing.

```bash
# Local (uvicorn)
export BASE_URL="http://localhost:8000"

# Modal deployment
# export BASE_URL="https://ln-tuttagunta--agentic-underwriter-fastapi-app.modal.run"
```

Start the local server if needed:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoint coverage

| # | Method | Path | Request model |
|---|--------|------|---------------|
| 1 | GET  | `/health` | – |
| 2 | GET  | `/stats` | – |
| 3 | POST | `/quote/run` | `QuoteRunRequest` |
| 4 | POST | `/quote/ho3` | `HO3RunRequest` |
| 5 | GET  | `/runs` | query: `limit`, `status` |
| 6 | GET  | `/runs/{run_id}` | – |
| 7 | GET  | `/runs/{run_id}/audit` | – |
| 8 | POST | `/runs/{run_id}/answers` | `MissingInfoAnswerRequest` |
| 9 | GET  | `/reviews/pending` | query: `limit` |
| 10 | GET | `/reviews/{run_id}` | – |
| 11 | POST | `/reviews/{run_id}/actions` | `ReviewActionRequest` |

---

## 1. Health check

```bash
curl -s "${BASE_URL}/health"
```

Expect `200` with a status payload.

## 2. System stats

```bash
curl -s "${BASE_URL}/stats"
```

Returns `total_runs`, `recent_runs_24h`, and `runs_by_status`. Cross-check with
`python scripts/db_inspect.py stats` (see below).

## 3. Process a quote (`/quote/run`)

`submission` is required; `use_agentic` and `additional_answers` are optional.

```bash
curl -s -X POST "${BASE_URL}/quote/run" \
  -H "Content-Type: application/json" \
  -d '{
    "submission": {
      "applicant_name": "Legacy User",
      "address": "456 Legacy Ln, Oakland, CA 94601",
      "property_type": "single_family",
      "coverage_amount": 300000,
      "construction_year": 2005,
      "roof_type": "composite"
    },
    "use_agentic": false
  }'
```

Capture the `run_id` for later steps:

```bash
RUN_ID=$(curl -s -X POST "${BASE_URL}/quote/run" \
  -H "Content-Type: application/json" \
  -d '{"submission":{"applicant_name":"Legacy User","address":"456 Legacy Ln","property_type":"single_family","coverage_amount":300000}}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
echo "RUN_ID=$RUN_ID"
```

## 4. Process an HO3 submission (`/quote/ho3`)

Full accept-path example:

```bash
curl -s -X POST "${BASE_URL}/quote/ho3" \
  -H "Content-Type: application/json" \
  -d '{
    "submission": {
      "applicant": {"full_name": "Jane Smith", "email": "jane@example.com"},
      "risk": {
        "property_address": "456 Oak Ave",
        "occupancy": "owner_occupied_primary",
        "dwelling_type": "single_family",
        "year_built": 2010,
        "roof_age_years": 5,
        "construction_type": "frame",
        "stories": 2
      },
      "coverage_request": {"coverage_a": 300000, "deductible": 1000}
    }
  }'
```

Missing-info path (omit `roof_age_years`) returns `status: "waiting_for_info"` and a
`required_questions` list. Save that `run_id` to resume in step 8.

```bash
HO3_RUN_ID=$(curl -s -X POST "${BASE_URL}/quote/ho3" \
  -H "Content-Type: application/json" \
  -d '{"submission":{"applicant":{"full_name":"Robert Johnson"},"risk":{"property_address":"789 Pine St","occupancy":"owner_occupied_primary","dwelling_type":"single_family","year_built":1995,"construction_type":"frame","stories":1},"coverage_request":{"coverage_a":350000,"deductible":1000}}}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
echo "HO3_RUN_ID=$HO3_RUN_ID"
```

## 5. List runs

```bash
curl -s "${BASE_URL}/runs?limit=10"
curl -s "${BASE_URL}/runs?limit=10&status=pending_review"
```

## 6. Get a single run

```bash
curl -s "${BASE_URL}/runs/${RUN_ID}"
```

## 7. Full audit trail

```bash
curl -s "${BASE_URL}/runs/${RUN_ID}/audit"
```

Returns node outputs for validation, enrichment, retrieval, assessment,
verification, rating, and decision packaging.

## 8. Answer missing info and resume (`/runs/{run_id}/answers`)

Use the `HO3_RUN_ID` from the missing-info path in step 4:

```bash
curl -s -X POST "${BASE_URL}/runs/${HO3_RUN_ID}/answers" \
  -H "Content-Type: application/json" \
  -d '{
    "answered_by": "underwriter",
    "answers": {"roof_age_years": 7}
  }'
```

The resumed response keeps the original `run_id`.

## 9. List pending reviews

```bash
curl -s "${BASE_URL}/reviews/pending?limit=10"
```

## 10. Get a review packet

```bash
curl -s "${BASE_URL}/reviews/${HO3_RUN_ID}"
```

## 11. Take a review action (`/reviews/{run_id}/actions`)

`action` is one of `approve`, `override`, `request_more_info`.

```bash
# Approve the AI recommendation
curl -s -X POST "${BASE_URL}/reviews/${HO3_RUN_ID}/actions" \
  -H "Content-Type: application/json" \
  -d '{"action": "approve", "reviewer": "senior_uw", "note": "Citations reviewed."}'

# Override with a different final decision (note required)
curl -s -X POST "${BASE_URL}/reviews/${HO3_RUN_ID}/actions" \
  -H "Content-Type: application/json" \
  -d '{"action": "override", "reviewer": "senior_uw", "note": "Manual decline.", "final_decision": "DECLINE"}'

# Request more information
curl -s -X POST "${BASE_URL}/reviews/${HO3_RUN_ID}/actions" \
  -H "Content-Type: application/json" \
  -d '{"action": "request_more_info", "reviewer": "senior_uw", "requested_info": ["prior_loss_history"]}'
```

---

## Automated sweep (optional)

Run all read endpoints plus a create round-trip in one shot:

```bash
set -e
echo "health:";  curl -s "${BASE_URL}/health";  echo
echo "stats:";   curl -s "${BASE_URL}/stats";   echo

RUN_ID=$(curl -s -X POST "${BASE_URL}/quote/run" \
  -H "Content-Type: application/json" \
  -d '{"submission":{"applicant_name":"Sweep","address":"1 Test St","property_type":"single_family","coverage_amount":250000}}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['run_id'])")
echo "created RUN_ID=$RUN_ID"

curl -s "${BASE_URL}/runs/${RUN_ID}"        > /dev/null && echo "GET /runs/{id} ok"
curl -s "${BASE_URL}/runs/${RUN_ID}/audit"  > /dev/null && echo "GET /runs/{id}/audit ok"
curl -s "${BASE_URL}/runs?limit=5"          > /dev/null && echo "GET /runs ok"
curl -s "${BASE_URL}/reviews/pending"       > /dev/null && echo "GET /reviews/pending ok"
```

---

## Verifying data in SQLite

The app stores everything in SQLite (`storage/underwriting.db` locally; the path is
configurable via `UNDERWRITING_DB_PATH` or `DATABASE_URL`). Use the read-only
inspector at `scripts/db_inspect.py` to confirm what each API call persisted.

```bash
# Every table with row counts
python scripts/db_inspect.py tables

# High-level run/review stats (compare against GET /stats)
python scripts/db_inspect.py stats

# Columns for a table
python scripts/db_inspect.py schema run_records

# Recent rows (newest first)
python scripts/db_inspect.py rows run_records --limit 5
python scripts/db_inspect.py rows human_review_records --limit 5

# Filter rows
python scripts/db_inspect.py rows run_records --where "status = 'pending_review'"

# Everything tied to one run_id (run_records, reviews, hitl_tasks, etc.)
python scripts/db_inspect.py run "$RUN_ID"

# Arbitrary read-only SELECT (writes are rejected)
python scripts/db_inspect.py query "SELECT status, COUNT(*) FROM run_records GROUP BY status"

# JSON output for any command
python scripts/db_inspect.py run "$RUN_ID" --json
```

### Suggested verification per endpoint

| After calling | Verify in SQLite |
|---------------|------------------|
| `POST /quote/run` / `POST /quote/ho3` | `db_inspect.py run <run_id>` shows a `run_records` row with the expected `status` |
| `POST /runs/{id}/answers` | `run_records.status` advanced past `waiting_for_info`; `hitl_tasks` row marked answered |
| referral/decline outcome | `human_review_records` row with `status = pending_review` |
| `POST /reviews/{id}/actions` | `human_review_records.final_decision` / `reviewer` populated; `hitl_tasks.status` updated |
| `GET /stats` | matches `db_inspect.py stats` |

### Tables

| Table | Holds |
|-------|-------|
| `run_records` | Full workflow state + node outputs per run |
| `human_review_records` | Human review status, reviewer, final decision |
| `quote_records` | Quote summary records |
| `hitl_tasks` | Human-in-the-loop tasks (missing info / review) |
| `tool_calls` | Append-only tool call event store |
| `retrieval_events` | RAG retrieval events for eval/debug |
| `idempotency_keys` | Idempotency key bookkeeping |

### Raw sqlite3 (alternative)

```bash
sqlite3 storage/underwriting.db ".tables"
sqlite3 storage/underwriting.db "SELECT run_id, status, created_at FROM run_records ORDER BY created_at DESC LIMIT 5;"
```
