# Agentic Underwriter

A governed agentic workflow that turns a homeowner-insurance application into a
cited `ACCEPT` / `REFER` / `DECLINE` decision, with human-in-the-loop review.

**Portfolio proof points:** CI-gated evals across 206 stratified cases with 100%
decision accuracy, 100% reason-code match, 100% retrieval recall@5, 100% citation
faithfulness, and 116 passing product tests.

This repo demonstrates a compact quote-to-underwrite workflow for homeowners
submissions, including HO3 homeowners policies. It is built to show the product
loop reviewers care about: normalize an intake, identify missing or uncertain
facts, ask targeted follow-up questions, resume the same quote run, produce a
decision packet, and route referrals to a human review queue.

Architecture note: deterministic underwriting rules remain the governed source
of truth for eligibility decisions. AI components assist with retrieval,
evidence grounding, rationale support, follow-up question workflows, and
orchestration around the decisioning layer.

## What It Shows

- FastAPI quote endpoints for legacy quote payloads and canonical HO3 payloads.
- Seven-step deterministic agent workflow orchestration for intake, routing,
  enrichment, retrieval, assessment, verification, rating, and decision
  packaging.
- Two interchangeable orchestration engines — a native explicit state machine
  (default) and a **LangGraph** `StateGraph` with durable, checkpointed
  human-in-the-loop pause/resume — producing identical, rules-owned decisions.
  Optional **LangChain** structured-output provider (`LLM_PROVIDER=langchain`).
  See [docs/orchestration.md](docs/orchestration.md).
- Missing-info loop for roof age, occupancy, applicant/address gaps, and
  wildfire mitigation evidence.
- Same-run resume through `/runs/{run_id}/answers` with audit events preserved.
- Human review queue for referred or declined risks.
- Decision packets with system recommendation, confidence, reason codes,
  citations, next steps, premium indication, facts used, and a trace reference.
- Versioned deterministic underwriting rules backed by lexical, semantic, or
  hybrid guideline retrieval.
- Structured LLM service boundary for producer-facing rationale and
  missing-info wording, with Pydantic validation and deterministic fallback.
- Streamlit demo for editing submissions, running the workflow, viewing
  citations, and inspecting audit/HITL status.

## Architecture

```mermaid
flowchart LR
    A["HO3 / homeowner application"] --> B["Intake normalizer"]
    B --> C{"Missing facts?"}
    C -->|Yes| D["Follow-up questions"]
    D --> E["Same-run resume"]
    E --> B
    C -->|No| F["Planner / router"]
    F --> G["Risk enrichment"]
    G --> H["Guideline retrieval"]
    H --> I["Rule-based assessment"]
    I --> J["Verifier guardrail"]
    J --> K["Rating"]
    K --> L["Decision packet"]
    L --> M{"Human review needed?"}
    M -->|Yes| N["HITL review queue"]
    M -->|No| O["Completed quote"]
```

## Corpus

The retrieval corpus is **synthetic** carrier underwriting guidelines (not
proprietary), authored in Markdown with explicit `MUST/SHALL/SHOULD/MAY` rule
language and `## / ###` section structure for citation-grade chunking. It lives
in [app/externaldata/docs](app/externaldata/docs) and currently holds 8
guideline documents:

- `uw_guidelines_homeowners.md` — eligibility, referral, and knockouts
- `hazards_guidance.md` — wildfire/flood/wind hazard signals → actions
- `endorsements_manual.md` — endorsement catalog and requirements
- `rating_rules.md` — rating plan, factors, deductible rules
- `uw_workflow_playbook.md` — triage workflow and missing-info questions
- `liability_exposures_guidance.md` — animal, pool/trampoline, home-business liability
- `water_damage_guidance.md` — non-flood water, sump/backup, freeze, leak mitigation
- `claims_history_guidance.md` — loss frequency/severity and coverage continuity

Owner of record: underwriting policy (synthetic for this demo). Refresh cadence:
versioned per document header (`Effective Date` / `Version`); re-ingested on
startup. Swap in real guidelines without code changes — chunking and citation
tracking key off the same Markdown structure.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env
python -m pytest
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## Modal Deployment

Live API docs:
[https://ln-tuttagunta--agentic-underwriter-fastapi-app-dev.modal.run/docs](https://ln-tuttagunta--agentic-underwriter-fastapi-app-dev.modal.run/docs)

List runs endpoint docs:
[https://ln-tuttagunta--agentic-underwriter-fastapi-app-dev.modal.run/docs#/default/list_runs_runs_get](https://ln-tuttagunta--agentic-underwriter-fastapi-app-dev.modal.run/docs#/default/list_runs_runs_get)

Smoke test:

```bash
curl https://ln-tuttagunta--agentic-underwriter-fastapi-app-dev.modal.run/health
```

This repo includes a Modal ASGI entrypoint for the existing FastAPI app:

```bash
modal setup
modal serve modal_app.py
modal deploy modal_app.py
```

`modal_app.py` builds a Python 3.13 image from `requirements.txt`, includes the
local application packages and underwriting guideline documents, mounts a
durable Modal Volume at `/data`, and points SQLite at
`sqlite:////data/underwriting.db`. The deployed endpoint serves the same API
routes as local `uvicorn`; deterministic underwriting rules remain the governed
decision layer. The Modal function is capped at one container because the
portfolio deploy uses SQLite on a Modal Volume; move `DATABASE_URL` to a managed
database before raising container count for higher write concurrency.

Useful Modal knobs:

```bash
MODAL_APP_NAME=agentic-underwriter
MODAL_DATA_VOLUME=agentic-underwriter-data
```

Structured LLM output is disabled by default for deploy parity. Enable it with a
Modal Secret or environment variables only when you want provider-backed wording:

```bash
LLM_STRUCTURED_OUTPUT_ENABLED=true
OPENAI_API_KEY=...
```

## Interactive Demo

Run the Streamlit demo locally:

```bash
pip install -r requirements.txt -r requirements-demo.txt
streamlit run demo_app.py
```

The demo loads sample homeowner submissions, lets you edit the JSON payload,
runs the workflow in-process, and shows the final decision packet, rationale,
reason codes, citations, premium indication, audit events, and follow-up
questions when more information is required.

## One-Command Walkthrough

Run the product walkthrough without starting a separate server:

```bash
python scripts/demo_walkthrough.py
```

The script uses FastAPI's in-process test client to exercise the real API
routes. It walks through a missing roof-age pause and same-run resume, then a
wildfire mitigation follow-up that moves into human review and approval.
The payloads live in `examples/demo_submissions.json` so the walkthrough is
separate from test fixtures.

Compare retrieval modes for a single query:

```bash
python scripts/compare_retrieval.py --query "high wildfire risk roof age referral"
```

The comparison CLI prints lexical, semantic, and hybrid results side by side
with source document, score, chunk ID, and snippet.

Compare chunking strategies (header-based vs fixed-size) on the same query set:

```bash
python scripts/compare_chunking.py --out docs/chunking_comparison.md
```

This ingests the corpus under each chunking strategy across all three retrieval
modes and reports chunk count, mean chunk size, hit@k, and MRR over a fixed
probe set. The lexical-vs-hybrid columns double as a reranking-impact view. The
generated report lives at [docs/chunking_comparison.md](docs/chunking_comparison.md).

Generate and run the labeled workflow eval harness:

```bash
python -m evals.generate_dataset
python -m evals.run --dataset evals/datasets/ho3_labeled.jsonl
```

The eval dataset contains 196 stratified HO3 submissions with expected
decisions, reason codes, workflow statuses, and gold citation chunk IDs. The
runner reports decision accuracy, reason-code exact match, retrieval recall@k,
citation faithfulness, and optional rationale quality.

Current CI-gated metrics:

| Metric | Value |
| --- | ---: |
| Labeled cases | 196 |
| Strata | 12 |
| Decision accuracy | 100% |
| Reason-code match | 100% |
| Retrieval recall@5 | 100% |
| Faithfulness | 100% |
| Product tests | 99 passing |

Faithfulness is a deterministic groundedness check: it verifies the decision
packet only cites chunks that were actually retrieved this run and only asserts
supporting facts the rules actually produced. It catches fabricated citations or
facts when the optional LLM wording path is enabled.

### LLM-as-judge calibrated against human labels

The deterministic metrics above are reproducible by construction — strong for
governance, but they do not test the one part of the system that is genuinely
stochastic: the LLM critic that judges whether a producer rationale is faithful
to its evidence (`workflows/critic.py`). That judge — **Claude
`claude-sonnet-4-6` by default, chosen independently of the generator to avoid
self-grading bias** — can be wrong, and it fails open. So we calibrate it
against a hand-labeled set rather than trusting it blindly.

[evals/judge_calibration.py](evals/judge_calibration.py) runs the judge over ~24
human-labeled rationales (balanced, with injected hallucinations and fabricated
citations) and reports judge↔human agreement, Cohen's kappa, and — the headline
— the **false-negative rate**, i.e. the unfaithful rationales the judge passed.

```bash
python -m evals.judge_calibration --record   # writes evals/reports/judge_calibration.md
```

| Metric | Value |
| --- | ---: |
| Labeled rationales | 24 |
| Agreement | 83.3% |
| Cohen's kappa | 0.667 |
| Recall (catch rate) | 83.3% |
| False-negative rate (fail-open) | 16.7% |

The numbers above were produced by a deterministic *simulated* stand-in judge
(this repo ships without API keys); the same harness recalibrates the real LLM
critic with `--backend llm`. How the simulated numbers were generated — and how
to regenerate against a hosted model — is documented in full in
[docs/judge_calibration_provenance.md](docs/judge_calibration_provenance.md).
See [docs/judge_calibration.md](docs/judge_calibration.md) for methodology.

Coverage breakdown:

| Outcome / status | Cases |
| --- | ---: |
| `ACCEPT` | 38 |
| `REFER` | 127 |
| `DECLINE` | 31 |
| `completed` | 38 |
| `pending_review` | 122 |
| `waiting_for_info` | 36 |

Run the same threshold gate used in CI:

```bash
python -m evals.run \
  --dataset evals/datasets/ho3_labeled.jsonl \
  --json \
  --min-decision-accuracy 1.0 \
  --min-reason-code-match 1.0 \
  --min-retrieval-recall 1.0 \
  --min-faithfulness 1.0
```

Exit code `0` means configured thresholds passed, `1` means metric thresholds
failed, and `2` means the dataset or eval run could not be loaded.

## Retrieval Config

Retrieval supports four modes plus an optional cross-encoder rerank stage.
Lexical is the default and the universal fallback, so the core workflow always
runs without network calls or model downloads.

```bash
RAG_RETRIEVAL_MODE=lexical|bm25|semantic|hybrid
RAG_EMBEDDINGS_ENABLED=true|false
EMBEDDING_MODEL=hashing-underwriting-v1
RAG_HYBRID_ALPHA=0.5          # semantic weight in RRF (0..1); BM25 gets 1 - alpha
RAG_RERANK_ENABLED=true|false # cross-encoder rerank of the top-N candidates
RAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RAG_RERANK_TOP_N=20
RAG_CHUNK_STRATEGY=header|fixed
```

- `lexical`: term-frequency scoring with a MUST/SHALL boost. Always available.
- `bm25`: Okapi BM25 over chunk tokens (`rank_bm25`). No embeddings required.
- `semantic`: cosine similarity over embeddings.
- `hybrid`: weighted **Reciprocal Rank Fusion** of the BM25 and semantic
  rankings, controlled by `RAG_HYBRID_ALPHA`.

When `RAG_RERANK_ENABLED=true`, retrieval pulls a `RAG_RERANK_TOP_N` candidate
pool in the chosen mode, then a cross-encoder rescores `(query, chunk)` pairs and
keeps the top-k. Reranked chunks carry `pre_rerank_rank` / `rerank_score` in
metadata so the reordering is visible in traces.

### Embedding providers — deterministic default, real model for production

`EMBEDDING_MODEL` selects one of three tiers. The default is deterministic *on
purpose*: it keeps CI and offline runs reproducible. Production should opt into a
real embedding model.

| `EMBEDDING_MODEL` | Use | Network/download |
|---|---|---|
| `hashing-underwriting-v1` (default) | Deterministic hash embeddings — CI, tests, offline reproducibility | none |
| `sentence-transformers:<model>` (e.g. `all-MiniLM-L6-v2`) | **Recommended for production** semantics | local model download |
| `nebius:<model>` (e.g. `BAAI/bge-en-icl`) | Hosted embeddings (OpenAI-compatible), needs `NEBIUS_API_KEY` | API call |

The hashing provider is a faithful *interface* stand-in but a weak retriever; a
real model measurably improves top-1 ranking. On the labeled eval, switching the
semantic retriever from hashing to `all-MiniLM-L6-v2` lifts `recall@1` from
**0.55 → 0.67** (see `docs/retrieval_eval.md`). If a provider is unavailable,
semantic/hybrid fall back to lexical (and `nebius:` falls back to hashing), so
runs never break.

Install the optional semantic/rerank stack with:

```bash
pip install -r requirements-rag.txt   # sentence-transformers (embeddings + cross-encoder)
```

Recommended production retrieval config:

```bash
RAG_RETRIEVAL_MODE=hybrid
RAG_EMBEDDINGS_ENABLED=true
EMBEDDING_MODEL=sentence-transformers:all-MiniLM-L6-v2
RAG_RERANK_ENABLED=true
```

`RAG_CHUNK_STRATEGY` switches between header-based chunking (default, keeps
related rules together) and fixed-size windows. Compare them with
`scripts/compare_chunking.py`.

### Nebius Token Factory

Nebius Token Factory backs at least one model call in this project (per the
cohort requirement), reachable for either embeddings or generation since it is
OpenAI-compatible:

```bash
# Embeddings through Nebius
RAG_EMBEDDINGS_ENABLED=true
EMBEDDING_MODEL=nebius:BAAI/bge-en-icl
NEBIUS_API_KEY=...

# Or producer-rationale generation through Nebius
LLM_STRUCTURED_OUTPUT_ENABLED=true
LLM_PROVIDER=nebius
LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct
NEBIUS_API_KEY=...
```

Both paths stay disabled by default for deterministic CI; deterministic
underwriting rules remain the governed decision layer regardless of provider.

## Configuration

Copy `.env.example` to `.env` for local configuration. The defaults run the
governed workflow with lexical retrieval, deterministic fallback wording, and
in-process trace recording. Enable semantic retrieval, provider-backed
structured LLM output, or OpenTelemetry export by changing the relevant
environment variables rather than editing workflow code.

## Structured LLM Output

LLM calls are intentionally narrow. Deterministic underwriting rules decide
`ACCEPT`, `REFER`, or `DECLINE`; the LLM service can only assist with
producer-facing rationale wording and missing-info follow-up wording after the
workflow has already identified the decision or the required fields.

```bash
LLM_STRUCTURED_OUTPUT_ENABLED=true|false
LLM_PROVIDER=openai|nebius|disabled
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=...
NEBIUS_API_KEY=...
LLM_PROMPT_VERSION=structured-llm-v1
```

Set `LLM_PROVIDER=nebius` to route the same structured calls through Nebius
Token Factory (OpenAI-compatible); it reuses the OpenAI SDK with a Nebius
`base_url` and `NEBIUS_API_KEY`.

Set `LLM_STRUCTURED_OUTPUT_ENABLED=true` with an API key to enable provider
calls. The `app/llm_service.py` provider wrapper sends prompt templates from
`app/prompt_templates.py` and validates responses against Pydantic models before
they enter the workflow state or decision packet. If no API key or provider is
available, the same deterministic fallback wording is validated and used.

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

The legacy `/quote/run` endpoint accepts a `QuoteSubmission` wrapped in a
`QuoteRunRequest` body (the `submission` object is required; `use_agentic` and
`additional_answers` are optional):

```bash
curl -X POST http://localhost:8000/quote/run \
  -H "Content-Type: application/json" \
  -d '{
    "submission": {
      "applicant_name": "Robert Johnson",
      "address": "789 Pine St, Los Angeles, CA 90001",
      "property_type": "single_family",
      "coverage_amount": 350000
    },
    "use_agentic": false
  }'
```

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
- `premium`: transparent premium indication

Use `/runs/{run_id}/audit` for the full workflow state, node outputs, required
questions, answer events, and final completion events.

## Engineering Notes

This repo is organized as a governed agentic workflow around deterministic
underwriting controls.

- **Why seven agents:** the workflow separates intake normalization, routing,
  enrichment, retrieval, underwriting assessment, verification, rating, and
  packaging so each step has a clear contract and can be tested or replaced
  independently. That mirrors the operating model of regulated underwriting:
  facts, evidence, decisioning, and review need distinct accountability.
- **Why deterministic rules first:** underwriting decisions are high-consequence
  and must be repeatable. The system uses deterministic rules and governed
  retrieval so referral reasons, citations, and tests are stable. An LLM can be
  added for question wording, document extraction, query formulation, or
  summarization without becoming the source of truth for eligibility.
- **LLM guardrails:** structured LLM output is constrained to two assistant
  tasks: producer rationale and follow-up wording. Provider responses must pass
  Pydantic validation, cannot alter question identifiers or answer contracts,
  and never set the eligibility outcome.
- **Auditability:** every run has a durable `run_id`; missing-info pauses,
  follow-up answers, review actions, node outputs, decision packets, and final
  outcomes are stored with the run. Human review decisions are recorded
  separately from the system recommendation so the audit trail does not rewrite
  the original decision.
- **Reliability boundaries:** validation and routing happen before strict HO3
  model construction so incomplete submissions can pause cleanly instead of
  failing. Referral and decline decisions require retrieved citations before the
  decision packet is finalized. Persistence is abstracted behind the storage
  layer so SQLite can be replaced by Postgres without changing workflow
  contracts.
- **Reliability and extension points:** hazard enrichment, retrieval,
  traceability, HITL assignment, auth, rate limits, idempotency, and PII
  redaction are intentionally separated behind modules that can be hardened
  independently as the application moves toward production deployment.
- **Production extension path:** replace deterministic enrichment with provider
  integrations, move persistence to Postgres, add idempotency and auth, persist
  OpenTelemetry traces, introduce queue-backed HITL tasks, and add LLM/tool
  calls behind the existing agent contracts for extraction, evidence gathering,
  and producer-facing explanations. Keep deterministic rule evaluation as the
  governed decision layer.

## Roadmap

- Add production-grade embedding providers and LLM-as-judge rationale evals
  alongside the existing deterministic retrieval and workflow metrics.
- Explore constrained LLM tool orchestration for non-binding assistance, while
  keeping deterministic rules as the governed eligibility decision layer.

## Scenario Coverage

The one-command walkthrough uses `examples/demo_submissions.json`. The product
tests also use curated scenarios in `tests/demo_scenarios.py`, including:

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
explicit rule triggers, lexical and semantic RAG behavior, fallback retrieval,
eval harness behavior, rating sanity checks, and end-to-end underwriting
scenarios.

## Traceability

`observability.py` provides a tracer abstraction with completed span records,
attributes, status, timing, and exporter boundaries. The default `memory`
backend records spans in-process for tests and audit inspection; `logging`
emits structured span records to application logs; `otel` routes spans through
OpenTelemetry when the host environment provides the SDK configuration.
Decision packets include trace references so workflow output can be joined to
span data.

On top of tracing, request-level **quality metrics** (latency p50/p95, failure
rate, citation coverage, LLM usage, and cost) are recorded per run and served at
`GET /metrics` (and embedded in `GET /stats`). An optional Langfuse sink mirrors
them when configured, and is a graceful no-op otherwise so CI stays hermetic.
See `docs/observability.md`.

## Production Hardening Roadmap

The next production hardening steps are clear and modular: connect external
hazard, claims, geocoding, and RCE providers; move persistence to Postgres; add
auth, idempotency, rate limits, and PII redaction; persist distributed traces;
and introduce queue-backed HITL assignment with SLA tracking.
