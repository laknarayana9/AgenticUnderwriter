# Agentic Underwriter — CTO Demo Walkthrough

End-to-end verification of every major feature. Run these steps in order. Each section is independent and can be shown on its own.

**Prerequisites:**
```bash
pip install -r requirements-demo.txt   # includes langsmith, streamlit
cp .env.example .env                   # fill in OPENAI_API_KEY (and optionally LANGSMITH_API_KEY)
```

---

## 1. Unit + Product Test Suite (CI gate)

*Shows: 116 passing product tests, fully deterministic, no LLM calls needed.*

```bash
pytest tests/ -q
```

**Expected output:**
```
116 passed in Xs
```

---

## 2. Core Workflow — Accept Path

*Shows: 7-step deterministic agent pipeline producing a cited ACCEPT decision.*

```bash
# Start the API server
uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```bash
curl -s -X POST http://localhost:8000/quote/ho3 \
  -H "Content-Type: application/json" \
  -d '{
    "submission": {
      "applicant": {"full_name": "Alex Kim", "email": "alex@example.com", "phone": "+1-555-100-0001"},
      "risk": {
        "property_address": "100 Pine St, Palo Alto, CA 94301",
        "occupancy": "owner_occupied_primary",
        "dwelling_type": "single_family",
        "year_built": 2005,
        "roof_age_years": 3,
        "construction_type": "frame",
        "stories": 1
      },
      "coverage_request": {
        "coverage_a": 500000, "coverage_b_pct": 10, "coverage_c_pct": 50,
        "coverage_d_pct": 20, "coverage_e": 300000, "coverage_f": 5000, "deductible": 1000
      }
    }
  }' | python3 -m json.tool
```

**Look for:**
- `"decision": "ACCEPT"`
- `"citations"` — list of guideline chunks that grounded the decision
- `"producer_rationale"` — LLM-generated, PII-free explanation
- `"run_id"` — save this for steps 3 and 4

---

## 3. Missing-Info Loop + Same-Run Resume

*Shows: workflow pauses for missing data, resumes on the same run ID with audit trail intact.*

**Step 1 — submit with missing roof age (copy the `run_id` from the response):**

```bash
curl -s -X POST http://localhost:8000/quote/ho3 \
  -H "Content-Type: application/json" \
  -d '{
    "submission": {
      "applicant": {"full_name": "Sam Lee", "email": "sam@example.com", "phone": "+1-555-200-0001"},
      "risk": {
        "property_address": "200 Elm St, San Jose, CA 95101",
        "occupancy": "owner_occupied_primary",
        "dwelling_type": "single_family",
        "year_built": 1998,
        "construction_type": "frame",
        "stories": 2
      },
      "coverage_request": {
        "coverage_a": 450000, "coverage_b_pct": 10, "coverage_c_pct": 50,
        "coverage_d_pct": 20, "coverage_e": 300000, "coverage_f": 5000, "deductible": 1000
      }
    }
  }' | python3 -m json.tool
```

**Step 2 — check status (paste your `run_id` in place of `<RUN_ID>`):**

```bash
curl -s http://localhost:8000/runs/<RUN_ID> | python3 -m json.tool
```

**Step 3 — resume with the missing answer:**

```bash
curl -s -X POST http://localhost:8000/runs/<RUN_ID>/answers \
  -H "Content-Type: application/json" \
  -d '{"answers": {"roof_age_years": 5}}' | python3 -m json.tool
```

**Look for:**
- Step 1 response: `"status": "waiting_for_info"` and `"follow_up_questions"` in the packet
- Step 3 response: completed decision with `"status": "completed"`

---

## 4. Audit Trail + HITL Review Queue

*Shows: every action is logged; referrals route to a human review queue.*

**Step 1 — write the wildfire payload to a temp file, then POST it:**

```bash
python3 -c "
import json
d = json.load(open('examples/demo_submissions.json'))
print(json.dumps(d['wildfire_high'], indent=2))
" > /tmp/wildfire.json

curl -s -X POST http://localhost:8000/quote/ho3 \
  -H "Content-Type: application/json" \
  -d @/tmp/wildfire.json | python3 -m json.tool
```

```bash
# View pending human review queue
curl -s http://localhost:8000/reviews/pending | python3 -m json.tool

# Pull audit trail for the run
# Replace <RUN_ID> with the run_id from the REFER response above
curl -s http://localhost:8000/runs/<RUN_ID>/audit | python3 -m json.tool
```

**Look for:**
- Decision `"REFER"` or `"DECLINE"` with reason codes
- `/reviews/pending` shows the queued case
- `/audit` shows every agent step with timestamps

---

## 5. Observability — Metrics + Streaming Monitor

*Shows: SRE-grade request metrics, latency budget, real-time anomaly detection.*

```bash
# Aggregate metrics across all runs so far
curl -s http://localhost:8000/metrics | python3 -m json.tool

# Streaming monitor summary (latency p50/p95, citation coverage, cost)
curl -s http://localhost:8000/monitor/summary | python3 -m json.tool

# Anomalies detected
curl -s http://localhost:8000/monitor/anomalies | python3 -m json.tool

# Per-stage latency budget for a specific run
curl -s http://localhost:8000/runs/<RUN_ID>/latency-budget | python3 -m json.tool
```

**Look for:**
- `latency_p50_ms`, `latency_p95_ms`, `citation_coverage`, `total_cost_usd`
- `latency_budget` breaking down time per agent stage

---

## 6. Retrieval — BM25 + Hybrid Reranking

*Shows: lexical, hybrid, and cross-encoder reranking modes for guideline retrieval.*

```bash
# Compare retrieval strategies on the same query
python3 scripts/compare_retrieval.py \
  --query "roof age eligibility requirements for homeowners policy" \
  --limit 5 --no-embeddings
```

**Look for:** side-by-side table of lexical vs BM25 vs RRF hybrid vs cross-encoder recall scores.

---

## 7. CI-Gated Eval Harness (206-case golden set)

*Shows: deterministic evaluation across 206 stratified HO3 cases with gated thresholds.*

```bash
python3 evals/run.py \
  --dataset evals/datasets/ho3_labeled.jsonl \
  --min-decision-accuracy 1.0 \
  --min-reason-code-match 0.95 \
  --min-retrieval-recall 0.75
```

**Expected output:**
```
HO3 Evaluation Report
=====================
cases: 206
decision_accuracy: 1.000
reason_code_match: 1.000
retrieval_recall@5: 1.000
faithfulness: 1.000
No eval failures.
```

---

## 8. LangSmith Tracing + Dataset + Evaluators

*Shows: experiment tracking, versioned dataset, three registered evaluators, comparison view.*

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_API_KEY=<your-key>
export LANGSMITH_PROJECT=AgenticUnderwriter

# Upload 206-case golden set as a versioned LangSmith dataset (idempotent)
python3 evals/langsmith_eval.py upload-dataset

# Run baseline experiment — streams per-case scores, prints experiment link
python3 evals/langsmith_eval.py run-eval --experiment-prefix baseline
```

**Look for:**
- Dataset URL → open in browser to see 206 examples with inputs/outputs
- Experiment URL → three score columns: `decision_accuracy`, `retrieval_recall@5`, `faithfulness`
- Run a second experiment (`--experiment-prefix improved`) to get the comparison view

---

## 9. Vision Intake — Photo → Submission Enrichment

*Shows: property photo flows into the workflow via GPT-4o vision; confident attributes fold into the submission; low-confidence fields pause for follow-up.*

```bash
# Contrast demo: same submission, with vs without a photo (fake provider, no API key needed)
python3 scripts/vision_workflow_demo.py --demo-contrast
```

**Look for:**
- Run 1 (no photo): `waiting_for_info` because roof age is unknown
- Run 2 (photo provided): vision evidence fills in roof condition → workflow proceeds to decision

**With a real photo — cloud (OpenAI GPT-4o):**
```bash
# Requires OPENAI_API_KEY + VISION_PROVIDER=openai (default)
python3 scripts/vision_workflow_demo.py --image /path/to/property_photo.jpg
```

**With a real photo — fully local / on-device (Ollama, no egress, no API key):**
```bash
# Requires Ollama running locally with llama3.2-vision pulled
ollama pull llama3.2-vision
VISION_PROVIDER=ollama python3 scripts/vision_workflow_demo.py --image /path/to/property_photo.jpg
```

**Look for:** `vision_evidence_applied` event in the audit trail, extracted attributes (roof condition, stories, construction type) folded into the submission before the rules run. The Ollama path keeps property photos fully on-device — the privacy-preserving option for sensitive PII.

**API endpoint (multipart):**
```bash
curl -s -X POST http://localhost:8000/quote/ho3/with-photo \
  -F 'submission={"submission":{"applicant":{"full_name":"Alex Kim","email":"alex@example.com","phone":"+1-555-100-0001"},"risk":{"property_address":"100 Pine St, Palo Alto, CA 94301","occupancy":"owner_occupied_primary","dwelling_type":"single_family","year_built":2005,"construction_type":"frame","stories":1},"coverage_request":{"coverage_a":500000,"coverage_b_pct":10,"coverage_c_pct":50,"coverage_d_pct":20,"coverage_e":300000,"coverage_f":5000,"deductible":1000}}};type=application/json' \
  -F 'photo=@/path/to/photo.jpg' | python3 -m json.tool
```

---

## 10. LLM-as-Judge Calibration (Critic Faithfulness)

*Shows: the generator-critic loop's faithfulness judge calibrated against 24 human-labeled rationales — agreement, Cohen's kappa, and false-negative rate.*

```bash
# Run calibration with the deterministic simulated stand-in (no API key needed)
python3 evals/judge_calibration.py \
  --backend simulated \
  --dataset evals/datasets/judge_calibration.jsonl
```

**Expected output:**
```
agreement:      0.xxx
cohens_kappa:   0.xxx
false_neg_rate: 0.xxx   ← fail-open rate (judge misses a bad rationale)
```

**With a real LLM judge (uses `ANTHROPIC_API_KEY`, Claude grades independently of the generator):**
```bash
python3 evals/judge_calibration.py \
  --backend llm \
  --dataset evals/datasets/judge_calibration.jsonl
```

**Look for:**
- `cohens_kappa > 0.6` — substantial judge/human agreement
- `false_neg_rate < 0.2` — judge rarely lets bad rationales through
- The critic defaults to Claude (`CRITIC_LLM_PROVIDER=claude`) independent of the generator to avoid self-grading bias

---

## 11. Fine-Tune Track (Nebius Token Factory)

*Shows: LoRA extraction workflow generating training data and submitting a fine-tune job.*

```bash
# View the fine-tune pipeline demo (dry-run, no API key needed)
python3 scripts/extraction_workflow_demo.py
```

**Look for:** structured extraction → JSONL training pairs → fine-tune submission steps.

---

## 12. Streamlit Interactive Demo

*Shows: full UI — edit a submission, run the workflow, view citations and audit trail.*

```bash
streamlit run demo_app.py
```

Open `http://localhost:8501` in a browser.

**Walk through:**
1. Edit the submission fields in the left panel
2. Click **Run Underwriting** 
3. View the decision, confidence, and reason codes
4. Expand **Citations** to see which guideline chunks grounded the decision
5. Expand **Audit Trail** to see the per-agent step log

---

## Quick Reference — What Each Section Proves

| # | Feature | Proof point |
|---|---------|-------------|
| 1 | Test suite | 116 passing product tests, CI-ready |
| 2 | Core workflow | 7-agent pipeline → cited ACCEPT/REFER/DECLINE |
| 3 | Missing-info loop | Pause → resume on same run ID with audit |
| 4 | HITL review queue | High-risk cases routed to human review |
| 5 | Observability | Latency budget, cost, citation coverage, anomaly detection |
| 6 | Retrieval | BM25 + RRF hybrid + cross-encoder reranking |
| 7 | CI eval harness | 206-case golden set, gated thresholds |
| 8 | LangSmith | Tracing, versioned dataset, registered evaluators, comparison view |
| 9 | Vision intake | Property photo → GPT-4o or local Ollama → submission enrichment via `/quote/ho3/with-photo` |
| 10 | LLM-as-judge | Critic calibrated against human labels, Cohen's kappa, fail-open rate |
| 11 | Fine-tuning | LoRA extraction pipeline on Nebius Token Factory |
| 12 | Streamlit UI | End-to-end interactive demo |
