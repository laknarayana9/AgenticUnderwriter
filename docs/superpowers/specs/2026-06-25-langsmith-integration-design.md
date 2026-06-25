# LangSmith Integration Design

**Date:** 2026-06-25  
**Status:** Approved

## Goal

Add LangSmith as a configurable, opt-in tracing and evaluation layer. Disabled by default (commercial). Produces a shareable experiment link with baseline→improvement comparison view for portfolio/cert purposes and closes the Week-4 assignment gap.

## Scope

- Configurable tracing of `UnderwritingWorkflow.run` via `@traceable`
- Upload of the 206-case golden dataset to LangSmith
- Three evaluators ported to `client.evaluate`: decision_accuracy, retrieval_recall@5, faithfulness
- CLI to run both upload and evaluation

Explicitly out of scope: modifying `evals/run.py` (CI harness stays independent), changing `observability.py`.

---

## Architecture

### Single module: `evals/langsmith_eval.py`

Mirrors the structure of `evals/run.py`. Contains three concerns:

#### 1. Traceable workflow wrapper

```python
from langsmith import traceable

@traceable(name="underwriting_workflow")
def run_workflow(submission: dict) -> dict:
    workflow = UnderwritingWorkflow()
    state = workflow.run(submission)
    packet = state.decision_packet
    return {
        "decision": packet.decision.value if packet else None,
        "reason_codes": packet.review_reason_codes if packet else [],
        "citations": [c.get("chunk_id") for c in (packet.citations or []) if isinstance(c, dict)],
        "status": state.status,
        "retrieved_chunks": (state.retrieval or {}).get("retrieved_chunks", []),
    }
```

LangSmith SDK auto-traces when `LANGSMITH_TRACING=true`; is a no-op otherwise. No custom guard needed.

#### 2. Dataset upload (`upload-dataset` command)

- Reads `evals/datasets/ho3_labeled.jsonl`
- Checks whether dataset `"ho3-golden-206"` already exists; skips creation if so (idempotent)
- Pushes examples with `inputs={"submission": case.submission}` and `outputs={"decision": ..., "reason_codes": ..., "gold_citations": ...}`
- Prints dataset URL on completion

#### 3. Evaluation runner (`run-eval` command)

Calls:
```python
client.evaluate(
    run_workflow,
    data="ho3-golden-206",
    evaluators=[eval_decision_accuracy, eval_retrieval_recall, eval_faithfulness],
    experiment_prefix=args.experiment_prefix,
    max_concurrency=1,  # avoid rate limits on free tier
)
```

**Evaluators** — each takes `(outputs: dict, reference_outputs: dict) -> dict`:

- `eval_decision_accuracy`: `score = 1.0 if outputs["decision"] == reference_outputs["decision"] else 0.0`
- `eval_retrieval_recall`: recall of `reference_outputs["gold_citations"]` in `outputs["citations"][:5]`; returns `key="retrieval_recall@5"`
- `eval_faithfulness`: reuses the deterministic logic from `evals/run.py` — every cited chunk_id must appear in `retrieved_chunks`; vacuously 1.0 when no citations

Prints experiment URL on completion.

---

## Env Vars

Added to `.env.example` (all default to disabled):

```
# LangSmith (optional, off by default)
LANGSMITH_TRACING=false
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=AgenticUnderwriter
```

---

## CLI Interface

```bash
# One-time: push golden set to LangSmith
python evals/langsmith_eval.py upload-dataset

# Run experiment (repeat with different model/prompt to get comparison view)
python evals/langsmith_eval.py run-eval --experiment-prefix baseline
python evals/langsmith_eval.py run-eval --experiment-prefix improved
```

Both commands require `LANGSMITH_API_KEY` to be set; they fail fast with a clear error if it isn't.

---

## Dependency

Add `langsmith` to `requirements-demo.txt` (already the heavier optional deps file). Not added to a core requirements file — keeps the default install free of commercial SDKs.

---

## Testing

No new product tests needed. The existing `tests/product/test_eval_harness.py` covers the logic being reused. Manual smoke test: run both CLI commands with real credentials, verify dataset and experiment appear in the LangSmith UI.

---

## What This Unlocks

- LangSmith tracing keyword on résumé
- Shareable experiment link with per-case scores
- Baseline vs. improved run comparison view (run `run-eval` twice with different prefixes)
- Versioned dataset in LangSmith UI
