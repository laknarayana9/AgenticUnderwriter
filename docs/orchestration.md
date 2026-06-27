# Orchestration: native state machine + LangGraph engine

The HO3 workflow runs on **two interchangeable orchestration engines** that
produce **identical decisions**:

1. **Native** (`workflows/agent_workflow.py`, default) — an explicit, hand-rolled
   state machine. The governed baseline: minimal dependencies, fully inspectable,
   DB-integrated (run records, HITL review queue).
2. **LangGraph** (`workflows/langgraph_workflow.py`) — the same workflow as a
   `StateGraph`, adding idiomatic graph orchestration and **durable
   human-in-the-loop pause/resume** via `interrupt()` + a SQLite checkpointer.

Both call the **same governed components** — `underwriting_rules`, the agents,
`RAGEngine`, the vision service, and the shared generator–critic loop
(`UnderwritingWorkflow.package_decision_with_critic`). There is **no logic
duplication**: the LangGraph nodes are thin wrappers over the native components,
and the ACCEPT/REFER/DECLINE decision is produced by the deterministic rules in
the assess node. ADR-0001 holds on both engines.

## Why two engines

It's a deliberate demonstration, not indecision. The native explicit state
machine is the right governed default (determinism, auditability, minimal deps —
see ADR-0001). The LangGraph engine shows the same workflow expressed in the
framework, and earns its keep on one axis the native engine handles manually:
**durable, checkpointed pause/resume**.

> **Note:** the LangGraph engine is still an *explicit graph with fixed edges* —
> the LLM is **not** the planner choosing the next step. This is exactly the
> distinction ADR-0001 draws when it rejects "LLM as orchestrator": a bounded
> workflow deserves an explicit graph, not an agentic ReAct loop. LangGraph here
> is the explicit-graph runtime, not an autonomous agent.

## Durable human-in-the-loop

On a missing-info pause, the LangGraph node calls `interrupt()`; the run state is
persisted to the SQLite checkpointer keyed by `thread_id`. A paused run **survives
a process restart** — a fresh engine instance on the same checkpoint DB resumes
from `Command(resume=answers)`. This is verified end-to-end in
`tests/product/test_langgraph_workflow.py::test_durable_resume_across_new_instance`.

```bash
# run -> may pause with interrupted=true + a thread_id
curl -X POST localhost:8000/quote/ho3/langgraph -H 'content-type: application/json' -d '{"submission": {...}}'
# resume later (even after a restart) by thread_id
curl -X POST localhost:8000/quote/ho3/langgraph/<thread_id>/resume -H 'content-type: application/json' -d '{"answers": {"roof_age_years": 9}}'
```

Config: `LANGGRAPH_CHECKPOINT_DB` (default `storage/langgraph_checkpoints.sqlite`,
gitignored). The native engine remains the default for the primary routes
(`/quote/ho3`), which are DB-integrated with the review queue.

## LangChain in the LLM path

`LLM_PROVIDER=langchain` selects a `LangChainOpenAIProvider`
(`app/providers/langchain_provider.py`) built on `langchain_openai.ChatOpenAI`.
It implements the same `StructuredJSONProvider` interface as every other adapter,
so it plugs **inside** `StructuredLLMService` — PII masking, the generator–critic
loop, and the deterministic fallback all still wrap it. LangChain is genuinely
used for structured generation without bypassing any governance.

## Design notes (LangGraph 1.x)

- Lean `TypedDict` state with `add` reducers for accumulators (events, verdicts);
  state is JSON-able so it checkpoints cleanly (the decision packet is stored as a
  dict).
- Nodes return partial updates and don't mutate inputs.
- Nodes around `interrupt()` are idempotent (a node re-runs from the top on
  resume; `interrupt()` then returns the resume value instead of pausing again).
- LangGraph auto-traces to LangSmith when `LANGSMITH_TRACING=true`, complementing
  `evals/langsmith_eval.py`.

## Parity guarantee

`test_langgraph_workflow.py` asserts the LangGraph engine yields the **same
decision** as native across accept / refer / decline cases. Because decisions come
from the shared deterministic rules, the two engines cannot diverge on outcomes —
the parity tests are the guard against drift.
