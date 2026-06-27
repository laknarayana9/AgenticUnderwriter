"""LangGraph engine tests: decision parity with the native engine and durable
human-in-the-loop pause/resume across a simulated process restart.

Hermetic — deterministic rules drive decisions, LLM disabled (fallback rationale),
no network. A temp SQLite checkpoint file backs the durable-resume test.
"""

import pytest

pytest.importorskip("langgraph")

from workflows.agent_workflow import UnderwritingWorkflow
from workflows.langgraph_workflow import LangGraphUnderwritingWorkflow


def _sub(**risk):
    base = {
        "property_address": "120 Market St, Palo Alto, CA 94301",
        "occupancy": "owner_occupied_primary", "dwelling_type": "single_family",
        "year_built": 2005, "roof_age_years": 8, "construction_type": "frame", "stories": 1,
    }
    base.update(risk)
    return {"applicant": {"full_name": "Avery Chen"}, "risk": base,
            "coverage_request": {"coverage_a": 500000, "deductible": 1000}}


@pytest.fixture(scope="module")
def native():
    return UnderwritingWorkflow()


@pytest.fixture
def lg(native, tmp_path):
    return LangGraphUnderwritingWorkflow(native=native, checkpoint_db=str(tmp_path / "ckpt.sqlite"))


@pytest.mark.parametrize("submission,label", [
    (_sub(), "accept_clean"),
    (_sub(roof_age_years=25), "roof_referral"),
    (_sub(dwelling_type="commercial"), "commercial_decline"),
])
def test_decision_parity_with_native(native, lg, submission, label):
    native_state = native.run(submission)
    native_decision = native_state.decision_packet.decision.value

    lg_result = lg.run(submission)
    assert lg_result["interrupted"] is False, label
    assert lg_result["decision"] == native_decision, f"{label}: {lg_result['decision']} != {native_decision}"


def test_clean_submission_completes_accept(lg):
    result = lg.run(_sub())
    assert result["status"] == "completed"
    assert result["decision"] == "ACCEPT"
    assert any(e["event"] == "workflow_completed" for e in result["events"])


def test_missing_roof_age_interrupts(lg):
    result = lg.run(_sub(roof_age_years=None))
    assert result["interrupted"] is True
    assert result["status"] == "waiting_for_info"
    # the durable interrupt surfaces the question to answer
    assert any("roof" in (q.get("question", "") + q.get("question_text", "")).lower()
               for q in result["questions"])


def test_langchain_provider_config_and_graceful_no_key():
    from app.llm_service import LLMServiceConfig, StructuredLLMService
    cfg = LLMServiceConfig(enabled=True, provider="langchain", model="gpt-4o-mini", api_key="")
    # No key -> provider unavailable -> None (graceful), wording falls back.
    assert StructuredLLMService(config=cfg).provider is None


def test_langchain_provider_registered(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "langchain")
    from app.llm_service import LLMServiceConfig
    cfg = LLMServiceConfig.from_env()
    assert cfg.provider == "langchain"
    assert cfg.model == "gpt-4o-mini"


def test_langgraph_endpoints_run_and_resume():
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    # Clean submission -> completes ACCEPT via the LangGraph engine.
    clean = {"submission": {"applicant": {"full_name": "Avery Chen"},
             "risk": {"property_address": "120 Market St, Palo Alto, CA 94301",
                      "occupancy": "owner_occupied_primary", "dwelling_type": "single_family",
                      "year_built": 2005, "roof_age_years": 8, "construction_type": "frame", "stories": 1},
             "coverage_request": {"coverage_a": 500000, "deductible": 1000}}}
    r = client.post("/quote/ho3/langgraph", json=clean)
    assert r.status_code == 200
    assert r.json()["decision"] == "ACCEPT"

    # Missing roof age -> interrupt -> resume by thread_id completes.
    missing = {"submission": {"applicant": {"full_name": "Avery Chen"},
               "risk": {"property_address": "120 Market St, Palo Alto, CA 94301",
                        "occupancy": "owner_occupied_primary", "dwelling_type": "single_family",
                        "year_built": 2005, "construction_type": "frame", "stories": 1},
               "coverage_request": {"coverage_a": 500000, "deductible": 1000}}}
    r1 = client.post("/quote/ho3/langgraph", json=missing).json()
    assert r1["interrupted"] is True
    tid = r1["thread_id"]
    r2 = client.post(f"/quote/ho3/langgraph/{tid}/resume", json={"answers": {"roof_age_years": 9}}).json()
    assert r2["interrupted"] is False
    assert r2["status"] == "completed"


def test_dual_engine_dataset_parity(native, tmp_path):
    """The anti-drift guarantee: every labeled eval case yields the SAME outcome
    on both engines — same decision when it completes, and both pause on the same
    missing-info cases. This is the dataset-level parity behind the dual-engine
    claim (decisions come from the shared deterministic rules, so they can't
    diverge)."""
    from pathlib import Path
    from evals.run import load_dataset

    cases = load_dataset(Path("evals/datasets/ho3_labeled.jsonl"))
    assert cases, "eval dataset is empty"
    lg = LangGraphUnderwritingWorkflow(native=native, checkpoint_db=str(tmp_path / "parity.sqlite"))

    mismatches = []
    for case in cases:
        cid = getattr(case, "id", "?")
        native_state = native.run(case.submission)
        lg_result = lg.run(case.submission)

        if native_state.status == "waiting_for_info":
            if not lg_result["interrupted"]:
                mismatches.append((cid, "native paused; langgraph did not"))
        else:
            native_decision = native_state.decision_packet.decision.value if native_state.decision_packet else None
            if lg_result["interrupted"] or lg_result["decision"] != native_decision:
                mismatches.append((cid, f"native={native_decision} langgraph={lg_result.get('decision')} "
                                        f"interrupted={lg_result['interrupted']}"))

    assert not mismatches, f"{len(mismatches)}/{len(cases)} parity mismatches (first 5): {mismatches[:5]}"


def test_durable_resume_across_new_instance(native, tmp_path):
    """A run paused in one engine instance resumes from a fresh instance on the
    same checkpoint DB — proving durable, cross-process HITL."""
    db = str(tmp_path / "durable.sqlite")

    engine1 = LangGraphUnderwritingWorkflow(native=native, checkpoint_db=db)
    paused = engine1.run(_sub(roof_age_years=None))
    assert paused["interrupted"] is True
    thread_id = paused["thread_id"]
    del engine1  # simulate process restart

    engine2 = LangGraphUnderwritingWorkflow(native=native, checkpoint_db=db)
    resumed = engine2.resume(thread_id, {"roof_age_years": 9})
    assert resumed["interrupted"] is False
    assert resumed["status"] == "completed"
    assert resumed["decision"] == "ACCEPT"  # roof 9 is clean
