"""Human-readable anomaly explanations with graceful degradation.

Tries an LLM explanation under a hard timeout; on timeout, error, or no provider,
falls back to a deterministic template. The monitoring path must never hang or
fail because an optional LLM is slow or down — that is the resilience point.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Optional

from streaming.stream_monitor import Anomaly

logger = logging.getLogger(__name__)

_DETERMINISTIC = {
    "failure_rate": (
        "Failure rate {value} exceeds the {threshold} budget. Recent runs are "
        "erroring before producing a decision — check the workflow error events."
    ),
    "latency_p95": (
        "p95 latency {value} ms exceeds the {threshold} ms budget. Inspect the "
        "per-run latency budget to find the slowest stage."
    ),
    "citation_coverage": (
        "Adverse-decision citation coverage {value} is below the required "
        "{threshold}. A REFER/DECLINE shipped without guideline evidence — the "
        "verifier guardrail should be reviewed immediately."
    ),
}


def _deterministic_explanation(anomaly: Anomaly) -> str:
    template = _DETERMINISTIC.get(
        anomaly.kind, "{message} (value {value}, threshold {threshold})"
    )
    return template.format(value=anomaly.value, threshold=anomaly.threshold, message=anomaly.message)


def explain_anomaly(anomaly: Anomaly, llm_service: Optional[object] = None, timeout_s: float = 3.0) -> dict:
    """Return {'explanation', 'source'} for an anomaly.

    source is 'llm' when a provider produced it within the timeout, else
    'deterministic'. Always returns quickly and never raises.
    """
    fallback = _deterministic_explanation(anomaly)
    provider = getattr(llm_service, "provider", None) if llm_service else None
    if provider is None:
        return {"explanation": fallback, "source": "deterministic"}

    def _call() -> str:
        from app.prompt_templates import PRODUCER_RATIONALE_SYSTEM_PROMPT  # reuse a terse system prompt
        raw = provider.generate_json(
            system_prompt="You are an SRE assistant. Explain the anomaly in one sentence for an on-call engineer.",
            user_prompt=(
                f"Anomaly: {anomaly.kind}. {anomaly.message} "
                f"value={anomaly.value} threshold={anomaly.threshold}. "
                'Reply as JSON {"explanation": "<one sentence>"}.'
            ),
            schema={"type": "object", "properties": {"explanation": {"type": "string"}}, "required": ["explanation"]},
        )
        text = raw.get("explanation") if isinstance(raw, dict) else None
        return text or fallback

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            explanation = pool.submit(_call).result(timeout=timeout_s)
        return {"explanation": explanation, "source": "llm"}
    except (FuturesTimeout, Exception) as exc:  # timeout OR any provider error → degrade
        logger.info("anomaly explanation fell back to deterministic: %s", exc)
        return {"explanation": fallback, "source": "deterministic"}
