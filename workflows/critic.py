"""Critic agent for the generator-critic rationale verification loop.

The critic uses a separate LLM provider (configured via CRITIC_LLM_PROVIDER /
CRITIC_LLM_MODEL) to verify that a ProducerRationaleOutput is grounded in the
retrieved evidence before it is released. This avoids self-grading bias.

Flow:
  1. Deterministic pre-check — any citation_chunk_id not present in
     retrieved_chunks fails immediately without an LLM call (fast + cheap).
  2. LLM faithfulness check — the critic is given the rationale summary,
     the supporting_facts, and the retrieved evidence excerpts and asked
     whether every claim is supported by the provided context.
  3. Returns a CriticVerdict with structured feedback for the generator to use
     on the next attempt (if retries remain).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from app.llm_service import LLMServiceConfig, LLMUnavailable, StructuredLLMService
from app.pii_masker import MaskMap, PIIMasker
from models.schemas import CriticVerdict, ProducerRationaleOutput

logger = logging.getLogger(__name__)

_CRITIC_SYSTEM_PROMPT = """
You are a strict faithfulness reviewer for insurance underwriting rationales.

Your job: given a producer rationale and the retrieved underwriting-guideline
excerpts that were used to produce it, determine whether every claim in the
rationale is directly supported by those excerpts.

Rules:
- A claim is "supported" if the text of at least one retrieved excerpt
  clearly backs it up.
- A claim is "unsupported" if you cannot find backing in the provided excerpts,
  even if you believe it is generally true.
- Do not use outside knowledge — only the provided context.
- Return JSON only, matching the supplied schema.
""".strip()

_CRITIC_USER_TEMPLATE = """
--- PRODUCER RATIONALE ---
Summary: {summary}
Supporting facts: {supporting_facts}
Cited chunk IDs: {cited_ids}

--- RETRIEVED EVIDENCE EXCERPTS ---
{evidence}

Task: Identify any claims in the summary or supporting_facts that are NOT
supported by the evidence excerpts above.

Return a JSON object with:
  "passed": true if all claims are supported, false otherwise
  "unsupported_facts": list of specific unsupported claim strings (empty if passed)
  "feedback_for_generator": a concise instruction string for the generator to
    improve the rationale on the next attempt (empty string if passed)
""".strip()


class _CriticResponseModel:
    """Lightweight schema for the critic's JSON output (not a Pydantic model
    so we avoid a schema-round-trip dependency)."""
    schema = {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean"},
            "unsupported_facts": {"type": "array", "items": {"type": "string"}},
            "feedback_for_generator": {"type": "string"},
        },
        "required": ["passed", "unsupported_facts", "feedback_for_generator"],
    }


class CriticAgent:
    """Verifies ProducerRationaleOutput faithfulness against retrieved evidence."""

    def __init__(self) -> None:
        # Default the judge to Claude, independent of the generator's provider,
        # so the critic does not grade output produced by the same model
        # (self-grading bias). Override with CRITIC_LLM_PROVIDER / CRITIC_LLM_MODEL.
        critic_provider = os.getenv("CRITIC_LLM_PROVIDER", "claude").strip().lower()
        critic_model = os.getenv("CRITIC_LLM_MODEL", "claude-sonnet-4-6").strip()

        config = LLMServiceConfig(
            enabled=bool(os.getenv("LLM_STRUCTURED_OUTPUT_ENABLED", "false").lower() in {"1", "true", "yes"}),
            provider=critic_provider,
            model=critic_model,
            api_key=self._api_key_for(critic_provider),
            base_url=os.getenv("OLLAMA_BASE_URL") if critic_provider == "ollama" else None,
        )
        self._service = StructuredLLMService(config=config)

    @staticmethod
    def _api_key_for(provider: str) -> str:
        mapping = {
            "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "nebius": "NEBIUS_API_KEY",
            "ollama": "",
        }
        env_var = mapping.get(provider, "OPENAI_API_KEY")
        return os.getenv(env_var, "") if env_var else ""

    def verify_rationale(
        self,
        rationale: ProducerRationaleOutput,
        retrieved_chunks: List[Dict[str, Any]],
        facts_used: Dict[str, Any],
        attempt: int = 0,
        mask_map: Optional[MaskMap] = None,
    ) -> CriticVerdict:
        """Run deterministic pre-check then (if available) LLM faithfulness check."""
        # --- 1. Deterministic citation check (no LLM call) ---
        chunk_ids = {
            chunk.get("chunk_id") or chunk.get("id") or ""
            for chunk in retrieved_chunks
            if isinstance(chunk, dict)
        }
        invalid_ids = [
            cid for cid in (rationale.citation_chunk_ids or [])
            if cid and cid not in chunk_ids
        ]
        if invalid_ids:
            feedback = (
                f"The following citation IDs do not exist in the retrieved evidence: "
                f"{invalid_ids}. Only cite chunk IDs from the provided evidence list."
            )
            logger.info("Critic pre-check failed (invalid citation IDs): %s", invalid_ids)
            return CriticVerdict(
                passed=False,
                invalid_citation_ids=invalid_ids,
                unsupported_facts=[],
                feedback_for_generator=feedback,
                attempt=attempt,
            )

        # --- 2. LLM faithfulness check ---
        if not self._service.provider:
            # No LLM critic configured — pass deterministic check only
            return CriticVerdict(passed=True, attempt=attempt)

        # Scrub PII before sending to the critic LLM using the mask_map that
        # was computed from the original submission in the workflow Step 8.
        # Without the original PII values there is nothing to scrub, so the
        # caller is responsible for passing mask_map.
        _masker = PIIMasker()
        _active_map: MaskMap = mask_map or {}

        evidence_text = "\n\n".join(
            f"[{chunk.get('chunk_id', chunk.get('id', '?'))}] "
            f"{chunk.get('text', chunk.get('excerpt', ''))[:500]}"
            for chunk in retrieved_chunks[:8]
            if isinstance(chunk, dict)
        ) or "(no retrieved evidence)"

        safe_summary = _masker.mask_text(rationale.summary[:800], _active_map)
        safe_facts = _masker.mask_text(json.dumps(rationale.supporting_facts or []), _active_map)
        evidence_text = _masker.mask_text(evidence_text, _active_map)

        user_prompt = _CRITIC_USER_TEMPLATE.format(
            summary=safe_summary,
            supporting_facts=safe_facts,
            cited_ids=json.dumps(rationale.citation_chunk_ids or []),
            evidence=evidence_text,
        )

        try:
            raw = self._service.provider.generate_json(
                system_prompt=_CRITIC_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=_CriticResponseModel.schema,
            )
            passed = bool(raw.get("passed", True))
            unsupported = raw.get("unsupported_facts", [])
            feedback = raw.get("feedback_for_generator", "")
            logger.info("Critic LLM check attempt=%d passed=%s unsupported=%s", attempt, passed, unsupported)
            return CriticVerdict(
                passed=passed,
                invalid_citation_ids=[],
                unsupported_facts=unsupported if isinstance(unsupported, list) else [],
                feedback_for_generator=feedback if isinstance(feedback, str) else "",
                attempt=attempt,
            )
        except (LLMUnavailable, ValueError, KeyError, TypeError) as exc:
            # Critic unavailable → pass through (don't block the workflow)
            logger.warning("Critic LLM check unavailable, passing through: %s", exc)
            return CriticVerdict(passed=True, attempt=attempt)
