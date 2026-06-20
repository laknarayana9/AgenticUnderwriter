"""Structured LLM boundary for narrow underwriting assistance tasks."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.prompt_templates import (
    MISSING_INFO_SYSTEM_PROMPT,
    MISSING_INFO_USER_TEMPLATE,
    PRODUCER_RATIONALE_SYSTEM_PROMPT,
    PRODUCER_RATIONALE_USER_TEMPLATE,
    PRODUCER_RATIONALE_RETRY_SUFFIX,
)
from app.pii_masker import MaskMap, PIIMasker
from models.schemas import (
    MissingInfoQuestionBatch,
    MissingInfoQuestionOutput,
    ProducerRationaleOutput,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """Raised when a configured LLM provider cannot produce structured output."""


class StructuredJSONProvider(Protocol):
    """Provider interface for JSON-only structured LLM calls."""

    provider_name: str
    model: str

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a JSON object that should validate against schema."""


# Nebius Token Factory exposes an OpenAI-compatible REST surface, so the same
# OpenAI SDK client works by pointing base_url at the Nebius endpoint.
NEBIUS_DEFAULT_BASE_URL = "https://api.studio.nebius.com/v1/"


@dataclass(frozen=True)
class LLMServiceConfig:
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    prompt_version: str = "structured-llm-v1"

    @classmethod
    def from_env(cls) -> "LLMServiceConfig":
        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        if provider == "nebius":
            api_key = os.getenv("NEBIUS_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("NEBIUS_BASE_URL", NEBIUS_DEFAULT_BASE_URL).strip()
            default_model = "meta-llama/Llama-3.3-70B-Instruct"
        elif provider == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            base_url = None
            default_model = "claude-sonnet-4-6"
        elif provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            base_url = None
            default_model = "gemini-1.5-flash"
        elif provider == "ollama":
            api_key = ""  # not used
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
            default_model = os.getenv("OLLAMA_MODEL", "llama3.2").strip()
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
            default_model = "gpt-4o-mini"
        return cls(
            enabled=_env_bool("LLM_STRUCTURED_OUTPUT_ENABLED", False),
            provider=provider,
            model=os.getenv("LLM_MODEL", default_model).strip(),
            api_key=api_key,
            base_url=base_url,
            prompt_version=os.getenv("LLM_PROMPT_VERSION", "structured-llm-v1").strip(),
        )


class OpenAIJSONProvider:
    """
    OpenAI-compatible chat-completions adapter with lazy dependency loading.

    A ``base_url`` override lets the same adapter target any OpenAI-compatible
    host, including Nebius Token Factory, without a second SDK.
    """

    provider_name = "openai"

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        if not api_key:
            raise LLMUnavailable("API key is not configured")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMUnavailable("openai package is not installed") from exc

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{user_prompt}\n\n"
                        f"JSON schema:\n{json.dumps(schema, sort_keys=True)}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)


class NebiusJSONProvider(OpenAIJSONProvider):
    """Nebius Token Factory adapter (OpenAI-compatible chat completions)."""

    provider_name = "nebius"

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        super().__init__(api_key=api_key, model=model, base_url=base_url or NEBIUS_DEFAULT_BASE_URL)


class StructuredLLMService:
    """
    Narrow LLM facade for wording assistance.

    Eligibility decisions are supplied by deterministic underwriting rules.
    This service can only create validated wording artifacts around that output.
    """

    def __init__(
        self,
        config: Optional[LLMServiceConfig] = None,
        provider: Optional[StructuredJSONProvider] = None,
    ):
        self.config = config or LLMServiceConfig.from_env()
        self.provider = provider or self._build_provider()

    def generate_producer_rationale(
        self,
        decision_data: Dict[str, Any],
        citations: List[Dict[str, Any]],
        fallback_summary: str,
        mask_map: Optional[MaskMap] = None,
    ) -> ProducerRationaleOutput:
        # _force_fallback is set by the critic loop after max retries are exhausted
        if decision_data.get("_force_fallback"):
            return self._fallback_rationale(decision_data, citations, fallback_summary)

        fallback = self._fallback_rationale(decision_data, citations, fallback_summary)
        if not self.provider:
            return fallback

        # Use the caller-supplied mask_map (built from the original submission) so
        # that PII values from risk_factors, facts_used, citations, and fallback_summary
        # are all scrubbed, not just fields that happen to sit under applicant.* / risk.*.
        masker = PIIMasker()
        active_map: MaskMap = mask_map or {}
        if active_map:
            logger.debug("PII masked before LLM call: %s", masker.fields_masked(active_map))

        prompt = PRODUCER_RATIONALE_USER_TEMPLATE.format(
            decision=decision_data.get("decision") or decision_data.get("preliminary_decision"),
            confidence=decision_data.get("confidence"),
            risk_factors=json.dumps(decision_data.get("risk_factors", []), sort_keys=True),
            facts_used=json.dumps(decision_data.get("facts_used", {}), sort_keys=True),
            citations=json.dumps(_summarize_citations(citations), sort_keys=True),
            fallback_summary=fallback_summary,
        )
        # Scrub the full rendered prompt in one pass using the real PII values
        prompt = masker.mask_text(prompt, active_map)

        # Append critic feedback when this is a retry attempt
        critic_feedback = decision_data.get("critic_feedback", "")
        if critic_feedback:
            prompt = prompt + "\n\n" + PRODUCER_RATIONALE_RETRY_SUFFIX.format(
                critic_feedback=masker.mask_text(critic_feedback, active_map)
            )

        try:
            output = self._call_and_validate(
                ProducerRationaleOutput,
                PRODUCER_RATIONALE_SYSTEM_PROMPT,
                prompt,
            )
        except (LLMUnavailable, ValidationError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.info("Using fallback producer rationale: %s", exc)
            return fallback

        return output.model_copy(update={"source": "llm"})

    def word_missing_info_questions(
        self,
        questions: List[Dict[str, Any]],
        submission_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        fallback = [self._fallback_question(question).model_dump() for question in questions]
        if not questions or not self.provider:
            return [_question_payload(question, source="fallback") for question in questions]

        masker = PIIMasker()
        masked_context, mask_map = masker.mask_submission_context(submission_context or {})
        if mask_map:
            logger.debug("PII masked before LLM call: %s", masker.fields_masked(mask_map))

        prompt = MISSING_INFO_USER_TEMPLATE.format(
            submission_context=json.dumps(masked_context, sort_keys=True),
            questions=json.dumps(fallback, sort_keys=True),
        )
        # Second-pass scrub: catch PII that leaked through serialized question context blobs
        prompt = masker.mask_text(prompt, mask_map)

        try:
            batch = self._call_and_validate(
                MissingInfoQuestionBatch,
                MISSING_INFO_SYSTEM_PROMPT,
                prompt,
            )
            return self._merge_question_wording(questions, batch.questions)
        except (LLMUnavailable, ValidationError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.info("Using fallback missing-info wording: %s", exc)
            return [_question_payload(question, source="fallback") for question in questions]

    def _build_provider(self) -> Optional[StructuredJSONProvider]:
        if not self.config.enabled or self.config.provider in {"", "disabled", "none"}:
            return None
        # Lazy imports so missing optional SDKs don't break the base install
        from app.providers.claude_provider import ClaudeJSONProvider
        from app.providers.gemini_provider import GeminiJSONProvider
        from app.providers.ollama_provider import OllamaJSONProvider
        provider_classes = {
            "openai": OpenAIJSONProvider,
            "nebius": NebiusJSONProvider,
            "claude": ClaudeJSONProvider,
            "gemini": GeminiJSONProvider,
            "ollama": OllamaJSONProvider,
        }
        provider_cls = provider_classes.get(self.config.provider)
        if provider_cls is None:
            logger.info("Unsupported LLM provider '%s'; using fallback wording", self.config.provider)
            return None
        try:
            return provider_cls(
                api_key=self.config.api_key or "",
                model=self.config.model,
                base_url=self.config.base_url,
            )
        except LLMUnavailable as exc:
            logger.info("Structured LLM provider unavailable: %s", exc)
            return None

    def _call_and_validate(
        self,
        output_model: Type[T],
        system_prompt: str,
        user_prompt: str,
    ) -> T:
        if not self.provider:
            raise LLMUnavailable("No structured LLM provider configured")
        raw = self.provider.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=output_model.model_json_schema(),
        )
        return output_model.model_validate(raw)

    def _fallback_rationale(
        self,
        decision_data: Dict[str, Any],
        citations: List[Dict[str, Any]],
        fallback_summary: str,
    ) -> ProducerRationaleOutput:
        facts = [
            f"{key}: {value}"
            for key, value in (decision_data.get("facts_used") or {}).items()
            if value is not None
        ][:6]
        return ProducerRationaleOutput(
            summary=fallback_summary,
            supporting_facts=facts,
            citation_chunk_ids=[
                citation.get("chunk_id", "")
                for citation in citations
                if isinstance(citation, dict) and citation.get("chunk_id")
            ][:8],
            source="fallback",
        )

    def _fallback_question(self, question: Dict[str, Any]) -> MissingInfoQuestionOutput:
        return MissingInfoQuestionOutput(
            question_id=question.get("question_id", "missing_info"),
            field_path=question.get("field_path", question.get("answer_key", "missing_info")),
            answer_key=question.get("answer_key"),
            question_text=question.get("question_text") or question.get("question") or "Please provide the required underwriting information.",
            question_type=question.get("question_type", "text"),
            required=question.get("required", True),
            options=question.get("options"),
            context=question.get("context", {}),
            source="fallback",
        )

    def _merge_question_wording(
        self,
        original_questions: List[Dict[str, Any]],
        generated_questions: List[MissingInfoQuestionOutput],
    ) -> List[Dict[str, Any]]:
        generated_by_id = {question.question_id: question for question in generated_questions}
        if set(generated_by_id.keys()) != {question.get("question_id") for question in original_questions}:
            raise ValueError("Provider question IDs did not match requested questions")

        merged = []
        for original in original_questions:
            generated = generated_by_id[original["question_id"]]
            if generated.field_path != original.get("field_path"):
                raise ValueError("Provider changed a question field path")
            if generated.question_type != original.get("question_type"):
                raise ValueError("Provider changed a question type")
            if generated.options != original.get("options"):
                raise ValueError("Provider changed question options")
            payload = dict(original)
            payload["question"] = generated.question_text
            payload["question_text"] = generated.question_text
            payload["wording_source"] = "llm"
            payload["prompt_version"] = self.config.prompt_version
            return_payload = _question_payload(payload, source="llm")
            merged.append(return_payload)
        return merged


def _question_payload(question: Dict[str, Any], source: str) -> Dict[str, Any]:
    payload = dict(question)
    payload.setdefault("question_text", payload.get("question", "Additional information requested."))
    payload.setdefault("question", payload["question_text"])
    payload["wording_source"] = source
    return payload


def _summarize_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": citation.get("chunk_id"),
            "doc_id": citation.get("doc_id"),
            "section": citation.get("section"),
            "excerpt": citation.get("excerpt") or citation.get("text", "")[:240],
        }
        for citation in citations
        if isinstance(citation, dict)
    ][:8]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
