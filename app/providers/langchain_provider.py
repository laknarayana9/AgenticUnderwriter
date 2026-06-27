"""LangChain structured-output provider for the fenced LLM service.

Implements the same `StructuredJSONProvider` interface as the other adapters,
using `langchain_openai.ChatOpenAI`. Because it plugs into `StructuredLLMService`,
every governance guarantee still applies — PII masking, the generator–critic
loop, and the deterministic fallback all wrap this provider. Selected with
`LLM_PROVIDER=langchain`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.llm_service import LLMUnavailable


class LangChainOpenAIProvider:
    """structured-JSON provider built on LangChain's ChatOpenAI."""

    provider_name = "langchain"

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        if not api_key:
            raise LLMUnavailable("API key is not configured")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - only without the package
            raise LLMUnavailable("langchain-openai is not installed") from exc

        kwargs: Dict[str, Any] = {"model": model, "api_key": api_key, "temperature": 0}
        if base_url:
            kwargs["base_url"] = base_url
        # Constrain the model to JSON output at the LangChain binding level.
        self._llm = ChatOpenAI(**kwargs).bind(response_format={"type": "json_object"})
        self.model = model

    def generate_json(self, system_prompt: str, user_prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self._llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{user_prompt}\n\nJSON schema:\n{json.dumps(schema, sort_keys=True)}"),
        ])
        content = response.content if isinstance(response.content, str) else "{}"
        return json.loads(content or "{}")
