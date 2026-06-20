"""Anthropic Claude provider implementing StructuredJSONProvider."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from app.llm_service import LLMUnavailable

CLAUDE_DEFAULT_MODEL = "claude-sonnet-4-6"


class ClaudeJSONProvider:
    """Anthropic Claude adapter using the Messages API with JSON extraction."""

    provider_name = "claude"

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        if not api_key:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not configured")
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise LLMUnavailable("anthropic package is not installed: pip install anthropic") from exc

        self._anthropic = _anthropic
        self.client = _anthropic.Anthropic(api_key=api_key)
        self.model = model or CLAUDE_DEFAULT_MODEL

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Claude doesn't have a native json_object mode, so we ask explicitly and
        # extract the first JSON object from the response text.
        combined_system = (
            f"{system_prompt}\n\n"
            "IMPORTANT: Respond with a single valid JSON object only. "
            "No markdown fences, no prose before or after."
        )
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=combined_system,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{user_prompt}\n\n"
                        f"JSON schema to conform to:\n{json.dumps(schema, sort_keys=True)}"
                    ),
                }
            ],
        )
        text = message.content[0].text.strip()
        # Strip accidental markdown code fences if the model adds them
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
