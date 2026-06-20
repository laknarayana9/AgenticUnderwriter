"""Google Gemini provider implementing StructuredJSONProvider."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.llm_service import LLMUnavailable

GEMINI_DEFAULT_MODEL = "gemini-1.5-flash"


class GeminiJSONProvider:
    """Google Gemini adapter using the GenerativeAI SDK with JSON response mode."""

    provider_name = "gemini"

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        if not api_key:
            raise LLMUnavailable("GOOGLE_API_KEY is not configured")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise LLMUnavailable("google-generativeai package is not installed: pip install google-generativeai") from exc

        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = model or GEMINI_DEFAULT_MODEL

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        import google.generativeai as genai

        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0,
        )
        model_instance = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=(
                f"{system_prompt}\n\n"
                "Return a single valid JSON object matching the schema below.\n"
                f"{json.dumps(schema, sort_keys=True)}"
            ),
            generation_config=generation_config,
        )
        response = model_instance.generate_content(user_prompt)
        return json.loads(response.text)
