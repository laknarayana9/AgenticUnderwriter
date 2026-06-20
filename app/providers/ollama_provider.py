"""Ollama local model provider implementing StructuredJSONProvider."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from app.llm_service import LLMUnavailable

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.2"


class OllamaJSONProvider:
    """Ollama adapter using the REST API directly (no SDK required)."""

    provider_name = "ollama"

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        # api_key unused for Ollama but accepted for interface compatibility
        try:
            import urllib.request  # stdlib, always available
        except ImportError as exc:
            raise LLMUnavailable("urllib not available") from exc

        self.base_url = (base_url or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
        self.model = model or OLLAMA_DEFAULT_MODEL

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        import urllib.request

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": (
                    f"{system_prompt}\n\n"
                    "Return only valid JSON. No markdown, no extra text.\n"
                    f"Schema:\n{json.dumps(schema, sort_keys=True)}"
                )},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read())
        except Exception as exc:
            raise LLMUnavailable(f"Ollama request failed: {exc}") from exc

        content = body.get("message", {}).get("content", "{}")
        return json.loads(content)
