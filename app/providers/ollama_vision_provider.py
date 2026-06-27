"""Ollama local vision adapter for structured property-evidence extraction.

Runs a local vision model (llama3.2-vision / llava) via Ollama's REST API — no
SDK, no network egress, so property photos stay on-device. This is the
privacy-preserving option for the PII concern in failure mode #11.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional

from app.vision_service import VisionUnavailable, vision_user_instruction

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_VISION_DEFAULT_MODEL = "llama3.2-vision"


class OllamaVisionProvider:
    """Vision provider over Ollama's /api/chat with image input."""

    provider_name = "ollama"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        try:
            import urllib.request  # stdlib, always available
        except ImportError as exc:  # pragma: no cover
            raise VisionUnavailable("urllib not available") from exc
        self.base_url = (base_url or OLLAMA_DEFAULT_BASE_URL).rstrip("/")
        self.model = model or OLLAMA_VISION_DEFAULT_MODEL

    def extract(self, image_bytes: bytes, system_prompt: str) -> Dict[str, Any]:
        import urllib.request

        b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": vision_user_instruction(), "images": [b64]},
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
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
        except Exception as exc:  # connection refused, model missing, timeout, ...
            raise VisionUnavailable(f"Ollama vision request failed: {exc}") from exc

        content = body.get("message", {}).get("content", "{}")
        return json.loads(content)
