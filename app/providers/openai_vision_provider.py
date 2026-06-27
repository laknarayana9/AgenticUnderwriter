"""OpenAI vision adapter for structured property-evidence extraction.

Sends the image as a base64 data URL to an OpenAI vision-capable model
(gpt-4o by default) and returns a JSON object keyed by the vision attributes.
Lazy SDK import so the base install stays lean; the OpenAI dependency is the same
one the text path already uses.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional

from app.vision_service import VisionUnavailable, vision_user_instruction


class OpenAIVisionProvider:
    """Vision provider over OpenAI-compatible chat completions with image input."""

    provider_name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: Optional[str] = None):
        if not api_key:
            raise VisionUnavailable("OPENAI_API_KEY is not configured")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the SDK
            raise VisionUnavailable("openai package is not installed") from exc
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model = model

    def extract(self, image_bytes: bytes, system_prompt: str) -> Dict[str, Any]:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_user_instruction()},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content or "{}")
