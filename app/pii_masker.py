"""PII masking for LLM call boundaries.

Masks applicant-identifying fields before they enter LLM prompts.
The mask map is ephemeral — never persisted or logged.
"""

from __future__ import annotations

import copy
import re
from typing import Dict, List, Optional, Tuple

# MaskMap: token -> original value (in-memory only, never stored)
MaskMap = Dict[str, str]

_FIELD_TOKENS: Dict[str, str] = {
    "applicant.full_name": "[APPLICANT_NAME]",
    "applicant.email": "[APPLICANT_EMAIL]",
    "applicant.phone": "[APPLICANT_PHONE]",
    "risk.property_address": "[PROPERTY_ADDRESS]",
}

# Reverse: token -> field path (for fields_masked lookup)
_TOKEN_TO_FIELD: Dict[str, str] = {v: k for k, v in _FIELD_TOKENS.items()}


class PIIMasker:
    """Masks PII fields in dicts before LLM calls; never persists original values."""

    def mask_submission_context(self, context: Dict) -> Tuple[Dict, MaskMap]:
        """Return a deep-copied masked dict and an ephemeral MaskMap."""
        masked = copy.deepcopy(context)
        mask_map: MaskMap = {}

        for field_path, token in _FIELD_TOKENS.items():
            parts = field_path.split(".")
            node = masked
            for part in parts[:-1]:
                if not isinstance(node, dict):
                    break
                node = node.get(part, {})
            else:
                leaf_key = parts[-1]
                if isinstance(node, dict) and node.get(leaf_key):
                    original = str(node[leaf_key])
                    mask_map[token] = original
                    node[leaf_key] = token

        return masked, mask_map

    def mask_text(self, text: str, mask_map: MaskMap) -> str:
        """Replace any literal PII values found in free text with their tokens."""
        for token, original in mask_map.items():
            if original and original in text:
                text = text.replace(original, token)
        return text

    def fields_masked(self, mask_map: MaskMap) -> List[str]:
        """Return the field paths that were actually masked (no values)."""
        return [_TOKEN_TO_FIELD[token] for token in mask_map if token in _TOKEN_TO_FIELD]

    def pii_in_text(self, text: str, mask_map: MaskMap) -> bool:
        """Return True if any original PII value appears literally in text."""
        return any(
            original and original in text
            for original in mask_map.values()
        )

    def pii_in_text_by_pattern(self, text: str, known_values: List[str]) -> bool:
        """Return True if any of the supplied PII strings appear in text."""
        return any(v and v in text for v in known_values)
