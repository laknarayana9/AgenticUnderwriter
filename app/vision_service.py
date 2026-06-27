"""Fenced vision evidence service (multimodal intake).

Extracts structured risk attributes from a property photo and folds the
high-confidence ones into the HO3 submission *before* the deterministic rules
run. Vision is upstream-and-guarded input, never the decision — exactly the
boundary ADR-0001 draws for the text LLM.

Phase 1 ships the schema, the fenced service with a deterministic stub (no
network), and the confidence-gated mapping. A real vision provider
(OpenAI/Claude/Ollama) plugs in behind `VisionProvider` later without changing
this boundary.
"""

from __future__ import annotations

import hashlib
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

from models.schemas import VisionAttribute, VisionEvidence

logger = logging.getLogger(__name__)


class VisionUnavailable(RuntimeError):
    """Raised when a configured vision provider cannot be constructed."""

# Attributes the vision model is asked to assess. Kept here so the prompt and the
# parser stay in sync; a real provider is told to return exactly these keys with
# {value, confidence, visible} each, abstaining (visible=false) when unsure.
VISION_ATTRIBUTES = (
    "roof_material",
    "roof_condition",
    "roof_damage",
    "tarp_present",
    "defensible_space_present",
    "hazards",
)

VISION_SYSTEM_PROMPT = (
    "You are a property-underwriting vision assistant. From the photo, assess only "
    "what is clearly visible. For each attribute return {value, confidence (0-1), "
    "visible (bool)}. If an attribute cannot be assessed from the image, set "
    "visible=false and value=null — do NOT guess. Return JSON only."
)


def vision_user_instruction() -> str:
    """Shared user instruction listing the required attributes and output shape.

    Used by every vision provider so the prompt and the parser stay in sync."""
    keys = ", ".join(VISION_ATTRIBUTES)
    return (
        "Assess this property photo for underwriting. Return a JSON object with "
        f"exactly these keys: {keys}. Each value is an object "
        '{"value": <bool|string|list|null>, "confidence": <0..1>, "visible": <bool>}. '
        "Set visible=false and value=null for anything you cannot assess from the "
        "image. hazards.value is a list of detected hazards "
        "(pool, trampoline, overhanging_trees, ...) or []. JSON only."
    )


@dataclass(frozen=True)
class VisionServiceConfig:
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # Only attributes at/above this confidence are folded into the submission;
    # below it, the workflow's missing-info gate asks a human instead.
    min_confidence: float = 0.6

    @classmethod
    def from_env(cls) -> "VisionServiceConfig":
        provider = os.getenv("VISION_PROVIDER", "openai").strip().lower()
        try:
            min_conf = float(os.getenv("VISION_MIN_CONFIDENCE", "0.6"))
        except ValueError:
            min_conf = 0.6
        if provider == "ollama":
            api_key = ""  # not used by Ollama
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
            default_model = os.getenv("VISION_MODEL", "llama3.2-vision").strip()
        else:
            api_key = os.getenv("OPENAI_API_KEY") if provider == "openai" else os.getenv("ANTHROPIC_API_KEY")
            base_url = None
            default_model = os.getenv("VISION_MODEL", "gpt-4o").strip()
        return cls(
            enabled=os.getenv("VISION_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
            provider=provider,
            model=default_model,
            api_key=api_key,
            base_url=base_url,
            min_confidence=max(0.0, min(1.0, min_conf)),
        )


class VisionProvider(Protocol):
    """Interface a real vision model implements (Phase 3)."""
    model: str

    def extract(self, image_bytes: bytes, system_prompt: str) -> Dict[str, Any]:
        """Return a dict keyed by VISION_ATTRIBUTES, each {value, confidence, visible}."""


class VisionEvidenceService:
    """Fenced facade. With no provider (default / unavailable), returns fully
    abstained evidence so the workflow degrades to asking a human."""

    def __init__(self, config: Optional[VisionServiceConfig] = None,
                 provider: Optional[VisionProvider] = None):
        self.config = config or VisionServiceConfig.from_env()
        # An injected provider wins (tests); otherwise build from config. With
        # vision disabled or the provider unavailable, this stays None and the
        # service returns abstained evidence.
        self.provider = provider or self._build_provider()

    def _build_provider(self) -> Optional[VisionProvider]:
        if not self.config.enabled or self.config.provider in {"", "none", "disabled"}:
            return None
        try:
            if self.config.provider == "openai":
                from app.providers.openai_vision_provider import OpenAIVisionProvider
                return OpenAIVisionProvider(api_key=self.config.api_key or "", model=self.config.model)
            if self.config.provider == "ollama":
                from app.providers.ollama_vision_provider import OllamaVisionProvider
                return OllamaVisionProvider(model=self.config.model, base_url=self.config.base_url)
            logger.info("Unsupported vision provider '%s'; vision disabled", self.config.provider)
            return None
        except Exception as exc:  # noqa: BLE001 - missing key/SDK → disabled, not fatal
            logger.info("Vision provider unavailable, using abstained evidence: %s", exc)
            return None

    def extract_evidence(self, image_bytes: bytes) -> VisionEvidence:
        sha = hashlib.sha256(image_bytes or b"").hexdigest()
        if not self.provider:
            return self._abstained(sha, model="stub", source="stub")
        try:
            raw = self.provider.extract(image_bytes, VISION_SYSTEM_PROMPT)
            return self._parse(raw, sha, model=getattr(self.provider, "model", self.config.model))
        except Exception as exc:  # noqa: BLE001 - any provider failure must degrade, not crash the workflow
            logger.info("Vision extraction failed; returning abstained evidence: %s", exc)
            return self._abstained(sha, model="stub", source="stub")

    def _parse(self, raw: Dict[str, Any], sha: str, model: str) -> VisionEvidence:
        attrs: Dict[str, Any] = {}
        for key in VISION_ATTRIBUTES:
            item = raw.get(key) or {}
            attrs[key] = VisionAttribute(
                value=item.get("value"),
                confidence=float(item.get("confidence", 0.0) or 0.0),
                visible=bool(item.get("visible", False)),
            )
        return VisionEvidence(image_sha256=sha, model=model, source="llm", **attrs)

    @staticmethod
    def _abstained(sha: str, model: str, source: str) -> VisionEvidence:
        return VisionEvidence(image_sha256=sha, model=model, source=source)


def fold_vision_into_submission(
    submission_raw: Dict[str, Any],
    evidence: VisionEvidence,
    min_confidence: float = 0.6,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fold confident, visible vision attributes into the submission.

    Conservative by design: only `defensible_space_present` maps to a field the
    rules actually consume (`risk.wildfire_mitigation_evidence`). Everything else
    is recorded as provenance, not as a decision input. Attributes that are not
    visible or below `min_confidence` are left untouched, so the deterministic
    missing-info gate still pauses to ask a human.

    Returns (updated_submission, applied) where `applied` records what changed
    and the image provenance.
    """
    updated = deepcopy(submission_raw)
    risk = updated.setdefault("risk", {})
    applied: Dict[str, Any] = {"image_sha256": evidence.image_sha256, "model": evidence.model, "fields": {}}

    ds = evidence.defensible_space_present
    if ds.visible and ds.confidence >= min_confidence and isinstance(ds.value, bool):
        # Only set when the field is not already provided by the producer.
        if risk.get("wildfire_mitigation_evidence") is None:
            risk["wildfire_mitigation_evidence"] = ds.value
            note = (
                f"Defensible space {'observed' if ds.value else 'not observed'} from "
                f"property photo (conf={ds.confidence:.2f}, img={evidence.image_sha256[:12]})."
            )
            existing = risk.get("mitigation_notes")
            risk["mitigation_notes"] = f"{existing} {note}".strip() if existing else note
            applied["fields"]["risk.wildfire_mitigation_evidence"] = ds.value

    # Record other confident, visible attributes as provenance only (no rule uses them).
    observed = {
        key: getattr(evidence, key).value
        for key in ("roof_material", "roof_condition", "roof_damage", "tarp_present", "hazards")
        if getattr(evidence, key).visible and getattr(evidence, key).confidence >= min_confidence
    }
    if observed:
        applied["observed"] = observed

    return updated, applied
