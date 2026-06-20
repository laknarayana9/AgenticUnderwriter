"""Fine-tuning track: messy producer note -> canonical HO3 JSON extraction.

A LoRA supervised fine-tune (Project 4 / Tier 3) for the structured-extraction
task the intake normalizer performs. Trained and served through Nebius Token
Factory's OpenAI-compatible fine-tuning API, so the resulting model is just
another LLM_MODEL pointed at the Nebius base URL.
"""
