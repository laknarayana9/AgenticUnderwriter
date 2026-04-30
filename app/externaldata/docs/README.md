# Underwriting RAG Starter Doc Pack (Synthetic)

**Purpose:** Starter documents for building an evidence-first underwriting copilot (RAG + citations + highlighting).
These documents are **synthetic** (not carrier-proprietary) and are designed to:
- Contain realistic section headers (## / ###) for header-based chunking
- Include explicit rule language (MUST/SHALL/REQUIRED/SHOULD/MAY)
- Include thresholds and referral triggers to test evidence verification
- Provide endorsement definitions and rating rule references

## Metadata
- Effective Date: 2026-01-01
- Version: v0.1-synthetic
- Generated: 2026-03-02

## Files
1. `uw_guidelines_homeowners.md` — eligibility + referral + knockouts
2. `hazards_guidance.md` — hazard signals → underwriting actions
3. `endorsements_manual.md` — endorsement catalog + eligibility + requirements
4. `rating_rules.md` — simplified rating plan + factors + deductible rules
5. `uw_workflow_playbook.md` — triage workflow + missing-info questions + escalation

## Notes
- You can ingest these with your Phase 1 ingestion pipeline (Markdown).
- For PDF ingestion testing, convert any Markdown to PDF later; the content is stable and highlightable.
