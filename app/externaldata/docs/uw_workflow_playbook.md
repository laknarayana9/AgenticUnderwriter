# Underwriting Workflow Playbook (Synthetic)
**Effective Date:** 2026-01-01  
**Version:** v0.1-synthetic

## 1. Triage Workflow
### 1.1 Straight-Through Processing
- If no referral/decline triggers apply and required fields are present, proceed to **ACCEPT** with standard endorsements optional.

### 1.2 Referral Packet
When referring, the system **MUST** generate:
- trigger list (each with citations)
- missing documents list
- suggested endorsements/conditions (each with citations)
- risk summary (hazards + key characteristics)

## 2. Missing-Info Loop
If required fields are missing, the system **MUST** ask targeted questions.
### 2.1 Required Fields (Minimum)
- roof_age
- construction_year
- occupancy_type
- distance_to_hydrant OR protection_class
- claims_count_36mo

### 2.2 Question Generation Rules
- Each question **MUST** reference the guideline section that requires it.
- Questions are prioritized:
  - P0: required to determine eligibility
  - P1: required to determine referral conditions
  - P2: optional underwriting detail

## 3. Conflicting Evidence Handling
If guidelines conflict across documents/versions:
- system **SHALL** prefer the most recent effective date
- if conflict remains, **SHALL** refer and include both citations

## 4. Evidence Policy
- Any decision statement **MUST** include at least one citation.
- If evidence confidence is below threshold, system **MUST** output **REFER** with “insufficient evidence” rationale.
