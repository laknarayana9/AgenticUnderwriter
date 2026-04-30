# Rating Rules (Synthetic)
**Carrier:** DemoCarrier  
**Product:** HO3 / HO5  
**State:** CA (demo)  
**Effective Date:** 2026-01-01  
**Version:** v0.1-synthetic

## 1. Base Premium
Base premium is computed from dwelling limit using a simplified curve:
- BaseRate = 0.0020 * DwellingLimit

## 2. Territory Factor (Demo)
- TerritoryFactor is based on county (proxy):
  - LowRiskCounty: 0.95
  - MediumRiskCounty: 1.00
  - HighRiskCounty: 1.10

## 3. Construction Age Factor
- Built ≥ 2000: 0.95
- Built 1980–1999: 1.00
- Built < 1980: 1.10

## 4. Hazard Surcharges
### 4.1 Wildfire
- Moderate: +5%
- High: +12%
- Severe: +20% (requires UW approval in practice)

### 4.2 Flood (SFHA)
- SFHA indicator adds +10% and triggers referral (see UW Guidelines §4.2)

## 5. Deductible Factors
- $1,000: 1.00
- $2,500: 0.92
- $5,000: 0.85

## 6. Endorsement Premium Impacts (Simplified)
- WBK-01: +$35 flat
- ORD-01: +3% of base
- WFD-10: -2% (deductible increase credit)
- FLX-ACK: $0

## 7. Pricing Integrity Rules
- Any premium impact **MUST** be computed via rating tool output; the agent **MUST NOT** invent premium numbers.
