# Demo Evaluation Report

Generated: `2026-04-30T19:22:05.066137+00:00`

## Summary

- Scenarios: 10 / 10
- Passed: 10
- Failed: 0
- Overall: PASS

## Guardrails

| Check | Result |
| --- | --- |
| Ran all 10 demo scenarios | PASS |
| Expected vs actual decision | PASS |
| Refer decline citations exist | PASS |
| No silent accept on missing critical info | PASS |

## Scenarios

| # | Scenario | Expected | Actual | Status | Citations | Missing-Info Guard | Result |
| ---: | --- | --- | --- | --- | ---: | --- | --- |
| 1 | Scenario 1: Standard Quote - Low Risk | ACCEPT | ACCEPT | completed | 5 | PASS | PASS |
| 2 | Scenario 2: Wildfire High Risk - Refer | REFER | REFER | waiting_for_info | 3 | PASS | PASS |
| 3 | Scenario 3: Missing Roof Age - Need Info | REFER | REFER | waiting_for_info | 3 | PASS | PASS |
| 4 | Scenario 4: Old Construction - Refer | REFER | REFER | pending_review | 5 | PASS | PASS |
| 5 | Scenario 5: Condo - Quote Eligible | ACCEPT | ACCEPT | completed | 5 | PASS | PASS |
| 6 | Scenario 6: Flood Risk - Refer | REFER | REFER | pending_review | 5 | PASS | PASS |
| 7 | Scenario 7: Townhouse - Quote Eligible | ACCEPT | ACCEPT | completed | 5 | PASS | PASS |
| 8 | Scenario 8: Claims History - Refer | REFER | REFER | waiting_for_info | 3 | PASS | PASS |
| 9 | Scenario 9: High Coverage - Quote Eligible | ACCEPT | ACCEPT | completed | 5 | PASS | PASS |
| 10 | Scenario 10: Tenant Occupied - Refer | REFER | REFER | pending_review | 5 | PASS | PASS |
