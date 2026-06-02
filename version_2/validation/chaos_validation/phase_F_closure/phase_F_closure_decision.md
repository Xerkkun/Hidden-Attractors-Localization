# Phase F closure decision

## Decision

Phase F is structurally complete but not strictly closed as chaos validation.

```text
status: F_closed_as_structured_diagnostics_not_chaos_certification
strict_chaos_validation_closed: false
structured_diagnostics_closed: true
chaos_verified: false
hiddenness_verified: false
```

## Closure Routes

| Route | Status | Reason |
|---|---|---|
| A: `fractional_variational_abm_qr` | `assessed_with_documented_validation_gap` | `rigorous_internal_controls_completed_but_published_validation_or_accepted_policy_not_achieved` |
| B: Fischer 2020 cloned dynamics | `assessed_with_documented_discrepancies` | `rigorous_fischer2020_reproduction_and_sensitivity_completed_with_documented_discrepancies` |
| C: diagnostic scope closure | `completed` | `F4 and F5 are structurally complete with documented limitations` |

## Documented Assessments

Route A is labeled
`assessed_with_documented_validation_gap`. The local
full-history QR implementation has F4 internal controls and sensitivity
evidence. The DK2018 long block-restart reproduction remains a separate
contract and does not promote full-history QR.

Route B is labeled
`assessed_with_documented_discrepancies`. The Fischer 2020
published GS lane records `24`
rows: `10`
quantitative passes, `6`
sign-pattern passes, and `8`
discrepancy rows. Bounded sensitivity sweeps were completed for the current
scope, while the discrepancies remain explicit.

## Strict Closure Boundary

Rigorous assessments were executed for the fractional Lyapunov lanes. They are
recorded as documented evidence, not discarded as failed work. Strict chaos
validation remains outside the current evidence scope because no accepted
fractional Lyapunov method has been applied to each fractional candidate:

```text
valid_fractional_lyapunov_method_per_candidate: not_strictly_validated_with_documented_attempts
```

F4 internal controls, published reproduction attempts, sensitivity sweeps, F5
standardized diagnostics, and optional F6/F7 integration outputs remain
reproducible numerical evidence. They do not certify mathematical chaos or
hiddenness.
