# Phase F closure decision

## Decision

Phase F is frozen as a structured finite-time chaos-evidence layer. It reports
strong, supported, inconclusive or regular dynamics according to Lyapunov,
0-1, spectral, boundedness and Poincare diagnostics.

```text
status: phase_F_frozen
phase_F_frozen: true
evidence_layer: finite_time_chaos_evidence
available_evidence_level: chaos_evidence_inconclusive
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

## Evidence Scope

Rigorous assessments were executed for the fractional Lyapunov lanes and are
recorded with their method controls and discrepancies:

```text
valid_fractional_lyapunov_method_per_candidate: not_strictly_validated_with_documented_attempts
```

F4 internal controls, published reproduction attempts, sensitivity sweeps, F5
standardized diagnostics, and optional F6/F7 integration outputs form the
finite-time evidence layer. Evidence levels are numerical and tied to the
recorded solver, memory and time horizon. Hiddenness is assessed separately by
the sampled-neighborhood candidate gate.
