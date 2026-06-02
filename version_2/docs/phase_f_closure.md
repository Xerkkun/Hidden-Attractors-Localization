# Phase F Closure Status

Phase F is closed as standardized diagnostic evidence, not as chaos
certification:

```text
F_closed_as_structured_diagnostics_not_chaos_certification
```

The closure assessment is generated with:

```powershell
python .\validation\python\run_phase_F_closure_assessment.py
```

Artifacts live under:

```text
validation/chaos_validation/phase_F_closure/
```

## Assessed Routes

Route A, `fractional_variational_abm_qr`, is labeled
`assessed_with_documented_validation_gap`. Internal full-history controls and
sensitivity evidence are retained, while published validation or an accepted
formal policy has not been achieved. DK2018 block-restart evidence remains a
separate contract.

Route B, `fractional_cloned_dynamics_abm_gs_published`, is labeled
`assessed_with_documented_discrepancies`. The long Fischer 2020 run and bounded
sensitivity sweeps are preserved: `24` rows, `10` quantitative passes, `6`
sign-pattern passes, and `8` discrepancy rows.

Route C closes the structured diagnostic scope because F4 and F5 have
auditable outputs with explicit limitations.

## Boundary

The rigorous A and B assessments are not discarded as false. They also are not
promoted to strict fractional-method validation. Current Phase F artifacts do
not certify mathematical chaos and do not certify hiddenness.
