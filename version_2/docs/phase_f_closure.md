# Phase F Closure Status

Phase F is frozen as a structured finite-time chaos-evidence layer. It reports
strong, supported, inconclusive or regular dynamics according to Lyapunov,
0-1, spectral, boundedness and Poincare diagnostics:

```text
phase_F_frozen
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

## Scope

Evidence levels are numerical and tied to the recorded solver, memory and time
horizon. Hiddenness is assessed separately through sampled-neighborhood
candidate gates.

