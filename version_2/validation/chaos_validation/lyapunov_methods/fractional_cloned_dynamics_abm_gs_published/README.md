# Fischer 2020 cloned-dynamics GS lane

This directory records the published-algorithm reproduction lane for
`fractional_cloned_dynamics_abm_gs_published`.

Current status: `published_benchmarks_pending_discrepancy`.

The long GS runner was executed on `2026-06-01`: `24` rows were recorded,
with `10` quantitative passes, `6` sign-pattern support passes, `8`
quantitative discrepancy rows, and no numerical failures. The strict all-row
sign-pattern test reports `14 passed, 10 failed`, including two near-zero
sign-crossing rows that still satisfy the quantitative absolute-error target.

The method uses ABM predictor-corrector integration with
`memory_protocol: published_block_restart` and modified Gram-Schmidt
orthonormalization. Results are finite-time local Lyapunov indicators. They do
not certify chaos or hiddenness, and they are not a full-memory Caputo-aware
claim.

Formal diagnostics are stored in
[discrepancy_diagnostics/](discrepancy_diagnostics/README.md). The
[row-level report](discrepancy_diagnostics/fischer2020_discrepancy_report.md)
classifies strict discrepancies and near-zero sign boundaries and records a
bounded opt-in sensitivity plan. These additions do not promote F3:
`validated=False`.
