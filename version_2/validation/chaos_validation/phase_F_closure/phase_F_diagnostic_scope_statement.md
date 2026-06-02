# Phase F diagnostic scope statement

## 1. Purpose

Phase F produces standardized numerical evidence about bounded dynamics,
chaos indicators, and method comparison. Its outputs are reproducible,
traceable, and conservative.

## 2. What F4/F5 currently establish

F4 establishes internal method benchmarks and comparisons against published
references while preserving documented discrepancies. It does not promote
strong fractional-method validation.

F5 establishes standardized outputs for:

- `boundedness`
- `zero_one`
- `psd_fft`
- `poincare`

F5 does not certify mathematical chaos. F5 does not evaluate or certify
hiddenness.

## 3. Why strict chaos validation is not closed

There is not yet an accepted fractional Lyapunov method applied to every
fractional candidate. This is a strict-closure limitation, not an assertion
that the completed numerical work is false or absent.

- `fractional_variational_abm_qr` has internal full-history controls and
  sensitivity evidence. It remains pending published validation or an
  accepted formal policy.
- `fractional_cloned_dynamics_abm_gs_published` retains Fischer 2020 status
  `published_benchmarks_pending_discrepancy` after a long published
  reproduction run and bounded sensitivity sweeps.
- `fractional_variational_dk2018_block_restart_abm_gs` retains the
  Rabinovich-Fabrikant `lambda_3` discrepancy after a long native run. This
  reproduction lane remains distinct from full-history QR.

Therefore Phase F cannot assert `chaos_verified`.

## 4. Current valid closure

The current valid closure is diagnostic, not certifying.

```text
F_closed_as_structured_diagnostics_not_chaos_certification
```

F4 validates internal consistency without promoting strong fractional-method
validation. F5 produces standardized diagnostics. Phase F does not certify
mathematical chaos. Phase F does not certify hiddenness. Current candidates
remain `mixed_diagnostics_inconclusive` unless additional evidence is
recorded.

The strict routes are labeled as assessed with documented limitations:

- Route A: `assessed_with_documented_validation_gap`
- Route B: `assessed_with_documented_discrepancies`

These labels preserve rigorous attempts without promoting the methods to
strictly validated status.

## 5. Requirements for future strict closure

1. Validate `fractional_variational_abm_qr` against a published benchmark or
   accepted formal policy.
2. Or formally resolve the Fischer 2020 F3 discrepancies.
3. Apply at least one valid fractional Lyapunov method to every fractional
   candidate.
4. Preferably obtain sign agreement between two valid methods.
5. Preserve compatibility with boundedness, zero-one, PSD/FFT, and Poincare.
6. Do not use any isolated diagnostic as proof of chaos.
