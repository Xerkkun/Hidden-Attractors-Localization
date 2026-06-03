# F6 Integrated Chaos Validator

F6 combines F5 dynamics diagnostics, the Lyapunov method registry, available
case-specific finite-time local spectra, and F4 internal-validation metadata.
It emits finite-time chaos-evidence labels. Hiddenness is evaluated separately
by `candidate_gate`.

Run:

```powershell
python .\validation\python\run_integrated_chaos_validator.py
```

Outputs live under:

```text
validation/chaos_validation/integrated_chaos_validator/
```

## Inputs

F6 reads the F5 summary and its four standardized diagnostic summaries. It
also reads `hidden_attractors/analysis/lyapunov_methods.py`. When present, F4
is recorded as internal consistency metadata. When absent, F6 reports:

```text
f4_internal_validation_missing_or_pending
```

F6 still produces its report in that state.

## Decision Rules

A strong chaos-evidence label requires bounded finite-time behavior and at least
two support indicators: applicable controlled positive
`lambda_max`, zero-one chaotic-candidate behavior, broadband PSD/FFT, or
cloud-like Poincare geometry. Contradictory F5 evidence remains inconclusive.

A regular-candidate label similarly requires bounded finite-time behavior and
at least two regular-like indicators. Fractional method status remains
explicit in the report.

Allowed per-case labels:

```text
chaotic_candidate_numerically_supported
regular_candidate_numerically_supported
mixed_diagnostics_inconclusive
insufficient_lyapunov_support
insufficient_f5_support
method_validation_pending
numerical_failure
not_evaluated
```

The generated `integrated_chaos_rules.json` is the auditable machine-readable
rule set.

## Current Interpretation

The three configured Chua cases remain `mixed_diagnostics_inconclusive`.
Their F5 summaries combine zero-one regular-like behavior, inconclusive
PSD/FFT, and cloud-like Poincare geometry. F6 reports
`chaos_evidence_inconclusive` for that conflict.
