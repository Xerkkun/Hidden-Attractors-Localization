# Freeze Status

## Interpretation of evidence levels

The library reports evidence with its tested numerical scope. Software checks,
published-reference reproductions, finite-time diagnostics, and sampled
neighborhood assessments are separate layers. The recorded solver, memory,
time horizon, tolerances, and sampling contract bound each result.

Allowed evidence levels:

- `software_validated`
- `published_reference_reproduced`
- `published_reference_partially_reproduced`
- `structured_diagnostic_ready`
- `strong_chaos_evidence`
- `chaotic_dynamics_supported`
- `chaos_evidence_inconclusive`
- `hiddenness_supported_under_tested_neighborhoods`
- `compatible_with_hiddenness_under_tested_radii`
- `self_excited_contact_detected`
- `candidate_rejected`

## Frozen / closed

- `scientific_scope`
- `literature_comparison_table`
- `algebraic_validation`
- `transfer_function_validation`
- `fractional_frequency_validation`
- `describing_function_validation_as_seed_generation`
- ABM fast software validation
- EFORK fast software validation
- `integer_qr_benettin` for `q=1`
- common Lyapunov API compatibility gate
- F5 structured diagnostics output format
- F6/F7 structured integration and comparison format

## Frozen with documented limitations

- Danca reproduction: partial where the paper does not publish an exact hidden initial condition.
- Wu reproduction: paper-style reported-IC reproduction where `omega`, `k`, and `a0` are not explicitly reported.
- `fractional_variational_abm_qr`: implemented and internally controlled, but not promoted as a published-validated reference method.
- Fischer cloned dynamics: implemented with documented reproduction discrepancies.

## Not frozen as universal claim

- Global hiddenness verification for arbitrary candidates.
- Strict universal chaos certification.
- Noncommensurate fractional systems.
- Non-Caputo derivatives.
- Arbitrary non-Lur'e systems.
