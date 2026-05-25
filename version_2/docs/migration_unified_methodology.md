# Migration To The Unified Methodology

`version_2` now has one official Caputo hidden-attractor protocol. Old route
names must not appear in new promoted outputs.

## Renamed Concepts

| Previous wording | Official representation |
|---|---|
| classical centered describing function | `seed_generation.family=lure_classical_centered` |
| centered Lur'e route | `seed_generation.family=lure_classical_centered` |
| biased Lur'e route | `seed_generation.family=lure_classical_biased` |
| Machado/FDF route | `machado_centered` or `machado_biased` seed family |
| epsilon or eta continuation | `ContinuationPlan(lambda_values=...)` with internal mapping in provenance |
| early periodicity filter | `soft_precheck`; periodic seeds use `pre_continuation_periodic` and continue |
| sphere-only controls | interior ball sampling in `hiddenness_tests` |

Describing functions and reconstructed Lur'e/Machado points are seed sources
only. They are not hiddenness evidence.

## Removed And Retained Material

`version_1/` is removed from the maintained repository after its relevant
integer reference artifacts were promoted to
`validation/reference_cases/chua_integer_q1/`.

`tools/legacy/` is temporarily retained for a narrow reason: current
maintained wrappers still import the full fractional Chua driver and Danca ABM
adapter, and existing transition tests exercise the precheck/continuation
adapter. It is no longer a documented route for new results. Its remaining
dependencies must be ported into `hidden_attractors/` before that directory
can be deleted safely.

Older generated validation stage trees under former directory names were
removed. The surviving `validation/reference_cases/` subtree is retained
only for independently motivated benchmark evidence. New promotions conform
to `configs/validation_contract.json` version 2 and use the official stage
envelope.

## EFORK Correction Invariant

Every executable EFORK implementation still reachable from `version_2`, in
Python or C and including retained compatibility engines, is checked against
the published third-stage ordering:

```text
K3 = F(... + a31*K1 + a32*K2)
```

`tests/test_efork_published_validation.py` tests the Python implementations,
the package-native C backends, and the C/Python engines temporarily retained
under `tools/legacy/`. No superseded executable EFORK tree is kept as a second
workflow surface.
