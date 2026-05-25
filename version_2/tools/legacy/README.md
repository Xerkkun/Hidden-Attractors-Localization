# Internal Transition Dependencies

This folder is not a public workflow surface. It contains only numerical
helpers still imported by maintained code while they are ported into
`hidden_attractors/`:

- `chua_initial_cond.py`, `biased_describing_function.py`,
  `harmonic_diagnostics.py` and `extended_search_utils.py` support current
  seed generation in `hidden_attractors.workflows.fractional_report_run`.
- `danca2017_chua_abm_replication.py`, `equilibria_analysis.py` and
  `parallel_policy.py` support the ABM full-history reference comparison.
- `early_periodicity_filter.py`, `lure_biased_multiparam_search.py`,
  `lure_biased_multiparam_continuation.py` and
  `lure_candidate_manifest.py` remain only for migration tests.

No command in this folder defines official evidence. Official runs use:

```bash
hidden-attractors-protocol --help
hidden-attractors-fractional-report-run --help
```

Any continuing internal use of EFORK is covered by the corrected third-stage
ordering `K3 = F(... + a31*K1 + a32*K2)`. The directory can be removed after
the remaining imports and migration tests are moved to package modules.
