# Internal Transition and Legacy Dependencies

> [!WARNING]
> This folder `tools/legacy` is not a public workflow surface.
> It contains only historical/internal numerical helpers still imported by maintained code while they are ported into `hidden_attractors/`.

No command in this folder defines official evidence. Standalone legacy CLI executables such as `hidden-attractors-protocol` or `hidden-attractors-fractional-report-run` are deprecated and not for active use.

All active runs must use the unified CLI command:

```bash
# Unified launcher replacing legacy entrypoints
hidden-attractors --help
```

## Legacy Components

- `chua_initial_cond.py`, `biased_describing_function.py`, `harmonic_diagnostics.py` and `extended_search_utils.py` support current seed generation.
- `danca2017_chua_abm_replication.py`, `equilibria_analysis.py` and `parallel_policy.py` support the ABM reference comparison.
- `early_periodicity_filter.py`, `lure_biased_multiparam_search.py`, `lure_biased_multiparam_continuation.py` and `lure_candidate_manifest.py` remain only for migration tests.

The directory can be removed after the remaining imports and migration tests are moved to package modules.
