# Chua fractional arctan canonical validation package

This package promotes the `chua_arctan_c590_q09999_seed9` candidate for version `1.0.0` under a radius-limited finite numerical contract.

## Status

- Public validation package: `version_2/validation/chua_fractional_arctan/`.
- Source evidence package: `version_2/validation/chua_fractional_arctan_c590/`.
- Label: `hiddenness_supported_under_tested_neighborhoods` for local radii `r <= 0.3`.
- Local finite probes: `8400`.
- Local target contacts: `0`.
- Tested equilibria: `E0, E+, E-`.
- Macro-radius audit: `610` contacts in `5100` probes at radii `1.0` and `2.0`.

## Boundary

The promoted claim is finite numerical evidence under the recorded Caputo ABM full-memory contract, the listed probe radii, the recorded target-contact classifier, and the documented equilibria. It is not a global mathematical proof of basin exclusion. The contacts at radii `1.0` and `2.0` are intentionally retained as an extended-radius audit and do not change the local-radius claim boundary.

## Files

- `hiddenness_validation_summary.json`: canonical machine-readable summary.
- `run_metadata.json`: parameters, seed, numerical contract, and source pointers.
- `hiddenness_decisions.csv`: per-radius and per-equilibrium decisions.
- `summary_by_radius.csv`: aggregate decisions by radius.
- `equilibria.json`: tested equilibria.
- `matignon_classification.json`: local Matignon classification snapshot.
- `config.json`: minimal reproducibility configuration for the promoted evidence.
- `figures_manifest.json`: presentation figure metadata linked to this case.
