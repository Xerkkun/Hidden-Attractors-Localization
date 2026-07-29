# Chua fractional arctan validation record

This directory is a fixed public projection of the c590 radius-limited
validation result.

## Result

- Local radii `r <= 0.3`: `8,400` finite probes and `0` target contacts.
- All three equilibria were tested.
- Macro radii `1.0` and `2.0`: `610` contacts in `5,100` probes.
- At `r = 2`, `2,569` trajectories reached the full horizon and `131`
  reached the declared divergence threshold.

## Boundary

The local conclusion is finite numerical evidence under the recorded Caputo
ABM full-memory contract and target-contact classifier. It is not a global
mathematical proof of basin exclusion. Macro-radius contacts are retained as
an extended basin audit and do not enlarge or invalidate the local contract.
This record does not certify chaos.

## Canonical files

- `hiddenness_validation_summary.json`
- `run_metadata.json`
- `hiddenness_decisions.csv`
- `hiddenness_decisions_status_contact.csv`
- `summary_by_radius.csv`
- `equilibria.json`
- `matignon_classification.json`
- `config.json`
- `figures_manifest.json`

The row-level source record is
`validation/chua_fractional_arctan_c590/validation_summary.json`.
