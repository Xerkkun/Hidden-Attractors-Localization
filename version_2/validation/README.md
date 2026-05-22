# Validation Evidence

This directory is reserved for generated validation evidence.

Keep the roles separate:

- JSON records traceability, run status, parameters, tolerances, software
  versions, seeds, and evidence paths.
- CSV stores numerical result tables.
- PNG/PDF stores visual evidence.
- MD explains one validation stage in human language.
- TEX/PDF is the final scientific report.

The detailed contract is documented in `docs/validation_evidence.md` and the
machine-readable layout contract is `configs/validation_contract.json`.

Do not store broad exploratory sweeps here by default. Use `outputs/` for
ordinary run products, then promote only the selected evidence needed for a
defensible validation package.
