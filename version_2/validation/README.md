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

## Reference Cases

Validated baselines that are not the main fractional candidate package live in
`reference_cases/`. The integer reference case is
`reference_cases/chua_integer_q1/`, the piecewise-linear Chua workflow
evaluated at `q=1`. Its integration-dependent evidence was deleted and
regenerated after the EFORK-3 third-stage order was aligned with the published
formula, `K3 = a31*K1 + a32*K2`.

Check that case independently:

```bash
hidden-attractors-check-validation \
  --contract configs/validation_chua_integer_q1.json \
  --validation-root validation/reference_cases/chua_integer_q1
```

The independent numerical-method evidence package is
`reference_cases/efork3_ghoreishi_ghaffari/`. It reproduces the published
manufactured-solution tables before the method is used in the regenerated
integer Chua case:

```bash
hidden-attractors-check-validation \
  --contract configs/validation_efork3_ghoreishi_ghaffari.json \
  --validation-root validation/reference_cases/efork3_ghoreishi_ghaffari
```
