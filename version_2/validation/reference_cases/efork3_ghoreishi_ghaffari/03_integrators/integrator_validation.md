# Three-Stage EFORK Validation

## Reference

Ghoreishi, Ghaffari, and Saad, *Fractional Order Runge-Kutta Methods*,
*Fractal and Fractional* **7** (2023), 245.

The validation evaluates the two scalar manufactured-solution examples and
reproduces the terminal absolute-error sequences published in Tables 3, 4, 9,
and 10 for `alpha = 1/4` and `alpha = 1/2`.

## Implementation Check

The reference solver in
`hidden_attractors.solvers.efork_published.efork3_caputo_integrate` applies
the published third stage

```text
K3 = h^alpha F_n(t_n + c3 h, y_n + a31 K1 + a32 K2)
```

and evaluates the known Caputo-history correction in all three stages.  This
ordering is also present in `sources/ejemplo1.py`, supplied by Dr. Luis
Gerardo de la Fraga (CINVESTAV Unidad Zacatenco).

## Evidence

- `benchmark_results.csv`: calculated and displayed published terminal errors.
- `convergence_errors.png`: convergence trace for both examples.
- `efork3_validation_summary.json`: acceptance result and maximum difference.
- `sources/constantes_efork.py` and `sources/ejemplo1.py`: provided source
  scripts with hashes registered in `../00_manifest/source_registry.json`.

The displayed paper values are rounded.  Therefore the acceptance comparison
uses an absolute tolerance of `6e-9`, sufficient to account for printed
rounding while remaining several orders of magnitude below the reported
errors.
