# Citation

Use the archived software metadata for academic attribution:

- **Archived software DOI**:
  [10.17605/OSF.IO/ZGK74](https://doi.org/10.17605/OSF.IO/ZGK74)
- **Repository**:
  [Xerkkun/Hidden-Attractors-Localization](https://github.com/Xerkkun/Hidden-Attractors-Localization)
- **Release metadata**: `CITATION.cff` in the repository root

When reporting a numerical result, also record:

- the package version and exact commit;
- the input configuration or API arguments;
- solver, step, horizon, transient removal, and memory policy;
- sampling and estimator settings;
- the result artifact and its checksum; and
- the primary references for the numerical method.

The completed integer-order Chua reference evidence is stored under
`validation/reference_cases/chua_integer_q1/`. Results that reproduce that
record should cite the sources listed in
[Integer Chua `q=1` Reference](integer_chua_reference.md).

The completed EFORK-3 method reproduction is stored under
`validation/reference_cases/efork3_ghoreishi_ghaffari/`. Numerical-method
claims based on that record should cite Ghoreishi, Ghaffari, and Saad (2023)
as documented in [EFORK-3 Published Validation](efork3_validation.md).

Time-series Lyapunov results delegated to `nolds` should cite the estimator
sources recorded by the package and report the installed backend version.
