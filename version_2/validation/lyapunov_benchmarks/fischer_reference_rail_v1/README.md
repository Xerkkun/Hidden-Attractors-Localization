# Fischer reference rail, audited article contract v1

These six derived configurations preserve every equation, parameter, initial
state, derivative-order row, and published target of the original six
`fractional_cloned_dynamics_abm_gs_published` YAML files. Each derivative
records its original file SHA-256, complete original integration settings,
source pages, execution version, and evaluation version. Originals and old
numerical results remain unchanged.

The separate reference has a fixed zero Caputo lower limit and full history.
Clones and their short fiducial restart each T_clone and then reanchor to the
separate reference at physical time k*T_clone. Section 2.2 of Fischer et al.
(2020), DOI 10.1016/j.apnum.2020.03.027, gives this reference-rail structure
(printed pp.189-191, equations 10,20,21). Financial reference h=0.05 and clone
h=0.01 remain distinct; a ten-second block samples reference index 200.
The full-history reference is an explicit choice among the article's
single-reference integration and its approximate chained-block discussion.

For Jerk, T_clone=10, N_clone=1000, and K=100 implement table 2 condition III,
supported by the final paragraph of section 3.1 on printed p.196. The original
T_clone=1 configurations remain available for historical and diagnostic use.
No row is chosen according to its outcome.

The V2 evaluator enforces the YAML's strict sign requirement as well as
absolute tolerances. Near-zero diagnostics do not modify the sign or verdict.
These are finite-time benchmark comparisons, without certification of chaos,
hiddenness, or a full-memory fractional variational exponent theory.

From `version_2`, run into a new directory:

```powershell
python validation/python/run_cloned_dynamics_benchmarks.py `
  --benchmarks-dir validation/lyapunov_benchmarks/fischer_reference_rail_v1 `
  --output-dir local_reports/fischer_article_contract_run_01
```

Add `--fast` only for reduced interface smoke checks. A smoke result is not
quantitative article reproduction. For the historical execution, select the
original directory and both explicit legacy options described in
`docs/published_benchmark_evaluation.md`.
