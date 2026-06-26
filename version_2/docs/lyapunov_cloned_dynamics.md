# Cloned Dynamics Lyapunov Indicators

## F3 scope

Phase F3 implements the cloned-dynamics method described by Fischer, Zourmba,
and Mohamadou (2020), DOI `10.1016/j.apnum.2020.03.027`.

The implementation returns finite-time local Lyapunov indicators. It does not
certify chaos by itself and it does not certify hiddenness.

## Mathematical formulation

Starting with a fiducial trajectory,

```text
X^(0)(0) = X0
```

create one perturbed clone per state direction:

```text
X^(j)(0) = X0 + delta e_j,  j = 1,...,n.
```

After a cloning interval `T_C`, calculate:

```text
v_j^(k) = X^(j)(T_C) - X^(0)(T_C).
```

Orthonormalize these difference vectors to obtain `u_j^(k)`, accumulate the
logarithmic growth factors, and restart each clone around the evolved fiducial
trajectory:

```text
lambda_j = 1/(K T_C) sum_k log(r_j^(k) / delta)
X^(j) = X^(0) + delta u_j^(k).
```

For modified Gram-Schmidt, `r_j^(k)` is the norm of the residual vector before
normalization. The QR variant uses `abs(diag(R))`. No Jacobian and no
variational system are used.

## Implemented methods

| Method | Orthonormalization | Status |
| --- | --- | --- |
| `fractional_cloned_dynamics_abm_gs_published` | Modified Gram-Schmidt | Implemented; `published_benchmarks_pending_discrepancy` |
| `fractional_cloned_dynamics_abm_qr` | QR | Experimental internal comparison |

The GS method is the Fischer 2020 reproduction lane. The QR method is a more
stable internal variant; it must be compared against the GS lane before any
promotion.

## Fractional memory contract

For Caputo fractional systems the initial F3 implementation uses:

```text
memory_protocol: published_block_restart
```

Each ABM clone interval has local history and restarts at the next interval.
This is not a full-memory Caputo-aware claim. It is intentionally separate from
`fractional_variational_abm_qr`.

The ABM implementation accepts one shared order or one order per component:

- `[q]`: commensurate fractional order.
- `[q1, ..., qn]`: component-wise incommensurate fractional orders.
- `[1.0]` or `[1.0, ..., 1.0]`: integer benchmark.

Incommensurate fractional execution is recorded as experimental in result
metadata.

## Fischer 2020 extracted values

| System | Orders | Published LE | Published K01 |
| --- | --- | --- | --- |
| Jerk | `[1,1,1]` | `[0.1899, 0.0413, -0.4246]` | `0.9866` |
| Jerk | `[0.9,0.9,0.9]` | `[-0.0042, -0.0267, -0.4656]` | `0.0758` |
| Jerk | `[0.8,0.8,0.8]` | `[-0.0047, -0.2864, -0.3634]` | `0.0151` |
| Jerk | `[0.7,0.7,0.7]` | `[-0.0527, -0.0528, -0.2763]` | `-0.1425` |
| Jerk | `[0.9,1,1]` | `[0.1744, 0.0246, -0.4600]` | `0.9928` |
| Jerk | `[0.8,1,1]` | `[0.0810, -0.0162, -0.4017]` | `0.3786` |
| Jerk | `[0.7,1,1]` | `[-0.0395, -0.0888, -0.2442]` | `0.1543` |
| Jerk | `[0.6,1,1]` | `[0.0179, -0.0971, -0.2816]` | `0.1329` |
| Financial | `[1,1,1]` | `[0.0891, 0.0058, -0.4348]` | `0.9984` |
| Financial | `[0.9,0.9,0.9]` | `[0.1133, -0.0030, -0.3370]` | `0.9974` |
| Financial | `[0.8,0.8,0.8]` | `[-0.0407, -0.0406, -0.2648]` | `0.1325` |
| Financial | `[0.7,0.7,0.7]` | `[-0.2125, -0.4539, -0.4539]` | `0.2491` |
| Financial | `[0.9,1,1]` | `[0.1445, 0.0012, -0.4419]` | `0.9980` |
| Financial | `[0.8,1,1]` | `[0.1235, -0.0026, -0.3412]` | `0.9975` |
| Financial | `[0.7,1,1]` | `[0.0935, -0.0476, -0.2900]` | `0.9986` |
| Financial | `[0.6,1,1]` | `[-0.0678, -0.0680, -0.3308]` | `0.1037` |
| Four-wing | `[1,1,1]` | `[0.3358, 0.0133, -1.2608]` | `0.9980` |
| Four-wing | `[0.9,0.9,0.9]` | `[0.2684, -0.0259, -0.9699]` | `0.9982` |
| Four-wing | `[0.8,0.8,0.8]` | `[-0.1543, -0.1531, -0.8010]` | `0.1352` |
| Four-wing | `[0.7,0.7,0.7]` | `[-0.4825, -0.4818, -0.6942]` | `0.1752` |
| Four-wing | `[0.9,1,1]` | `[0.3623, -0.0501, -1.0984]` | `0.9983` |
| Four-wing | `[0.8,1,1]` | `[0.3460, -0.1048, -0.9196]` | `0.9978` |
| Four-wing | `[0.7,1,1]` | `[0.3598, -0.0345, -0.9160]` | `0.9976` |
| Four-wing | `[0.6,1,1]` | `[0.3371, -0.0628, -0.9076]` | `0.9969` |

The financial system contains `abs(x)`. It is supported because cloned
dynamics does not require a Jacobian; its YAML metadata records
`nonsmooth_rhs: true`.

## Sensitivity and validation

The estimates depend on `delta`, `T_C`, `K`, `h`, and the order vector. The
initial absolute tolerance is `0.15`; the target quantitative tolerance is
`0.05`. A sign-pattern-only match is recorded separately and is not a strong
quantitative pass.

Published runs are opt-in:

```text
RUN_PUBLISHED_CLONED=1 pytest tests/test_cloned_dynamics_fischer_published.py -v -m "slow and published"
```

Until those benchmarks pass, both F3 registry entries remain `validated=False`.

Passing F3 Fischer tests does not validate `fractional_variational_abm_qr`.
Passing F3 does not certify chaos or hiddenness. F3 and F2 remain separate
validation lanes.

## Fischer 2020 published benchmark execution status

The published GS lane was executed on `2026-06-01`. The first-row pytest gate
reported `5 passed, 1 failed`; the exhaustive runner then recorded all rows
with:

```text
python validation/python/run_cloned_dynamics_benchmarks.py \
  --official-summary-dir validation/chaos_validation/lyapunov_methods/fractional_cloned_dynamics_abm_gs_published
```

| Item | Result |
| --- | --- |
| Rows executed | `24` |
| Quantitative passes (`all abs_error < 0.05`) | `10` |
| Sign-pattern support passes | `6` |
| Quantitative discrepancy rows | `8` |
| Numerical failures | `0` |
| Strict sign-pattern gate failures | `10` |
| Final status | `published_benchmarks_pending_discrepancy` |

The strict all-row test reports `14 passed, 10 failed`. Two rows have
`all abs_error < 0.05` but still cross the sign boundary for a near-zero
exponent, so they count as quantitative passes in the summary and as failures
under the stricter sign-pattern gate.

Discrepancies remain in the financial incommensurate lane, the four-wing
incommensurate lane, and jerk commensurate/incommensurate rows. The method
therefore remains `validated=False`.

Formal row-level diagnostics are tracked in the
[Fischer 2020 discrepancy report](https://github.com/Xerkkun/Hidden-Attractors-Localization/blob/main/version_2/validation/chaos_validation/lyapunov_methods/fractional_cloned_dynamics_abm_gs_published/discrepancy_diagnostics/fischer2020_discrepancy_report.md).
The report records the explicit `near_zero_sign_policy`, separates strict
sign discrepancies from near-zero boundary crossings, and provides an opt-in
sensitivity plan. These diagnostics do not change the official validation
status.

Bounded sensitivity diagnostics recorded `164` runs across `delta`, `T_clone`,
`h`, `K`, `q1_mode`, and `gs_policy`. Persistent strict discrepancies remain
in one financial incommensurate row and four jerk rows. The lane is
`implemented_with_documented_published_discrepancies`: exact reproduction
requires protocol and numerical details not reported in the article or
access to the authors' implementation. No further bounded sweep is required
in the current scope. See the
[formal reproduction limitations](https://github.com/Xerkkun/Hidden-Attractors-Localization/blob/main/version_2/validation/chaos_validation/lyapunov_methods/fractional_cloned_dynamics_abm_gs_published/discrepancy_diagnostics/fischer2020_reproduction_limitations.md).

This closure does not certify chaos, certify hiddenness, validate F2, or
promote the internal QR variant.

Auditable runtime outputs:

```text
validation/outputs/lyapunov_benchmarks/fractional_cloned_dynamics_abm_gs_published/
```

The directory contains `published_benchmark_results.csv`,
`published_benchmark_all_rows_results.csv`, `validation_summary.json`,
`run_metadata.json`, and representative `convergence_by_block.csv`.
