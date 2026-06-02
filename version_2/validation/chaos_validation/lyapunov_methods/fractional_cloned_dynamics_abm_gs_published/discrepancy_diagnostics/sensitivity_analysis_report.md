# F3 sensitivity analysis report

## Executive summary

Controlled diagnostic sweeps recorded `164` unique runs.
Execution timestamp: `2026-06-01T18:25:15.122385+00:00`.
Recorded integration time: `2928.588` seconds.
These runs assess sensitivity only. They do not promote F3 validation,
certify chaos, or certify hiddenness.
The unlimited sweep was not executed because bounded runs already showed
substantial cost; partial outputs were preserved after each row.

## Executed commands

- `validation\python\run_cloned_dynamics_sensitivity.py --max-runs 12`
- `validation\python\run_cloned_dynamics_sensitivity.py --axis q1_mode --max-runs 20`
- `validation\python\run_cloned_dynamics_sensitivity.py --axis delta --max-runs 40`
- `validation\python\run_cloned_dynamics_sensitivity.py --case-file fischer2020_jerk_commensurate.yaml --row-index 2 --axis h --max-runs 3`
- `validation\python\run_cloned_dynamics_sensitivity.py --case-file fischer2020_jerk_incommensurate.yaml --row-index 1 --row-index 2 --row-index 3 --axis h --max-runs 9`
- `validation\python\run_cloned_dynamics_sensitivity.py --case-file fischer2020_four_wing_incommensurate.yaml --row-index 1 --row-index 2 --row-index 3 --axis h --max-runs 9`
- `validation\python\run_cloned_dynamics_sensitivity.py --case-file fischer2020_financial_incommensurate.yaml --row-index 0 --axis h --max-runs 3`
- `validation\python\run_cloned_dynamics_sensitivity.py --axis k --max-runs 40`
- `validation\python\run_cloned_dynamics_sensitivity.py --axis gs_policy --max-runs 40`
- `validation\python\run_cloned_dynamics_sensitivity.py --case-file fischer2020_financial_incommensurate.yaml --row-index 0 --axis t_clone --max-runs 2`
- `validation\python\run_cloned_dynamics_sensitivity.py --case-file fischer2020_four_wing_incommensurate.yaml --row-index 1 --row-index 2 --row-index 3 --axis t_clone --max-runs 9`
- `validation\python\run_cloned_dynamics_sensitivity.py --case-file fischer2020_jerk_commensurate.yaml --row-index 2 --axis t_clone --max-runs 3`
- `validation\python\run_cloned_dynamics_sensitivity.py --case-file fischer2020_jerk_incommensurate.yaml --row-index 1 --axis t_clone --max-runs 3`
- `validation\python\run_cloned_dynamics_sensitivity.py --case-file fischer2020_jerk_incommensurate.yaml --row-index 2 --axis t_clone --max-runs 3`
- `validation\python\run_cloned_dynamics_sensitivity.py --case-file fischer2020_jerk_incommensurate.yaml --row-index 3 --axis t_clone --max-runs 3`

## Results by axis

| axis | runs | best row | best max_abs_error | best lambda_max_abs_error | improved rows | degraded rows |
|---|---:|---|---:|---:|---:|---:|
| delta | 44 | financial Incomm [0.9, 1.0, 1.0] (delta=0.01) | 0.0545538 | 0.0545538 | 3 | 10 |
| gs_policy | 33 | financial Incomm [0.9, 1.0, 1.0] (orthonormalization=qr) | 0.054767 | 0.054767 | 0 | 5 |
| h | 24 | financial Incomm [0.9, 1.0, 1.0] (h_clone=0.01) | 0.054767 | 0.054767 | 2 | 6 |
| k | 33 | financial Incomm [0.9, 1.0, 1.0] (K=250) | 0.0540135 | 0.0540135 | 1 | 8 |
| q1_mode | 2 | jerk Comm [1.0, 1.0, 1.0] (integration_mode=integer_rk4_reference) | 0.263303 | 0.0014197 | 0 | 0 |
| t_clone | 28 | jerk Comm [1.0, 1.0, 1.0] (t_clone=10) | 0.0257811 | 0.00244794 | 2 | 8 |

## Results by system

| system | runs | best row | best max_abs_error | strict sign matches | tolerant sign matches |
|---|---:|---|---:|---:|---:|
| financial | 15 | financial Incomm [0.9, 1.0, 1.0] (K=250) | 0.0540135 | 0 | 15 |
| four_wing | 48 | four_wing Incomm [0.6, 1.0, 1.0] (h_clone=0.02) | 0.0720654 | 11 | 32 |
| jerk | 101 | jerk Comm [1.0, 1.0, 1.0] (t_clone=10) | 0.0257811 | 35 | 71 |

## Best improvements

| system | axis | variant | baseline class | improved class | max_abs_error | lambda_max_abs_error |
|---|---|---|---|---|---:|---:|
| four_wing | delta | delta=0.0001 | `strict_discrepancy` | `sign_pattern_supported_not_quantitative` | 0.101345 | 0.0596167 |
| four_wing | delta | delta=1e-05 | `strict_discrepancy` | `sign_pattern_supported_not_quantitative` | 0.103297 | 0.0628357 |
| four_wing | delta | delta=0.0001 | `strict_discrepancy` | `sign_pattern_supported_not_quantitative` | 0.0917988 | 0.0917988 |
| four_wing | delta | delta=1e-05 | `strict_discrepancy` | `sign_pattern_supported_not_quantitative` | 0.121506 | 0.0919814 |
| four_wing | delta | delta=0.0001 | `strict_discrepancy` | `sign_pattern_supported_not_quantitative` | 0.0896146 | 0.0896146 |
| four_wing | delta | delta=1e-05 | `strict_discrepancy` | `sign_pattern_supported_not_quantitative` | 0.0895219 | 0.0895219 |
| jerk | t_clone | t_clone=10 | `sign_pattern_supported_not_quantitative` | `quantitative_abs_pass_strict_sign_pass` | 0.0257811 | 0.00244794 |
| four_wing | t_clone | t_clone=2.5 | `strict_discrepancy` | `sign_pattern_supported_not_quantitative` | 0.653358 | 0.0821909 |
| four_wing | h | h_clone=0.005 | `strict_discrepancy` | `sign_pattern_supported_not_quantitative` | 0.103723 | 0.0700538 |
| four_wing | h | h_clone=0.02 | `strict_discrepancy` | `sign_pattern_supported_not_quantitative` | 0.0801678 | 0.0743327 |
| four_wing | h | h_clone=0.005 | `strict_discrepancy` | `sign_pattern_supported_not_quantitative` | 0.0767615 | 0.0767615 |
| four_wing | k | K=100 | `strict_discrepancy` | `sign_pattern_supported_not_quantitative` | 0.0976668 | 0.0739462 |

## Persistent strict discrepancies

| system | orders | case file | row index |
|---|---|---|---:|
| financial | `[0.9, 1.0, 1.0]` | `fischer2020_financial_incommensurate.yaml` | 0 |
| jerk | `[0.8, 0.8, 0.8]` | `fischer2020_jerk_commensurate.yaml` | 2 |
| jerk | `[0.8, 1.0, 1.0]` | `fischer2020_jerk_incommensurate.yaml` | 1 |
| jerk | `[0.7, 1.0, 1.0]` | `fischer2020_jerk_incommensurate.yaml` | 2 |
| jerk | `[0.6, 1.0, 1.0]` | `fischer2020_jerk_incommensurate.yaml` | 3 |

## Hypothesis assessment

| hypothesis | evidence for | evidence against | status |
|---|---|---|---|
| H1 | T_clone improved 2 row classifications; jerk q=1 reaches quantitative agreement at T_clone=10. | No protocol variant is proven equivalent to the article. | `supported_by_t_clone_sensitivity` |
| H2 | T_clone changes some classes strongly; h_clone and K produce smaller row-specific shifts. | Parameter sensitivity alone does not identify the article convention. | `supported_but_not_identified` |
| H3 | GS modified, GS classical, and QR outputs are compared. | The maximum cross-policy max_abs_error spread is 4.11227e-13; policy outputs are numerically equivalent for the swept rows. | `weakened_as_primary_explanation` |
| H4 | Delta sweeps improved 3 row classifications and near-zero sign crossings remain explicitly classified. | Large jerk lambda_3 gaps cannot be explained by near-zero signs. | `supported_for_subset_only` |
| H5 | Incommensurate four-wing and jerk rows retain targeted sweeps. | Commensurate jerk discrepancies show this is not the only cause. | `open` |
| H6 | Rounded published values can affect near-zero sign interpretation. | Rounding cannot explain large lambda_3 gaps. | `supported_for_subset_only` |

Favored hypotheses: `H1`, `H2`, and `H5` remain the main audit paths.
Weakened hypothesis: `H3` is not supported as a primary explanation
because GS modified, GS classical, and QR remain numerically equivalent
for the swept rows. `H4` and `H6` explain only near-zero subsets.

## Conservative conclusion

The sweep evidence is diagnostic and partial. It does not justify promotion
to validation. The official F3 state remains
`published_benchmarks_pending_discrepancy` with `validated=False`.

## Diagnostic closure

The bounded partial sweeps covered six axes: `delta`, `T_clone`, `h`, `K`,
`q1_mode`, and `gs_policy`. The `gs_policy` hypothesis is weakened as the
primary cause because modified GS, classical GS, and QR are numerically
equivalent for the swept rows. `T_clone`, protocol parameters, and
incommensurate handling remain the main audit paths.

An unlimited sweep is not required in the current scope. Its cost is high,
and the article does not report enough protocol detail to distinguish
conclusively among plausible conventions. F3 remains
`published_benchmarks_pending_discrepancy` and is not promoted.
