# Fischer 2020 F3 discrepancy report

## Executive summary

F3 was executed against 24 published Fischer 2020 rows. It produced 10
quantitative passes, 6 sign-pattern passes, and 8 failures. Therefore the
method remains `validated=False`.

The official status remains `published_benchmarks_pending_discrepancy`.
This diagnostic layer classifies the recorded results; it does not tune
parameters to force agreement and it does not promote validation.

## Runtime CSV consistency

- `outputs_csv_present = true`
- `outputs_csv_consistent = true`

## Global table

| system | commensurate rows | incommensurate rows | quantitative | sign-pattern | failed | sign mismatches |
|---|---:|---:|---:|---:|---:|---:|
| financial | 4 | 4 | 5 | 2 | 1 | 2 |
| four_wing | 4 | 4 | 5 | 0 | 3 | 4 |
| jerk | 4 | 4 | 0 | 4 | 4 | 4 |

## Row-level classification

| system | type | orders | computed LE | published LE | abs error | diagnostic class |
|---|---|---|---|---|---|---|
| financial | Comm | `[1.0, 1.0, 1.0]` | `[0.0620426, 0.00274922, -0.361116]` | `[0.0891, 0.0058, -0.4348]` | `[0.0270574, 0.00305078, 0.073684]` | `sign_pattern_supported_not_quantitative` |
| financial | Comm | `[0.9, 0.9, 0.9]` | `[0.0742229, 0.00480853, -0.345161]` | `[0.1133, -0.003, -0.337]` | `[0.0390771, 0.00780853, 0.00816124]` | `quantitative_abs_pass_near_zero_sign_boundary` |
| financial | Comm | `[0.8, 0.8, 0.8]` | `[-0.0546788, -0.0549282, -0.263117]` | `[-0.0407, -0.0406, -0.2648]` | `[0.0139788, 0.0143282, 0.00168343]` | `quantitative_abs_pass_strict_sign_pass` |
| financial | Comm | `[0.7, 0.7, 0.7]` | `[-0.210743, -0.243196, -0.243221]` | `[-0.2125, -0.4539, -0.4539]` | `[0.00175715, 0.210704, 0.210679]` | `sign_pattern_supported_not_quantitative` |
| financial | Incomm | `[0.9, 1.0, 1.0]` | `[0.089733, -0.0281534, -0.492609]` | `[0.1445, 0.0012, -0.4419]` | `[0.054767, 0.0293534, 0.0507088]` | `strict_discrepancy` |
| financial | Incomm | `[0.8, 1.0, 1.0]` | `[0.111191, -0.0360465, -0.371452]` | `[0.1235, -0.0026, -0.3412]` | `[0.012309, 0.0334465, 0.0302524]` | `quantitative_abs_pass_strict_sign_pass` |
| financial | Incomm | `[0.7, 1.0, 1.0]` | `[0.12035, -0.060955, -0.305799]` | `[0.0935, -0.0476, -0.29]` | `[0.0268496, 0.013355, 0.0157988]` | `quantitative_abs_pass_strict_sign_pass` |
| financial | Incomm | `[0.6, 1.0, 1.0]` | `[-0.0850938, -0.0856937, -0.356619]` | `[-0.0678, -0.068, -0.3308]` | `[0.0172938, 0.0176937, 0.0258185]` | `quantitative_abs_pass_strict_sign_pass` |
| four_wing | Comm | `[1.0, 1.0, 1.0]` | `[0.32694, 0.00194533, -1.29598]` | `[0.3358, 0.0133, -1.2608]` | `[0.00885968, 0.0113547, 0.035175]` | `quantitative_abs_pass_strict_sign_pass` |
| four_wing | Comm | `[0.9, 0.9, 0.9]` | `[0.223289, 0.0031331, -0.974095]` | `[0.2684, -0.0259, -0.9699]` | `[0.0451111, 0.0290331, 0.00419512]` | `quantitative_abs_pass_near_zero_sign_boundary` |
| four_wing | Comm | `[0.8, 0.8, 0.8]` | `[-0.155208, -0.154515, -0.803355]` | `[-0.1543, -0.1531, -0.801]` | `[0.000907578, 0.00141463, 0.00235528]` | `quantitative_abs_pass_strict_sign_pass` |
| four_wing | Comm | `[0.7, 0.7, 0.7]` | `[-0.482011, -0.481475, -0.693107]` | `[-0.4825, -0.4818, -0.6942]` | `[0.000488727, 0.000324596, 0.00109279]` | `quantitative_abs_pass_strict_sign_pass` |
| four_wing | Incomm | `[0.9, 1.0, 1.0]` | `[0.316758, -0.0176838, -1.07271]` | `[0.3623, -0.0501, -1.0984]` | `[0.0455422, 0.0324162, 0.0256904]` | `quantitative_abs_pass_strict_sign_pass` |
| four_wing | Incomm | `[0.8, 1.0, 1.0]` | `[0.282531, 0.0125641, -0.936367]` | `[0.346, -0.1048, -0.9196]` | `[0.063469, 0.117364, 0.0167666]` | `strict_discrepancy` |
| four_wing | Incomm | `[0.7, 1.0, 1.0]` | `[0.268966, 0.0224327, -0.975833]` | `[0.3598, -0.0345, -0.916]` | `[0.0908335, 0.0569327, 0.0598326]` | `strict_discrepancy` |
| four_wing | Incomm | `[0.6, 1.0, 1.0]` | `[0.24554, 0.00135524, -0.923721]` | `[0.3371, -0.0628, -0.9076]` | `[0.0915601, 0.0641552, 0.0161213]` | `strict_discrepancy` |
| jerk | Comm | `[1.0, 1.0, 1.0]` | `[0.197016, 0.0347942, -0.759424]` | `[0.1899, 0.0413, -0.4246]` | `[0.00711594, 0.00650581, 0.334824]` | `sign_pattern_supported_not_quantitative` |
| jerk | Comm | `[0.9, 0.9, 0.9]` | `[-0.0239961, -0.138025, -1.34484]` | `[-0.0042, -0.0267, -0.4656]` | `[0.0197961, 0.111325, 0.879239]` | `sign_pattern_supported_not_quantitative` |
| jerk | Comm | `[0.8, 0.8, 0.8]` | `[0.00233586, -0.561171, -1.66322]` | `[-0.0047, -0.2864, -0.3634]` | `[0.00703586, 0.274771, 1.29982]` | `strict_discrepancy` |
| jerk | Comm | `[0.7, 0.7, 0.7]` | `[-0.000336589, -0.298177, -1.55048]` | `[-0.0527, -0.0528, -0.2763]` | `[0.0523634, 0.245377, 1.27418]` | `sign_pattern_supported_not_quantitative` |
| jerk | Incomm | `[0.9, 1.0, 1.0]` | `[0.196033, 0.0227063, -0.735419]` | `[0.1744, 0.0246, -0.46]` | `[0.021633, 0.00189365, 0.275419]` | `sign_pattern_supported_not_quantitative` |
| jerk | Incomm | `[0.8, 1.0, 1.0]` | `[0.188783, 0.0201008, -0.717387]` | `[0.081, -0.0162, -0.4017]` | `[0.107783, 0.0363008, 0.315687]` | `strict_discrepancy` |
| jerk | Incomm | `[0.7, 1.0, 1.0]` | `[0.181268, 0.0276048, -0.71032]` | `[-0.0395, -0.0888, -0.2442]` | `[0.220768, 0.116405, 0.46612]` | `strict_discrepancy` |
| jerk | Incomm | `[0.6, 1.0, 1.0]` | `[0.171535, 0.0204265, -0.683484]` | `[0.0179, -0.0971, -0.2816]` | `[0.153635, 0.117526, 0.401884]` | `strict_discrepancy` |

## Diagnosis by system

### Financial

- Results are mostly close to the published rows.
- The `[0.9, 1, 1]` incommensurate row crosses the sign boundary for an exponent close to zero.
- The RHS is nonsmooth because it contains `abs(x)`.

### Four-wing

- Several commensurate rows reproduce closely.
- Incommensurate rows retain discrepancies, including second-exponent sign changes.

### Jerk

- Several rows retain large `lambda_3` discrepancies.
- Review the exponential nonlinearity scale, `T_clone`, and ABM protocol interpretation.

## Ordered hypotheses

1. H1: cloning protocol is not identical to the published protocol.
2. H2: interpretation of `T_C`, `N_C`, `h_C`, or `K` differs.
3. H3: classical GS, modified GS, QR, or accumulated norm convention differs.
4. H4: near-zero exponents are strongly sensitive.
5. H5: incommensurate ABM handling requires a specific audit.
6. H6: published rounded values may retain ambiguity.

These are hypotheses, not conclusions. The current evidence does not justify
claiming that the article is incorrect.

## Near-zero sign policy

A quantitative absolute-error pass may still cross the strict sign boundary
for near-zero exponents. Use the additional diagnostic field
`near_zero_sign_boundary` before interpreting sign failures.

## Recommendations

- Do not promote F3.
- Review the completed bounded `T_clone` and `delta` sweeps before extending costly variants.
- Validate ABM `q=1` against an exact solution.
- Compare integer jerk against the independent reusable RK4 integrator.
- Review whether the article uses a transient before clone accumulation.
- Review whether clones restart on a long fiducial trajectory or restart block-locally.

## Methodological declaration

Passing or failing F3 does not validate or invalidate F2.
F3 does not certify chaos. F3 does not certify hiddenness.
The current state is `published_benchmarks_pending_discrepancy`.

## Reproduction limitations and missing article data

See `fischer2020_reproduction_limitations.md`.

No further bounded diagnostic sweep is required in the current scope; exact
reproduction requires protocol details not reported in the article or access
to the authors' implementation.

## Sensitivity sweep results

Controlled diagnostic sweeps recorded `164` unique runs across `delta, gs_policy, h, k, q1_mode, t_clone`. See the [sensitivity analysis report](sensitivity_analysis_report.md) for the run inventory, best improvements, persistent discrepancies, and hypothesis assessment. These sweeps do not promote validation.
