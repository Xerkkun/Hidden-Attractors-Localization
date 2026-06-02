# Fischer 2020 reproduction limitations for F3

## 1. Status

F3 is implemented. All 24 published rows and 164 bounded sensitivity runs
across six axes were executed. Reproduction is partial, and the method remains
not validated.

```text
Official status: published_benchmarks_pending_discrepancy
validated: false
validated_against_published_benchmarks: false
```

The diagnostic closure is
`closed_with_documented_discrepancies`. No further bounded sweep is required
in the current scope. Exact reproduction requires protocol details not
reported in the article or access to the authors' implementation.

## 2. Numerical comparison summary

| System | Rows | Quantitative passes | Sign-pattern passes | Failures | Main discrepancy |
|---|---:|---:|---:|---:|---|
| financial | 8 | 5 | 2 | 1 | near-zero boundary crossing in one incommensurate row |
| four-wing | 8 | 5 | 0 | 3 | incommensurate second-exponent sign changes |
| jerk | 8 | 0 | 4 | 4 | persistent large `lambda_3` gaps and sign mismatches |

## 3. Persistent discrepancies

The following rows remain strict discrepancies after the bounded sweeps.
Values are taken from `fischer2020_discrepancy_matrix.csv`.

| System | Orders | Published LCE | Calculated LCE | Absolute error | Discrepancy class | Near-zero crossing | Most plausible hypothesis |
|---|---|---|---|---|---|---|---|
| financial | `[0.9, 1.0, 1.0]` | `[0.1445, 0.0012, -0.4419]` | `[0.089733, -0.0281534, -0.492609]` | `[0.054767, 0.0293534, 0.0507088]` | `strict_discrepancy` | yes | H5: incommensurate ABM handling |
| jerk | `[0.8, 0.8, 0.8]` | `[-0.0047, -0.2864, -0.3634]` | `[0.00233586, -0.561171, -1.66322]` | `[0.00703586, 0.274771, 1.29982]` | `strict_discrepancy` | yes | H1/H2: clone protocol and interval interpretation |
| jerk | `[0.8, 1.0, 1.0]` | `[0.081, -0.0162, -0.4017]` | `[0.188783, 0.0201008, -0.717387]` | `[0.107783, 0.0363008, 0.315687]` | `strict_discrepancy` | yes | H1/H2/H5: protocol and incommensurate handling |
| jerk | `[0.7, 1.0, 1.0]` | `[-0.0395, -0.0888, -0.2442]` | `[0.181268, 0.0276048, -0.71032]` | `[0.220768, 0.116405, 0.46612]` | `strict_discrepancy` | no | H1/H2/H5: protocol and incommensurate handling |
| jerk | `[0.6, 1.0, 1.0]` | `[0.0179, -0.0971, -0.2816]` | `[0.171535, 0.0204265, -0.683484]` | `[0.153635, 0.117526, 0.401884]` | `strict_discrepancy` | no | H1/H2/H5: protocol and incommensurate handling |

## 4. Sensitivity conclusion

- `T_clone` improves some cases, especially integer jerk with `T_clone=10`.
- `delta` moves several four-wing incommensurate rows from strict discrepancy
  to sign-pattern support.
- `h` and `K` produce smaller or localized changes.
- `gs_policy` does not explain the differences: modified GS, classical GS,
  and QR are numerically equivalent in the bounded sweeps.
- `q1_mode` improves the maximum exponent for integer jerk but does not
  resolve the full spectrum.

The sensitivity evidence supports protocol ambiguity, not validation.

## 5. Missing information in Fischer et al. 2020 needed for exact reproduction

### 5.1. Clone initialization and restart protocol

- Whether the fiducial trajectory is integrated from `X_0` through a transient
  before accumulation starts.
- Whether clones are reinitialized at each block around the current fiducial
  trajectory.
- Whether fractional state memory is retained between blocks or restarted.
- Whether each block uses local Caputo history or accumulated history.
- Whether clones share prior history with the fiducial trajectory or only its
  initial point at the start of a block.

### 5.2. Treatment of fractional memory

- The effective lower bound of the Caputo derivative in each block.
- Whether ABM uses full memory from `t_0`, truncated memory, or block restart.
- How clone memory is handled after Gram-Schmidt.
- Whether a history window or only the final state is transported.
- How ABM weights are implemented for incommensurate orders.

### 5.3. ABM predictor-corrector convention

- Exact predictor and corrector weights and their indexing.
- Whether PECE correction is applied once or iterated.
- How `q=1` is treated: fractional ABM or a classical integer integrator.
- Whether `h`, `h_C`, `N`, and `N_C` match the implementation convention.

### 5.4. Clone interval parameters

- The exact relationship between `T_clone`, `N_C`, and `h_C`.
- Whether `T_C = N_C h_C` is exact or approximate.
- Whether jerk cases A/B/C apply to every row or selected rows.
- Whether a transient precedes the counted `K` blocks.
- Whether `K` includes or excludes burn-in blocks.

### 5.5. Orthonormalization convention

Although the sweeps weaken this as the primary explanation, the article does
not specify classical or modified Gram-Schmidt, column order, whether norms
are accumulated before or after projection subtraction, whether an equivalent
QR convention is used, or whether each clone is rescaled exactly by `delta`.

### 5.6. Log accumulation and ordering of exponents

- Whether accumulated norms are raw difference norms or equivalent GS/QR
  diagonal factors.
- Whether exponents are reordered at the end.
- Whether reported LCEs use descending order, clone order, or post-processing.
- Whether transient blocks are discarded before averaging.

### 5.7. Incommensurate fractional systems

- Whether each component uses its own independent ABM kernel.
- How components with different orders are synchronized.
- Whether every order uses the same time step.
- Whether an equivalent commensurate-system transformation is used.
- How per-component memory is handled after clone restart.

### 5.8. Reported numerical precision

- Arithmetic precision, software, and internal tolerances.
- Rounding convention for reported LCEs.
- Whether LCEs are from one run or averaged across runs.
- Whether the 0-1 test uses the same trajectory as the LCE calculation.

### 5.9. Jerk system implementation details

Large jerk `lambda_3` gaps require checking the exact `I_c=10e-9` scale,
whether that notation means `10 * 10^-9`, the use of `V_T=0.026`, exponential
treatment, possible saturation or overflow, and any undocumented scaling or
normalization. This does not establish an error in the article.

## 6. Reproduction status statement

The current implementation reproduces part of Fischer et al. 2020 but does
not reproduce all reported LCE values. The remaining discrepancies cannot be
resolved unambiguously with the information reported in the paper alone.
Therefore F3 remains implemented but not validated against the published
benchmarks.
