# Scientific Software Validation

This page records the fast scientific-software validation layer. These tests
check algebra, numerical consistency, and operational classifiers. They do not
certify chaos or hidden attractors. Long reproductions remain explicit opt-in
tests marked `published`, `slow`, or `native`.

## Transfer-Function Sign Convention

The generic seed-generation API intentionally preserves the convention used by
the promoted Kuznetsov integer-reference MATLAB/Wolfram artifacts:

```text
W_code(s) = c^T (P - s I)^(-1) b
1 + k * W_code(s) = 0
k = -1 / Re(W_code(s))
```

The normalized thesis/report form is:

```text
W_report(s) = c^T (s I - P)^(-1) b = -W_code(s)
1 - k * W_report(s) = 0
k = 1 / Re(W_report(s))
```

Both forms are valid only when the closure relation and gain sign are changed
together. The Python API is not silently inverted because the Kuznetsov 2017
Nyquist branch, gain, amplitude, and harmonic seed depend on the preserved
`P-sI` convention. Fractional evaluations always use the principal branch
`s=(j*omega)^q=omega^q*exp(j*q*pi/2)`, never the integer shortcut `j*omega`
when `q<1`.

## Validation Matrix

| Function | File | Implemented test | Reference | Tolerance | Pytest marker |
| --- | --- | --- | --- | --- | --- |
| `solve_equilibria` | `hidden_attractors/verification/equilibria.py` | Non-smooth Kuznetsov/Danca and arctan Wu 2023 substitution plus symmetry | Danca 2017; Wu et al. 2023 | `norm(F(E)) < 1e-8` | default |
| `compute_jacobian` | `hidden_attractors/verification/jacobian.py` | Inner/outer non-smooth regions and smooth arctan derivative against central differences | Chua split; Wu et al. 2023 | `1e-12` analytic, `1e-5` finite difference | default |
| `classify_equilibrium_stability` | `hidden_attractors/verification/stability.py` | Controlled real and complex spectra, including `q=1` | Matignon; Danca 2017 | `1e-12` measure | default |
| `lure_transfer_function` | `hidden_attractors/seed_generation/lure.py` | Synthetic `2x2` cofactor formula, cross-tool Chua artifact, sign normalization | Kuznetsov et al. 2017 | `1e-10` synthetic, `1e-8` cross-tool | default |
| `fractional_iomega_power` | `hidden_attractors/seed_generation/core.py` | Principal-branch values and invalid input rejection | Weyl-Caputo frequency rule | `1e-12` | default |
| `lure_describing_function` | `hidden_attractors/seed_generation/lure.py` | Closed form against first-harmonic quadrature for saturation and arctan | Kuznetsov et al. 2017; Wu et al. 2023 | `1e-6` | default |
| `find_lure_omega_gain_candidates`, `find_lure_harmonic_seed` | `hidden_attractors/seed_generation/lure.py` | Kuznetsov 2017 integer branch, gain, amplitude, seed, and closure residual | Kuznetsov et al. 2017 | `1e-8` scalars, `1e-7` seed | `published` |
| `caputo_abm_integrate` | `hidden_attractors/integrations/abm.py` | Scalar Caputo decay against Mittag-Leffler on refined meshes; window-memory separation | Diethelm-Ford-Freed; Danca 2017 | monotone error decrease | default |
| `efork_integrate` | `hidden_attractors/integrations/efork.py` | Full-memory scalar decay against ABM; native backend against Python fallback; documented `q=1` route | Ghoreishi et al. 2023; ABM cross-check | monotone difference decrease, `1e-12` native | default, `native` |
| `evaluate_target_match` | `hidden_attractors/verification/hiddenness.py` | Close and separated synthetic point clouds with `nn_percentile` | Leonov-Kuznetsov operational basin distinction | `1e-2` cloud distance | default |
| `classify_hiddenness_verdict` | `hidden_attractors/verification/classifiers.py` | Sampled-radii compatible, self-excited contact, unsupported, and numerical-failure states | Leonov-Kuznetsov operational basin distinction | exact label | default |
| `validate_run_metadata`, `validate_hiddenness_promotion_metadata` | `hidden_attractors/reproducibility.py` | Complete auditable run envelope, full-history promotion requirement, and conservative fallback | repository reproducibility contract | exact required fields and labels | default |

## Scope Boundary

Describing-function, Nyquist, bounded-trajectory, and synthetic cloud tests are
candidate-generation or software-validation evidence. They do not establish a
global basin result. The Machado fractional describing-function family remains
an auxiliary extension and is not a hiddenness proof. Any operational
hiddenness promotion must continue to use the full neighborhood and basin
contract documented in `hiddenness_verification.md`. Every maintained run
writes `run_metadata.json`; an incomplete envelope blocks the strong label and
emits `compatible_with_hiddenness_under_tested_radii`.

## References

- G. A. Leonov and N. V. Kuznetsov, hidden-attractor definition and the
  hidden-versus-self-excited basin distinction.
- N. V. Kuznetsov et al. (2017), localization of hidden Chua attractors by the
  describing-function method.
- M.-F. Danca (2017), fractional-order hidden chaotic attractors.
- K. Diethelm, N. J. Ford, and A. D. Freed, Caputo ABM predictor-corrector.
- D. Matignon, fractional-order stability criterion.
- Wu et al. (2023), fractional Chua system with arctan nonlinearity.
- Machado-family FDF: auxiliary extension only, not hiddenness certification.
