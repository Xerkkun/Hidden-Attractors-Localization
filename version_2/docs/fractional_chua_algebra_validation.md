# Fractional Non-Smooth Chua Algebra Validation

This page records the algebraic audit of the non-smooth fractional Chua case
at `q=0.9998`. It validates equations, equilibria, regional Jacobians, the
Lur'e split, the transfer-function convention, and harmonic seed branches.
It does not by itself validate time integration, chaos, or hiddenness.

## Registered Model

The commensurate Caputo system is

```text
^C D_t^q x = alpha * (y - x - f(x))
^C D_t^q y = x - y + z
^C D_t^q z = -beta * y - gamma * z
f(x) = m1*x + (m0-m1)*sat(x)
```

with

| Parameter | Value |
|-----------|------:|
| `alpha` | `8.4562` |
| `beta` | `12.0732` |
| `gamma` | `0.0052` |
| `m0` | `-0.1768` |
| `m1` | `-1.1468` |
| `q` | `0.9998` |

Danca (2017), Section 3.3, supplies the published external reference for
this model, parameter set, order, ABM integration route, and
equilibrium-neighborhood interpretation. Petras (2008) supplies a circuit
derivation of the fractional PWL Chua model but not this exact validation
parameter set. Sene (2021) is methodologically relevant for equilibrium,
local-stability, Lyapunov, and bifurcation reporting, but its Chua variant
does not contain the `-gamma*z` damping term and is therefore not a direct
numeric reference for this case.

## Cross-Tool Results

Python, MATLAB, and Wolfram reproduce the same equilibria:

| Equilibrium | State |
|-------------|-------|
| `E0` | `(0, 0, 0)` |
| `E+` | `(6.588307886539, 0.002836402256, -6.585471484283)` |
| `E-` | `(-6.588307886539, -0.002836402256, 6.585471484283)` |

At `q=0.9998`, the Matignon threshold is `1.570482167530` radians. The
central equilibrium is stable and the external equilibria are unstable:

| Region | Eigenvalues | Result |
|--------|-------------|--------|
| Inner, `E0` | `-7.9587261113`, `-0.0038088643 +/- 3.2494460858 i` | stable |
| Outer, `E+`, `E-` | `2.2193492642`, `-0.9915895521 +/- 2.4067596392 i` | unstable |

## Completion Matrix

| Element | Acceptance tolerance | Recorded maximum error | Status |
|---------|---------------------:|-----------------------:|:------:|
| Equilibrium substitution in Python, MATLAB, Wolfram | `||F(E_i)||_2 < 1e-10` | `1.10e-15` | passed |
| Analytic Jacobian cross-tool agreement | relative error `< 1e-10` | `0.00e+00` | passed |
| Python analytic vs. central finite-difference Jacobian | relative error `< 1e-6` | `2.47e-09` | passed |
| Eigenvalues in Python, MATLAB, Wolfram | relative error `< 1e-8` | below `1e-15` | passed |
| Matignon classification at `q=0.9998` | same signed margin/classification | `E0` stable; `E+`, `E-` unstable | passed |

The machine-readable evidence is in
`validation/01_algebra/equilibria_cross_tool_residuals.csv`,
`jacobian_cross_tool_comparison.csv`,
`jacobian_finite_difference_check.csv`, and
`eigenvalues_cross_tool_comparison.csv`.

## Transfer Convention

The MATLAB/Wolfram derivation uses

```text
W_report(z) = r^T (z I - P)^(-1) q_v
```

whereas the Python API evaluates

```text
W_code(omega) = r^T (P - z I)^(-1) q_v = -W_report(z),
z = (i omega)^q.
```

Consequently, the equivalent closure relations are

```text
1 - k * W_report(z) = 0
1 + k * W_code(omega) = 0
```

after setting the describing-function gain to `k = N(A)`. Comparisons must
normalize this sign before claiming agreement.

## Harmonic Branches

The two centered describing-function branches reproduced by MATLAB and
Python are:

| Branch | `omega0` | `k` | `a0` | Initial seed |
|--------|---------:|----:|-----:|--------------|
| 1 | `2.040286051079` | `0.210022792962` | `5.851767785486` | `(5.8517677855, 0.3704086003, -8.3609729344)` |
| 2 | `3.244926730975` | `0.956945404928` | `1.053016610257` | `(1.0530166103, 0.8532234830, -1.5095086807)` |

MATLAB records modal residual norms of `5.207e-16` and `1.948e-15`.

## Wolfram Source Correction

The supplied Wolfram Language source confirms the equilibrium equations,
regional Jacobian, characteristic polynomial identity, Lur'e residual,
transfer-function identity, modified determinant, describing function, and
modal-seed construction. Its real-imaginary separation block does not execute
cleanly as supplied because it assigns to the protected Mathematica symbol
`Tr`. The promoted source must rename that variable, for example to `Treal`,
before the source can be registered as a completely passing symbolic run.

## Correction Required In `170526.pdf`

The fractional non-smooth derivation later uses the internally consistent
report convention `1 - k*W_report = 0` and `k = 1/Re(W_report)`. However,
earlier general text in the same PDF writes `1 + W_q*N = 0` while defining
`W_q = r^T(s^q I-P)^(-1)b` with `b = (-alpha,0,0)^T`. The smooth arctan
discussion repeats this `1 + W_q*N` expression after its own derivation uses
`1 - k*W_q = 0`. Those occurrences should be changed to the minus convention,
or the transfer must be redefined explicitly with the opposite sign.

## Remaining Validation Stages

The algebraic validation stage is complete. The following remain independent
validation stages:

1. integrator convergence and comparison against an independent method such as
   ABM;
2. dynamic checks such as FFT/PSD, Lyapunov estimates, sections, bounded
   ranges, and bifurcations;
3. hiddenness tests from equilibrium neighborhoods and refined basin maps,
   including Danca's published neighborhood radius `delta=0.01` as a
   reference control.

## References

- M. F. Danca, "Hidden Chaotic Attractors in Fractional-Order Systems,"
  *Nonlinear Dynamics*, 89, 577--586, 2017.
- I. Petras, "A Note on the Fractional-Order Chua's System," *Chaos,
  Solitons and Fractals*, 38, 140--147, 2008.
- N. Sene, "Mathematical Views of the Fractional Chua's Electrical Circuit
  Described by the Caputo-Liouville Derivative," *Revista Mexicana de
  Fisica*, 67(1), 91--99, 2021.
