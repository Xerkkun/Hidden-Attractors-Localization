# Wolfram Language Algebraic and Numerical Validation

This directory contains the Wolfram Language (`.wl`) scripts that provide
**algebraic and numerical certification** of the mathematical formulas
used by the seed workflows and by independently validated fractional
operators.

---

## Purpose

These scripts **certify formulas**, not attractor existence:

1. **Lur'e form** — verify that `F(X) = P X + b ψ(rᵀ X)` is algebraically exact.
2. **Equilibria** — compute and verify `F(Xeq) = 0`.
3. **Jacobian** — verify analytical Jacobian against definition.
4. **Matignon criterion** — compute `|arg(λ)| > q π/2` margin for every equilibrium.
5. **Fractional transfer function** — verify `Ŵ_q(z) = rᵀ (zI − P)⁻¹ b` symbolically.
6. **Frequency evaluation** — evaluate at `z = (jω)^q = ω^q exp(j q π/2)`.
7. **Describing function** — first harmonic approximation `N(A₀)`.
8. **Frequency/Nyquist condition** — solve `Im[Ŵ_q(z)] = 0` numerically.
9. **Initial condition (seed)** — construct `X_seed = a₀ S[:,0]`.
10. **Wolfram–Python comparison** — export JSON/CSV for cross-checking.
11. **Fractional operators** — validate transformations, analytic identities,
    sampled weights, and manufactured finite-grid problems without promoting
    them to dynamical certification.

> **Mathematical warning**: The describing function and harmonic balance
> generate seeds and candidate parameters. They do **not** prove the
> existence of exact periodic orbits, nor do they verify attractor hiddenness.
> No script in this directory declares `hidden_verified`.

---

## Wolfram Engine Is Optional

Wolfram Engine / `wolframscript` is **not** a dependency of the main library.

- Simulations, integrators, and the full workflow run **without** Mathematica.
- These validation scripts run **only on demand** for algebraic certification.
- If `wolframscript` is absent, pytest tests are **skipped**, not failed.

---

## Directory Structure

```text
validation/wolfram/
├── common/
│   ├── ha_validation_common.wl          # Shared helpers (I/O, JSON, CSV)
│   ├── chua_saturation_validation.wl    # Saturation validator (S via similarity)
│   └── chua_arctan_validation.wl        # Arctan validator (S via similarity)
├── cases/
│   ├── chua_integer_saturation.wl       # q=1, nonsmooth
│   ├── chua_fractional_saturation.wl    # q=0.9998, nonsmooth
│   ├── chua_fractional_arctan.wl        # q=0.99, Wu arctan, rho=1
│   ├── chua_fractional_arctan_c590.wl   # q=0.9999, c590, rho!=1
│   ├── kalman_fitts_integer.wl          # source polynomial -> switching seed
│   ├── mavpd_integer.wl                 # source ODE -> direct harmonic seeds
│   ├── pll_lead_lag_integer.wl          # source H(s) -> zero-gain running seed
│   ├── sali_gali_integer.wl              # exact integer tangent alignment indices
│   ├── covariant_lyapunov_integer.wl     # exact q=1 Ginelli CLV fixtures
│   ├── gl_fractional_operator_validation.wl # GL/RL formulas and finite grids
│   ├── hadamard_fractional_operator.wl  # log transform, CQ, manufactured ABM
│   ├── atangana_baleanu_operator.wl     # ABC sampled operator at alpha=1/2
│   ├── abc_predictor_corrector.wl       # conventional ABC finite recurrence
│   ├── variable_order_caputo_type3_l1.wl # Type III formula, L1 and recurrence
│   ├── distributed_order_caputo_l1.wl   # integrated multinode L1 kernel
│   ├── multi_term_caputo_l1.wl          # finite Caputo sum, alpha=1 and recurrence
│   ├── correlation_dimension.wl         # q=2 counts and explicit log-log fit
│   └── permutation_entropy.wl           # ordinal ranks, ties, and finite entropy
└── template/
    └── new_lure_system_template.wl      # Blank template for new systems

validation/outputs/wolfram/              # Generated outputs (git-ignored)
validation/python/
├── run_wolfram_validations.py           # Python runner / CLI
├── compare_with_library.py              # Chua Python ↔ Wolfram comparison
├── compare_integer_nonchua_wolfram.py   # Non-Chua Python ↔ Wolfram comparison
├── sali_gali_integer_compare_wolfram.py # integer SALI/GALI public API comparison
├── covariant_lyapunov_integer_compare_wolfram.py # integer CLV public API comparison
├── hadamard_compare_wolfram.py          # Hadamard Python ↔ Wolfram comparison
├── abc_predictor_corrector_compare_wolfram.py # ABC solver comparison
├── variable_order_caputo_type3_compare_wolfram.py # Type III comparison
├── distributed_order_caputo_l1_compare_wolfram.py # distributed L1 comparison
├── multi_term_caputo_l1_compare_wolfram.py # finite multi-term facade comparison
├── correlation_dimension_compare_wolfram.py # finite correlation comparison
└── permutation_entropy_compare_wolfram.py # finite ordinal-pattern comparison

tests/
├── test_wolfram_validations.py          # Pytest suite (smoke + wolframscript)
├── test_wolfram_python_consistency.py   # Chua consistency tests (tolerances)
├── test_sali_gali_integer_wolfram.py # integer alignment-index consistency
├── test_covariant_lyapunov_wolfram.py # integer CLV consistency
├── test_hadamard_wolfram.py             # Hadamard identities + consistency
├── test_abc_predictor_corrector_wolfram.py # ABC solver consistency
├── test_variable_order_caputo_type3_wolfram.py # Type III consistency
├── test_distributed_order_caputo_wolfram.py # distributed L1 consistency
├── test_multi_term_caputo_wolfram.py # finite multi-term facade consistency
├── test_correlation_dimension_wolfram.py # finite counts and fit consistency
└── test_permutation_entropy_wolfram.py # ranks, ties, and entropy consistency
```

---

## How to Install WolframScript

1. Download **Wolfram Engine** (free for developers):
   [https://www.wolfram.com/engine/](https://www.wolfram.com/engine/)

2. Activate with a free Wolfram account.

3. Verify: `wolframscript -version`

---

## Running Validations

### All Included Cases

```bash
python validation/python/run_wolfram_validations.py --all
```

Outputs are written to `validation/outputs/wolfram/<system_id>/`.

The three non-Chua integer files are self-contained at the algebraic level.
They do not read the comparative LaTeX report, previously generated JSON, or
a published target-attractor initial condition. Their inputs are the cited
source model and exact parameter values. Each script exports
`report_input_used=false` and records its derivation steps.

After generating the Wolfram artifacts, compare them independently with the
Python implementation:

```bash
python validation/python/compare_integer_nonchua_wolfram.py
```

The integer SALI/GALI case constructs exact tangent histories with
`MatrixPower` and `MatrixExp`, derives GALI independently from Gram and
Cauchy--Binet volumes, and checks the equivalent SVD/LDI formulation. It uses
orthogonal and hyperbolic linear fixtures without labelling the latter as
chaotic attractors. Compare its temporary or retained summary through every
agreed public alignment API using:

```bash
python validation/python/sali_gali_integer_compare_wolfram.py \
    --summary validation/outputs/wolfram/sali_gali_integer_verified/sali_gali_integer_validation_summary.json
```

Only that explicit `sali_gali_integer_verified` promotion path is used as the
persisted comparison oracle. A failed or provisional summary retained under
`sali_gali_integer/` is evidence to preserve, not a promoted validation result;
until a new passing summary is promoted, the live test is authoritative.

This establishes finite integer tangent-algebra and propagation consistency
only. It does not classify general nonlinear dynamics, validate a Lyapunov
spectrum, or prove chaos, attraction, hiddenness, or fractional SALI/GALI.

The integer CLV case independently implements 80-digit modified Gram--Schmidt,
positive-diagonal QR histories, the Ginelli backward triangular recursion,
projective covariance, and unoriented pair angles for a nonnormal map and a
constant flow. It does not call Wolfram `QRDecomposition`/`Eigensystem`, HAFO,
or generated reports. Run the case, the public-API comparison, and its focused
test with:

```powershell
python validation/python/run_wolfram_validations.py `
    --case validation/wolfram/cases/covariant_lyapunov_integer.wl `
    --out C:\tmp
python validation/python/covariant_lyapunov_integer_compare_wolfram.py `
    --summary validation/outputs/wolfram/covariant_lyapunov_integer_verified/covariant_lyapunov_integer_validation_summary.json `
    --require-core
python -m pytest tests/test_covariant_lyapunov_wolfram.py -q
```

The verified promotion is restricted to
`validation/outputs/wolfram/covariant_lyapunov_integer_verified/`. Its passing
summary records 17/17 Wolfram checks and SHA-256
`C2366BC0379B8C73863431CAC8B72742286869293DA6A5CC6C6428C5D9DDB211`; the
passing `--require-core` comparison has SHA-256
`24BFCFBECDEAC466F3CBFFF7F7AE4869792D507A6C53E09629D0975A2A729AF0`.
These artifacts establish finite constant-cocycle `q=1` consistency only, not
nonlinear CLV convergence, hyperbolicity, chaos, attraction, hiddenness, or a
fractional-order CLV formulation.

The Hadamard case independently derives the exact change of variable
`u=log(t/a)`, Gamma/Beta identities for logarithmic powers and constants,
BDF1/BDF2 convolution-quadrature values, the `q -> 1` dilation-derivative
limit, and a manufactured constant-forcing Caputo--Hadamard IVP. Compare its
finite-grid output with the public HAFO implementation using:

```bash
python validation/python/hadamard_compare_wolfram.py \
    --summary validation/outputs/wolfram/hadamard_fractional_operator/hadamard_fractional_operator_validation_summary.json
```

The variable-order case fixes the Caputo Type III convention, derives the L1
weights from symbolic interval integrals, checks the constant-order reduction,
and reconstructs one manufactured finite recurrence. Compare its retained
artifact with the public HAFO API using:

```bash
python validation/python/variable_order_caputo_type3_compare_wolfram.py \
    --summary validation/outputs/wolfram/variable_order_caputo_type3_l1/variable_order_caputo_type3_l1_validation_summary.json
```

Its finite operator and recurrence errors are diagnostics, not convergence
rates or evidence of nonlinear stability, chaos, attraction, or hiddenness.

The distributed-order case derives every node contribution from a symbolic
Caputo-kernel integral, aggregates the multinode L1 kernel, and solves an
independently written linear recurrence. Compare the retained artifact with the
public HAFO solver using:

```bash
python validation/python/distributed_order_caputo_l1_compare_wolfram.py \
    --summary validation/outputs/wolfram/distributed_order_caputo_l1/distributed_order_caputo_l1_validation_summary.json
```

This verifies finite kernel and trajectory consistency only. It does not infer
quadrature convergence, nonlinear stability, chaos, attraction, or hiddenness.

The multi-term Caputo case treats the finite coefficients as equation
coefficients rather than a normalized order-density quadrature. It derives the
fractional interval weights with `Integrate`, includes the exact `alpha=1`
backward-Euler branch, verifies permutation and duplicate-order coalescence,
and advances an affine manufactured recurrence. Compare its temporary or
retained output with the semantic facade using:

```bash
python validation/python/multi_term_caputo_l1_compare_wolfram.py \
    --summary validation/outputs/wolfram/multi_term_caputo_l1/multi_term_caputo_l1_validation_summary.json
```

The coefficients in this fixture sum to `37/20`; therefore the validation also
detects any silent normalization. It remains finite consistency evidence and
does not prove convergence, stability, chaos, attraction, or hiddenness.

The correlation-dimension case independently enumerates unordered pairs after
a positive Theiler exclusion, applies the strict `distance < radius` rule, and
fits only the caller-declared log--log interval. Compare its retained artifact
with the public HAFO API using:

```bash
python validation/python/correlation_dimension_compare_wolfram.py \
    --summary validation/outputs/wolfram/correlation_dimension/correlation_dimension_validation_summary.json
```

This establishes finite-set count and regression consistency only. It does not
certify an asymptotic scaling region, fractal dimension, chaos, attraction, or
hiddenness.

The permutation-entropy case independently enumerates forward ordinal windows,
uses lexicographic Lehmer ranks, and covers delay plus both stable-index and
omitted-tie conventions. Compare the retained artifact with the public HAFO API
using:

```bash
python validation/python/permutation_entropy_compare_wolfram.py \
    --summary validation/outputs/wolfram/permutation_entropy/permutation_entropy_validation_summary.json
```

This establishes agreement for the declared finite fixtures and plug-in Shannon
calculation only. It does not establish an asymptotic entropy rate or certify
chaos, attraction, or hiddenness.

This validates formulas and finite sampled values only. It does not certify
nonlinear solver stability, chaos, attractor existence, or hiddenness.

This covers the Lur'e matrices, a complex transfer-function sample, and the
route-specific seed: the exact sign-switching point map for Kalman--Fitts, all
direct MAVPD branches and phases, and the analytic zero-gain PLL running cycle.

The fourth case validates the exact c590 parameterization used in Paper 07.
It checks the general nonlinearity
`a1*x + a2*ArcTan[rho*x]` with
`rho=1.7984259332820332`.  Its recorded fractional seed is tagged as the
result of bounded integer-order search and independent Caputo refinement; it
is not presented as a describing-function seed.
Its reviewable passing summaries and numeric exports are tracked in
`validation/chua_fractional_arctan_c590/algebraic_validation/`.

For `chua_fractional_saturation`, the official algebraic validator consumes
the generated prefixed CSV files directly from that ignored output directory.
It records their hashes and both Wolfram/Python consistency summaries in
`validation/02_algebraic_validation/algebraic_validation_validation_summary.json`.
The official stage is closed only when those resolved artifacts exist and
their comparisons pass.

### Single Case

```bash
python validation/python/run_wolfram_validations.py \
    --case validation/wolfram/cases/chua_fractional_saturation.wl \
    --out validation/outputs/wolfram
```

### Direct WolframScript Call

```powershell
$env:WOLFRAM_OUT = "validation/outputs/wolfram/chua_fractional_saturation"
wolframscript -file validation/wolfram/cases/chua_fractional_saturation.wl
```

The Python runner is preferred: `--out` denotes the base directory and the
runner appends `<system_id>`. It passes the final case directory through
`WOLFRAM_OUT`, which is reliable on installations where `wolframscript`
does not preserve trailing script arguments.

---

## Running Pytest

```bash
# Only Wolfram-marked tests (skipped if wolframscript absent)
pytest -m wolfram -v

# Skip Wolfram tests (for CI without Wolfram Engine)
pytest -m "not wolfram"

# All tests (Wolfram tests skip gracefully if wolframscript is absent)
pytest
```

---

## Generated Outputs

Each `.wl` case script writes:

| File                            | Content                                     |
| ------------------------------- | ------------------------------------------- |
| `<id>_validation_summary.json`  | Overall pass/fail + test list               |
| `<id>_symbolic_summary.json`    | P, b, r, W(z) as Mathematica expressions    |
| `<id>_equilibria_residuals.csv` | ‖F(Xeq)‖ per equilibrium                    |
| `<id>_jacobians.csv`            | Jacobian matrix entries at each equilibrium |
| `<id>_eigenvalues_matignon.csv` | Eigenvalues + Matignon margin per q         |
| `<id>_seed_data.json`           | ω₀, k, a₀, d, S, X_seed per candidate       |
| `<id>_seed_summary.csv`         | Tabular summary of seed data                |
| `<id>_recorded_candidate.json`  | c590 parameters, recorded seed, provenance, and RHS |

The `passed` field in `*_validation_summary.json` must be `true` before
the corresponding algebraic result is cited.  For c590, this status validates
the system algebra and recorded parameter/seed consistency; it does not
validate how the seed was dynamically selected.

---

## How S Is Constructed (Mathematical Constraint)

The transformation matrix **S is never built from eigenvectors**.
It is obtained by solving the similarity relation:

```text
P₀ S = S Hq
```

where

```text
P₀ = P + k b rᵀ

Hq = [[zr, -zi, 0],
      [zi,  zr, 0],
      [0,   0, -d]]

zr = ω₀^q cos(q π/2)
zi = ω₀^q sin(q π/2)
```

The initial seed is then:

```text
X_seed = a₀ · S[:, 0]
```

i.e., **a₀ times the first column of S**.

---

## Adding a New Lur'e System

Copy `validation/wolfram/template/new_lure_system_template.wl` and fill in:

- `system_id`
- State vector and field `F(X)`
- Matrices `P`, `b` (`bvec`), `r` (`rvec`)
- Nonlinearity `psi[s_]`
- Numerical parameter values in `params`
- Fractional order(s) in `qCases`
- Describing function `Npsi[a]` (or set to `None` and skip)
- Expected equilibria or seed initial guesses

The template automatically:

- Verifies the Lur'e form residual
- Computes `Ŵ(z) = rᵀ (zI − P)⁻¹ b`
- Sets up the similarity equation `P₀ S = S Hq` to solve
- Exports a JSON summary

---

## Tolerances and Verification Scope

### Tolerances

| Quantity                                          | Symbolic                   | Numeric                                       |
| ------------------------------------------------- | -------------------------- | --------------------------------------------- |
| Lur'e residual                                    | Exactly 0 via FullSimplify | —                                             |
| Chua saturation equilibrium residual              | —                          | < 1e-20 (high-precision Mathematica)          |
| Chua arctan equilibrium residual                  | —                          | < 1e-12 (high-precision Mathematica)          |
| ω₀ Python vs Wolfram                              | —                          | < 1e-8                                        |
| k and W_q differences                             | —                          | < 1e-8                                        |
| Describing-function residual \|N_py(a₀_WL)−k_WL\| | —                          | < 1e-8                                        |
| X_seed components                                 | —                          | < 1e-7                                        |
| Eigenvalues                                       | —                          | < 1e-7                                        |
| W(z) transfer function evaluation                 | —                          | < 1e-8                                        |
| Similarity residual ‖P₀ S − S Hq‖                 | —                          | Matching case `SimilarityTolerance` (< 1e-16) |

### Validation Scope vs. Consistency Verification

It is critical to distinguish between the scope of the Wolfram Language algebraic validation and the Python consistency checks:

1. **Wolfram Language Validation**:
   - Focuses on mathematical and symbolic proofs (Lur'e form equivalence, exact transfer function derivation, symbolic similarity formulation).
   - Solves the frequency equation to high precision (using 70-digit working precision) to find candidate seed frequencies ($\omega_0$) and parameters ($k$, $d$, $h$).
   - Verifies numerical equilibria and similarity transformation residuals.

2. **Python Consistency Checks**:
   - Cross-checks the exported Wolfram quantities (matrices, equilibria, eigenvalues, transfer function evaluations, seed vectors) against the Python library's implementations to ensure consistency.
   - Evaluates the describing function and checks that the amplitude residual $|N(a_0) - k|$ satisfies the tolerance ($< 1\text{e-}8$).
   - Direct matching of eigenvalues and equilibria using permutation distance metrics.

3. **No Attractor Hiddenness Certification**:
   - Neither the Wolfram validation nor the Python consistency check certifies `hidden_verified` on its own.
   - These scripts verify the mathematical validity of the seeds and system forms. The verification of the attractor being hidden requires complete simulation, integration, and basin of attraction checks.

## Tempered BDF convolution quadrature

`cases/tempered_convolution_quadrature.wl` is an independent 80-digit oracle
for BDF1/BDF2 tempered RL and exponentially conjugated Caputo sampled
operators. It builds ordinary BDF fractional weights through both a recurrence
and a separate generalized-binomial factor expansion, then compares damped
weights against explicit conjugation. It does not import HAFO source, formulas,
reports, or retained outputs.

The case checks 18 assertions covering scalar and componentwise fixtures,
`lambda=0`, `q=1`, both definitions, both BDF orders, and manufactured endpoint
convergence. The public-core comparator is:

```powershell
python validation/python/tempered_convolution_quadrature_compare_wolfram.py `
  --summary validation/outputs/wolfram/tempered_convolution_quadrature_verified/tempered_convolution_quadrature_validation_summary.json `
  --require-core
```

The retained run passed 18/18 Wolfram assertions. Its independent Python
maximum difference was `1.30841518175551e-14`; the required public HAFO
comparison maximum was `4.44089209850063e-15`. This is finite
algebraic/numerical consistency only, not a general convergence or stability
theorem and not evidence of chaos, attraction, or hiddenness.
