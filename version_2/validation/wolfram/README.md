# Wolfram Language Algebraic and Numerical Validation

This directory contains the Wolfram Language (`.wl`) scripts that provide
**algebraic and numerical certification** of the mathematical formulas
used to locate initial conditions (seeds) for the hidden attractor search.

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
│   └── chua_fractional_arctan.wl        # q=0.9998, arctan
└── template/
    └── new_lure_system_template.wl      # Blank template for new systems

validation/outputs/wolfram/              # Generated outputs (git-ignored)
validation/python/
├── run_wolfram_validations.py           # Python runner / CLI
└── compare_with_library.py             # Python ↔ Wolfram comparison

tests/
├── test_wolfram_validations.py          # Pytest suite (smoke + wolframscript)
└── test_wolfram_python_consistency.py   # Consistency tests (tolerances)
```

---

## How to Install WolframScript

1. Download **Wolfram Engine** (free for developers):
   https://www.wolfram.com/engine/

2. Activate with a free Wolfram account.

3. Verify: `wolframscript -version`

---

## Running Validations

### All Three Cases

```bash
python validation/python/run_wolfram_validations.py --all
```

Outputs are written to `validation/outputs/wolfram/<system_id>/`.

### Single Case

```bash
python validation/python/run_wolfram_validations.py \
    --case validation/wolfram/cases/chua_fractional_saturation.wl \
    --out validation/outputs/wolfram/chua_fractional_saturation
```

### Direct WolframScript Call

```bash
wolframscript -file validation/wolfram/cases/chua_fractional_saturation.wl \
    --out validation/outputs/wolfram/chua_fractional_saturation
```

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

| File | Content |
|------|---------|
| `<id>_validation_summary.json` | Overall pass/fail + test list |
| `<id>_symbolic_summary.json` | P, b, r, W(z) as Mathematica expressions |
| `<id>_equilibria_residuals.csv` | ‖F(Xeq)‖ per equilibrium |
| `<id>_jacobians.csv` | Jacobian matrix entries at each equilibrium |
| `<id>_eigenvalues_matignon.csv` | Eigenvalues + Matignon margin per q |
| `<id>_seed_data.json` | ω₀, k, a₀, d, S, X_seed per candidate |
| `<id>_seed_summary.csv` | Tabular summary of seed data |

The `passed` field in `*_validation_summary.json` must be `true` before
any seed is used in simulations.

---

## How S Is Constructed (Mathematical Constraint)

The transformation matrix **S is never built from eigenvectors**.
It is obtained by solving the similarity relation:

```
P₀ S = S Hq
```

where

```
P₀ = P + k b rᵀ

Hq = [[zr, -zi, 0],
      [zi,  zr, 0],
      [0,   0, -d]]

zr = ω₀^q cos(q π/2)
zi = ω₀^q sin(q π/2)
```

The initial seed is then:

```
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

| Quantity | Symbolic | Numeric |
|---------|---------|---------|
| Lur'e residual | Exactly 0 via FullSimplify | — |
| Chua saturation equilibrium residual | — | < 1e-20 (high-precision Mathematica) |
| Chua arctan equilibrium residual | — | < 1e-12 (high-precision Mathematica) |
| ω₀ Python vs Wolfram | — | < 1e-8 |
| k and W_q differences | — | < 1e-8 |
| a₀ Python vs Wolfram | — | < 1e-8 |
| X_seed components | — | < 1e-7 |
| Eigenvalues | — | < 1e-7 |
| W(z) transfer function evaluation | — | < 1e-8 |
| Similarity residual ‖P₀ S − S Hq‖ | — | Matching case `SimilarityTolerance` (< 1e-16) |

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

