# Integrator Method Validation Phase

## Purpose

This phase validates each numerical integration method against **exact solutions**,
**manufactured solutions**, and **observed convergence order**.

It is **distinct** from `validation/integrator_crosscheck`:

| Phase | What it answers |
|---|---|
| **integrator_method_validation** | Is the integrator mathematically correct? Does it converge at the right rate against known solutions? |
| **integrator_crosscheck** | Are different integrators consistent with each other on chaotic systems? |

Method validation answers: *"Is the solver right?"*
Crosscheck answers: *"Do independent solvers agree on chaotic dynamics?"*

---

## EFORK3 — Already Validated Elsewhere

EFORK3 (Caputo fractional, 3-stage) is already validated at a high level in:

```
tools/validation/validate_efork_integrator.py
```

That script:
- Reproduces published terminal errors from Ghoreishi, Ghaffari, and Saad (2023) to within 6×10⁻⁹.
- Verifies the q=1 limit stage formula to floating-point precision (<10⁻¹⁵).
- Compares EFORK3 against ABM on short Chua fractional integrations.
- Compares against `scipy.solve_ivp` for the q=1 exponential decay problem.

**This phase does not re-validate EFORK3** (to avoid duplication). It records its status as:

```
validated_elsewhere_against_published_errors
reference_script: tools/validation/validate_efork_integrator.py
```

---

## ABM (Adams-Bashforth-Moulton Caputo) — Validated Here

ABM is used as the primary reference integrator for the Danca (2017) fractional Chua case.
It is validated here independently against:

### A1. Mittag-Leffler Exact Solution

Equation: `D^q y = λ y`, `y(0) = y₀`

Exact solution: `y(t) = y₀ · E_q(λ · t^q)`

where `E_q(z) = E_{q,1}(z)` is the one-parameter Mittag-Leffler function.

Cases: `q ∈ {0.25, 0.5, 0.8, 0.9998}`, `λ ∈ {-1, -5}`, `t_final=1`.

> **Note:** Caputo solutions often have an initial singularity of the form `t^q`, which
> degrades convergence order near `t=0`. We therefore check that error *decreases* as
> `h` is refined, and that the *observed order is positive*, without enforcing a strict
> theoretical rate.

### A2. Manufactured Solution t^m

Equation: `D^q y = -y + forcing(t)`, `y(0) = 0`

where `forcing(t) = Γ(m+1)/Γ(m+1-q) · t^{m-q} + t^m` is chosen so that `y(t) = t^m`
is the exact solution.

Cases: `q ∈ {0.25, 0.5, 0.8, 0.9998}`, `m ∈ {4, 5}`.

### A3. Diagonal Vector Linear System

Equation: `D^q X = A X`, `A = diag(-1, -2, -5)`, `X₀ = [1, 0.5, -0.25]`

Exact solution: `X_i(t) = X_i(0) · E_q(λᵢ · t^q)`

Cases: `q ∈ {0.5, 0.8, 0.9998}`.

### A4. External References (Optional — Not Required for CI)

Potential future comparisons (not implemented here, no download required):

- **FractionalDiffEq.jl**: Julia package for FDEs (external, not needed for CI).
- **FDE12 (Garrappa)**: MATLAB/Octave ABM implementation (external, not needed for CI).

---

## RK4 — Validated Here (q=1 only)

Classical 4th-order Runge-Kutta for integer-order ODEs. **Not used for q<1.**

### B1. Exponential Decay

`y' = -y`, `y(0) = 1`. Exact: `y(t) = exp(-t)`.

Expected observed convergence order: `~4`.

### B2. Harmonic Oscillator

`x' = y`, `y' = -x`, `[x(0), y(0)] = [1, 0]`.

Exact: `x(t) = cos(t)`, `y(t) = -sin(t)`.

Checks: error decreases, energy drift `|x²+y²-1|` decreases.

### B3. Diagonal Linear System (3D)

`X' = A X`, `A = diag(-1,-2,-3)`, `X₀ = [1, 0.5, -0.25]`.

Exact: `X_i(t) = X_i(0) · exp(λᵢ · t)`.

### B4. Comparison with scipy.solve_ivp (Optional)

`y' = -y + sin(t)`, `y(0) = 0`, `t_final = 5`.

Uses `solve_ivp(method='DOP853', rtol=1e-11, atol=1e-13)` as reference.

Skipped automatically if `scipy` is unavailable (though `scipy` is a core dependency).

---

## Dependencies

| Dependency | Required? | Purpose |
|---|---|---|
| `numpy` | **Required** | All numerical computations |
| `scipy` | Optional (but installed) | `solve_ivp` comparison for RK4/B4 |
| `pytest` | Required for tests | Test runner |
| `pyyaml` | Required | YAML config loading |

Nothing is downloaded at runtime.

---

## What This Phase Does NOT Certify

- It does **not** declare `hidden_verified`.
- It does **not** certify that any attractor is hidden or chaotic.
- Method validation and chaotic robustness are **separate concerns**:
  - A solver that is numerically correct on linear problems may still show
    integrator-specific step-size requirements on chaotic attractors.
  - See `validation/integrator_crosscheck` for chaotic robustness analysis.

---

## Commands

Run the full method validation:

```bash
python validation/python/run_integrator_method_validation.py --all
```

Run only ABM:

```bash
python validation/python/run_integrator_method_validation.py --method ABM
```

Run only RK4:

```bash
python validation/python/run_integrator_method_validation.py --method RK4
```

Run the test suite:

```bash
pytest tests/test_integrator_method_validation.py -v
```
