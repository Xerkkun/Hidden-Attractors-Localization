# Lyapunov Methods Audit

> **F0 — integer_qr_benettin (frozen)**
> Phase F0 freezes the existing integer-order QR-Benettin method and
> establishes the methodological boundary between integer-order and
> fractional Caputo Lyapunov analysis.

---

## Frozen method: `integer_qr_benettin`

| Property | Value |
|---|---|
| Canonical identifier | `integer_qr_benettin` |
| Implemented in | `hidden_attractors/analysis/lyapunov.py` |
| Entry points | `integer_lyapunov_exponents`, `integer_qr_benettin_lyapunov_exponents`, `integer_system_lyapunov_exponents` |
| Derivative model | Integer-order ODE (q = 1) |
| Variational equation | Φ' = J(X) Φ (first-order, memoryless) |
| Orthonormalisation | QR decomposition (`numpy.linalg.qr`) |
| Result type | Finite-time local Lyapunov exponent estimates |
| Jacobian required | Yes (analytic or finite differences) |
| q support | q = 1.0 **only** |
| F0 status | Implemented ✓ · Validated ✓ (for q=1) |

### Algorithm summary

1. Optionally integrate a burn-in phase (state only, no accumulation).
2. Propagate the variational basis: `Φ ← Φ + h · J(X) · Φ`.
3. Advance the state: `X ← efork_q1_step(F, X, h)`.
4. Every `reorthonormalize_every` steps:
   - Compute `Q, R = QR(Φ)`.
   - Accumulate `Σ += log|diag(R)|`.
   - Reset `Φ = Q`.
5. Divide by elapsed time: `λᵢ = Σᵢ / T`.

### References

- **Benettin et al. 1980** — G. Benettin, L. Galgani, A. Giorgilli, J.-M. Strelcyn,
  "Lyapunov Characteristic Exponents for Smooth Dynamical Systems and for
  Hamiltonian Systems", *Meccanica* 15, 1980.

- **Wolf et al. 1985** — A. Wolf, J.B. Swift, H.L. Swinney, J.A. Vastano,
  "Determining Lyapunov Exponents from a Time Series", *Physica D* 16, 1985.

- **Danca & Kuznetsov 2018** — M.-F. Danca, N. Kuznetsov,
  "Matlab Code for Lyapunov Exponents of Fractional-Order Systems",
  *Int. J. Bifurcation Chaos* 28(5), 2018.
  — Establishes that fractional Caputo spectra require integrating the
  **extended fractional original–variational system with memory**.
  The integer QR method is **not valid for q < 1**.

---

## Scope

### ✅ In scope (F0)

- Integer-order ODE systems (q = 1).
- Requires Jacobian: analytic or central finite differences.
- Finite-time, local Lyapunov exponent estimates.
- QR reorthonormalisation every `reorthonormalize_every` steps.

### ❌ Out of scope (F0)

- **NOT** valid for Caputo fractional systems (q < 1).
- **NOT** handling fractional memory.
- **DOES NOT** certify chaos.
- **DOES NOT** certify hiddenness of attractors.

```
chaos_certified_by_this_pipeline: false
hiddenness_certified_by_this_pipeline: false
```

> **Methodological warning:** This routine is **not a validated Caputo
> fractional Lyapunov method**. It is restricted to **q = 1**.
> Fractional Caputo spectra require a dedicated extended-memory variational
> method integrating the full original–variational system with Caputo memory.

---

## q-Validation gate

Both `integer_lyapunov_exponents` and `integer_qr_benettin_lyapunov_exponents`
accept an optional `q` parameter.  If `abs(q - 1.0) > 1e-9`, a `ValueError`
is raised:

```
ValueError: integer_qr_benettin is valid only for q=1 (integer-order ODE);
received q=0.99.  Use a fractional Lyapunov method for Caputo q<1
(e.g., fractional_variational_abm_qr — not yet implemented in F0).
```

---

## Future phases (not implemented in F0)

| Method ID | Model | Status |
|---|---|---|
| `fractional_variational_abm_qr` | Caputo q<1 | NOT implemented · NOT validated |
| `fractional_cloned_dynamics_abm` | Caputo q<1 | NOT implemented · NOT validated |
| `zero_one_test` | — | NOT implemented |
| PSD/FFT analysis | — | NOT implemented |
| Boundedness checks | — | NOT implemented |

### Fractional methods — design intent

Fractional Caputo Lyapunov spectra will require:
1. An Adams–Bashforth–Moulton (ABM) predictor-corrector integrating the
   **extended (original + variational) system** with full Caputo memory.
2. QR or Gram–Schmidt reorthonormalisation applied to the variational block.
3. Validation against published results (Danca & Kuznetsov 2018).

These are tracked as `fractional_variational_abm_qr` and
`fractional_cloned_dynamics_abm` in the method registry
(`hidden_attractors/analysis/lyapunov_methods.py`).

---

## Method registry

Static metadata for all known methods is in:

```
hidden_attractors/analysis/lyapunov_methods.py
```

```python
from hidden_attractors.analysis.lyapunov_methods import LYAPUNOV_METHODS

info = LYAPUNOV_METHODS["integer_qr_benettin"]
print(info.implemented)  # True
print(info.validated)    # True
print(info.q_support)    # "q=1 only"
```

---

## What F0 does NOT implement

- `fractional_variational_abm_qr`
- `fractional_variational_abm_gs`
- `fractional_cloned_dynamics_abm`
- 0–1 test
- PSD/FFT
- Poincaré sections
- `chaos_validation_summary`
- Any modification to `validation/wolfram`
- Any modification to `validation/fractional_memory_validation`
- Any modification to `validation/continuation_memory_validation`
- Any modification to `validation/published_continuation_comparison`
- Hiddenness verification
