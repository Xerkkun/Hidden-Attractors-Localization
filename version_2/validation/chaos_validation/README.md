# Chaos Validation — Phase F

> **F0 — Audit and freeze of integer_qr_benettin**
> This directory will house phase-F validation artefacts for Lyapunov
> exponent methods.  F0 freezes the existing integer-order method and
> establishes the methodological boundary.

---

## Phase F0 — integer_qr_benettin (frozen)

**Frozen method identifier:** `integer_qr_benettin`

**Status:** Implemented ✓ · Validated for q=1 ✓

| Property | Value |
|---|---|
| Scope | q = 1 (integer-order ODE) only |
| Variational | Φ' = J(X) Φ (first-order, no Caputo memory) |
| Orthonormalization | QR |
| Result | Finite-time local Lyapunov exponents |
| Jacobian | Required (analytic or finite differences) |
| Certifies chaos | **No** (`chaos_certified_by_this_pipeline: false`) |
| Certifies hiddenness | **No** (`hiddenness_certified_by_this_pipeline: false`) |

### Not in scope

- NOT Caputo q < 1
- NOT fractional memory
- Does NOT certify chaos
- Does NOT certify hiddenness

> **Methodological warning:** This routine is **not a validated Caputo
> fractional Lyapunov method**.  It is restricted to **q = 1**.
> Fractional Caputo spectra require a dedicated extended-memory variational
> method.

### References

- Benettin et al. 1980 / Wolf et al. 1985 (integer-order QR-Benettin)
- Danca & Kuznetsov 2018: fractional Caputo spectra require extended-memory
  variational integration; integer QR is NOT valid for q < 1.

---

## Future phases (not implemented)

| Phase | Method ID | Status |
|---|---|---|
| F1 | Common API / dispatcher | **Initiated** ✓ |
| F1 | `integer_qr_benettin` dispatch | **Implemented** ✓ |
| F1 | `fractional_variational_abm_qr` | NOT implemented · NOT validated |
| F1 | `fractional_cloned_dynamics_abm` | NOT implemented · NOT validated |
| F2 | `zero_one_test` | NOT implemented |
| F3 | PSD/FFT | NOT implemented |
| F4 | Boundedness | NOT implemented |

---

## F1 — Common API and method dispatcher

**Status: Initiated**

| Item | Status |
|---|---|
| `LyapunovComputationRequest` dataclass | ✓ Implemented |
| `LyapunovComputationSummary` dataclass | ✓ Implemented |
| `validate_lyapunov_method_request` | ✓ Implemented |
| `compute_lyapunov_spectrum` dispatcher | ✓ Implemented |
| `integer_qr_benettin` dispatch path | ✓ Implemented |
| `fractional_variational_abm_qr` dispatch | NOT implemented (raises `NotImplementedError`) |
| `fractional_cloned_dynamics_abm` dispatch | NOT implemented (raises `NotImplementedError`) |

### F1 does NOT certify

```
chaos_certified_by_this_pipeline: false
hiddenness_certified_by_this_pipeline: false
```

Fields `hidden_verified`, `chaos_verified`, `fractional_lyapunov_validated`,
and `caputo_lyapunov_validated` are **not declared** in F0 or F1.

---

## Notes

`chaos_certified_by_this_pipeline: false`
`hiddenness_certified_by_this_pipeline: false`

Fields `hidden_verified`, `chaos_verified`, `fractional_lyapunov_validated`,
and `caputo_lyapunov_validated` are **not declared** in F0 or F1.

