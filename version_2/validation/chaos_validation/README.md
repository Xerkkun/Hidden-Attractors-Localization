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
| F1 | `fractional_variational_abm_qr` | NOT implemented · NOT validated |
| F2 | `fractional_cloned_dynamics_abm` | NOT implemented · NOT validated |
| F3 | `zero_one_test` | NOT implemented |
| F4 | PSD/FFT | NOT implemented |
| F5 | Boundedness | NOT implemented |

---

## Notes

`chaos_certified_by_this_pipeline: false`
`hiddenness_certified_by_this_pipeline: false`

Fields `hidden_verified`, `chaos_verified`, `fractional_lyapunov_validated`,
and `caputo_lyapunov_validated` are **not declared** in F0.
