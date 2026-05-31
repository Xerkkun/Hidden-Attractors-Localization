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

## Current implementation status of Phase F methods

| Phase | Method ID | Status |
|---|---|---|
| F1 | Common API / dispatcher | **Completed** ✓ |
| F1 | `integer_qr_benettin` dispatch | **Implemented & Validated** ✓ (F0/F1) |
| F2 | `fractional_variational_abm_qr` | **Implemented ✓ · NOT validated against published benchmarks (F2)** |
| F2 | `fractional_variational_dk2018_block_restart_abm_gs` | **Implemented native reproduction lane · published validation pending** |
| F2 | `fractional_cloned_dynamics_abm` | NOT implemented · NOT validated |
| F2 | `zero_one_test` | NOT implemented |
| F3 | PSD/FFT | NOT implemented |
| F4 | Boundedness | NOT implemented |

---

## F1 — Common API and method dispatcher

**Status: Completed ✓**

| Item | Status |
|---|---|
| `LyapunovComputationRequest` dataclass | ✓ Implemented |
| `LyapunovComputationSummary` dataclass | ✓ Implemented |
| `validate_lyapunov_method_request` | ✓ Implemented |
| `compute_lyapunov_spectrum` dispatcher | ✓ Implemented |
| `integer_qr_benettin` dispatch path | ✓ Implemented |
| `fractional_variational_abm_qr` dispatch | ✓ Implemented & Routed (F2) |
| `fractional_cloned_dynamics_abm` dispatch | NOT implemented (raises `NotImplementedError`) |

---

## F2 — `fractional_variational_abm_qr` Validation

**Status: Implemented, pending benchmark validation**

The `fractional_variational_abm_qr` method estimates the Lyapunov spectrum for Caputo fractional systems $0 < q < 1$. It uses an extended original–variational system solver and history-consistent QR reorthonormalisation.

Validation configurations and metadata are defined in:
[fractional_variational_abm_qr_validation.yaml](file:///c:/Users/moren/Desktop/Codes/Hidden%20Attractors%20Fractional%20Order/version_2/validation/chaos_validation/lyapunov_methods/fractional_variational_abm_qr_validation.yaml)

### Certifications & Status

* `chaos_certified_by_this_pipeline: false`
* `hiddenness_certified_by_this_pipeline: false`
* `validated_against_published_benchmarks: false` (F2 pending)

### F1/F2 does NOT certify

```
chaos_certified_by_this_pipeline: false
hiddenness_certified_by_this_pipeline: false
```

Fields `hidden_verified`, `chaos_verified`, `fractional_lyapunov_validated`, and `caputo_lyapunov_validated` are **not declared** in F0, F1, or F2.

---

## F2.1 — `fractional_variational_abm_qr` Benchmark Validation Layer

The F2.1 benchmark layer provides synthetic checks for the local full-history
QR method and explicit RF/Lorenz published-value cases for the separate DK2018
block-restart ABM-GS reproduction lane.

### Status
- Synthetic benchmarks: **Implemented & Validated** (Zero RHS, Linear Stable)
- Published RF/Lorenz values: **Extracted into YAML; extensive native runs pending**.
- Published 4D non-smooth case: **Qualitative only** because the quantitative article data are incomplete.
- Passing the DK2018 lane does not promote `fractional_variational_abm_qr`.

### Methodological boundary
- `chaos_certified_by_this_pipeline: false`
- `hiddenness_certified_by_this_pipeline: false`

