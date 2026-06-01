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
| F2 | `fractional_variational_dk2018_block_restart_abm_gs` | **Implemented native reproduction lane · `published_benchmarks_pending_reproduced_discrepancy` (`RF lambda_3` only)** |
| F2 | `fractional_cloned_dynamics_abm` | Legacy placeholder; NOT implemented |
| F3 | `fractional_cloned_dynamics_abm_gs_published` | **Implemented · `published_benchmarks_pending_discrepancy`** |
| F3 | `fractional_cloned_dynamics_abm_qr` | **Implemented experimental QR variant · benchmark comparison pending** |
| F2 | `zero_one_test` | NOT implemented |
| F3 | PSD/FFT validation | Partial; diagnostics are not complete |
| F4 | Boundedness checks | Pending; diagnostics are not complete |

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
| `fractional_cloned_dynamics_abm_gs_published` dispatch | Implemented and routed (F3) |
| `fractional_cloned_dynamics_abm_qr` dispatch | Implemented and routed (F3 experimental) |

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
- Published RF/Lorenz values: **Long native run executed on 2026-05-31**. Lorenz passes quantitatively; RF remains `published_benchmarks_pending_reproduced_discrepancy` because only `lambda_3` exceeds tolerance.
- Published 4D non-smooth case: **Qualitative only** because the quantitative article data are incomplete.
- Passing the DK2018 lane does not promote `fractional_variational_abm_qr`.
- Fast CI executes native smoke tests only. Published quantitative checks require `RUN_PUBLISHED_LYAPUNOV=1`.

### Methodological boundary
- `chaos_certified_by_this_pipeline: false`
- `hiddenness_certified_by_this_pipeline: false`

---

## F3 - Fischer 2020 cloned dynamics

**Status: `published_benchmarks_pending_discrepancy`**

The published GS lane and the internal QR comparison lane are implemented.
They use a fiducial trajectory plus perturbed clones and do not require a
Jacobian. Fractional execution uses `memory_protocol: published_block_restart`;
this is not a full-memory Caputo-aware claim.

The six Fischer YAML specifications include extracted LE and K01 values for
jerk, financial, and four-wing systems in commensurate and incommensurate
cases. Long numerical reproduction is protected by `RUN_PUBLISHED_CLONED=1`.
Until those runs pass, both F3 methods remain `validated=False`, and diagnostics
remain partial.

Passing F3 Fischer tests does not validate `fractional_variational_abm_qr`.
Passing F3 does not certify chaos or hiddenness. F3 and F2 remain separate
validation lanes.

### Fischer 2020 published benchmark execution status

The long GS runner was executed on `2026-06-01` and recorded `24` rows:

| Result class | Count |
|---|---:|
| Quantitative passes (`all abs_error < 0.05`) | `10` |
| Sign-pattern support passes | `6` |
| Quantitative discrepancy rows | `8` |
| Numerical failures | `0` |
| Strict sign-pattern gate failures | `10` |

Final status: `published_benchmarks_pending_discrepancy`.

The strict all-row test reports `14 passed, 10 failed`. Two near-zero
exponents cross sign while remaining below the quantitative absolute-error
tolerance. The GS lane remains `validated=False`; the QR lane remains
experimental.

Auditable CSV/JSON outputs are stored under
`validation/outputs/lyapunov_benchmarks/fractional_cloned_dynamics_abm_gs_published/`.
