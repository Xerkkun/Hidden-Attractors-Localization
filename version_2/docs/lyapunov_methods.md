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
(e.g., fractional_variational_abm_qr).
```

---

## Implementation status of methods

| Method ID | Model | Status |
|---|---|---|
| `integer_qr_benettin` | q=1 ODE | Implemented ✓ · Validated ✓ (F0) |
| `fractional_variational_abm_qr` | Caputo q<1 | Implemented ✓ · NOT validated (F2) |
| `fractional_variational_dk2018_block_restart_abm_gs` | Caputo q<1 | Implemented native reproduction lane · `published_benchmarks_pending_reproduced_discrepancy` (`RF lambda_3` only) |
| `fractional_cloned_dynamics_abm_gs_published` | integer / Caputo `0 < q <= 1` | Implemented F3 Fischer 2020 GS lane; `published_benchmarks_pending_discrepancy` |
| `fractional_cloned_dynamics_abm_qr` | integer / Caputo `0 < q <= 1` | Implemented F3 experimental QR lane; pending comparison |
| `fractional_cloned_dynamics_abm` | legacy placeholder | NOT implemented |
| `zero_one_test` | diagnostic | NOT implemented |
| PSD/FFT validation | diagnostic | Partial; diagnostics are not complete |
| Boundedness checks | diagnostic | Pending; diagnostics are not complete |

### Fractional methods — design and integration

The F2 variational lane computes fractional Caputo Lyapunov spectra as follows:
1. An Adams–Bashforth–Moulton (ABM) predictor-corrector integrates the
   **extended (original + variational) system** with Caputo memory.
2. History-aware QR reorthonormalisation is applied to the variational block.
3. Validation is performed against published results (Danca & Kuznetsov 2018).

These are tracked in the method registry
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

## Historical F0 exclusions

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

---

## F1 — Common Lyapunov API

**Phase F1** adds a method-agnostic dispatch layer on top of the frozen F0
implementation.

### Entry point

```python
from hidden_attractors.analysis import compute_lyapunov_spectrum

summary = compute_lyapunov_spectrum(
    rhs=my_rhs,
    x0=x0,
    q=1.0,
    method="integer_qr_benettin",
    h=0.01,
    t_final=100.0,
)
print(summary.result.exponents)
print(summary.compatibility_status)   # 'compatible'
print(summary.method_info.method_id)  # 'integer_qr_benettin'
```

### How it works

1. Build a `LyapunovComputationRequest` from the arguments.
2. Call `validate_lyapunov_method_request(request)` → `(ok, status, warnings)`.
3. If invalid: raise `ValueError` (or `NotImplementedError` for registered-but-unimplemented methods).
4. Route to the correct implementation.
5. Return `LyapunovComputationSummary`.

### Method compatibility table (F1/F2/F3)

| Method | q | memory_mode | Jacobian | Implemented | Status |
|---|---:|---|---|---|---|
| `integer_qr_benettin` | `1` | `not_applicable` | Required | Yes | Validated for integer ODE |
| `integer_qr_benettin` | `< 1` | any | Required | Yes | Invalid; raises `ValueError` |
| `fractional_variational_abm_qr` | `0 < q < 1` | `full` / `window` | Required | Yes | Pending published validation |
| `fractional_cloned_dynamics_abm_gs_published` | `0 < q <= 1` | `published_block_restart` | Not required | Yes | Fischer 2020 discrepancy pending |
| `fractional_cloned_dynamics_abm_qr` | `0 < q <= 1` | `experimental_qr_block_restart` | Not required | Yes | Internal experimental comparison |
| `fractional_cloned_dynamics_abm` | `0 < q < 1` | legacy | Not required | No | Placeholder only |

### reorthonormalization_time → reorthonormalize_every

Pass `reorthonormalization_time` (physical time units) instead of
`reorthonormalize_every` (step count):

```python
summary = compute_lyapunov_spectrum(
    ...,
    reorthonormalization_time=1.0,   # convert: every = round(1.0 / h)
    h=0.01,
)
```

If **both** are provided, `reorthonormalize_every` wins and a warning
`'both_reorthonormalization_time_and_every_provided_using_every'` is added
to `summary.warnings`.

### Request summary

`summary.request_summary` is a plain dict:

```python
{
  "method": "integer_qr_benettin",
  "q": 1.0,
  "h": 0.01,
  "t_final": 100.0,
  "t_burn": 0.0,
  "reorthonormalize_every": 10,
  "reorthonormalization_time": None,
  "memory_mode": "not_applicable",
  "memory_window": None,
}
```

### F1/F2/F3 does NOT implement

- `fractional_cloned_dynamics_abm` legacy placeholder
- 0–1 test
- completed PSD/FFT validation
- Poincaré sections
- `chaos_validation_summary`

### F1/F2/F3 does NOT certify

```
chaos_certified_by_this_pipeline: false
hiddenness_certified_by_this_pipeline: false
```

Fields `hidden_verified`, `chaos_verified`, `fractional_lyapunov_validated`,
and `caputo_lyapunov_validated` are **not present** in
`LyapunovComputationRequest` or `LyapunovComputationSummary`.

---

## F2 — fractional_variational_abm_qr

**Phase F2** implements the first formal method for estimating Caputo fractional-order Lyapunov exponents by integrating the extended original–variational system.

### Mathematical formulation

We integrate the original–variational system:
$$
\begin{aligned}
{}^C D_t^q X &= F(X), \quad X(0) = X_0 \\
{}^C D_t^q \Phi &= J(X)\Phi, \quad \Phi(0) = I
\end{aligned}
$$
where $0 < q < 1$, and $J(X)$ is the Jacobian matrix.
Under Caputo fractional derivatives, the future state depends on the **entire history**.

### Integration with Caputo ABM

Both state $X$ and variational basis $\Phi$ are integrated stepwise using a history-dependent Caputo Adams–Bashforth–Moulton (ABM) predictor-corrector. The integration weights mirror exactly those in the standard fractional solver:
* Predictor scale: $h^q / \Gamma(q+1)$
* Corrector scale: $h^q / \Gamma(q+2)$
* $b_{j,n+1} = (n+1-j)^q - (n-j)^q$
* $a_0 = n^{q+1} - (n-q)(n+1)^q$
* $a_j = (n-j+2)^{q+1} + (n-j)^{q+1} - 2(n-j+1)^{q+1}$ for $j > 0$.

### History-consistent (History-aware) QR

At each orthonormalisation step (every `reorthonormalize_every` steps):
1. Compute the QR decomposition of the current variational block:
   $$ \Phi(t_k) = Q R $$
2. Extract the diagonal elements of $R$ to accumulate the Lyapunov exponents:
   $$ \lambda_i = \frac{1}{T} \sum_{k} \log |R_{ii}^{(k)}| $$
3. Since Caputo fractional ODEs are non-local, resetting $\Phi(t_k) \leftarrow Q$ without updating the historical steps would create a **history-inconsistent** trajectory. 
   To maintain coherence with the Caputo memory, we apply the inverse rotation to the **entire history** of variational states stored in the memory window:
   $$ \Phi_j \leftarrow \Phi_j \cdot R^{-1}, \quad \text{for } j \in [0, k] $$
4. Recalculate all historical derivative values:
   $$ G(Y_j) \leftarrow \text{rhs\_ext}(X_j, \Phi_j \cdot R^{-1}) $$

If `history_aware_qr=False`, only the current $\Phi$ is updated, which behaves as a standard block-restart approximation (not Caputo-coherent).

### Methodological warning

> [!WARNING]
> This routine is **not yet validated against published benchmarks**.
> Results are finite-time local Lyapunov exponent estimates.
> Caputo memory requires transforming the entire stored variational history at each QR step (`history_aware_qr=True`). If `history_aware_qr=False` (block-restart), the method is NOT full-memory Caputo-aware; label results accordingly.
> Does not certify chaos; does not certify hiddenness of attractors.

---

## F2.1 — Benchmark validation

The benchmark layer keeps two contracts separate:

- `fractional_variational_abm_qr`: fixed-lower-limit, full-history QR. The C
  backend provides direct and FFT-block convolution modes with short parity
  tests. Published validation remains pending.
- `fractional_variational_dk2018_block_restart_abm_gs`: native C reproduction
  of the supplied `FO_Lyapunov.m` block-restart ABM-GS contract. RF and Lorenz
  published-value runs can promote only this lane.

- **Methodological Rules**:
  - Extensive published runs must use the native C backend.
  - Short C-versus-Python ABM parity does not replace comparison with the
    exact published `FDE12.m` output.
  - Setting `validated=False` in the main registry means that the published benchmarks replication has not yet been fully completed.
  - Passing synthetic benchmarks (such as zero RHS or linear stable systems) does **not** count as full published validation.
  - A published quantitative benchmark requires complete reference data (coefficients, step sizes, simulation times, initial conditions).
  - If a published benchmark is missing any of these required data fields, it is marked as `published_reference_data_missing` and global validation remains pending.

`FO_LE.m` (Danca, 2026) is recorded as a future QR/LIL_nc reference for
non-commensurate orders. `LIL_nc.m` was not supplied and that third contract is
not implemented here.

---

## F3 - Fischer 2020 cloned dynamics

F3 adds a second fractional Lyapunov family that does not use a Jacobian or a
variational system. A fiducial trajectory and one perturbed clone per state
direction are integrated over a cloning interval. Their endpoint differences
are orthonormalized and restarted around the evolved fiducial state.

For block `k`, the published lane uses:

```text
X^(0)(0) = X0
X^(j)(0) = X0 + delta e_j,  j = 1,...,n
v_j^(k) = X^(j)(T_C) - X^(0)(T_C)
{v_1^(k),...,v_n^(k)} -> {u_1^(k),...,u_n^(k)}
lambda_j = 1/(K T_C) sum_k log(||v_j^(k)|| / delta)
```

The published reproduction lane is
`fractional_cloned_dynamics_abm_gs_published`. It uses ABM
predictor-corrector integration, modified Gram-Schmidt, and
`memory_protocol: published_block_restart`. This is a block-local memory
contract, not a full-memory Caputo-aware claim.

The separate `fractional_cloned_dynamics_abm_qr` method is an experimental
internal QR variant. It is not the published algorithm and cannot be promoted
from Fischer decimal agreement alone.

Both methods produce finite-time local Lyapunov indicators. They do not certify
chaos, do not certify hiddenness, and do not close integrated diagnostics.

Passing F3 Fischer tests does not validate `fractional_variational_abm_qr`.
Passing F3 does not certify chaos or hiddenness. F3 and F2 remain separate
validation lanes.

See [Cloned Dynamics Lyapunov Indicators](lyapunov_cloned_dynamics.md).

### Fischer 2020 published benchmark execution status

The long GS runner was executed on `2026-06-01`. It recorded `24` rows:
`10` quantitative passes, `6` sign-pattern support passes, `8` quantitative
discrepancy rows, and `0` numerical failures. The stricter all-row sign gate
reports `14 passed, 10 failed` because two near-zero exponents cross sign even
though their absolute errors remain below `0.05`.

Final status:

```text
published_benchmarks_pending_discrepancy
```

The GS lane remains `validated=False`. The QR lane remains experimental, and
F3 does not validate `fractional_variational_abm_qr`.

Audit outputs are stored under:

```text
validation/outputs/lyapunov_benchmarks/fractional_cloned_dynamics_abm_gs_published/
```

The tracked
[Fischer 2020 discrepancy report](https://github.com/Xerkkun/Hidden-Attractors-Localization/blob/main/version_2/validation/chaos_validation/lyapunov_methods/fractional_cloned_dynamics_abm_gs_published/discrepancy_diagnostics/fischer2020_discrepancy_report.md)
adds row-level classifications, an explicit near-zero sign policy, and a
reproducible opt-in sensitivity plan. F3 is implemented and the all-row
benchmark has been executed, but reproduction remains partial and
`validated=False`.
