# Continuation Memory Validation

**Validation phase:** `continuation_memory_validation`
**Stage in pipeline:** Phase D — numerical continuation memory sensitivity

---

## 1. What This Phase Validates

This phase evaluates the numerical sensitivity of memory-preservation strategies during multi-segment integrations. It operates across two distinct validation layers:

### A. Deformed Lur'e Continuation (`deformed_lure_continuation`)
*Only available if a describing-function gain seed $k \neq \text{null}$ is provided (e.g., Chua saturation).*
This layer validates the parameter continuation in $\eta \in [0, 1]$ between the auxiliary linear system ($\eta=0$) and the original Chua system ($\eta=1$). It compares:
1. **`last_point_restart`**: Restarts the simulation at each $\eta_i \to \eta_{i+1}$ step using only the last computed point (Caputo history reset).
2. **`history_window_transport`**: Carries the discrete history window $H_k = \{X(t_{k-M}), \dots, X(t_k)\}$ to the next continuation segment and recomputes the RHS samples under the new field $F_{\eta_{i+1}}(X_j)$.

### B. Original System Strategy Comparison (`original_system_strategy_comparison`)
*Available even if $k = \text{null}$ (e.g., Chua arctan).*
This layer compares the `last_point_restart` vs. `history_window_transport` integration strategies on the **original (undeformed) system** directly. It does **not** claim to be a Lur'e parameter continuation (since there is no $k$ to deform the field), but rather evaluates the sensitivity of the original fractional system integration to Caputo history resets across segment transitions. This comparison is particularly useful when analyzing papers that report initial conditions but do not specify or use a continuation auxiliary system.

The phase compares these strategies in terms of:
- dynamic classification of each trajectory segment;
- attractor statistics (`rho_attractor`, `rho_max`, range, centroid);
- jump norms between consecutive steps;
- sensitivity to grid refinement ($N_{\eta} = 10, 25, 50, 100$ segments).

This phase does **not** certify hidden attractors, chaos, or Lyapunov exponents.

---

## 2. Why Caputo Requires History

For `0 < q < 1`, the Caputo derivative is non-local:

$${}^C D_t^q X(t) = \frac{1}{\Gamma(1-q)} \int_{t_0}^{t} (t - \tau)^{-q}\, X'(\tau)\, d\tau$$

This integral depends on the **full history** of `X'` from `t_0` to `t`. As a consequence,
the instantaneous state `X(t_k)` alone is insufficient to continue a fractional simulation.
The effective numerical state at step `k` is the discrete history:

$$H_k = \{X(t_{k-M}),\, \ldots,\, X(t_k)\}$$

When integrating with the Adams-Bashforth-Moulton (ABM) scheme, the corrector step depends
on all previous right-hand-side evaluations from the window start. Discarding this history
(as in `last_point_restart`) introduces a **Caputo memory reset** that is not equivalent
to the true Caputo continuation.

---

## 3. Why `last_point_restart` Is Not a Complete Caputo Continuation

In the `last_point_restart` strategy:
- Only the final state `X(t_final)` of the previous segment is used as `x0`.
- All historical derivative samples are discarded.
- The new integration starts with empty history, as if `t_0 = t_final` of the previous segment.

This is labeled `caputo_history_reset: true` in all output rows.

The result is a **re-initialization** of the Caputo integral, not a true continuation
of the fractional dynamics. This strategy may appear in some published works as a
computational convenience, but it does **not** reproduce the exact Caputo trajectory.

Outputs explicitly report `caputo_history_reset: true` and never claim this strategy
yields the correct Caputo answer.

---

## 4. What `history_window_transport` Does

In the `history_window_transport` strategy:
- The last `M` discrete states `{X(t_{k-M}), ..., X(t_k)}` are extracted from the
  previous segment's trajectory.
- When eta changes from `eta_i` to `eta_{i+1}`, the RHS history samples are
  **recomputed** using the new field `F_{eta_{i+1}}`:

$$f_j^{\text{new}} = F_{eta_{i+1}}(X_j), \quad j = k-M, \ldots, k$$

- These recomputed samples are passed as `history_states` to `caputo_abm_integrate`,
  so the ABM scheme starts the new segment with a non-empty history.

Outputs report:
- `history_transported: true`
- `rhs_history_recomputed_after_eta_change: true`
- `history_length: M`

---

## 5. Why the RHS History Is Recomputed at Each eta Transition

The deformed vector field changes with eta:

$$F_\eta(X) = P X + b \left[ k\,\sigma + \eta\,(\psi(\sigma) - k\,\sigma) \right],
\quad \sigma = r^T X$$

When `eta` changes from `eta_i` to `eta_{i+1}`, the derivative samples
`f_j = F_{eta_i}(X_j)` are no longer consistent with the new field `F_{eta_{i+1}}`.
Using the old samples with the new RHS would mix two different vector fields,
introducing a systematic error.

The recomputation:

$$f_j^{\text{new}} = F_{eta_{i+1}}(X_j)$$

is a necessary correction. It uses the actual historical state trajectory `X_j`
(which is transported), but evaluates it under the new field.

This approach is labeled `experimental_history_transport_abm` in the code because
the official ABM integrator's `history_states` argument was designed primarily
for initial prehistory, not for mid-run field changes. The recomputed history is
a valid numerical approximation but is documented as experimental.

---

## 6. eta Grid: N = 10, 25, 50, 100

The continuation parameter `eta` runs from 0 to 1:

$$\eta_i = \frac{i}{N}, \quad i = 0, 1, \ldots, N$$

where `N` is the number of eta steps.

A coarse grid (N=10) may miss bifurcations or allow large jumps between eta values.
A fine grid (N=100) resolves the parameter space more carefully but requires more computation.

The phase tests N = 10, 25, 50, 100 and compares the final dynamic classification
and attractor metrics across grids:

| `eta_refinement_status` | Meaning |
|------------------------|---------|
| `continuation_stable_under_eta_refinement` | N=25,50,100 all agree in class and metrics |
| `continuation_requires_eta_refinement` | N=10 fails but N=50,100 agree |
| `continuation_unstable` | All grids give inconsistent classes or large jumps |
| `continuation_inconclusive` | Too many failures to determine |

---

## 7. What Is `jump_norm`?

At each eta transition from `eta_i` to `eta_{i+1}`, the jump norm measures the
relative change in the final state:

$$\text{jump\_norm} = \frac{\|X_{\text{final}}^{i+1} - X_{\text{final}}^{i}\|}
{\|X_{\text{final}}^{i}\| + \varepsilon}$$

Similarly:

$$\text{rho\_jump} = \frac{|\rho^{i+1} - \rho^i|}{\rho^i + \varepsilon}$$

$$\text{range\_jump} = \frac{\|\text{range}^{i+1} - \text{range}^i\|}
{\|\text{range}^i\| + \varepsilon}$$

Large jump norms indicate that the continuation path is numerically sensitive.

---

## 8. Dynamic Classification

Each trajectory segment is classified using the same classifier as Phase C:

| Class | Meaning |
|-------|---------|
| `nan_detected` | NaN or Inf in trajectory |
| `diverged` | max `\|X\|` exceeded divergence threshold |
| `too_short` | Fewer than 2 post-transient points |
| `collapsed_to_equilibrium` | Variance and range below threshold |
| `bounded_nontrivial` | Bounded, non-collapsed trajectory |
| `periodic_candidate` | Strong periodicity detected (geometric heuristic) |
| `chaotic_candidate_by_geometry` | High-dimensional spread, bounded, non-periodic |
| `inconclusive` | None of the above matched |

> **Note:** `chaotic_candidate_by_geometry` is a **geometric heuristic only**.
> It does NOT certify chaos or Lyapunov exponents. A definitive chaos certification
> requires a full Lyapunov exponent spectrum, which is done in a separate phase.

---

## 9. What Is Not Certified

This phase does **not** certify:
- Hidden attractors (`hidden_verified` is absent from all outputs)
- Chaos or Lyapunov exponents (`chaos_certified_by_this_pipeline: false`)
- That `history_window_transport` reproduces exact full Caputo memory
- That `last_point_restart` is correct for Caputo
- That the paper's continuation was performed using the exact same strategy

All outputs carry:
```json
{
  "hiddenness_certified_by_this_pipeline": false,
  "chaos_certified_by_this_pipeline": false,
  "no_hidden_verified_claim": true,
  "pointwise_comparison_used": false
}
```

---

## 10. Relation to Guan & Xie (2025) on Hidden Attractors

Guan & Xie (2025) review methods for the localization of hidden attractors, identifying
numerical continuation as an important family of approaches. However, they note that
continuation methods are **neither automatic nor universal**: the result depends on
the path chosen, the auxiliary system used, and the specific parameter route.

This phase validates the **numerical sensitivity** of the continuation path to:
- memory transport strategy (restart vs. history),
- eta grid resolution (N = 10, 25, 50, 100),
- history window length (M = 256, 512, 1024).

It does **not** claim to reproduce or extend any specific result from the literature.
The classification `bounded_nontrivial` or `chaotic_candidate_by_geometry` is a
necessary but not sufficient step toward a hidden attractor localization.

---

## References

1. **Caputo (1967)**: original Caputo derivative definition.
2. **Guan & Xie (2025)**: review on methods for localization of hidden attractors.
   Numerical continuation is an important family of methods but depends on the route
   and auxiliary system; not automatic or universal.
3. **Yoon & You (2017)**, arXiv:1711.10071: adaptive memory method for Caputo derivative;
   memory reduction and associated errors.
4. **Hai, Ren, Yu, Mo & Xu (2020)**, arXiv:2007.05755: stability of short-memory
   fractional differential equations related to Caputo.
5. **Danca & Fečkan (2024)**, arXiv:2406.04686: memory principle in Caputo-based
   fractional-order numerical codes.

> **Methodological note:** This phase is grounded in the Caputo definition.
> References based solely on the Grünwald-Letnikov discretization are not used
> as primary justification.

### Detailed Notes on $k$ and Continuation Layers
1. **Gain Equivalent $k$**: For the Chua saturation system, the parameter $k = 0.20986735451508398$ represents the equivalent gain from describing-function analysis used for the seed. It is **not** the local nonlinear slopes ($m_0, m_1$) nor the exterior slope. Sign conventions are strictly preserved to ensure consistency with the deformed vector field $F_{\eta}(X) = P X + b [k \sigma + \eta(\psi(\sigma) - k \sigma)]$.
2. **Handling $k = \text{null}$ (Chua Arctan)**: No artificial gain $k$ is invented or assumed. The `deformed_lure_continuation` is marked as `continuation_auxiliary_unavailable`. However, strategy comparison (`original_system_strategy_comparison`) is still fully executed on the original system directly to evaluate numerical restart vs history transport sensitivities under Caputo integrations.
3. **Conservative Status Aggregation**: Los estados `restart_and_history_consistent` / `original_restart_and_history_consistent` solo pueden asignarse si ninguna comparación relevante excede tolerancias ni produce warnings. Si cualquier fila excede tolerancias, el estado agregado se degrada conservadoramente a `differs_from_history` o `artifact_possible` (o sus equivalentes con prefijo `original_`).
4. **Partial Original System Comparison Meaning**: `continuation_validation_partial_original_only` no significa que la continuación Lur’e haya sido reproducida. Solo significa que, al no existir k, se ejecutó una comparación de estrategias sobre el sistema original.
5. **No-Claim Invariants**:
   - Discarding history (`last_point_restart`) is a control/restart strategy and is not a complete or exact Caputo continuation.
   - Using finite window transport (`history_window_transport`) is a numerical approximation, not an exact full-history Caputo continuation.
   - No hidden attractors are certified (`hidden_verified` is never claimed or set in any output).
   - No chaos is certified (`chaos_certified_by_this_pipeline: false`). If `chaotic_candidate_by_geometry` is reported, it is a geometric classification and does not represent proven mathematical chaos.

---

## Cases Validated

| `case_id` | `system_id` | `q` | `k` | Continuation Modes |
|-----------|-------------|-----|-----|--------------------|
| `chua_fractional_saturation_continuation` | `chua_fractional_saturation` | 0.9998 | 0.20986735451508398 | deformed_lure & original_system |
| `chua_fractional_arctan_continuation` | `chua_fractional_arctan` | 0.99 | null | original_system comparison only |

---

## Running the Validation

```bash
# Run all cases (full simulation):
python validation/python/run_continuation_memory_validation.py --all

# Run all cases in fast mode:
python validation/python/run_continuation_memory_validation.py --all --fast

# Run a single case:
python validation/python/run_continuation_memory_validation.py \
    --case validation/continuation_memory_validation/chua_fractional_saturation_continuation.yaml

# Save history arrays:
python validation/python/run_continuation_memory_validation.py --all --save-histories

# Custom output directory:
python validation/python/run_continuation_memory_validation.py --all \
    --output-dir validation/outputs/continuation_memory_validation
```

## Running the Tests

```bash
# Unit tests for this phase only:
python -m pytest tests/test_continuation_memory_validation.py -v

# All tests except wolfram:
python -m pytest -m "not wolfram" -v

# Full suite:
python -m pytest
```

---

## Output Structure

```
validation/outputs/continuation_memory_validation/<case_id>/
├── continuation_grid_summary.csv
├── restart_vs_history_comparison.csv
├── continuation_validation_summary.json
└── histories/    # only if --save-histories
```

### `overall_status` values

| Value | Meaning |
|-------|---------|
| `continuation_validation_passed` | Full pass: stable eta grids, consistent strategies |
| `continuation_validation_passed_with_eta_refinement` | Passes with fine N; coarse N insufficient |
| `continuation_validation_sensitive_to_history` | Restart vs history differ significantly |
| `continuation_validation_inconclusive` | Too many failures or missing data |
| `continuation_validation_failed` | All grids or strategies fail consistently |
