# Fractional Memory Validation

**Validation phase:** `fractional_memory_validation`
**Stage in pipeline:** Phase C — numerical memory sensitivity analysis

---

## Purpose

This phase validates **numerically** the sensitivity of fractional-order Chua dynamics
to the amount of historical memory retained in the Adams-Bashforth-Moulton (ABM)
Caputo integrator.

It does **NOT** validate hiddenness or certify chaotic behavior.
All outputs carry:

```json
{
  "hiddenness_certified_by_this_pipeline": false,
  "no_hidden_verified_claim": true
}
```

---

## Mathematical Background

### Caputo Derivative

For $0 < q < 1$, the Caputo fractional derivative is defined as:

$${}^C D_t^q x(t) = \frac{1}{\Gamma(1-q)} \int_{t_0}^{t} (t - \tau)^{-q}\, x'(\tau)\, d\tau$$

This integral involves the **full history** of $x'(\tau)$ from $t_0$ to $t$.
Unlike integer-order ODEs, the state vector $X(t_k)$ alone is **insufficient** to
continue a fractional simulation. The effective numerical state is the discrete history:

$$H_k = \{X(t_{k-M}),\, \ldots,\, X(t_k)\}$$

### Full Memory vs. Finite-Window Approximation

**Full memory** (`memory_mode="full"`):
The ABM scheme uses all derivative samples from $t_0$ to the current step.
This is the reference implementation of the Caputo derivative.

**Finite-window approximation** (`memory_mode="window"`, window $M$ steps):
Only the most recent $M$ derivative samples are retained. This is a **computational
approximation**, not an equivalent formula. Results must always be labeled accordingly.

> **Warning:** Short memory is NOT equivalent to full Caputo memory.
> The two agree only when the historical contribution from outside the window is negligible.
> This phase measures that empirically.

### Tail-Defect Bound

For a window of length $L = M \cdot h$ in time, the **tail defect** (the contribution
discarded by the truncation) satisfies:

$$E_L(t) = \frac{1}{\Gamma(1-q)} \int_{t_0}^{t-L} (t - \tau)^{-q}\, x'(\tau)\, d\tau$$

If $\|x'(\tau)\| \leq K$ on $[t_0, t]$, a crude upper bound is:

$$|E_L(t)| \leq \frac{K}{\Gamma(2-q)}\left[(t - t_0)^{1-q} - L^{1-q}\right],
\quad \text{for } t - t_0 > L$$

This bound decays as $L$ grows, but without additional hypotheses on $x'$
(e.g., rapid decay, cancellation), a **large $L$ alone does not guarantee**
that the window approximation is accurate. This is why the validation compares
empirically against the full-memory reference instead of relying solely on the
theoretical bound.

The bound is saved as a diagnostic quantity `tail_bound_estimate` in the output
and is **not** used as a pass/fail criterion.

---

## What is Measured

### `rho_attractor`

For a post-transient trajectory $\{X_n : t_n \geq t_{\text{burn}}\}$, define the
mean vector:

$$\bar{X} = \frac{1}{N} \sum_{n} X_n$$

Then:

$$\rho_{\text{attractor}} = \sqrt{\frac{1}{N} \sum_n \|X_n - \bar{X}\|^2}$$

This is the root-mean-square spread of the post-transient trajectory about its centroid.
It is **translation-invariant**: shifting all states by a constant does not change $\rho$.

### Dynamic Classification

Each trajectory is classified as one of:

| Class | Meaning |
|-------|---------|
| `nan_detected` | NaN or Inf appeared in the trajectory |
| `diverged` | $\max \|X_n\| > \text{divergence\_norm}$ |
| `too_short` | Fewer than 2 post-transient points |
| `collapsed_to_equilibrium` | Variance or range below threshold |
| `bounded_nontrivial` | Bounded, non-collapsed trajectory |
| `inconclusive` | None of the above criteria matched |

> **Note:** `bounded_nontrivial` does **not** imply chaotic or hidden behavior.
> Lyapunov exponent computation and hiddenness certification require separate phases.

### Window Comparison Status

Each finite-window run is compared to the full-memory reference:

| Status | Meaning |
|--------|---------|
| `full_memory_reference` | This row IS the full-memory reference |
| `window_memory_sufficient` | Class unchanged, all metrics within tolerance |
| `window_memory_sensitive` | Class unchanged, but some metrics exceed tolerance |
| `window_memory_insufficient` | Dynamic class changed relative to full memory |

---

## References

1. **Caputo derivative definition** (primary mathematical reference):
   M. Caputo, "Linear models of dissipation whose Q is almost frequency independent — II,"
   *Geophys. J. R. Astron. Soc.*, 13(5):529–539, 1967.

2. **Yoon & You (2017)** — adaptive memory method for Caputo derivative:
   S. Yoon and C. You, "An adaptive memory method for accurate and efficient computation
   of the Caputo fractional derivative," *arXiv:1711.10071*, 2017.
   Used as reference for memory reduction in Caputo schemes and numerical error analysis.

3. **Hai, Ren, Yu, Mo & Xu (2020)** — stability of short-memory fractional systems:
   P. Hai, S. Ren, H. Yu, Q. Mo, and S. Xu, "Stability Analysis of Short Memory
   Fractional Differential Equations," *arXiv:2007.05755*, 2020.
   Used as reference for short-memory fractional systems related to Caputo.

4. **Danca & Fečkan (2024)** — memory principle in Caputo fractional-order code:
   M.-F. Danca and M. Fečkan, "Memory principle of the Matlab code for Lyapunov Exponents
   of fractional order," *arXiv:2406.04686*, 2024.
   Used as reference for the memory principle in Caputo-based numerical codes.

> **Methodological note:** This phase is grounded exclusively in the Caputo definition.
> References based solely on the Grünwald-Letnikov discretization are not used as
> the primary basis for this validation.

---

## Cases Validated

| `case_id` | `system_id` | `q` |
|-----------|-------------|-----|
| `chua_fractional_saturation_memory` | `chua_fractional_saturation` | 0.9998 |
| `chua_fractional_arctan_memory` | `chua_fractional_arctan` | 0.99 |

Windows tested: **M = 256, 512, 1024 steps** (plus full-memory reference).

---

## Running the Validation

```bash
# Run all cases (full simulation):
python validation/python/run_fractional_memory_validation.py --all

# Run all cases in fast mode (short t_final from fast_test config):
python validation/python/run_fractional_memory_validation.py --all --fast

# Run a single case:
python validation/python/run_fractional_memory_validation.py \
    --case validation/fractional_memory_validation/chua_fractional_saturation_memory.yaml

# Run with trajectory saving:
python validation/python/run_fractional_memory_validation.py --all --save-trajectories

# Custom output directory:
python validation/python/run_fractional_memory_validation.py --all \
    --output-dir validation/outputs/fractional_memory_validation
```

## Running the Tests

```bash
# Unit tests for this phase only:
python -m pytest tests/test_fractional_memory_validation.py -v

# All tests except wolfram (fast):
python -m pytest -m "not wolfram" -v

# Full suite:
python -m pytest
```

---

## Output Structure

```
validation/outputs/fractional_memory_validation/<case_id>/
├── memory_window_summary.csv      # one row per (IC, memory_mode)
├── memory_comparison.csv          # window vs. full comparison per IC
├── memory_validation_summary.json # structured summary
└── trajectories/                  # optional: saved trajectory arrays
```

### `overall_status` values

| Value | Meaning |
|-------|---------|
| `memory_validation_passed` | Full memory OK; all large windows sufficient |
| `memory_validation_passed_with_sensitive_windows` | Full memory OK; some windows sensitive but no class changes |
| `memory_validation_inconclusive` | Full memory itself failed or insufficient data |
| `memory_validation_failed` | All windows change class or diverge |

---

## Important Limitations

- This phase certifies **numerical memory sensitivity only**.
- It does **not** certify chaos, hidden attractors, or Lyapunov exponents.
- `bounded_nontrivial` classification is a **necessary but not sufficient** condition
  for chaotic or hidden behavior. The label `chaotic_candidate` may only be used
  in later pipeline phases that include a full Lyapunov exponent spectrum.
- Finite-window results must always be reported as **approximations**, never as
  exact Caputo equivalents.
