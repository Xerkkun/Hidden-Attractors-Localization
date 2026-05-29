# Integrator Crosscheck Validation Phase

## Purpose

This validation phase performs **numerical robustness cross-validation** of candidate chaotic
trajectories under different integrators, step sizes `h`, and memory modes (full vs. windowed
Caputo history).

It does **not** certify hidden attractors.  All outputs explicitly state:

```json
{
  "hiddenness_certified_by_this_pipeline": false,
  "no_hidden_verified_claim": true
}
```

---

## Why We Do NOT Compare Chaotic Trajectories Point-by-Point

In a **chaotic system**, two trajectories starting from the same initial condition but integrated
with different methods or step sizes will diverge **exponentially** due to sensitivity to
perturbations (positive Lyapunov exponent).  This means:

```
||X_ABM(t) - X_EFORK(t)|| → ∞  as t → ∞
```

even when both trajectories describe the **same attractor**.

Demanding pointwise coincidence as the acceptance criterion would always produce **false
negatives** for chaotic attractors.  It is therefore mathematically incorrect as a primary
validation criterion.

---

## What We Compare Instead

We compare **geometric and statistical properties** of the **post-transient cloud** of points
(after the burn-in phase `t_burn`):

| Metric                  | Description                                               |
|------------------------|-----------------------------------------------------------|
| **Boundedness**         | Is `max ‖X(t)‖ < divergence_norm`?                        |
| **Non-collapse**        | Does variance remain above `collapse_variance_tolerance`? |
| **Dynamic class**       | `bounded_nontrivial`, `collapsed_to_equilibrium`, etc.    |
| **Range vector**        | Coordinate ranges `[max_i - min_i]` per dimension         |
| **Center (mean)**       | Mean state over the post-transient cloud                  |
| **Scale (std)**         | Standard deviation per dimension                          |
| **Cloud distance**      | Percentile-based distance between marginal distributions  |

The key metric is **geometric consistency**: do different integrators produce clouds with
similar ranges, centers, and scales?

---

## Trajectory Classification States

| State                      | Meaning                                                      |
|---------------------------|--------------------------------------------------------------|
| `bounded_nontrivial`       | Bounded and non-collapsed — candidate attractor behavior     |
| `diverged`                 | `max ‖X‖ > divergence_norm` — integrator blew up            |
| `nan_detected`             | NaN or Inf found in trajectory                               |
| `collapsed_to_equilibrium` | Variance < tolerance — trajectory converged to a fixed point |
| `integrated_ok`            | Short trajectory, no classification yet applied              |
| `too_short`                | Too few post-transient points for statistics                  |
| `inconclusive`             | Ambiguous; neither clearly bounded nor clearly diverged      |

---

## Step-Size Sensitivity States

| State                                   | Meaning                                              |
|----------------------------------------|------------------------------------------------------|
| `h_stable`                              | All tested `h` values yield the same dynamic class   |
| `requires_smaller_h`                    | Large `h` fails, refined `h` gives bounded_nontrivial|
| `sensitive_to_h`                        | Dynamic class varies with h, no clear convergence    |
| `reference_not_stable`                  | The reference run itself changes class under h sweep |
| `inconclusive`                          | Cannot determine pattern                             |

**Important**: Requiring a smaller `h` than another integrator is **not a failure**.  Different
integrators have different stability regions.  This is recorded as
`crosscheck_passed_with_integrator_specific_h`.

---

## Memory Sensitivity States (for q < 1)

| State                       | Meaning                                                |
|----------------------------|--------------------------------------------------------|
| `memory_window_sufficient`  | Windowed memory produces same class as full history    |
| `memory_window_insufficient`| Windowed memory produces different class               |
| `memory_sensitive`          | Class depends on window length                         |
| `no_memory_window_runs`     | No windowed runs in the grid                           |
| `not_applicable_q1`         | q=1, no fractional memory needed                       |

---

## Overall Crosscheck Status

| Status                                          | Meaning                                          |
|------------------------------------------------|--------------------------------------------------|
| `crosscheck_passed`                             | All integratos agree at their respective h       |
| `crosscheck_passed_with_integrator_specific_h`  | Some integrators need smaller h; acceptable      |
| `crosscheck_inconclusive`                       | Reference itself unstable; cannot certify        |
| `crosscheck_failed`                             | Systematic disagreement across all refinements   |
| `crosscheck_partial_integrator_unavailable`     | One or more methods could not be run             |

---

## Commands

Run all cases (fast mode, suitable for development and CI):

```bash
python validation/python/run_integrator_crosscheck.py --all --fast
```

Run a specific case:

```bash
python validation/python/run_integrator_crosscheck.py \
  --case validation/integrator_crosscheck/chua_fractional_saturation.yaml \
  --fast
```

Full run saving trajectories and figures:

```bash
python validation/python/run_integrator_crosscheck.py --all --save-trajectories --make-figures
```

Run the test suite:

```bash
pytest tests/test_integrator_crosscheck.py -v
```

---

## Cases Included

| File                              | System                         | q      | Note                        |
|----------------------------------|-------------------------------|--------|-----------------------------|
| `chua_integer_saturation.yaml`    | Chua non-smooth (integer)      | 1.0    | Kuznetsov seed              |
| `chua_fractional_saturation.yaml` | Chua non-smooth (Caputo)       | 0.9998 | Danca/Kuznetsov seed        |
| `chua_fractional_arctan.yaml`     | Chua arctan Wu2023 (Caputo)    | 0.99   | Wu et al. 2023 initial cond |

---

## What This Phase Does NOT Certify

- It does **not** declare `hidden_verified`.
- It does **not** certify that a trajectory is a hidden attractor.
- It does **not** perform Lyapunov exponent computation (planned for a later phase).
- It does **not** certify basin of attraction geometry.
