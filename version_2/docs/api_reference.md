# Public API Reference

This reference describes the installed `hidden-attractors-fo` 1.1.0 surface.
Repository-only validation runners, case-specific validation records, paper
figure generators, and non-public comparison modules are intentionally
excluded.

The library supports two independent uses:

1. dynamical-system and time-series characterization;
2. hidden-attractor seed generation, continuation, and finite-neighborhood
   verification.

Neither a diagnostic nor a successful continuation certifies chaos or
hiddenness by itself.

## Runtime Stability Introspection

```python
import hidden_attractors as ha

print(ha.PUBLIC_API_STABLE)
print(ha.PUBLIC_API_EXPERIMENTAL)
print(ha.get_tier(ha.compute_trajectory_metrics))
```

Every name in `PUBLIC_API_STABLE` and `PUBLIC_API_EXPERIMENTAL` returns the
matching tier through `get_tier`.

## Stable Models and Parameters

| Symbol | Purpose |
| --- | --- |
| `ChuaParameters` | Immutable Chua parameter record. |
| `chua_parameters` | Build an explicit parameter record. |
| `chua_nonsmooth_parameters` | Validated non-smooth Chua parameter preset. |
| `chua_arctan_wu2023_parameters` | Parameter record for the maintained published-reference model. |
| `rhs_nonsmooth`, `rhs_arctan` | Evaluate the vector field. |
| `jacobian_nonsmooth`, `jacobian_arctan` | Evaluate analytic Jacobians. |
| `equilibria_nonsmooth`, `equilibria_arctan` | Compute named equilibria. |

Example:

```python
import numpy as np
from hidden_attractors import (
    chua_nonsmooth_parameters,
    equilibria_nonsmooth,
    jacobian_nonsmooth,
    rhs_nonsmooth,
)

parameters = chua_nonsmooth_parameters()
state = np.array([0.1, 0.0, 0.0])
field = rhs_nonsmooth(state, parameters)
jacobian = jacobian_nonsmooth(state, parameters)
equilibria = equilibria_nonsmooth(parameters)
```

## Stable System Registry and Portable IO

| Symbol | Purpose |
| --- | --- |
| `ChaoticSystem`, `LureSystem` | Structured system contracts. |
| `register_system`, `get_system`, `list_systems` | Register and retrieve systems. |
| `check_system_capability`, `known_workflows`, `requirements_for` | Inspect workflow requirements. |
| `load_trajectory_csv` | Load a user-supplied numeric trajectory. |
| `CLASS_LABELS`, `TARGET_CLASS_IDS`, `class_label`, `is_target_class` | Basin-classification vocabulary. |

Candidate records are not loaded implicitly from the repository. The optional
`hidden_attractors.candidates` module requires an explicit JSON source so its
behavior is identical in a checkout and in a wheel installation.

## Generic Trajectory Characterization

```python
from hidden_attractors import compute_trajectory_metrics

metrics = compute_trajectory_metrics(
    times,
    states,
    equilibria=None,
    t_start=10.0,
    divergence_norm=120.0,
)
```

`times` is one-dimensional and strictly increasing. `states` has shape
`(len(times), dimension)`. Explicit arrays prevent a four-dimensional state
trajectory from being mistaken for `(t, x, y, z)`.

The result contains:

- finite-time boundedness and divergence status;
- state dimension and retained sample count;
- per-component ranges and variances;
- component-0 FFT peak and spectral entropy;
- optional final-state proximity to supplied equilibria.

`trajectory_metrics_for_system` is a compatibility wrapper for combined
trajectory matrices. Use its `has_time` flag explicitly for pure-state arrays.
`trajectory_metrics` remains a Chua-specific compatibility diagnostic.

## Boundedness, Spectral, 0–1, Poincaré, and Bifurcation APIs

| Symbol | Input | Main output |
| --- | --- | --- |
| `compute_boundedness_metrics` | time and state arrays | boundedness, growth, divergence metrics |
| `compute_fft_psd` | time and scalar signal | FFT/PSD arrays and spectral summaries |
| `zero_one_test` | scalar signal | finite-data 0–1 diagnostic |
| `detect_poincare_crossings` | time and state arrays | `PoincareCrossingResult` |
| `bifurcation_points_from_trajectories` | parameter/trajectory scans | `BifurcationPoint` records |
| `bifurcation_summary` | bifurcation points | aggregate counts and ranges |

These functions do not require a hidden-attractor workflow.

## Lyapunov Exponents From Equations

For an integer registered system:

```python
import numpy as np
from hidden_attractors import get_system, integer_system_lyapunov_exponents

system = get_system("chua-nonsmooth")
result = integer_system_lyapunov_exponents(
    system,
    np.array([0.1, 0.2, 0.3]),
    h=0.01,
    t_final=50.0,
)
```

The common dispatcher validates method, `q`, memory mode, numeric finiteness,
Jacobian requirements, and reorthonormalization parameters:

```python
from hidden_attractors import compute_lyapunov_spectrum

summary = compute_lyapunov_spectrum(
    system=system,
    x0=np.array([0.1, 0.2, 0.3]),
    q=1.0,
    method="integer_qr_benettin",
    h=0.01,
    t_final=50.0,
)
```

Returned exponents are finite-time numerical estimates. Method metadata states
the derivative model, memory contract, validation scope, and warnings.

## Lyapunov Exponents From a Scalar Time Series

The time-series function is fully integrated as an experimental public API:

```python
from hidden_attractors import estimate_time_series_lyapunov

result = estimate_time_series_lyapunov(
    signal,
    sample_interval=0.01,
    time_unit="s",
    observable="x",
    random_seed=0,
)

print(result.largest_exponent)
print(result.spectrum)
print(result.kaplan_yorke_dimension)
```

It uses the optional `nolds` backend:

- Rosenstein for the largest exponent;
- Eckmann for a finite-dimensional spectrum;
- Kaplan–Yorke dimension from the ordered finite spectrum.

The result records units, backend version, estimator parameters, fit
diagnostics, memory estimate, evidence status, and warnings. The implementation
rejects non-uniformly described inputs, non-finite data, constant signals,
insufficient samples, inconsistent sample rate/interval, and requests whose
estimated pairwise-distance matrix exceeds the configured memory limit.

These outputs depend on sampling, reconstruction, fit parameters, and the
finite data window. They do not certify chaos, an asymptotic spectrum, or
hiddenness.

## Optional Complexity Backends

```python
from hidden_attractors.integrations import compute_complexity_measures

values = compute_complexity_measures(
    signal,
    backend="auto",
    sample_rate=100.0,
    measures=["permutation_entropy", "sample_entropy"],
)
```

The adapter routes each measure to an installed backend that implements it.
Unknown measures, unsupported backend/measure combinations, and unavailable
requested measures raise explicit errors.

## Integration Selector

```python
from hidden_attractors.integrations import integrate

times, states, status = integrate(
    rhs,
    x0,
    q=1.0,
    h=0.01,
    t_final=10.0,
    integrator="rk4",
    use_c_backend=False,
    allow_python_fallback=True,
)
```

The selector normalizes values numerically equivalent to `q=1`, respects
`use_c_backend`, propagates `allow_python_fallback`, and rejects incompatible
integer/fractional integrator choices.

## Hidden-Attractor Workflow APIs

The experimental top-level surface includes structured contracts and the
validated integer reference workflow:

- `HarmonicSeed`, `find_harmonic_seed`, `find_lure_harmonic_seed`;
- `find_integer_lure_harmonic_seed_direct`,
  `find_integer_lure_omega_gain_candidates_direct`;
- `find_lure_omega_gain_candidates`, `find_omega_gain_candidates` (explicit
  scan alternatives, not the integer-reference default);
- `NumericalContract`, `FullWorkflowContract`, `WorkflowInputSpec`;
- `ContinuationPlan`, `ContinuationTrace`, `DynamicReference`;
- `integer_lure_seed`, `continue_integer_lure_seed`,
  `final_integer_lure_attractor`, `run_integer_lure_hiddenness_controls`;
- `load_config`, `save_effective_config`;
- `run_attractor_only_workflow`, `run_bifurcation_workflow`,
  `run_basin_workflow`, `run_simple_workflow`.

Seed generation and continuation produce candidate initial conditions. Only a
separate, declared equilibrium-neighborhood contract can support a finite
hiddenness statement.

For integer Lur'e systems, `integer_lure_seed` defaults to
`search_route="direct_integer_transfer"`.  A dense frequency scan is performed
only when `search_route="frequency_scan"` or the caller explicitly enables
`fallback_route="frequency_scan"`; fallback is never silent.

## CLI Characterization Commands

```text
hidden-attractors inspect systems
hidden-attractors bifurcation --help
hidden-attractors lyapunov compute --help
hidden-attractors lyapunov spectrum --help
hidden-attractors lyapunov validate --help
hidden-attractors chaos-test --help
```

The JSON emitted by `lyapunov spectrum` is accepted by `lyapunov validate`.
Case-specific validation and editorial commands are not part of the installed
CLI.
