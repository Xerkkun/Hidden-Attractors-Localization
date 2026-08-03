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

## Tempered BDF Convolution Quadrature

The experimental module-qualified API evaluates sampled tempered RL or
exponentially conjugated Caputo operators; it is not an FDE solver:

```python
from hidden_attractors.fractional import (
    TEMPERED_CAPUTO_INITIAL_CONDITION,
    tempered_convolution_quadrature,
)

result = tempered_convolution_quadrature(
    samples,
    [0.58, 0.84],
    tempering=[0.35, 1.10],
    bdf_order=2,
    definition="tempered_caputo",
    step=0.002,
    initial_condition_semantics=TEMPERED_CAPUTO_INITIAL_CONDITION,
    backend="fft",
)
```

`TemperedConvolutionQuadratureResult` retains base and exponentially damped
weights, component orders and tempering, the Caputo anchor correction,
starting convention, backend complexity, and the explicit absence of a
(-\lambda^q x) normalization. Python/Numba are direct (O(dN^2)) paths;
FFT is an offline (O(dN\log N)) linear batch convolution, not fast history.
See [Tempered BDF Convolution Quadrature](tempered_convolution_quadrature.md).

## Fast Recurrent Tempered Multistep History

The experimental module-qualified API evaluates the same sampled tempered RL
or conjugated-Caputo convention with an exact local window and a real
recurrent compression of older samples:

```python
from hidden_attractors.fractional import (
    TEMPERED_CAPUTO_INITIAL_CONDITION,
    tempered_fast_multistep_history,
)

result = tempered_fast_multistep_history(
    samples,
    [0.48, 0.67, 0.83],
    tempering=[0.10, 0.35, 0.70],
    multistep_method="gngf2",
    definition="tempered_caputo",
    step=0.005,
    initial_condition_semantics=TEMPERED_CAPUTO_INITIAL_CONDITION,
    local_history_steps=50,
    relative_tolerance=1.0e-9,
    backend="numba",
)
```

`TemperedFastHistoryResult` retains exact local weights, real quadrature nodes
and signed weights, final recurrent state, complete finite-grid L1 coefficient
calibration, a compression-only operator bound, complexity, anchor/startup
semantics, and references. The supported generators are FBDF1 and GNGF2;
GNGF2 is not fractional BDF2, although its `q=1` limit is exactly ordinary
BDF2. Python and Numba implement the same `O(d*(Q+n0)*N)` recurrence. FFT is
intentionally rejected because it is a batch convolution, not a recurrent
history backend. This API is an operator, not an FDE solver. See
[Fast Recurrent Tempered Multistep History](tempered_fast_multistep_history.md).

## Multi-Term Caputo L1 Facade

The experimental module-qualified API represents a finite equation sum, not an
inferred continuous order density:

```python
from hidden_attractors.fractional import integrate_multi_term_caputo_l1

result = integrate_multi_term_caputo_l1(
    rhs,
    initial_state=[0.8],
    orders=[1/3, 2/3, 1.0],
    coefficients=[0.4, 0.7, 0.75],
    step=0.01,
    n_steps=100,
    initial_regularity="nonsmooth",
)
```

`canonicalize_multi_term_caputo_terms` validates and records the original
terms, removes exact zero coefficients under a declared policy, and coalesces
only exactly equal `float64` orders with `math.fsum`. The coefficients are
never normalized. `MultiTermCaputoResult` retains the semantic method name and
delegates its trajectory and combined kernel to the existing distributed-order
L1 result. See [Multi-Term Caputo L1](multi_term_caputo_l1.md).

## Common Trajectory and Analysis Contract

`TrajectoryInput`, `PrehistorySpec`, and `AnalysisResult` provide the common
experimental envelope for diagnostics that accept either integer- or
fractional-order trajectories. The contract records time coordinate, sampling,
projection, derivative definition, order, memory policy, prehistory, solver
metadata, immutable arrays, and a deterministic SHA-256 fingerprint. For a
fractional simulation, physical time and any transformed integration coordinate
remain distinct; a finite state-vector projection is not described as the full
hereditary state.

## Bandt--Pompe Permutation Entropy

```python
from hidden_attractors import permutation_entropy

result = permutation_entropy(
    signal,
    embedding_dimension=5,
    delay=2,
    tie_policy="stable_index",
    log_base=2.0,
    backend="auto",
    sampling="uniform samples after transient",
    projection="x(t)",
)
print(result.entropy, result.normalized_entropy, result.backend)
```

`ordinal_pattern_distribution` exposes the underlying dense `m!` histogram.
Python, Numba, and native C/OpenMP share forward windows, stable lexicographic
Lehmer ranks, and explicit `stable_index`, `omit`, or `raise` tie policies for
`2 <= m <= 10`. Normalization is by `log(m!)`, not the observed support. The
plug-in estimate is finite-sample and projection-dependent; it does not certify
an entropy rate, chaos, attraction, or hiddenness.

`backend="auto"` uses a dimension-aware measured policy: native C begins at
131,072 windows for `m in {2, 8}`, at 32,768 for `3 <= m <= 7`, and remains on
Numba for `m in {9, 10}` because the factorial histogram makes the measured
OpenMP atomic path slower. The experimental
`PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS` mapping exposes the exact
policy; explicit `backend="native_c"` remains available.

## Correlation-Sum Curve and Explicit D2 Fit

```python
from hidden_attractors import (
    correlation_sum_curve,
    fit_correlation_dimension,
)

curve = correlation_sum_curve(
    points,
    radii,
    theiler_window=25,
    backend="auto",
    sampling="uniform samples after transient",
    projection="selected coordinates",
)
fit = fit_correlation_dimension(
    curve,
    fit_radius_range=(0.02, 0.15),
)
```

The experimental top-level API also exports `estimate_correlation_dimension`
for the combined call and the structured result types. Python, Numba, and
native-C/OpenMP backends share the strict `distance < radius` and Theiler
contract. The fit interval is mandatory and never selected silently. Results
from fractional trajectories characterize only the supplied projection, not a
complete hereditary state, and do not certify chaos or hiddenness.

## Integer SALI, GALI, and LDI Alignment Indices

The experimental top-level surface exposes the complete integer alignment
contract:

```python
from hidden_attractors import integer_system_alignment_indices

result = integer_system_alignment_indices(
    system,
    initial_state,
    q=1.0,
    method="variational",
    t_final=50.0,
    renormalization_time=0.25,
    gali_orders=(2, 3),
    backend="auto",
)
```

`AlignmentIndexResult`, `smaller_alignment_index`,
`generalized_alignment_index`, `linear_dependence_index`,
`alignment_indices_from_tangent_history`,
`integer_flow_alignment_indices`, `integer_map_alignment_indices`, and
`integer_system_alignment_indices` are public experimental symbols.
Instantaneous matrices use `(dimension, n_vectors)`, while public tangent
histories use `(n_samples, n_vectors, dimension)`. Flows use DOP853 with
either a variational equation or a declared multiparticle construction; maps
use an analytic/numeric Jacobian or neighboring particles. NumPy/SVD is the
reference backend and warmed Numba/Householder is the compiled batch backend.

Every executable facade requires `q=1`. Fractional SALI/GALI/LDI remains
`research_required`: a nonlocal derivative needs an operator-specific tangent
equation, history-space perturbation, and justified renormalization rule. The
returned finite-time indices do not automatically classify chaos, attraction,
or hiddenness. See [Integer SALI, GALI, and LDI](sali_gali_integer.md) and
`examples/sali_gali_henon_heiles.py`.

## Integer Covariant Lyapunov Vectors

The following eight names are top-level experimental API:

- result types `CovariantQRHistoryResult`, `CovariantLyapunovResult`, and
  `CovariantAngleResult`;
- `integer_covariant_vectors_from_qr_history`;
- `integer_flow_covariant_lyapunov_vectors`;
- `integer_map_covariant_lyapunov_vectors`;
- `integer_system_covariant_lyapunov_vectors`; and
- `covariant_lyapunov_angles`.

Flow and map facades implement the forward-QR/backward-triangular Ginelli
algorithm for memoryless `q=1` tangent cocycles. Flows propagate the state and
variational matrix with SciPy DOP853; maps use the Jacobian recurrence. Both
accept an analytic Jacobian or explicit central finite differences. NumPy/
LAPACK provides QR and the reference backward solve; Numba accelerates the
batched backward reconstruction.

```python
from hidden_attractors import (
    covariant_lyapunov_angles,
    integer_map_covariant_lyapunov_vectors,
)

result = integer_map_covariant_lyapunov_vectors(
    map_function,
    map_jacobian,
    initial_state,
    iterations=1000,
    transient_iterations=1000,
    forward_transient_iterations=500,
    backward_transient_iterations=500,
    n_vectors=2,
    backend="auto",
    q=1.0,
)
if result.status != "ok":
    raise RuntimeError(f"{result.status}: {result.error_message}")

angles = covariant_lyapunov_angles(
    result.vectors,
    coordinates=result.coordinates,
    pairs=((0, 1),),
)
```

Public CLV histories use `(samples, n_vectors, dimension)`. The algebraic QR
helper instead receives bases with vectors in columns,
`(samples, dimension, n_vectors)`, and positive-diagonal upper-triangular
factors. Pair angles are orientation-free by default; subspace comparisons use
principal angles. Repeated or nearly repeated finite-time exponents make
individual columns nonunique, so projector or subspace comparisons are safer.

Every CLV-construction facade rejects `q != 1`. Fractional CLV remains
`research_required` for each derivative definition because the memory
operator, history-space tangent cocycle, norm, and renormalization must be
specified and validated. The geometric angle postprocessor does not validate
the provenance of arbitrary supplied vectors. See
[Integer Covariant Lyapunov Vectors](covariant_lyapunov_vectors.md) and
`examples/covariant_lyapunov_henon_map.py`.

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

### Top-level tiered surface

Only names listed by `hidden_attractors.PUBLIC_API_STABLE` or
`hidden_attractors.PUBLIC_API_EXPERIMENTAL` belong to the tiered top-level
surface. The hidden-attractor portion of the experimental top-level API
includes:

- `HarmonicSeed`, `find_harmonic_seed`, `find_lure_harmonic_seed`;
- `find_lure_omega_gain_candidates`, `find_omega_gain_candidates` (explicit
  scan alternatives, not the integer direct-route default);
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

For integer Lur'e systems, `integer_lure_seed` is the direct rational-transfer
route. It has no `nscan`, `search_route`, or scan-fallback argument. A caller
that wants a dense scan must invoke a separately named scan function such as
`find_lure_omega_gain_candidates` or `find_lure_harmonic_seed` explicitly.

### Module-qualified experimental helpers

The maintained research examples also use narrower qualified helpers. These
names are importable from their modules, but they are not members of the
top-level `PUBLIC_API_EXPERIMENTAL` tuple and therefore do not acquire a
top-level compatibility guarantee:

| Qualified surface | Current use |
| --- | --- |
| `hidden_attractors.seed_generation.find_integer_lure_harmonic_seed_direct` and `find_integer_lure_omega_gain_candidates_direct` | Direct polynomial-root implementation beneath `integer_lure_seed`; no frequency grid. |
| `hidden_attractors.workflows.find_sign_switching_cycle_seed`, `continue_integer_lure_nonlinearity`, `sign_nonlinearity` | Explicit non-DF alternative for systems such as `kalman-fitts-2019`. |
| `hidden_attractors.workflows.continue_integer_parameter_path`, `deterministic_unit_directions`, `run_integer_hidden_chaos_controls`, `summarize_integer_hidden_chaos_controls` | Structured parameter transport and finite sampled-hiddenness controls. |
| `hidden_attractors.solvers.dop853_q1_integrate`, `efork_q1_integrate` | Adaptive and fixed-step memoryless `q=1` integration. |
| `hidden_attractors.analysis.lyapunov_adaptive.integer_system_dop853_variational_qr` | Finite-time adaptive variational-QR estimate; intentionally not a top-level analysis contract. |
| `hidden_attractors.verification.calibrate_attractor_reference`, `classify_cloud_against_reference` | Calibrated finite cloud comparison. |
| `hidden_attractors.verification.candidate_gate.evaluate_candidate_gate` | Joint evidence gate; it does not integrate or prove a global basin statement. |
| `hidden_attractors.systems.modified_van_der_pol_duffing.mavpd_2023_system`, `mavpd_hopf_gamma_boundaries` | System-specific factory and algebraic Hopf calculation used by the MAVPD example. |

`find_sign_switching_cycle_seed` and
`continue_integer_lure_nonlinearity` do not change that default. They form a
separately invoked alternative for systems whose direct gain is
describing-function incompatible, such as `kalman-fitts-2019`.

### MAVPD stage-to-integrator map

The executable route is
`examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/run_example.py`.
Its stages use different numerical contracts deliberately:

| Stage | Functions | Integrator or operation |
| --- | --- | --- |
| Direct localization | `mavpd_2023_system`, `integer_lure_seed` | Algebra and rational polynomial roots; no integration and no frequency grid. |
| Base `lambda` continuation | `continue_integer_lure_seed` | Fixed-step `efork_q1_integrate`. |
| Base and Hopf-offset chaos screens | `integer_system_dop853_variational_qr` | Adaptive DOP853 state/variational integration with QR. |
| `xi`/`gamma` state continuation | `continue_integer_parameter_path` | Adaptive `dop853_q1_integrate`, rebuilding the full system at every node. |
| Candidate cloud and equilibrium probes | `dop853_q1_integrate`, `run_integer_hidden_chaos_controls` | Adaptive DOP853. |
| Independent solver control | `efork_q1_integrate` | Fixed-step `q=1` EFORK cloud cross-check. |
| 0--1, Poincare, FFT, calibration, and joint gate | analysis/verification helpers above | Post-processing only; no additional ODE integrator. |

Neither direct base branch exceeds its declared finite-time Lyapunov screening
threshold. That observation triggers the separately declared Hopf-relative
parameter continuation; it does not establish an asymptotic classification.
The final label
`chaotic_hidden_under_tested_neighborhoods` remains conditional on the stored
finite solver, horizon, classifier, radii, and directions, and the locally
derived candidate has no Julia reproduction yet.

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
