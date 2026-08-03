# Supported Examples

The examples below exercise supported package interfaces. Numerical results
remain finite-time or finite-data diagnostics unless an explicit validation
record states a narrower evidence contract.

## Small Python examples

| Script | Supported use |
| --- | --- |
| `examples/quickstart_equilibria.py` | Evaluate registered equilibria and the vector field. |
| `examples/custom_system_definition.py` | Register an external `ChaoticSystem`. |
| `examples/new_system_workflow_spec.py` | Define and validate a `WorkflowInputSpec`. |
| `examples/minimal_chua_protocol.py` | Exercise the public localization stages with explicit settings. |
| `examples/integer_lure_chua_protocol.py` | Run the reusable integer-order Lur'e route. |
| `examples/dynamical_analysis_gallery.py` | Analyze an existing trajectory and generate diagnostic outputs. |
| `examples/fractional_core_catalog.py` | Compare sampled operators and finite manufactured GL, conformable, ABC, tempered-Caputo, variable-order Type III, and distributed-order Caputo L1 solvers. |
| `examples/caputo_hadamard_chua.py` | Run an experimental Caputo--Hadamard Chua IVP on a logarithmic grid. |
| `examples/chua_advanced_analysis.py` | Apply delay/RQA to Chua and basin diagnostics to a bistable flow. |
| `examples/correlation_dimension_integer_fractional.py` | Apply one explicit q=2/Theiler/fit contract to small integer and fractional trajectories. |
| `examples/permutation_entropy_integer_fractional.py` | Apply one declared Bandt--Pompe contract to small integer and Caputo trajectories while retaining derivative and memory provenance. |
| `examples/multi_term_caputo_relaxation.py` | Solve a forced multi-scale relaxation with a non-unit finite Caputo sum, exact duplicate canonization, an alpha=1 term, and an affine manufactured control. |
| `examples/tempered_convolution_quadrature.py` | Evaluate BDF2 tempered-Caputo CQ on a two-component manufactured nonlinear system with distinct orders and tempering parameters. |
| `examples/tempered_fast_history_chua.py` | Apply recurrent GNGF2 tempered-Caputo history to a uniformly sampled integer Chua trajectory and compare it with an independent direct convolution; this is post-processing, not a fractional Chua solve. |
| `examples/sali_gali_henon_heiles.py` | Compare variational and multiparticle SALI/GALI/LDI on a finite integer Hénon--Heiles trajectory without assigning a chaos label. |
| `examples/covariant_lyapunov_henon_map.py` | Compute finite `q=1` Ginelli CLV, projective-covariance residuals, and unoriented pair angles for the nonlinear Hénon map. |

Run an example from the `version_2` directory:

```bash
python examples/quickstart_equilibria.py
python examples/custom_system_definition.py
python examples/dynamical_analysis_gallery.py
python examples/fractional_core_catalog.py
python examples/caputo_hadamard_chua.py
python examples/chua_advanced_analysis.py
python examples/correlation_dimension_integer_fractional.py
python examples/permutation_entropy_integer_fractional.py
python examples/multi_term_caputo_relaxation.py
python examples/tempered_convolution_quadrature.py
python examples/tempered_fast_history_chua.py
python examples/sali_gali_henon_heiles.py --duration 4
python examples/covariant_lyapunov_henon_map.py --iterations 1000
```

## Independent characterization

Characterization functions do not require a hidden-attractor search:

```python
from hidden_attractors import (
    compute_trajectory_metrics,
    estimate_time_series_lyapunov,
    zero_one_test,
)

metrics = compute_trajectory_metrics(times, states)
zero_one = zero_one_test(states[:, 0])
lyapunov = estimate_time_series_lyapunov(
    states[:, 0],
    sample_interval=float(times[1] - times[0]),
    observable="x",
)
```

The Lyapunov time-series result contains the Rosenstein largest-exponent
estimate, an Eckmann reconstructed spectrum, and the associated Kaplan--Yorke
dimension. These estimates depend on sampling, embedding, retained data, and
backend parameters.

For equation-based integer tangent diagnostics, the Hénon--Heiles example
uses the analytic Jacobian and compares variational propagation against the
multiparticle construction for GALI orders 2--4. The four-second reference run
produced 17 samples, no censored cells, maximum SALI/GALI differences
`1.965e-8`/`7.336e-9`, and relative energy drift `8.421e-16` in
both routes. These are finite method-comparison outputs, not a standalone
chaos, attraction, or hiddenness decision.

The Hénon-map CLV example exercises the analytic-Jacobian map facade, explicit
state/forward/backward transients, NumPy-or-Numba backward reconstruction, and
the angle postprocessor. Its covariance residual and finite-time exponents are
implementation diagnostics; they do not establish an asymptotic spectrum,
hyperbolicity, attraction, hiddenness, or a fractional-memory CLV theory.

The equivalent CLI entry points include:

```bash
hidden-attractors chaos-test zero-one --help
hidden-attractors lyapunov spectrum --help
hidden-attractors bifurcation run --help
```

## Localization interfaces

The maintained localization commands expose seed generation, continuation,
finite trajectory checks, and equilibrium-neighborhood controls:

```bash
hidden-attractors seed lure-centered --help
hidden-attractors seed lure-biased --help
hidden-attractors continuation run --help
```

A seed or a continuation result is only an initialization result. Hiddenness
labels remain conditional on the stored solver, horizon, memory policy,
equilibria, radii, samples, classifier, and numerical-failure policy.

## Completed integer reference validation

The repository includes a completed `q=1` Chua reference record used to test
the executable Lur'e workflow:

```bash
python examples/chua_integer_lure_reference/run_example.py --quick
```

Its machine-readable evidence is stored under
`validation/reference_cases/chua_integer_q1/`. The recorded classification is
limited to its finite numerical contract and is not a global mathematical
proof.

## Non-Chua integer Lur'e reference

The Kalman--Fitts example exercises a fourth-order exact scalar Lur'e system
with a unique stable equilibrium and hidden periodic attractors:

```bash
python examples/kalman_fitts_integer_lure_reference/run_example.py --quick
```

Its direct transfer calculation is retained even though it fails to produce a
DF-compatible amplitude.  The example then invokes the separately named
published alternative: an Andronov switching map for the `sign` precursor and
`sign`-to-`tanh` nonlinearity continuation.  It does not use the published
target point as an input seed.

The modified autonomous Van der Pol--Duffing audit exercises a non-Chua
third-order direct scalar Lur'e route. It derives both integer-order branches
without a frequency grid, continues every branch, and uses a shared
108-condition contract in Python and Julia:

```bash
python examples/modified_van_der_pol_duffing_integer_lure_audit/run_example.py
```

The `xi=3.1` execution supports a periodic hidden candidate under that finite
contract. The `xi=3.5` execution is deliberately retained as a negative
hiddenness audit.

The separate MAVPD hidden-chaos example starts from that same direct
equation-to-seed route. Neither base branch exceeds the declared finite-time
Lyapunov screening threshold; this is the recorded trigger for the alternative
and not an asymptotic classification of the base trajectories. Only then does
the example invoke structured continuation relative to a Hopf boundary
derived from the active equations:

```bash
python examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/run_example.py
```

The alternative transports the continued state first in `xi` and then in
`gamma`, rebuilding the equations and Lur'e declaration at every node. It
screens only the declared Hopf offsets; it is not a frequency grid, a blind
initial-condition sweep, or a numerical claim that every transition boundary
was found.

The maintained candidate at `xi=2.85` and
`gamma=0.1538037983994911` is locally derived rather than copied from the
source article. Its Python evidence keeps boundedness, finite-time chaos
diagnostics, calibrated neighborhood probes, robustness controls, and the
joint gate separate. The resulting label is limited to
`chaotic_hidden_under_tested_neighborhoods`: it is neither a proof of global
basin separation nor a Julia reproduction. Every timing row written to
`phase_timings.csv` declares its `timing_source`; that run-specific file must
not be interpreted as a general performance benchmark. See
[Integer hidden-chaos search](integer_hidden_chaos_search.md).

The lead--lag PLL example covers a compatible equilibrium shift and a
cylindrical state space. It first rejects the direct transfer route
analytically, then continues the exact zero-gain running cycle to the target
loop gain without using a published initial condition:

```bash
python examples/pll_lead_lag_integer_lure_reference/run_example.py --quick
```

Its current reproducible evidence is Python-only; the Julia counterpart is
still pending.

See the [Integer-order Lur'e test catalog](integer_lure_test_catalog.md) for
the candidate queue and the route required by each system.

## Explicit CLI configurations

The package contains no runnable research-case presets. It provides one
non-runnable structural contract:

```bash
hidden-attractors init --example workflow_contract
hidden-attractors run --config my_workflow.yaml
hidden-attractors inspect-config --config my_workflow.yaml
```

Use `hidden-attractors inspect systems` and
`hidden-attractors inspect workflow-requirements` to inspect the supported
model and workflow contracts.
