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

Run an example from the `version_2` directory:

```bash
python examples/quickstart_equilibria.py
python examples/custom_system_definition.py
python examples/dynamical_analysis_gallery.py
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
