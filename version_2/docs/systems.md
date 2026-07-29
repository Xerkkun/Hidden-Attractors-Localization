# Systems

The package exposes a registry for built-in and user-defined dynamical
systems. A registered vector field can be characterized directly; it is not
automatically a complete hidden-attractor workflow.

## Inspect registered systems

```bash
hidden-attractors inspect systems
hidden-attractors inspect workflow-requirements
hidden-attractors inspect workflow-requirements --workflow sphere-controls
```

Aliases accepted by older configuration files are normalized by the registry.
New code should use the canonical names returned by `inspect systems`.

## Register a system

```python
from typing import Any, Mapping

import numpy as np

from hidden_attractors.systems import ChaoticSystem, register_system


def rhs(state: np.ndarray, p: Mapping[str, Any]) -> np.ndarray:
    x, y, z = state
    sigma = float(p["sigma"])
    rho = float(p["rho"])
    beta = float(p["beta"])
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


register_system(
    ChaoticSystem(
        name="lorenz63-demo",
        dimension=3,
        rhs=rhs,
        parameters={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        description="Minimal external system registration example.",
    ),
    replace=True,
)
```

See `examples/custom_system_definition.py` for a runnable example.

## Characterization requirements

Trajectory and scalar-series functions can be used without a Lur'e
decomposition or hiddenness workflow. Depending on the requested calculation,
a user supplies one of:

- sampled times and states;
- a uniformly sampled scalar observable and its sample interval;
- a vector field and parameters;
- equilibria or a Jacobian when the selected calculation requires them.

The numerical output remains conditional on the integration, sampling,
transient removal, embedding, and estimator settings supplied by the caller.

## Lur'e workflow requirements

The seed-continuation route uses the scalar form

```text
^C D_t^q X = P X + b psi(r^T X)
```

It requires:

- a stable system identifier, dimension, vector field, and parameters;
- all equilibria and, when required, a Jacobian;
- explicit `P`, `b`, `r`, and scalar `psi`;
- a describing-function convention and fractional frequency branch;
- an integrator, order, step, horizon, burn-in, and memory policy;
- a target reference and classifier thresholds;
- equilibrium-neighborhood radii, samples, and failure policy.

The package does not infer these scientific inputs silently.

## `WorkflowInputSpec`

Reusable numerical workflows can be represented with
`hidden_attractors.workflows.specs.WorkflowInputSpec`. Its component specs
record the integrator, classifier, target reference, neighborhood controls,
basin slices, strict refinement, trajectory diagnostics, parameter sweeps, and
robustness cases.

Passing spec validation means that the inputs are structurally auditable. It
does not prove chaos or hiddenness.

## Integer-order Lur'e route

The reusable order-one functions are:

```python
from hidden_attractors import get_system
from hidden_attractors.workflows.integer_lure import (
    continue_integer_lure_seed,
    final_integer_lure_attractor,
    integer_lure_seed,
    run_integer_lure_hiddenness_controls,
)

system = get_system("chua-nonsmooth")
seed = integer_lure_seed(system)
steps = continue_integer_lure_seed(system, seed)
target_seed, trajectory, status = final_integer_lure_attractor(
    system,
    steps[-1].x_out,
)
probes = run_integer_lure_hiddenness_controls(system, trajectory)
```

The completed reference validation for this route is documented in
[Integer Chua `q=1` Reference](integer_chua_reference.md).

## Fractional numerical contracts

For `0 < q < 1`, the memory policy must be explicit. Full-history Caputo,
finite-window Caputo, and local recurrence methods are different numerical
contracts and must not be treated as interchangeable.

See [API Reference](api_reference.md) for the exported system and analysis
interfaces.
