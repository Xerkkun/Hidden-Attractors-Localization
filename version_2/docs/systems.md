# Systems

The package exposes a registry for chaotic systems. Built-in systems and user
systems use the same `ChaoticSystem` contract, but a registered vector field is
not automatically a complete hidden-attractor workflow.

## Inspect registered systems

```bash
hidden-attractors inspect systems
hidden-attractors inspect workflow-requirements
hidden-attractors inspect workflow-requirements --workflow sphere-controls
```

Built-in Chua routes currently include the non-smooth saturation model and the
arctan model. Aliases used by older configs are normalized by the registry, but
new documentation should use the canonical names shown by `inspect systems`.

## Register a new system

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

See `examples/custom_system_definition.py` for a runnable version.

## Lur'e requirements

For the full methodology, a user must provide more than `rhs`. The route uses
an explicit scalar Lur'e form:

```text
^C D_t^q X = P X + b psi(r^T X)
```

Required inputs for DF/Nyquist and hiddenness workflows:

- stable system identifier and default parameters;
- state dimension and vector field;
- all equilibria and a Jacobian when stability/Matignon/Lyapunov checks are used;
- explicit `P`, `b`, `r`, and scalar `psi`;
- describing-function convention, including the `(j omega)^q` branch for fractional seeds;
- numerical contract: `q`, integrator, `h`, memory policy, time horizon, burn-in, thresholds;
- target reference and classifier thresholds;
- neighborhood radii and sample counts around every equilibrium;
- output paths and provenance metadata.

The package does not infer equilibria, Lur'e form, memory policy, or basin
targets silently.

## WorkflowInputSpec

Reusable workflows should be backed by `hidden_attractors.workflows.specs.WorkflowInputSpec`.
It records:

- `IntegratorSpec`
- `DestinationClassifierSpec`
- `TargetReferenceSpec`
- `SphereControlSpec`
- `BasinSliceSpec`
- `StrictRefinementSpec`
- `TrajectoryDiagnosticsSpec`
- `ParameterSweepSpec`
- `RobustnessCaseSpec`

The spec is a reproducibility contract. Passing spec validation means the run is
auditable; it does not prove hiddenness.

## Integer-order route

For `q=1`, the reusable functions live under
`hidden_attractors.workflows.integer_lure`:

```python
from hidden_attractors import get_system
from hidden_attractors.workflows.integer_lure import (
    integer_lure_seed,
    continue_integer_lure_seed,
    final_integer_lure_attractor,
    run_integer_lure_hiddenness_controls,
)

system = get_system("chua-nonsmooth")
seed = integer_lure_seed(system)
steps = continue_integer_lure_seed(system, seed)
target_seed, trajectory, status = final_integer_lure_attractor(system, steps[-1].x_out)
probes = run_integer_lure_hiddenness_controls(system, trajectory)
```

The official report example is:

```bash
python examples/chua_integer_lure_reference/run_example.py --quick
```

Promoted integer-reference artifacts are stored under
`validation/reference_cases/chua_integer_q1/`.

## Fractional route

For `0 < q < 1`, the memory policy must be explicit. Full-history Caputo,
finite-window Caputo, and local ADM-style recurrence are different contracts and
must not be merged in reports or manifests.

The official non-smooth BDF example is:

```bash
python examples/chua_nonsmooth_biased_hidden_attractor/run_example.py --quick
```

The arctan example is:

```bash
python examples/chua_arctan_wu2023/run_example.py --quick
```

The arctan lane remains non-promoted for hiddenness: the Wu2023 bibliographic
lane is a local ADM reproduction, and the c590 Caputo lane remains under review.

## API inventory

All functions, classes, and methods defined in `hidden_attractors` are listed in
[API Reference](api_reference.md). Update that inventory whenever a new symbol is
added for release.

