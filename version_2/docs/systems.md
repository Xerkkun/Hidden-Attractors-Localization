# Systems

The package exposes a registry for chaotic systems. Built-in workflows use the
same registry that user-defined systems can use.

## Inspect Registered Systems

```bash
hidden-attractors-systems
hidden-attractors-systems --system chua-piecewise --equilibria --state 0,0,0
```

The built-in Chua systems are `chua-piecewise` and `chua-arctan`. The Chua
piecewise system advertises the current full workflow:

```bash
hidden-attractors-unified-chua --help
```

## Define a New System

User-defined systems register a vector field, parameters, optional equilibria,
and optional workflow commands:

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
        name="lorenz63",
        dimension=3,
        rhs=rhs,
        parameters={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        description="Classic Lorenz 63 system.",
    ),
    replace=True,
)
```

See `examples/custom_system_definition.py` for a runnable example.
See also `examples/new_system_workflow_spec.py` for the next layer: writing a
`WorkflowInputSpec` that records solver, classifier, target-reference, basin,
and refinement inputs before launching reusable workflows.

## Workflow Contract

A system definition is not automatically a full hidden-attractor workflow. A
complete workflow still needs:

- a manual Lur'e form when the DF/Nyquist route is used:
  `D^q x = A x + b psi(c^T x)`;
- a classical describing function `N(A)` for the scalar nonlinearity;
- a Machado describing-function branch `N_mu(A, mu)` when Machado seeds or
  sweeps are requested;
- a numerical continuation map from the harmonic linearization to the original
  nonlinear system;
- equilibrium or target-neighborhood checks;
- hiddenness controls from equilibrium neighborhoods;
- basin classification or a documented replacement criterion;
- trajectory diagnostics and report outputs;
- an explicit numerical contract: model, parameters, time horizon, step size,
  memory length if fractional, burn-in, backend, and thresholds.

For heavy numerical work, add a C backend or an adapter to a proven external
solver before exposing the workflow as stable.

The package-level readiness checker makes this distinction explicit:

```bash
hidden-attractors-workflow-requirements --workflow basin --system chua-piecewise
hidden-attractors-workflow-requirements --workflow strict-refinement --system chua-piecewise
```

If the checker reports missing `integrator`, `target-reference`, or
`basin-slice`, add those fields to a `WorkflowInputSpec`; do not add them to
the vector-field definition itself.

## Lur'e Requirement

For the full route used by the Chua examples, the user must provide more than
the equations.  The system must be entered both as a vector field and as a
Lur'e split:

```text
D^q x = A x + b psi(c^T x)
```

The package does not infer that split automatically.  The user must provide:

- `A`, `b`, and `c`;
- the scalar nonlinearity `psi(sigma)`;
- the classical describing function `N(A)`;
- the Machado branch `N_mu(A, mu)`;
- amplitude/gain compatibility rules when closed-form bounds are known;
- equilibria for hiddenness tests;
- a Jacobian if Lyapunov exponents should be robust and fast.

The generic Machado branch currently expects a real-valued branch.  The
standard admitted form is:

```text
N_mu(A, mu) = N(A)^mu,     mu > 0,     N(A) > 0
```

If a system has a complex, sign-changing, or multi-branch describing function,
the user must define the branch manually through `machado_describing_function`
and document the convention.  The package should not silently guess it.

## Integer-Order Systems

Systems may be integer-order systems.  For order one, the reusable API is under
`hidden_attractors.workflows.integer_lure`:

```python
from hidden_attractors import get_system
from hidden_attractors.workflows.integer_lure import (
    integer_lure_seed,
    continue_integer_lure_seed,
    final_integer_lure_attractor,
    run_integer_lure_hiddenness_controls,
)

system = get_system("chua-piecewise")
seed = integer_lure_seed(system)
steps = continue_integer_lure_seed(system, seed)
target_seed, trajectory, status = final_integer_lure_attractor(system, steps[-1].x_out)
probes = run_integer_lure_hiddenness_controls(system, trajectory)
```

`examples/integer_lure_chua_protocol.py` is the small runnable Chua integer
example.  The old Chua integer run in
`version_1/legacy_root/chua_integer_runs/balanced` is the reference artifact
set for what an integer-order workflow should be able to reproduce or adapt:

- `fig01`: Nyquist/describing-function closure;
- `fig02`: continuation in epsilon;
- `fig03`: final attractor and linearized-versus-original comparison;
- `fig04` and `fig05`: reference section and hiddenness controls;
- `fig06`, `fig10`, and `fig12`: basin cuts and 3D basin summaries;
- `fig08` and `fig09`: bifurcation sweeps;
- `fig11`: FFT and PSD spectral diagnostics;
- `fig13`: Lyapunov convergence.

The generic library now exposes the reusable pieces for seed generation,
continuation, final trajectories, hiddenness controls, plotting, and
integer-order Lyapunov estimates.  Basin C backends and fractional C backends
still require a system-specific native implementation or adapter.
