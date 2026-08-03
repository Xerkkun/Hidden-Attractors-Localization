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

The default seed route is `direct_integer_transfer`: it recomputes the
integer Nyquist crossings from the declared `(P, b, r)` realization and then
solves the registered describing-function relation. `integer_lure_seed` has
no frequency-grid, `nscan`, or scan-fallback argument. Frequency scans remain
available only through separately named seed-generation functions that the
caller must invoke explicitly.

The completed reference validation for this route is documented in
[Integer Chua `q=1` Reference](integer_chua_reference.md).

For systems whose direct transfer root is incompatible with the registered
describing function, the package exposes a separate switching construction:

```python
from hidden_attractors import get_system
from hidden_attractors.workflows import (
    ContinuationPlan,
    continue_integer_lure_nonlinearity,
    find_sign_switching_cycle_seed,
    sign_nonlinearity,
)

system = get_system("kalman-fitts-2019")
source = find_sign_switching_cycle_seed(system, [-4.0, -4.0, 0.0, -4.0])
steps = continue_integer_lure_nonlinearity(
    system,
    source,
    sign_nonlinearity,
    plan=ContinuationPlan.uniform(51, internal_parameter="mu"),
)
```

This is an alternative theoretical route, not a successful direct-DF seed.
The complete example is under
`examples/kalman_fitts_integer_lure_reference/`.

Two additional integer examples keep their system declarations separate from
their executable workflows and tests:

- `modified-van-der-pol-duffing` uses the direct scalar Lur'e route and
  independently derives both frequency/gain/amplitude branches. Its separate
  `modified_van_der_pol_duffing_integer_hidden_chaos_search` example enables a
  Hopf-relative parameter continuation only after the direct base branches
  do not exceed the declared finite-time chaos-screen threshold. That trigger
  does not establish an asymptotic classification;
- `pll-lead-lag-2015` uses an exact equilibrium shift on
  \(\mathbb{R}\times\mathbb{S}^1\), an analytic direct-route rejection, and a
  separately named Andronov continuation in loop gain.

### MAVPD route ownership and evidence

The reusable direct entry points `integer_lure_seed` and
`continue_integer_lure_seed` are top-level experimental API names. The
adaptive hidden-chaos extension deliberately remains module-qualified:

| Responsibility | Qualified implementation |
| --- | --- |
| Rebuild the parameterized MAVPD equations and exact Lur'e declaration | `hidden_attractors.systems.modified_van_der_pol_duffing.mavpd_2023_system` |
| Derive the nonzero-equilibrium Hopf values | `hidden_attractors.systems.modified_van_der_pol_duffing.mavpd_hopf_gamma_boundaries` |
| Transport the state through complete parameter dictionaries | `hidden_attractors.workflows.continue_integer_parameter_path` |
| Generate deterministic directions and run finite neighborhood probes | `hidden_attractors.workflows.deterministic_unit_directions` and `run_integer_hidden_chaos_controls` |
| Calibrate and compare candidate clouds | `hidden_attractors.verification.calibrate_attractor_reference` and `classify_cloud_against_reference` |

These qualified functions are used by
`examples/modified_van_der_pol_duffing_integer_hidden_chaos_search/run_example.py`.
They do not enlarge either tiered top-level tuple (`PUBLIC_API_STABLE` or
`PUBLIC_API_EXPERIMENTAL`), and the example's locally derived `xi=2.85` point
is not a published MAVPD parameter tuple. Its strongest
hiddenness statement remains finite and conditional on the declared
equilibria, radii, directions, horizons, solver tolerances, and cloud
classifier. The prior Python--Julia comparison covers the periodic
`gamma=0.1` audit, not this chaotic candidate.

La cola canónica de sistemas directos o transformables que se probarán con
`q = 1` está en el [catálogo de pruebas Lur'e de orden
entero](integer_lure_test_catalog.md). El catálogo separa referencias
reproducidas, siguientes implementaciones, controles negativos, casos
bloqueados y sistemas que requieren el contrato multicanal.

## Fractional numerical contracts

For `0 < q < 1`, the memory policy must be explicit. Full-history Caputo,
finite-window Caputo, and local recurrence methods are different numerical
contracts and must not be treated as interchangeable.

See [API Reference](api_reference.md) for the exported system and analysis
interfaces.
