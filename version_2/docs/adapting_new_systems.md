# Adapting New Systems

The installable library should treat Chua and Danca as examples, not as the
only possible systems.  New systems enter the package through two explicit
layers:

1. a registered `ChaoticSystem`, which defines the mathematical model;
2. a `WorkflowInputSpec`, which defines the numerical experiment.

This separation is intentional.  A vector field is not enough to claim that
equilibrium-ball controls, basin cuts, strict refinement, continuation, Lyapunov
estimates, or hiddenness checks are meaningful.  Each workflow must record the
extra ingredients it needs.

## Where To Add A System

For built-in systems, add the system definition in:

```text
hidden_attractors/systems/builtins.py
```

For external or project-specific systems, register the system from a user
package or script:

```python
from hidden_attractors.systems import ChaoticSystem, register_system

register_system(ChaoticSystem(...), replace=True)
```

Every new system should provide:

- `name`: stable lowercase identifier, for example `lorenz63` or
  `fractional-rossler`;
- `dimension`: state dimension;
- `rhs(state, parameters)`: vector field for the integer system or the right
  side of the Caputo equation;
- `parameters`: numerical parameter set used by default;
- `equilibria(parameters)`: named equilibria when they exist;
- `jacobian(state, parameters)`: analytic Jacobian when stability, Matignon,
  or fast Lyapunov diagnostics are needed;
- `lure`: manual Lur'e split only when Nyquist/describing-function workflows
  are requested.

The library must not infer equilibria, Lur'e form, fractional order, memory
policy, or basin targets silently.

## Workflow Input Spec

Reusable CLI commands and migrated legacy scripts should accept or build a
`WorkflowInputSpec` from:

```text
hidden_attractors/workflows/specs.py
```

The spec records the experiment-level inputs:

- `IntegratorSpec`: solver implementation, order kind, `q`, step size `h`,
  horizon, burn-in, memory policy, and output columns;
- `DestinationClassifierSpec`: finite-time labels and thresholds for target,
  infinity, equilibrium, and unknown outcomes;
- `TargetReferenceSpec`: candidate attractor seed, recorded reference
  trajectory, symmetry rule, and target definition;
- `SphereControlSpec`: compatibility type name for equilibrium-centered ball
  samples, radii, and sample growth per radius;
- `BasinSliceSpec`: plane/grid definition and fixed coordinates;
- `StrictRefinementSpec`: trajectory-similarity thresholds and negative
  control policy;
- `TrajectoryDiagnosticsSpec`: retained tail window, observables, spectra,
  sections, and metric policy;
- `ParameterSweepSpec`: bifurcation/sweep parameter, values or range, seed
  policy, and plotted observable;
- `RobustnessCaseSpec`: allowed numerical or parameter perturbations.

The spec is a reproducibility contract, not a theorem.  Passing validation
means that the numerical run is auditable; it does not prove hiddenness.

## Requirements By Workflow

Inspect requirements from the command line:

```bash
hidden-attractors-workflow-requirements
hidden-attractors-workflow-requirements --workflow sphere-controls
hidden-attractors-workflow-requirements --workflow strict-refinement --system chua-nonsmooth
hidden-attractors-workflow-requirements --example-spec
```

The same information is available in Python:

```python
from hidden_attractors.systems import get_system, requirements_for, check_system_capability

system = get_system("chua-nonsmooth")
for item in requirements_for("sphere-controls"):
    print(item.key, item.add_where)
print(check_system_capability(system, "sphere-controls").as_lines())
```

Package-level capability checks only inspect hooks on `ChaoticSystem`.
Integrator, target-reference, basin-slice, and refinement thresholds live in
`WorkflowInputSpec`, so they are reported as missing until a workflow spec is
provided.

## CLI Policy

New maintained CLI commands should follow this pattern:

1. accept `--system` for registered systems when the workflow is generic;
2. accept `--spec path/to/workflow_spec.json` when the run needs explicit
   solver, basin, target, or threshold configuration;
3. write the effective spec next to outputs as JSON;
4. write a stage summary using only the official envelope and verdict labels
   from `hidden_attractors.workflows.protocol`;
5. use the same spec pattern for robustness, bifurcation, Lyapunov,
   continuation, and diagnostics, not only basins/refinement/spheres;
6. avoid environment-only configuration except as a compatibility layer.

Chua/Danca-specific CLIs may remain while their numerical details are being
migrated, but new reusable logic should live under `hidden_attractors/`, not
inside `tools/legacy/`.

## Legacy Policy

Legacy adapters may keep installed command names for reproducibility. They may
not publish new runs with historical methodology labels. When adding behavior
to a retained adapter:

1. move reusable mathematical or numerical logic into `hidden_attractors/`;
2. make the legacy script a thin wrapper around that package function;
3. build or load a `WorkflowInputSpec` and save it in the output directory;
4. declare any fixed-system assumption in the help text and output metadata;
5. route new summaries through the official protocol contract.

This keeps old artifacts reproducible while preventing the library API from
becoming a collection of one-off scripts.

## Minimum Inputs For Hiddenness Evidence

For an integer-order system:

- vector field and parameter set;
- equilibria and Jacobian;
- integrator contract with `order_kind="integer"`;
- candidate reference trajectory or seed;
- equilibrium-neighborhood controls;
- basin or alternative destination classifier;
- strict refinement thresholds if unresolved cells are revisited.

For a fractional Caputo system:

- all integer-system inputs above;
- fractional order `q`;
- solver type and memory policy: full history, finite memory, or external;
- memory length if finite memory is used;
- burn-in and post-transient sampling windows;
- documentation of any Weyl/Liouville-Weyl seed approximation before Caputo
  validation.

For describing-function or Nyquist routes:

- manual Lur'e split;
- branch convention for `(j omega)^q`;
- scalar nonlinearity and describing function;
- seed interpretation as heuristic or Weyl-asymptotic, followed by Caputo or
  integer-system validation.

## Existing System-Specific Workflows

The current strict refinement and Danca ABM command names remain available
because they encode published or recorded numerical comparisons. Official
hiddenness evidence nevertheless uses interior ball samples:

```bash
hidden-attractors-strict-target-refinement --help
hidden-attractors-danca-abm-sphere-controls --help
```

They are compatibility adapters, not competing methodologies. New runs should
enter through `hidden-attractors-protocol` and dispatch to the registered
system, solver backend, classifier, and dynamic reference.
