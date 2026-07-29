# Supported System Adapter Contract

The installed library accepts user-defined dynamical systems through the
public contracts described here. Chua is the validated reference family, not
a hard-coded limit of the software.

Trajectory and scalar-time-series analysis does not require model
registration. Functions such as `compute_trajectory_metrics` and
`estimate_time_series_lyapunov` operate directly on supplied numerical data.

## Entry layers

A model-based workflow uses two public layers:

1. `ChaoticSystem`: mathematical model registration.
2. `WorkflowInputSpec`: numerical and evidence contract for a particular run.

A vector field alone is not enough to run or claim hiddenness checks. The
workflow must record the solver, memory policy, target reference, classifier,
robustness cases, basin slices, and equilibrium-neighborhood sampling
contract.

## Minimum model inputs

Built-in systems are defined in `hidden_attractors/systems/builtins.py`.
User systems are registered from the calling script or package without
modifying the installed library.

A registered system defines:

- `name`: stable lowercase identifier;
- `dimension`: state dimension;
- `rhs(state, parameters)`: vector field or Caputo right-hand side;
- `parameters`: default parameter dictionary;
- `equilibria(parameters)`: named equilibria when available;
- `jacobian(state, parameters)`: analytic Jacobian when stability or Lyapunov diagnostics are needed;
- `lure`: explicit `LureSystem` only when DF/Nyquist workflows are requested.

## Lur'e methodology inputs

The supported seed-continuation-hiddenness route additionally requires:

- matrices/vectors `P`, `b`, `r`;
- scalar nonlinearity `psi(sigma)`;
- classical describing function `N(A)` or a numerical quadrature contract;
- sign and branch convention for `lambda=(j omega)^q`;
- seed interpretation as a heuristic Weyl/harmonic approximation;
- Caputo or integer validation after seed generation;
- all equilibria and the radii/samples used for neighborhood probes.

## WorkflowInputSpec checklist

Use `hidden_attractors.workflows.specs.WorkflowInputSpec` to record:

| Spec | What it fixes |
| --- | --- |
| `IntegratorSpec` | solver, order kind, `q`, `h`, horizon, burn-in, memory policy, output columns |
| `DestinationClassifierSpec` | target/equilibrium/divergence/unknown labels and thresholds |
| `TargetReferenceSpec` | target seed, reference trajectory, symmetry, target cloud definition |
| `SphereControlSpec` | equilibrium-centered radii, samples, sampling mode |
| `BasinSliceSpec` | planes, grid, fixed coordinates |
| `StrictRefinementSpec` | similarity thresholds and negative controls |
| `TrajectoryDiagnosticsSpec` | retained tail, spectra, sections, metric policy |
| `ParameterSweepSpec` | sweep parameter and observable |
| `RobustnessCaseSpec` | allowed numerical or parameter perturbations |

Inspect requirements with:

```bash
hidden-attractors inspect workflow-requirements
hidden-attractors inspect workflow-requirements --workflow sphere-controls
hidden-attractors inspect workflow-requirements --example-spec
```

## Evidence boundary

These interfaces define executable numerical contracts. They do not convert a
seed, continuation endpoint, finite trajectory, or single diagnostic into a
claim of chaos or hiddenness. Public validation records remain separate and
identify the solver, memory policy, classifier, equilibria, sampling contract,
and completed probe counts used for each reported result.
