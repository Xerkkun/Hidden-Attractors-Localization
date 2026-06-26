# Adapting New Systems

This page is the release checklist for applying the library methodology to a
new scalar Lur'e-compatible system. Chua is the reference family, not a hard
limit of the architecture.

## Entry layers

A new system enters through two layers:

1. `ChaoticSystem`: mathematical model registration.
2. `WorkflowInputSpec`: numerical and evidence contract for a particular run.

A vector field alone is not enough to run or claim hiddenness checks. The
workflow must record the solver, memory policy, target reference, classifier,
robustness cases, basin slices, and equilibrium-neighborhood sampling plan.

## Minimum model inputs

For a built-in system, add the definition in `hidden_attractors/systems/builtins.py`.
For a project/user system, register it from an external script or package.

Every maintained system should define:

- `name`: stable lowercase identifier;
- `dimension`: state dimension;
- `rhs(state, parameters)`: vector field or Caputo right-hand side;
- `parameters`: default parameter dictionary;
- `equilibria(parameters)`: named equilibria when available;
- `jacobian(state, parameters)`: analytic Jacobian when stability or Lyapunov diagnostics are needed;
- `lure`: explicit `LureSystem` only when DF/Nyquist workflows are requested.

## Lur'e methodology inputs

For a full seed-continuation-hiddenness route, also provide:

- matrices/vectors `P`, `b`, `r`;
- scalar nonlinearity `psi(sigma)`;
- classical describing function `N(A)` or a numerical quadrature contract;
- sign and branch convention for `lambda=(j omega)^q`;
- seed interpretation as a heuristic Weyl/harmonic approximation;
- Caputo or integer validation after seed generation;
- all equilibria and the radii/samples used for neighborhood probes.

Machado/FDF variants are currently documented as theory/planned seed families,
not promoted public workflows.

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
hidden-attractors inspect workflow-requirements --workflow strict-refinement --system chua-nonsmooth
hidden-attractors inspect workflow-requirements --example-spec
```

## Recommended extension order

```text
register ChaoticSystem
-> add equilibria and Jacobian
-> add Lur'e split if DF/Nyquist is needed
-> create WorkflowInputSpec
-> run small integration smoke checks
-> run seed generation
-> run continuation
-> build dynamic reference
-> run robustness checks
-> sample neighborhoods around all equilibria
-> generate diagnostics and figures
-> write validation manifest/report
```

## Public documentation rule

When a new system adds public functions, classes, methods, examples, or CLI
behavior, update:

- [API Reference](api_reference.md), which lists every symbol defined in the library;
- [Quick Start](quick_start.md), if the new route is a first-run path;
- [Examples](examples.md), if a script becomes part of the maintained example set;
- [Validation Evidence](validation_evidence.md), if evidence is promoted;
- `docs/reporte_unificado_chua_fraccionario.tex`, if the report discusses the route.

## Legacy boundary

Legacy adapters may preserve old computations for traceability, but new reusable
logic should live under `hidden_attractors/` and use the unified CLI contract.
Historical scripts must not be presented as competing release workflows.

