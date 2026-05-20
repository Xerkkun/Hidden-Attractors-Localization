# Workflows

Workflows are reusable modules under `hidden_attractors.workflows`. Command
wrappers live in `tools/cli/` and console scripts are declared in
`pyproject.toml`.

Registered systems can advertise their workflow commands:

```bash
hidden-attractors-systems --system chua-piecewise
```

Reusable workflows must also declare their experiment-level inputs through a
`WorkflowInputSpec`: solver contract, destination classifier, target reference,
sphere controls, basin slice, strict-refinement thresholds, trajectory
diagnostics, parameter sweeps, and robustness cases.  See
[Adapting New Systems](adapting_new_systems.md) and inspect requirements with:

```bash
hidden-attractors-workflow-requirements
hidden-attractors-workflow-requirements --workflow strict-refinement --system chua-piecewise
hidden-attractors-workflow-requirements --example-spec
```

## Robustness Overlay

Module:

```python
hidden_attractors.workflows.robustness_overlay
```

CLI:

```bash
python tools/cli/robustness_overlay_c_trajectories.py --help
hidden-attractors-robustness-overlay --help
```

Purpose: compare trajectory geometry and spectra under changes in `h`, `Lm`,
and integration time. This is a robustness workflow, not a hiddenness proof.

## Sphere Controls

Module:

```python
hidden_attractors.workflows.sphere_controls
```

CLI:

```bash
python tools/cli/lure_top3_sphere_robustness.py --help
hidden-attractors-sphere-controls --help
```

Purpose: probe equilibrium-neighborhood initial conditions on spheres and
record whether they enter target-attractor basins.

For new systems, the required reusable inputs are `ChaoticSystem.equilibria`,
`WorkflowInputSpec.integrator`, `DestinationClassifierSpec`, and
`TargetReferenceSpec`.  System-specific scripts may keep historical defaults,
but generic runs should write the effective spec next to the output artifacts.

## Refined Basin Classification

Module:

```python
hidden_attractors.workflows.refined_basin
```

CLI:

```bash
python tools/cli/refine_project_basin_classification.py --help
hidden-attractors-refined-basin --help
```

Purpose: revisit `unknown` basin cells and compare trajectory geometry against
reference target trajectories.

The stricter target-reference refinement used by recent Chua/Danca runs is
available as:

```bash
python tools/cli/strict_target_refinement.py --help
hidden-attractors-strict-target-refinement --help
```

It is still Chua/Danca-aware internally, but its inputs mirror the reusable
`TargetReferenceSpec` and `StrictRefinementSpec` contract.

## Danca ABM Sphere Controls

Module:

```python
hidden_attractors.workflows.danca_abm_sphere_controls
```

CLI:

```bash
python tools/cli/danca_abm_sphere_controls.py --help
hidden-attractors-danca-abm-sphere-controls --help
```

Purpose: run Danca's published fractional Chua example with ABM full memory,
sample the same kind of equilibrium-centered spheres used by the project
candidate controls, then refine only unresolved outcomes.  This remains a
system-specific compatibility workflow and should not be used as a generic
adapter without replacing the system, solver, classifier, and target reference.

## Unified Fractional Chua Pipeline

Module:

```python
hidden_attractors.workflows.unified_chua
```

CLI:

```bash
hidden-attractors-unified-chua --help
python -m hidden_attractors.workflows.unified_chua --help
```

Purpose: run the Chua fractional workflow with explicit command-line or Python
configuration instead of manual `$env:HIDDEN_ATTRACTORS_*` setup. Heavy stages
must use the packaged C backends: EFORK integration, basin classification,
hiddenness verification, bifurcation sweeps, and Lyapunov estimation.

## Generic Integer Lur'e Workflow

Module:

```python
hidden_attractors.workflows.integer_lure
```

Example:

```bash
python examples/integer_lure_chua_protocol.py
hidden-attractors-integer-chua --help
```

Purpose: make the Chua integer route reusable for other order-one systems in
manual Lur'e form. The reusable pieces include:

- Nyquist/describing-function seed generation;
- classical and Machado DF seed branches;
- epsilon continuation;
- final attractor integration;
- equilibrium-neighborhood hiddenness controls;
- reusable Nyquist, continuation, attractor, and hiddenness figures;
- integer-order Lyapunov estimates through `hidden_attractors.analysis`.

The existing `hidden-attractors-integer-chua` command remains a compatibility
wrapper for the full historical Chua run. For a different system, register a
`ChaoticSystem` with `lure=...`, equilibria, and preferably an analytic
Jacobian.

## Historical Workflows

Historical scripts are exposed through a common installable facade:

```bash
hidden-attractors-legacy --list
hidden-attractors-legacy extended-search --help
hidden-attractors-extended-search --help
hidden-attractors-danca2017 --help
hidden-attractors-nyquist-pipeline --help
```

These commands preserve reproducibility. New reusable workflow logic should go
under `hidden_attractors.workflows` and, when it is system-specific, be attached
to a registered `ChaoticSystem`.

Legacy scripts that gain new behavior should build or load a
`WorkflowInputSpec`, write that spec to the output directory, and delegate
reusable calculations to package modules.  This keeps historical defaults
traceable without letting the stable API become script-specific.

## Numerical Contract

Every workflow output should record:

- model and parameter set;
- order type (`integer` or `fractional`) and fractional order `q` when used;
- step size `h`;
- finite memory length `Lm` for fractional runs;
- integration horizon and burn-in;
- backend and compiler policy;
- thresholds used for classification or scoring.

Do not add new heavy Python integration or basin routes when a suitable C
backend exists. For non-Chua systems, add a system-specific native backend or
an adapter implementing `hidden_attractors.native.NativeIntegrationBackend` and
`NativeLyapunovBackend`.
