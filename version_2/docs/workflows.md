# Workflows

Workflows are reusable modules under `hidden_attractors.workflows`. Command
wrappers live in `tools/cli/` and console scripts are declared in
`pyproject.toml`.

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

## Numerical Contract

Every workflow output should record:

- model and parameter set;
- fractional order `q`;
- step size `h`;
- finite memory length `Lm`;
- integration horizon and burn-in;
- backend and compiler policy;
- thresholds used for classification or scoring.
