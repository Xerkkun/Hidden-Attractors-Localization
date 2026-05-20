# CLI Wrappers

These scripts preserve command-line access to maintained workflows:

```bash
python tools/cli/robustness_overlay_c_trajectories.py --help
python tools/cli/lure_top3_sphere_robustness.py --help
python tools/cli/refine_project_basin_classification.py --help
python tools/cli/strict_target_refinement.py --help
python tools/cli/danca_abm_sphere_controls.py --help
```

After installation, prefer the console entry points:

```bash
hidden-attractors-robustness-overlay --help
hidden-attractors-sphere-controls --help
hidden-attractors-refined-basin --help
hidden-attractors-strict-target-refinement --help
hidden-attractors-danca-abm-sphere-controls --help
hidden-attractors-workflow-requirements --help
```

## Contract For New CLIs

New maintained CLIs should accept either a registered `--system` plus explicit
workflow options, or a `--spec` JSON file compatible with
`hidden_attractors.workflows.WorkflowInputSpec`.  The effective spec should be
written next to every output directory.  This applies to sphere controls,
basins, strict refinement, continuation, robustness checks, and future
Lyapunov/bifurcation commands.

System-specific CLIs may remain as compatibility wrappers, but reusable
mathematical logic belongs in `hidden_attractors/`, not in the wrapper script.
