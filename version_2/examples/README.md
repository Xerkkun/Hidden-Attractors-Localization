# Examples

These examples are the user-facing entry points for the library. They should be
small, runnable from the repository root, and should not launch long numerical
jobs unless the filename and help text make that clear.

## Comandos

```bash
python examples/quickstart_equilibria.py
python examples/list_final_candidates.py
python examples/dynamical_analysis_gallery.py
python examples/create_robustness_overlay_config.py
python examples/aggregate_existing_robustness_overlay.py outputs/robustness_overlay_c_trajectories_20260517
```

`dynamical_analysis_gallery.py` can also plot a real trajectory from `outputs/`
when passed `--trajectory-csv path/to/trajectory.csv`.

## Rules

When adding a new example:

1. import from `hidden_attractors` whenever possible;
2. avoid duplicating long workflow logic;
3. write outputs to a new folder under `outputs/` or require
   `--output-dir`;
4. update this README if the example becomes part of the normal workflow.
