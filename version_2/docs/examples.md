# Examples

Examples are intentionally small and import from `hidden_attractors`.

## Quick Equilibria Check

```bash
python examples/quickstart_equilibria.py
```

Purpose: verify that the piecewise Chua equilibria are zeros of the vector
field.

## List Final Candidates

```bash
python examples/list_final_candidates.py
```

Purpose: load the reference candidates from `outputs/` using the public
candidate API.

## Create a Robustness Overlay Config

```bash
python examples/create_robustness_overlay_config.py
```

Purpose: write a workflow configuration without launching long simulations.

## Aggregate Existing Robustness Output

```bash
python examples/aggregate_existing_robustness_overlay.py outputs/robustness_overlay_c_trajectories_20260517
```

Purpose: regenerate summary tables and plots from an existing output folder.

## Adding Examples

New examples should:

- import from `hidden_attractors`;
- avoid duplicating workflow internals;
- write to a new folder under `outputs/` or require `--output-dir`;
- document whether they launch long numerical jobs.
