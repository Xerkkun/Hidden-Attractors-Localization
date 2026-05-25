# Examples

Examples are intentionally small and import from `hidden_attractors`.

## Quick Equilibria Check

```bash
python examples/quickstart_equilibria.py
```

Purpose: verify that the non-smooth Chua equilibria are zeros of the vector
field.

## List Final Candidates

```bash
python examples/list_final_candidates.py
```

Purpose: load the reference candidates from `outputs/` using the public
candidate API.

## Minimal Chua Protocol

```bash
python examples/minimal_chua_protocol.py
python examples/minimal_chua_protocol.py --run
```

Purpose: create a small, explicit contract for the unified fractional Chua
workflow, keeping heavy numerical stages delegated to the packaged C-backed
entry point. The default command only writes the JSON contract and shell command;
`--run` launches the workflow.

## Custom System Definition

```bash
python examples/custom_system_definition.py
hidden-attractors-systems
```

Purpose: show how a user registers a new chaotic system through the public
`ChaoticSystem` contract.

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

## Dynamical Analysis Gallery

```bash
python examples/dynamical_analysis_gallery.py
```

Purpose: generate phase-space, phase-projection, time-series, and
post-processed bifurcation figures using the public API.

With an existing project trajectory:

```bash
python examples/dynamical_analysis_gallery.py --trajectory-csv <trajectory_csv>
```

The example writes figures and tabular bifurcation points under
`outputs/examples/dynamical_analysis_gallery/`.

## Adding Examples

New examples should:

- import from `hidden_attractors`;
- register new systems through `hidden_attractors.systems` when they introduce
  a model not already built in;
- avoid duplicating workflow internals;
- write to a new folder under `outputs/` or require `--output-dir`;
- document whether they launch long numerical jobs.
