# Version 2 manifest

V2 is the active library-style distribution. It should run from this directory
without reading files from the old project root.

## Public Package

- `hidden_attractors/models/chua.py`: Chua parameters, vector field, and
  equilibria.
- `hidden_attractors/systems/`: registry for built-in and user-defined chaotic
  systems.
- `hidden_attractors/native/backends.py`: C/EFORK and basin-classifier wrappers.
- `hidden_attractors/native/csrc/`: C sources bundled with the package.
- `hidden_attractors/parallel.py`: native compilation and OpenMP policy.
- `hidden_attractors/analysis/trajectory.py`: trajectory metrics, FFT, clouds,
  sections, and robustness cases.
- `hidden_attractors/basins/classification.py`: basin labels and helpers.
- `hidden_attractors/plotting/overlays.py`: reusable plotting helpers.
- `hidden_attractors/workflows/robustness_overlay.py`: robustness trajectories.
- `hidden_attractors/workflows/sphere_controls.py`: equilibrium-neighborhood
  controls.
- `hidden_attractors/workflows/refined_basin.py`: geometry-based basin
  refinement.
- `configs/validation_contract.json`: contract for promoted validation
  evidence, manifests, stage summaries, and final reports.
- `validation/`: generated validation evidence promoted from ordinary outputs
  into a defensible review package.

## User-Facing Examples

- `examples/quickstart_equilibria.py`
- `examples/list_final_candidates.py`
- `examples/minimal_chua_protocol.py`
- `examples/custom_system_definition.py`
- `examples/create_robustness_overlay_config.py`
- `examples/aggregate_existing_robustness_overlay.py`

## Command Wrappers

- `tools/cli/robustness_overlay_c_trajectories.py`
- `tools/cli/lure_top3_sphere_robustness.py`
- `tools/cli/refine_project_basin_classification.py`
- `hidden-attractors-legacy`: installable facade for historical scripts.
- `hidden-attractors-extended-search`, `hidden-attractors-danca2017`,
  `hidden-attractors-nyquist-pipeline`: common command shape for legacy
  workflows that are still being migrated.

## Legacy Research Scripts

Long scripts that are not yet clean public API live under `tools/legacy/`, but
they are packaged and exposed through installable commands. Extend the package
first before growing those scripts further.
