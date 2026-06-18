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
- `hidden_attractors/validation_contract.py`: checker used by
  `hidden-attractors validate contract`.
- `validation/`: generated validation evidence promoted from ordinary outputs
  into a defensible review package.

## User-Facing Examples

- `examples/quickstart_equilibria.py`
- `examples/list_final_candidates.py`
- `examples/minimal_chua_protocol.py`
- `examples/custom_system_definition.py`
- `examples/create_robustness_overlay_config.py`
- `examples/aggregate_existing_robustness_overlay.py`

## CLI Boundary

The package exposes a single public console script:

```text
hidden-attractors
```

All maintained workflows are reached as subcommands, for example
`hidden-attractors validate contract`, `hidden-attractors validate bibliography`,
`hidden-attractors validate cpc-readiness`, and `hidden-attractors report
fractional-run`. The historical standalone command names are not public API in
this distribution.

`tools/cli/` contains maintained helper wrappers that are invoked by tests or
development workflows. They do not define additional public entry points.

## Legacy Research Scripts

Long scripts that are not yet clean public API live under `tools/legacy/`, but
they are historical material, not public API. The current setuptools package
find configuration still includes `tools*` and `benchmarks*` so traceability
helpers and benchmark modules remain importable for reproducibility audits.
They must not grow into new installed commands without first being routed
through the unified `hidden-attractors` CLI.
