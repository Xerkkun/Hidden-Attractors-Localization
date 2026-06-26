# Repository Layout

```text
hidden_attractors/              importable package
hidden_attractors/systems/      chaotic-system registry and extension point
hidden_attractors/native/csrc/  C sources bundled with the package
examples/                       small runnable examples
tests/                          smoke tests
configs/                        workflow configuration files
docs/                           documentation
figure_scripts/                 centralized folder for active figure generation scripts
validation/                     promoted validation evidence and final reports
tools/cli/                      legacy internal CLI wrappers (not public entry points)
tools/legacy/                   historical compatibility material (internal only)
outputs/                        exploratory outputs and unpromoted evidence
artifacts/                      migrated prebuilt or runtime artifacts
library_figures/                canonical promoted figure repository
```

## Public API Boundary

Only `hidden_attractors/`, `examples/`, `tests/`, `configs/`, `validation/`, and `library_figures/` are part of the active library workflow. The unified CLI `hidden-attractors` is the sole public command surface.

To keep the active layer clean and free of temporary files, strict script naming and location rules are enforced. See the [Development Hygiene Policy](development_hygiene.md) for more details.

Scripts in `tools/legacy/` are preserved for historical compatibility and are for internal use only. New reusable behavior should be added directly to `hidden_attractors/` and dispatched via the unified CLI in `hidden_attractors.cli.main`.

`outputs/` remains the default place for ordinary generated products and exploratory runs. Promoted arctan evidence lives under `validation/chua_fractional_arctan/`; older output folders remain non-canonical provenance unless explicitly copied into validation with a manifest.

`validation/` is reserved for promoted evidence: stage summaries, selected CSV tables, short stage notes, manifests, the final validation report, and promoted system verification outputs (such as `validation/chua_integer_saturation/`, `validation/chua_fractional_saturation/`, and `validation/chua_fractional_arctan/`).

`figure_scripts/` is the centralized repository for all active figure generation scripts. It holds scripts like `chua_arctan_wu2023_plot_basins.py` and `chua_nonsmooth_memory_matrix_run_figure_tasks.py`.

## Release packaging layout

Release packaging material belongs under `release_package/` and records software/archive metadata, sample commands, and remaining readiness items. It is documentation and packaging metadata, not new scientific evidence.

`library_figures/` is the canonical store for promoted scientific figures generated through `hidden_attractors.plotting.export.export_figure`. `docs/assets/` is reserved for documentation or web-only assets; it is not the promoted scientific figure store.
