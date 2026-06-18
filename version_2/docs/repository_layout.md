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
tools/cli/                      maintained CLI wrappers
tools/legacy/                   packaged historical scripts behind facade commands
outputs/                        exploratory outputs and unpromoted evidence
artifacts/                      migrated prebuilt or runtime artifacts
library_figures/                canonical promoted figure repository
```

## Public API Boundary

Only `hidden_attractors/`, `examples/`, `tests/`, `configs/`, `validation/`,
`library_figures/`, and documented `tools/cli/` commands are part of the active library workflow.

To keep the active layer clean and free of temporary files, strict script naming and location rules are enforced. See the [Development Hygiene Policy](development_hygiene.md) for more details.

Scripts in `tools/legacy/` are preserved for research traceability and are
packaged so installed commands can still run them. They can be mined for logic,
but new reusable behavior should be added to `hidden_attractors/` first.

`outputs/` remains the default place for ordinary generated products, exploratory runs, and unpromoted evidence (such as `outputs/chua_fractional_arctan/`). 

`validation/` is reserved for promoted evidence: stage summaries, selected CSV tables, short stage notes, manifests, the final validation report, and promoted system verification outputs (such as `validation/chua_integer_saturation/` and `validation/chua_fractional_saturation/`).

`figure_scripts/` is the centralized repository for all active figure generation scripts. It holds scripts like `chua_arctan_wu2023_plot_basins.py` and `chua_nonsmooth_memory_matrix_run_figure_tasks.py`.

All figures used as promoted evidence are stored in the canonical `library_figures/` repository. For detailed guidelines on figure generation, refer to the [Figure Export Policy](figure_export_policy.md) and [Figure Scripts Policy](figure_scripts_policy.md).

