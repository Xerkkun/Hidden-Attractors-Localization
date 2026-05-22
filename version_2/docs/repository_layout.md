# Repository Layout

```text
hidden_attractors/              importable package
hidden_attractors/systems/      chaotic-system registry and extension point
hidden_attractors/native/csrc/  C sources bundled with the package
examples/                       small runnable examples
tests/                          smoke tests
configs/                        workflow configuration files
docs/                           documentation
validation/                     promoted validation evidence and final reports
tools/cli/                      maintained CLI wrappers
tools/legacy/                   packaged historical scripts behind facade commands
outputs/                        reference outputs for examples/loaders
artifacts/                      migrated prebuilt or runtime artifacts
```

## Public API Boundary

Only `hidden_attractors/`, `examples/`, `tests/`, `configs/`, `validation/`,
and documented `tools/cli/` commands are part of the active library workflow.

Scripts in `tools/legacy/` are preserved for research traceability and are
packaged so installed commands can still run them. They can be mined for logic,
but new reusable behavior should be added to `hidden_attractors/` first.

`outputs/` remains the default place for ordinary generated products and
exploratory runs. `validation/` is reserved for promoted evidence: stage
summaries, selected CSV tables, selected figures, short stage notes, manifests,
and the final validation report.
