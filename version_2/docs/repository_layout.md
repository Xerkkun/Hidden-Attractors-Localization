# Repository Layout

```text
hidden_attractors/              importable package
hidden_attractors/native/csrc/  C sources bundled with the package
examples/                       small runnable examples
tests/                          smoke tests
configs/                        workflow configuration files
docs/                           documentation
tools/cli/                      maintained CLI wrappers
tools/legacy/                   historical scripts outside public API
outputs/                        reference outputs for examples/loaders
artifacts/                      migrated prebuilt or runtime artifacts
```

## Public API Boundary

Only `hidden_attractors/`, `examples/`, `tests/`, `configs/`, and documented
`tools/cli/` commands are part of the active library workflow.

Scripts in `tools/legacy/` are preserved for research traceability. They can be
mined for logic, but new reusable behavior should be added to
`hidden_attractors/` first.
