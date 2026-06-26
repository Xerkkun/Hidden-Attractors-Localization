# Internal Legacy CLI Scripts

> [!WARNING]
> This folder `tools/cli` contains internal legacy command-line script wrappers.
> These are **not** public entry points and are deprecated.
> All active development and public executions must use the unified CLI launcher: `hidden-attractors`.

## Unified CLI Alternatives (Migration Map)

The legacy standalone scripts have been migrated to the unified `hidden-attractors` tool as subcommands:

- Legacy `hidden-attractors-protocol` -> Use the unified subcommands:
  - `hidden-attractors seed`
  - `hidden-attractors validate contract`
  - `hidden-attractors run`
  - `hidden-attractors inspect`
  - `hidden-attractors validate release-readiness`
- Legacy `hidden-attractors-check-validation` -> Use `hidden-attractors validate contract` or `hidden-attractors validate release-readiness`.
- Legacy `hidden-attractors-robustness-overlay` -> Use `hidden-attractors run` with robustness configs.
- Legacy `hidden-attractors-sphere-controls` -> Use equilibrium-centered seeds inside the unified workflow registry.
- Legacy `hidden-attractors-refined-basin` -> Use unified basin/continuation subcommands.
- Legacy `hidden-attractors-danca-abm-sphere-controls` -> Deprecated historical script; not part of the active public surface.

Do not run these standalone legacy wrappers directly. They are kept for historical compatibility/migration only.
