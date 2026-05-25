# Versioning Policy

`version_2/` is the only maintained and executable library surface. New code,
documentation, configurations and validation summaries must use the official
Caputo protocol defined in `hidden_attractors.workflows.protocol`.

Historical numerical evidence that remains useful as a benchmark is promoted
under `validation/reference_cases/` and identified as archived evidence. It is
not a competing methodology and it is not relabelled as a run under the new
protocol.

`tools/legacy/` remains temporarily packaged only because a small set of
maintained adapters still imports its numerical engines. Those engines are
covered by the corrected EFORK regression tests and must not publish new
methodology names. New entry points use `hidden-attractors-protocol`.

## Migration Rule

When a compatibility adapter duplicates official logic:

1. Move the reusable implementation into `hidden_attractors/`.
2. Preserve only a narrow adapter if an installed command still depends on it.
3. Write outputs with the official JSON envelope and verdict vocabulary.
4. Delete the compatibility source when no maintained import or evidence build
   uses it.
