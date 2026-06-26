# Repository Cleanup Status

The repository has one maintained library surface: `version_2/`.

- `hidden_attractors/` contains public and reusable implementations.
- `configs/unified_caputo_protocol.json` defines the only protocol for new runs.
- `docs/` is the single documentation and report tree.
- `validation/` contains promoted evidence and explicitly marked reference cases.
- `tools/legacy/` is retained only where an installed compatibility adapter
  still imports an engine; it is not a separate supported methodology.

The previous duplicate source tree is removed. Reference artifacts needed for
traceability are already stored under `validation/reference_cases/`; their
stored provenance paths document origin and do not make that deleted source
tree executable.

All executable EFORK sources still reachable from `version_2/`, both Python
and C, are guarded by tests enforcing the published third stage:

```text
K3 = F(... + a31*K1 + a32*K2)
```

