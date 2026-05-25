# CLI Wrappers

The official command surface is:

```bash
hidden-attractors-protocol generate-seeds --help
hidden-attractors-protocol soft-precheck --help
hidden-attractors-protocol continue --help
hidden-attractors-protocol filter-survivors --help
hidden-attractors-protocol build-reference --help
hidden-attractors-protocol robustness --help
hidden-attractors-protocol hiddenness --help
hidden-attractors-protocol diagnostics --help
hidden-attractors-check-validation --help
```

The scripts below are computation adapters while their numerical engines are
migrated behind the canonical stages:

```bash
hidden-attractors-robustness-overlay --help
hidden-attractors-sphere-controls --help
hidden-attractors-refined-basin --help
hidden-attractors-strict-target-refinement --help
hidden-attractors-danca-abm-sphere-controls --help
```

Commands containing `sphere` retain old executable names only for previous
job manifests. Their current plans sample inside equilibrium-centred balls;
they are not an alternative methodology.

## Contract For New CLIs

New maintained CLIs emit the official JSON envelope and use only the stage
vocabulary in `hidden_attractors.workflows.protocol`. This applies to
equilibrium-ball controls, basin slices, strict refinement, continuation,
robustness and diagnostics.

System-specific adapters may calculate payloads, but promotion into official
evidence must pass through `hidden-attractors-protocol`.
