# API Reference

This is the current stable public surface. Scripts in `tools/legacy/` are not
public API.

## Top-Level Imports

```python
from hidden_attractors import (
    ChuaParameters,
    CandidateRecord,
    RobustnessCase,
    chua_piecewise_parameters,
    equilibria_piecewise,
    load_final_candidate_records,
    rhs_piecewise,
    trajectory_metrics,
)
```

## Models

- `hidden_attractors.models.ChuaParameters`
- `hidden_attractors.models.chua_piecewise_parameters`
- `hidden_attractors.models.equilibria_piecewise`
- `hidden_attractors.models.rhs_piecewise`

## Candidates

- `hidden_attractors.candidates.CandidateRecord`
- `hidden_attractors.candidates.load_final_candidate_records`

## Analysis

- `hidden_attractors.analysis.RobustnessCase`
- `hidden_attractors.analysis.default_robustness_cases`
- `hidden_attractors.analysis.trajectory_metrics`
- `hidden_attractors.analysis.cloud_median_distance`

## Basins

- `hidden_attractors.basins.CLASS_LABELS`
- `hidden_attractors.basins.TARGET_CLASS_IDS`
- `hidden_attractors.basins.class_label`
- `hidden_attractors.basins.is_target_class`

## Native Backends

- `hidden_attractors.native.FractionalChuaBackend`
- `hidden_attractors.native.BasinBackend`

Native backends compile C sources from `hidden_attractors/native/csrc/` into
`.runtime_native/`.

## Workflows

- `hidden_attractors.workflows.robustness_overlay`
- `hidden_attractors.workflows.sphere_controls`
- `hidden_attractors.workflows.refined_basin`

Use the workflow modules for reusable Python calls and `tools/cli/` or console
entry points for command-line execution.
