# Testing

## Smoke Checks

These commands do not require `pytest`:

```bash
python -m compileall hidden_attractors examples tests tools/cli
python examples/quickstart_equilibria.py
python examples/list_final_candidates.py
python tools/cli/robustness_overlay_c_trajectories.py --help
python tools/cli/lure_top3_sphere_robustness.py --help
python tools/cli/refine_project_basin_classification.py --help
```

## Pytest

After installing development dependencies:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Current tests verify:

- Chua equilibria are zeros of the vector field;
- final candidate loading returns the expected reference records.

## Native Backends

Native backend tests should be added cautiously. Keep them short, write to a
temporary output directory, and record whether OpenMP was active.
