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
python tools/cli/check_validation_contract.py --help
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

## Validation Evidence

Tests answer whether the package still behaves as expected. Validation evidence
answers whether a scientific claim is backed by traceable numerical artifacts.

Use `outputs/` for ordinary generated run products. Promote only selected
evidence into `validation/`, following `configs/validation_contract.json`.
Each validation stage should include:

- one short `*_validation.md` interpretation;
- one `*_validation_summary.json` or equivalent summary JSON;
- CSV tables for numerical checks;
- PNG/PDF figures for visual evidence when relevant.

The final report should cite the stage summaries and selected artifacts instead
of embedding all raw data.

Run the contract checker from `version_2/` after evidence has been promoted:

```bash
hidden-attractors-check-validation
```

The checker intentionally fails on the initial template-only tree because the
real CSV, JSON, figures, and final report have not been generated yet.
