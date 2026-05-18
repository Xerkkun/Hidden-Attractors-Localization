# Notebooks

Notebook-style examples are useful for teaching, inspection, and report
preparation. Keep them light enough to run without launching long numerical
jobs.

## Available Notebooks

- `examples/notebooks/hidden_attractors_quickstart.ipynb`

This notebook covers:

- importing the package;
- checking Chua equilibria;
- loading final candidate records;
- computing a lightweight trajectory diagnostic on synthetic sample data.

For phase-space plots, bifurcation post-processing, and optional complexity
metrics, mirror the documented workflow in [Dynamical Analysis](dynamical_analysis.md).

## Notebook Rules

- Keep the first cells self-contained and explanatory.
- Avoid long C/EFORK runs unless the notebook name clearly says so.
- Store generated figures under `outputs/notebooks/<notebook-name>/`.
- Mirror important notebook examples as `.py` scripts when possible.
