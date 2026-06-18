# Reproducibility

## Installation

```bash
cd version_2
python -m pip install -e ".[dev,analysis,legacy]"
```

## Quick checks

```bash
hidden-attractors --help
hidden-attractors validate contract --allow-pending
hidden-attractors validate cpc-readiness
python -m pytest -q -m "not slow"
```

Exact test counts can change as the suite evolves. The frozen source for reported counts is `version_2/validation/freeze_audit/`.

## Evidence boundary

Promoted evidence is stored under `version_2/validation/`. Promoted scientific figures are stored under `version_2/library_figures/` and must be generated through `hidden_attractors.plotting.export.export_figure`.

Local outputs, exploratory runs, regenerated artifacts, and arctan exploratory outputs belong under `version_2/outputs/`, `version_2/validation_outputs/`, `version_2/runs*/`, or `version_2/figures/`, all outside Git.

The arctan system is implemented algebraically but is pending full validation and must not be cited as a promoted validated hidden attractor.

## Code provenance

The EFORK implementation and integer-order Lyapunov algorithm include code provenance from material provided by Dr. Luis Gerardo de la Fraga. This provenance should remain traceable in publication and archival metadata.
