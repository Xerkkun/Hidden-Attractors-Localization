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

## Freeze audit status after CPC cleanup

`freeze_audit_status: pending_after_cpc_cleanup` is intentional. The last recorded freeze audit corresponds to commit `2bcea3430c50d3fb4e5eb70c8621cb3550dcc59a` and must be regenerated before CPC submission. Do not treat the current CPC-preparation metadata changes as covered by that older freeze audit until the full audit is rerun and recorded.
