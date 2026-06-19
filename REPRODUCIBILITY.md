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
hidden-attractors validate cpc-readiness --strict
python -m pytest -q -m "not slow"
```

Exact test counts can change as the suite evolves. The frozen source for reported scientific counts is `version_2/validation/freeze_audit/`.

## Evidence boundary

Promoted evidence is stored under `version_2/validation/`. Promoted scientific figures are stored under `version_2/library_figures/` and must be generated through `hidden_attractors.plotting.export.export_figure`.

Local outputs, exploratory runs, regenerated artifacts, and arctan exploratory outputs belong under `version_2/outputs/`, `version_2/validation_outputs/`, `version_2/runs*/`, or `figures/`, all outside Git.

Editorial drafts, the official Elsevier/CPC template, and the final manuscript are prepared locally under ignored `paper/`. They are intentionally not tracked as part of the software repository readiness contract. The repository tracks the software package, promoted validation evidence, citation metadata, reproducibility notes, and CPC submission scaffolding under `version_2/cpc_submission/`.

The arctan system is implemented algebraically, pending full validation, and must not be cited as a promoted validated hidden attractor.

## CI and freeze audit boundary

The GitHub Actions CI matrix for the CPC cleanup has passed. This confirms package hygiene and cross-platform test execution for the current repository state. It does not replace the full scientific freeze audit, which remains a separate artifact to regenerate once final promoted validation cases are fixed for submission.

The project keeps a small hygiene/readiness test suite because numerical tests do not protect repository publication boundaries. These tests guard against retracking local outputs, local manuscripts, absolute paths, legacy CLI entry points, unpromoted validation outputs, and overclaimed CPC metadata.

To run these tests specifically:
```bash
python -m pytest -q -m "hygiene"
python -m pytest -q -m "cpc_readiness"
python -m pytest -q -m "not hygiene and not cpc_readiness"
```

CI status: passed for current CPC cleanup. Freeze audit: last full scientific freeze audit corresponds to commit `2bcea3430c50d3fb4e5eb70c8621cb3550dcc59a` and must be regenerated only when the final scientific evidence set is frozen.

## Code provenance

The EFORK implementation and integer-order Lyapunov algorithm include code provenance from material provided by Dr. Luis Gerardo de la Fraga. This provenance should remain traceable in publication and archival metadata.
