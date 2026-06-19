# Reproducibility

## Installation

```bash
cd version_2
python -m pip install -e ".[dev,analysis,legacy]"
```

## Checks

```bash
hidden-attractors --help
hidden-attractors validate contract --allow-pending
hidden-attractors validate release-readiness
python -m pytest -q
```

## Evidence layout

* Promoted validation evidence: `version_2/validation/`
* Promoted figures: `version_2/library_figures/`
* Local/generated outputs: `version_2/outputs/`, `version_2/validation_outputs/`, `version_2/runs*/`, `version_2/figures/`

## Archive metadata

Citation and archive metadata are provided in `CITATION.cff`, `.zenodo.json`, and `codemeta.json`.
