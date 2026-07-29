# Reproducibility

## Installation

```bash
cd version_2
python -m pip install -e ".[dev,analysis,docs]"
```

## Minimal checks

```bash
hidden-attractors --help
hidden-attractors inspect systems
python -m pytest -q tests/test_package_smoke.py
python validation/paper07_chua/scripts/sync_paper07_evidence.py --verify
```

## Validated example

```bash
python examples/chua_integer_lure_reference/run_example.py --quick
```

## Evidence layout

- Closed validation records: `version_2/validation/`
- API inventory: `version_2/docs/api_reference.md`

## Claim boundary

Use [version_2/docs/scientific_scope.md](version_2/docs/scientific_scope.md)
for the evidence boundary. Diagnostics and seed-generation artifacts do not
prove hiddenness.

## Archive metadata

Citation and archive metadata are provided in `CITATION.cff`, `.zenodo.json`,
and `codemeta.json`.
