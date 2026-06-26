# Reproducibility

## Installation

```bash
cd version_2
python -m pip install -e ".[dev,analysis,docs,legacy]"
```

## Minimal checks

```bash
hidden-attractors --help
hidden-attractors inspect systems
hidden-attractors validate contract --allow-pending
hidden-attractors validate release-readiness
python -m pytest -q -m "release_readiness"
```

## Official examples

```bash
python examples/chua_integer_lure_reference/run_example.py --quick
python examples/chua_nonsmooth_biased_hidden_attractor/run_example.py --quick
python examples/chua_arctan_wu2023/run_example.py --quick
```

## Evidence layout

- Promoted validation evidence: `version_2/validation/`
- Promoted figures: `version_2/library_figures/`
- API inventory: `version_2/docs/api_reference.md`
- Local/generated outputs: `version_2/outputs/`, `version_2/validation_outputs/`, `version_2/runs*/`, `version_2/figures/`
- Release package metadata: `version_2/release_package/`

## Claim boundary

Use [version_2/THESIS_CLAIMS.md](version_2/THESIS_CLAIMS.md) for the current
claim matrix. Diagnostics and seed-generation artifacts do not prove hiddenness.

## Archive metadata

Citation and archive metadata are provided in `CITATION.cff`, `.zenodo.json`,
and `codemeta.json`.
