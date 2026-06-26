# Release package

`hidden-attractors-fo` provides reproducible workflows for theoretical-numerical
search, localization, audit, and conservative classification of hidden-attractor
candidates in integer- and commensurate Caputo fractional-order Chua/Lur'e
systems.

## Install

```bash
cd version_2
python -m pip install -e ".[dev,analysis,docs,legacy]"
```

## Minimal checks

```bash
hidden-attractors --help
hidden-attractors validate release-readiness
hidden-attractors validate release-readiness --strict
hidden-attractors validate contract --allow-pending
python -m pytest -q -m "release_readiness"
```

`--strict` is expected to pass when only declared final pending items remain.
`--submission-strict` is the strict archival readiness gate and may fail while
remaining validations, final scientific freeze audit, or executed sample outputs
are pending.

## Evidence included

- Promoted evidence: `validation/`
- Promoted figures: `library_figures/`
- Full API inventory: `docs/api_reference.md`
- Release sample templates: `release_package/sample_input/` and `release_package/sample_output/`
- Arctan c590 promotion boundary: `release_package/ARCTAN_C590_PROMOTION_BOUNDARY.md`
- Ordinary local outputs: `outputs/`, `validation_outputs/`, `runs*/`, `figures/`

## What is claimed

- The package exposes a single unified CLI, `hidden-attractors`.
- The integer Chua `q=1` route is the reproduced software reference.
- The non-smooth fractional Chua BDF example is a proposed methodology lane.
- The arctan Wu2023/c590 example separates bibliographic reproduction from the
  c590 Caputo candidate promoted for local radii `r <= 0.3`.
- API functions, classes, and methods are inventoried in `docs/api_reference.md`.
- The package is declared as `v1.0.0`; arctan promotion is radius-limited and recorded in the archive manifest.

## What is not claimed

- No global mathematical proof of hiddenness.
- No full Danca 2017 fractional hidden-attractor trajectory reproduction.
- No global arctan basin proof beyond the promoted local-radius contract.
- No chaos/hiddenness certification from Lyapunov, FFT/PSD, Poincare, 0-1, or
  phase portraits alone.
- No public Machado/FDF seed CLI; that route remains theory/internal planned support.

## Authorship, supervision, and code provenance

Maria Fernanda Moreno Lopez is the principal author and maintainer. Dr. Esteban
Tlelo Cuautle is acknowledged as doctoral thesis director and research guide.
Dr. Oscar Martinez-Fuentes is acknowledged for reviewing the theoretical
fractional-calculus component. Dr. Luis Gerardo de la Fraga is acknowledged for
code provenance related to EFORK and the integer-order Lyapunov algorithm.
