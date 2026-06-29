# Release package

`hidden-attractors-fo` provides reproducible workflows for theoretical-numerical
search, localization, audit, and conservative classification of hidden-attractor
candidates in integer- and commensurate Caputo fractional-order Chua/Lur'e
systems.

## PyPI installation

```bash
python -m pip install hidden-attractors-fo
```

The Python import name is:

```python
import hidden_attractors
```

The only public console script is:

```bash
hidden-attractors
```

## Development installation

```bash
cd version_2
python -m pip install -e ".[dev,analysis,docs,legacy]"
```

## Minimal checks

```bash
hidden-attractors --help
hidden-attractors seed --help
hidden-attractors validate release-readiness --submission-strict --json
hidden-attractors validate contract --allow-pending
python -m pytest -q -m "release_readiness"
```

## Package build checks

```bash
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
python tools/release/validate_wheel_install.py
```

The wheel smoke test must confirm:

- `hidden-attractors --help` works from the installed wheel.
- `hidden-attractors seed --help` works from the installed wheel.
- `hidden-attractors seed --help` does not expose Machado/FDF.
- `import hidden_attractors` works from the installed wheel.

## Evidence included

- Promoted evidence: `validation/`
- Promoted figures: `library_figures/`
- Full API inventory: `docs/api_reference.md`
- Release sample templates: `release_package/sample_input/` and `release_package/sample_output/`
- Arctan c590 local-radius boundary: `release_package/ARCTAN_C590_PROMOTION_BOUNDARY.md`
- PyPI release checklist: `release_package/PYPI_RELEASE_CHECKLIST.md`
- Publishing policy: `release_package/PUBLISHING_POLICY.md`
- Ordinary local outputs: `outputs/`, `validation_outputs/`, `runs*/`, `figures/`

PyPI distributes the software package. The GitHub repository and archived DOI
record remain the evidence locations for promoted scientific artifacts.

## What is claimed

- The package exposes a single unified CLI, `hidden-attractors`.
- The PyPI package name is `hidden-attractors-fo`.
- The Python import name is `hidden_attractors`.
- The package version is `1.0.0`.
- The integer Chua `q=1` route is the reproduced software reference.
- The non-smooth fractional Chua BDF example is a proposed methodology lane.
- The arctan Wu2023/c590 example separates bibliographic reproduction from the
  c590 Caputo candidate, which remains finite-time local/radius-limited evidence.
- API functions, classes, and methods are inventoried in `docs/api_reference.md`.

## What is not claimed

- No global mathematical proof of hiddenness.
- No full Danca 2017 fractional hidden-attractor trajectory reproduction.
- No global basin proof beyond any recorded local-radius contract.
- No chaos/hiddenness certification from Lyapunov, FFT/PSD, Poincare, 0-1, or
  phase portraits alone.
- No public Machado/FDF seed CLI; that route remains theory/internal planned support.

## Authorship, supervision, and code provenance

Maria Fernanda Moreno Lopez is the principal author and maintainer. Dr. Esteban
Tlelo Cuautle is acknowledged as doctoral thesis director and research guide.
Dr. Oscar Martinez-Fuentes is acknowledged for reviewing the theoretical
fractional-calculus component. Dr. Luis Gerardo de la Fraga is acknowledged for
code provenance related to EFORK and the integer-order Lyapunov algorithm.
