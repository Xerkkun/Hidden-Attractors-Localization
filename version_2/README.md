# hidden-attractors-fo

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research%20alpha-orange)
![Package](https://img.shields.io/badge/package-editable%20install-green)
![License](https://img.shields.io/badge/license-pending-lightgrey)

`hidden-attractors-fo` is a Python research library for reproducing, auditing,
and extending numerical workflows for hidden-attractor candidates in
fractional-order Chua/Lur'e systems.

The package turns the previous script collection into a reusable library:
models, candidate loaders, trajectory diagnostics, basin labels, plotting
helpers, native C/EFORK backends, and reproducible workflows live under
`hidden_attractors/`. Research scripts that are still useful but not yet part
of the public API are kept under `tools/legacy/`.

## Features

- Chua piecewise model definitions and equilibrium helpers.
- Final candidate loaders for the current reference outputs.
- Geometry, spectral, phase-space, and bifurcation post-processing diagnostics.
- Basin classification labels and plotting helpers.
- Optional adapters for external nonlinear time-series tools such as `nolds`
  and `antropy`, with PyDSTool documented as a companion continuation tool.
- C/EFORK wrapper classes with local build policy.
- Workflow entry points for robustness overlays, sphere controls, and refined
  basin classification.
- Small examples and smoke tests for users who install from GitHub.

## Installation

From this directory:

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
```

The native workflows compile C sources on demand. On Windows this expects a C
compiler such as `gcc` on `PATH`; on macOS, OpenMP builds may require Homebrew
`libomp` unless the workflow explicitly disables OpenMP.

## Quick Start

```python
from hidden_attractors import chua_piecewise_parameters
from hidden_attractors.models import equilibria_piecewise, rhs_piecewise

params = chua_piecewise_parameters()
for name, point in equilibria_piecewise(params).items():
    residual = rhs_piecewise(point, params)
    print(name, point, residual)
```

Run an included example:

```bash
python examples/quickstart_equilibria.py
python examples/list_final_candidates.py
python examples/dynamical_analysis_gallery.py
```

After editable installation, the same candidate listing is available as:

```bash
hidden-attractors-list-candidates
```

## Documentation Map

- [Documentation Home](docs/index.md)
- [Installation](docs/installation.md)
- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [Examples](docs/examples.md)
- [Dynamical Analysis](docs/dynamical_analysis.md)
- [External Tools](docs/external_tools.md)
- [Unified Report](docs/unified_report.md)
- [Figure Gallery](docs/figure_gallery.md)
- [Code Reference Map](docs/code_reference_map.md)
- [Notebooks](docs/notebooks.md)
- [Workflows](docs/workflows.md)
- [Testing](docs/testing.md)
- [Repository Layout](docs/repository_layout.md)
- [Contributing](docs/contributing.md)
- [Citation](docs/citation.md)

## Repository Layout

```text
hidden_attractors/      importable Python package
examples/               short, user-facing examples
tests/                  package smoke tests
configs/                workflow configurations
docs/                   user and developer documentation
tools/cli/              compatibility command wrappers
tools/legacy/           historical scripts not yet public API
outputs/                reference outputs used by examples/loaders
artifacts/              migrated runtime/prebuilt artifacts
```

## Notebook Tutorial

A narrative notebook is included for users who prefer an exploratory workflow:

```text
examples/notebooks/hidden_attractors_quickstart.ipynb
```

It follows the same style as scientific Python notebooks: short explanation,
imports, model inspection, candidate loading, and lightweight diagnostics.

## Tests

```bash
python -m compileall hidden_attractors examples tests tools/cli
python examples/quickstart_equilibria.py
python examples/list_final_candidates.py
python examples/dynamical_analysis_gallery.py
python -m pytest -q
```

If `pytest` is not installed, the first three commands still provide a useful
smoke check.

## Documentation Site

The docs can be served locally with MkDocs:

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
```

The site navigation is defined in `mkdocs.yml`.

## Scientific Scope

The library records numerical contracts. It does not turn finite-time
simulations into mathematical proofs of hiddenness. Any reported result must
state the model, `q`, step size, memory length, integration time, burn-in,
backend, and classification thresholds.

## Citation

Citation metadata is pending. Until a public release exists, cite the repository
URL, commit hash, and the project reports in `docs/`.
