# hidden-attractors-fo

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research%20alpha-orange)
![Package](https://img.shields.io/badge/package-editable%20install-green)
![License](https://img.shields.io/badge/license-pending-lightgrey)
[![CI](https://github.com/Xerkkun/Hidden-Attractors-Localization/actions/workflows/ci.yml/badge.svg)](https://github.com/Xerkkun/Hidden-Attractors-Localization/actions/workflows/ci.yml)

`hidden-attractors-fo` is a Python research library for reproducing, auditing,
and extending numerical workflows for hidden-attractor candidates in
integer- and fractional-order Chua/Lur'e systems.

The package turns the previous script collection into a reusable library:
models, candidate loaders, trajectory diagnostics, basin labels, plotting
helpers, native C/EFORK backends, and reproducible workflows live under
`hidden_attractors/`. Historical scripts are exposed through installable
compatibility commands while their reusable pieces are migrated into package
modules.

## Features

- Chua piecewise model definitions and equilibrium helpers.
- Final candidate loaders for the current reference outputs.
- Geometry, spectral, phase-space, and bifurcation post-processing diagnostics.
- Basin classification labels and plotting helpers.
- Optional adapters for external nonlinear time-series tools such as `nolds`
  and `antropy`, with PyDSTool documented as a companion continuation tool.
- C/EFORK wrapper classes with local build policy and backend contracts for
  future system-specific native engines.
- Workflow entry points for robustness overlays, sphere controls, refined basin
  classification, and the unified Chua pipeline without manual environment
  variables.
- A system registry for built-in and user-defined chaotic systems. Full
  Nyquist/DF workflows require a manual Lur'e form.
- Public seed-generation helpers for classical, biased, and Machado
  describing-function seeds.
- Generic order-one Lur'e helpers for seed generation, epsilon continuation,
  hiddenness controls, reusable figures, and Lyapunov estimates.
- Installable compatibility commands for historical workflows.
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

For the historical research scripts kept under `tools/legacy/`:

```bash
python -m pip install -e ".[legacy]"
```

The native workflows compile C sources on demand. On Windows this expects a C
compiler such as `gcc` on `PATH`; on macOS, OpenMP builds may require Homebrew
`libomp` unless the workflow explicitly disables OpenMP.

### Platform Limitations (C backends)

| Platform | Compiler | OpenMP | Status |
|----------|----------|--------|--------|
| Linux | `gcc` (system) | Available via `-fopenmp` | ✅ CI tested |
| Windows | `gcc` (MinGW, must be in `PATH`) | Available via `-fopenmp` | ✅ CI tested |
| macOS | `clang` (Xcode) | Requires `brew install libomp` | ⚠️ Manual CI only |

- **Windows**: The `PATH` on GitHub Actions runners already includes MinGW `gcc`.
  For local installs, ensure `gcc --version` works before calling any workflow
  that triggers native compilation.  The shared library is compiled as `.dll`;
  pure-Python smoke tests and `pytest` do not require the C compiler.
- **macOS**: The default `clang` does not bundle OpenMP.  Either install
  `libomp` via Homebrew (`brew install libomp`) or set `ALLOW_NO_OPENMP=1` to
  compile without parallelism.  Set `LIBOMP_PREFIX` to override the Homebrew
  prefix.  macOS is excluded from the automatic CI matrix but can be triggered
  manually via `workflow_dispatch` on the `ci.yml` workflow.
- **ALLOW_NO_OPENMP**: Setting `ALLOW_NO_OPENMP=1` lets `compile_c_target`
  retry the build without `-fopenmp` instead of raising `RuntimeError`.  This
  is always set in CI to avoid hard failures on platforms where OpenMP is absent.

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
python examples/minimal_chua_protocol.py
python examples/custom_system_definition.py
python examples/integer_lure_chua_protocol.py
python examples/dynamical_analysis_gallery.py
```

After editable installation, the same candidate listing is available as:

```bash
hidden-attractors-list-candidates
hidden-attractors-systems
hidden-attractors-unified-chua --help
hidden-attractors-integer-chua --help
hidden-attractors-legacy --list
```

Heavy numerical stages use the packaged C backends. Python helpers are for
configuration, seed construction, post-processing, plotting, and IO.

For systems beyond the built-in Chua cases, the user must provide the equations
and, for the full DF route, the Lur'e split `A, b, c, psi(c^T x)`, the
classical describing function, the Machado branch, equilibria, and preferably a
Jacobian. Current C backends are Chua-specific; reusable native contracts are
available for adding system-specific integer or fractional engines.

## Documentation Map

- [Documentation Home](docs/index.md)
- [Installation](docs/installation.md)
- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [Examples](docs/examples.md)
- [Systems](docs/systems.md)
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
hidden_attractors/systems/ extensible chaotic-system registry
hidden_attractors/workflows/integer_lure.py generic order-one Lur'e workflow
examples/               short, user-facing examples
tests/                  package smoke tests
configs/                workflow configurations
docs/                   user and developer documentation
tools/cli/              compatibility command wrappers
tools/legacy/           packaged historical scripts behind installable commands
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
python examples/minimal_chua_protocol.py
python examples/custom_system_definition.py
python examples/integer_lure_chua_protocol.py
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
