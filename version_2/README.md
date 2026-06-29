# hidden-attractors-fo

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Status](https://img.shields.io/badge/status-stable%201.0.0-green)
![Package](https://img.shields.io/badge/package-PyPI%20ready-green)
![License](https://img.shields.io/badge/license-MIT-blue)
[![CI](https://github.com/Xerkkun/Hidden-Attractors-Localization/actions/workflows/ci.yml/badge.svg)](https://github.com/Xerkkun/Hidden-Attractors-Localization/actions/workflows/ci.yml)

`hidden-attractors-fo` provides reproducible workflows for locating, simulating,
auditing, and conservatively classifying hidden-attractor candidates in integer-
and commensurate Caputo fractional-order Lur'e-compatible systems. Maintained
Chua lanes cover an integer reference, non-smooth fractional BDF/saturation,
and smooth arctan fractional validation example.

The PyPI project name is `hidden-attractors-fo`; the Python import name remains
`hidden_attractors`.

## PyPI installation

```bash
python -m pip install hidden-attractors-fo
```

Verify the public CLI:

```bash
hidden-attractors --help
hidden-attractors inspect systems
hidden-attractors seed --help
```

Use the package from Python as:

```python
import hidden_attractors
```

## Development installation

From a repository checkout:

```bash
python -m pip install -e ".[dev,analysis,docs,legacy]"
```

The package exposes one public console command:

```bash
hidden-attractors
```

## Quick path for new users

From `version_2/` after installation:

```bash
hidden-attractors --help
hidden-attractors inspect systems
python examples/chua_integer_lure_reference/run_example.py --quick
hidden-attractors validate contract --allow-pending
hidden-attractors validate release-readiness --submission-strict --json
```

## Research workflows

```bash
hidden-attractors inspect candidates
hidden-attractors run -p chua_integer
hidden-attractors run -p chua_fractional
```

Run a YAML file:

```bash
hidden-attractors init -e chua_fractional
hidden-attractors run -c configs/examples/chua_fractional_centered_lure_df.yaml
```

## Official examples

```bash
python examples/chua_integer_lure_reference/run_example.py --quick
python examples/chua_nonsmooth_biased_hidden_attractor/run_example.py --quick
python examples/chua_arctan_wu2023/run_example.py --quick
```

| Example | Role | Evidence status |
| --- | --- | --- |
| `examples/chua_integer_lure_reference/` | Integer `q=1` Lur'e reference | Reproduced software reference/control |
| `examples/chua_nonsmooth_biased_hidden_attractor/` | Biased-DF methodology for non-smooth fractional Chua | Candidate/compatible under tested local radii; not full Danca reproduction |
| `examples/chua_arctan_wu2023/` | Smooth arctan Wu2023 lane plus c590 local lane | Wu2023 bibliographic; c590 is finite-time local/radius-limited evidence under the recorded contract |

## Article reproduction status

| Source case | Library coverage |
| --- | --- |
| Integer Chua reference | Reproduced as the maintained `q=1` software route |
| Danca 2017 non-smooth fractional Chua | Partial implementation; missing published numerical details prevent full trajectory reproduction |
| Official nearby non-smooth candidate | Classified under the recorded local-neighborhood contract with target contacts from equilibrium neighborhoods |
| Wu2023 arctan Chua | Algebra/ADM local lane implemented as bibliographic reproduction; c590 is one smooth-nonlinearity Caputo validation lane with finite-time local/radius-limited evidence |
| DK2018/Fischer Lyapunov lanes | Diagnostic comparison lanes with documented discrepancies |

## API reference

[docs/api_reference.md](docs/api_reference.md) is generated from the active
`hidden_attractors` package and lists every defined function, class, and method.
Private helpers are included for auditability, but stable public usage should
prefer the unified CLI, top-level exports, and documented workflow specs.

Common programmatic entry points:

```python
from hidden_attractors import get_system, register_system
from hidden_attractors.systems import ChaoticSystem
from hidden_attractors.workflows.specs import WorkflowInputSpec
from hidden_attractors.workflows.config_loader import load_config
from hidden_attractors.integrations.selector import integrate
```

## New Lur'e systems

A new system needs more than a vector field before it can enter the full
methodology. Provide equilibria, Jacobian, an explicit Lur'e split `(P, b, r,
psi)`, describing-function convention, solver/memory contract, target reference,
classifier thresholds, and all-equilibrium neighborhood sampling settings. See
[docs/adapting_new_systems.md](docs/adapting_new_systems.md).

## Scientific Scope

DF, BDF, Nyquist, and continuation generate or transport candidate seeds.
FFT/PSD, 0-1, Poincare, phase portraits, and Lyapunov estimates are diagnostics.
Hiddenness is only a finite numerical label under a recorded contract; there is no global mathematical proof.

The Chua arctan c590 lane is finite-time evidence under a local/radius-limited
contract, not a global basin proof. The Wu2023 ADM lane remains bibliographic
and does not replace full-history Caputo validation.

Machado/FDF remains theory/internal planned support and is not exposed as a
public seed workflow in this release.

## Advanced validation workflows

These checks are for maintainers and release verification, not the first user route.

## Tests and release checks

```bash
python -m compileall hidden_attractors examples tests tools/cli
python -m pytest -q
python -m pytest -q -m "hygiene"
python -m pytest -q -m "release_readiness"
hidden-attractors validate release-readiness --submission-strict --json
```

Packaging checks:

```bash
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
python tools/release/validate_wheel_install.py
```

Release packaging metadata lives in `release_package/`. Promoted evidence lives
under `validation/`; ordinary run products stay under `outputs/`.

## Documentation map

- [User Manual](USER_MANUAL.md)
- [Claims Matrix](THESIS_CLAIMS.md)
- [Freeze Audit](validation/freeze_audit/)
- [Quick Start](docs/quick_start.md)
- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [Examples Index](docs/examples_index.md)
- [Scientific Scope](docs/scientific_scope.md)
- [Validation Evidence](docs/validation_evidence.md)
- [Unified Report](docs/unified_report.md)
- [Citation](docs/citation.md)

## Citation

Citation metadata is provided at the repository root in `CITATION.cff`,
`.zenodo.json`, and `codemeta.json`.

Archived DOI: `10.17605/OSF.IO/ZGK74`.

## License

MIT.

## Source

GitHub: <https://github.com/Xerkkun/Hidden-Attractors-Localization>
