# hidden-attractors-fo 1.1.0

[![PyPI](https://img.shields.io/pypi/v/hidden-attractors-fo)](https://pypi.org/project/hidden-attractors-fo/)
[![CI](https://github.com/Xerkkun/Hidden-Attractors-Localization/actions/workflows/ci.yml/badge.svg)](https://github.com/Xerkkun/Hidden-Attractors-Localization/actions/workflows/ci.yml)

`hidden-attractors-fo` provides reproducible numerical components for
integer-order and commensurate Caputo fractional-order Lur'e-compatible
systems. The installed surface includes system definitions, integer and
fractional integrators, seed construction, continuation, finite-time
diagnostics, sampled-neighborhood verification, plotting, and a unified CLI.

This public distribution documents implemented software and completed
validation only. It does not ship exploratory runs, internal study notes,
project plans, or unvalidated parameter searches.

## Install

```bash
python -m pip install hidden-attractors-fo
```

```python
import hidden_attractors
```

```bash
hidden-attractors --help
hidden-attractors inspect systems
```

For development from this directory:

```bash
python -m pip install -e ".[dev,analysis,docs]"
```

## Independent dynamical characterization

The library can characterize supplied systems, trajectories, and scalar time
series without running a hidden-attractor search. Public analysis entry points
cover generic trajectory and boundedness metrics, FFT/PSD, Poincare sections,
the 0-1 statistic, bifurcation post-processing, and equation-based Lyapunov
spectra.

Version 1.1.0 fully integrates Lyapunov estimation from a uniformly sampled
scalar time series. The structured result combines Rosenstein's largest
exponent, an Eckmann reconstructed spectrum, and a Kaplan--Yorke dimension
with units, estimator parameters, backend provenance, fit diagnostics, memory
guards, and finite-data warnings:

```bash
python -m pip install "hidden-attractors-fo[analysis]"
```

```python
from hidden_attractors import estimate_time_series_lyapunov

result = estimate_time_series_lyapunov(
    signal,
    sample_interval=0.01,
    time_unit="s",
    observable="x",
)
print(result.largest_exponent)
print(result.spectrum)
print(result.kaplan_yorke_dimension)
```

These calculations return finite numerical characteristics; they do not by
themselves certify chaos or hiddenness.

## Validated end-to-end example

From `version_2/`:

```bash
python examples/chua_integer_lure_reference/run_example.py --quick
```

The example executes seed construction, continuation, integration, sampled
equilibrium-neighborhood controls, and structured JSON/CSV output. It is an
integer-order software reference/control. Its finite numerical result is not a
global proof of hiddenness.

Programmatic starting points:

```python
from hidden_attractors import get_system
from hidden_attractors.integrations.selector import integrate
from hidden_attractors.workflows.config_loader import load_config
```

## Scientific Scope

The supported scope and evidence boundaries are summarized in
[`docs/scientific_scope.md`](docs/scientific_scope.md).

Describing-function and Nyquist calculations construct seeds. Continuation
transports those seeds. Phase portraits, spectra, Poincare sections, 0-1
statistics, and Lyapunov estimates are finite-time diagnostics. None of these
operations alone establishes hiddenness.

A hiddenness label requires the declared sampled-neighborhood or basin
contract, including all relevant equilibria, recorded solver settings, and
reproducible classifier thresholds. Such a label remains finite numerical
evidence rather than a global mathematical proof.

## Runtime paths

By default, generated outputs are written under `./outputs`. Override this
with `HIDDEN_ATTRACTORS_OUTPUT_DIR`. Runtime caches use the operating system
user-cache location or `HIDDEN_ATTRACTORS_CACHE_DIR`. The library does not
write into its installed `site-packages` directory.

## Validation and distribution

The wheel contains the importable library and its supported configuration
resources. The source distribution additionally contains the manual set and
the validated integer example. Full validation manifests and reproducibility
records live in the tagged repository and the archived DOI snapshot, not in
the installed package.

Maintainer checks:

```bash
python -m pytest -q -m "hygiene or release_readiness"
python -m build
python -m twine check dist/*
python tools/release/validate_wheel_install.py
```

The canonical freeze record is `validation/freeze_audit/`.

## Documentation

- [User Manual](USER_MANUAL.md)
- [Installation](docs/installation.md)
- [Quick Start](docs/quick_start.md)
- [API Stability](docs/api_stability.md)
- [Scientific Scope](docs/scientific_scope.md)
- [Citation](docs/citation.md)

## Citation and license

Archived DOI: `10.17605/OSF.IO/ZGK74`.

The software is licensed under the MIT License.
