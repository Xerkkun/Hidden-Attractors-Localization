# hidden-attractors-fo

`hidden-attractors-fo` is a Python library for reproducible numerical
integration, Lur'e seed construction, continuation, diagnostics, and
finite-neighborhood verification in integer- and commensurate Caputo
fractional-order systems.

The public documentation describes implemented software and completed
validation only. Exploratory runs, internal study notes, project plans, and
unvalidated parameter searches are not part of the public distribution.

## Installation

```bash
python -m pip install hidden-attractors-fo
```

The import name and public command are:

```python
import hidden_attractors
```

```bash
hidden-attractors --help
hidden-attractors inspect systems
```

## Independent characterization

The library can calculate finite-time characteristics independently of a
hidden-attractor search: trajectory and boundedness metrics, FFT/PSD,
Poincare sections, the 0-1 statistic, bifurcation post-processing, and
equation-based Lyapunov spectra. Version 1.1.0 fully integrates Lyapunov
estimation from uniformly sampled scalar time series using Rosenstein and
Eckmann reconstruction plus Kaplan--Yorke dimension.

## Validated example

The maintained end-to-end example is the integer-order Chua Lur'e
reference/control:

```bash
cd version_2
python examples/chua_integer_lure_reference/run_example.py --quick
```

It exercises seed construction, continuation, integration, sampled
equilibrium-neighborhood controls, and structured output generation. Its
finite numerical decision is not a global mathematical proof.

## Distribution boundary

The wheel contains the importable library and supported package resources.
The source distribution additionally contains the user manual and validated
example. Complete validation manifests and reproducibility records are
preserved in the tagged repository and archived snapshot; they are not
duplicated inside the installed package.

Canonical validation records live under `version_2/validation/`. Ordinary
runtime products are written to `./outputs` unless the user selects another
directory. The installed package directory is never used as an output or cache
location.

## Documentation

- User manual: `version_2/USER_MANUAL.md`
- Installation: `version_2/docs/installation.md`
- Quick start: `version_2/docs/quick_start.md`
- API stability: `version_2/docs/api_stability.md`
- Scientific scope: `version_2/docs/scientific_scope.md`
- Validation freeze: `version_2/validation/freeze_audit/`

## Citation and license

Citation metadata is provided in `CITATION.cff`, `.zenodo.json`, and
`codemeta.json`. Archived DOI: `10.17605/OSF.IO/ZGK74`.

The software is licensed under the MIT License.
