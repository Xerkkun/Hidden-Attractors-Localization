# Release package 1.2.0

`hidden-attractors-fo` provides reproducible numerical components for
integer-order and commensurate Caputo fractional-order Lur'e-compatible
systems. The current 1.2.0 surface also exposes independent characterization of dynamical
systems, trajectories, and scalar time series. It fully integrates
Rosenstein/Eckmann Lyapunov reconstruction and Kaplan--Yorke dimension for a
uniformly sampled scalar signal.

## Publication state

The 1.2.0 source is a locally verified release candidate. This record does not
claim a PyPI publication that has not been independently observed. The public
package version recorded at verification time is 1.0.0.

The release contract is machine-readable in `archive_manifest.json`.

## Installation

PyPI package name:

```bash
python -m pip install hidden-attractors-fo
```

Python import and public console script:

```python
import hidden_attractors
```

```bash
hidden-attractors --help
hidden-attractors inspect systems
```

Development installation from `version_2/`:

```bash
python -m pip install -e ".[dev,analysis,docs]"
```

## Release verification

```bash
python tools/release/validate_release_readiness.py --submission-strict --json
python -m pytest -q -m "not slow"
python -m build
python -m twine check dist/*
python tools/release/validate_wheel_install.py
```

The protected workflow additionally checks that the tag equals the package
version and that verification does not modify tracked files.

## Comprehensive software-validation sample

The tracked release control executes seed construction, continuation, final
integration, sampled-neighborhood verification, and structured output:

```bash
python examples/chua_integer_lure_reference/run_example.py \
  --config release_package/sample_input/chua_integer_comprehensive.yaml \
  --quick \
  --steps search continuation verification \
  --output-dir <empty-output-directory>
```

`sample_output/comprehensive_sample_summary.json` records two independent runs
with identical deterministic outputs. It is a software-validation control, not
promoted scientific evidence and not a global proof of hiddenness.

## Distribution boundary

The wheel contains the importable package and supported runtime resources. The
source distribution additionally contains selected user documentation, the
mathematical-diagnostics reference, the complete 33-function plotting catalog,
its generator and 62 real-system numerical PNG outputs, and the validated
integer reference example. Both exclude repository tests, maintainer release
files, exploratory configurations, ordinary workflow outputs, validation-side
figures, and the validation archive.

The catalog images are computed from the registered `chua-nonsmooth` system
under the parameters and numerical contracts recorded in
`catalog_results.json`. They are reproducible examples, not experimental
measurements or new validation evidence.

Scientific validation records remain in the repository and DOI archive. PyPI
distributes executable software; it is not the evidence archive.

## Evidence boundary

Describing-function and Nyquist calculations construct seeds. Continuation
transports them. Lyapunov, spectral, Poincare, boundedness, bifurcation, and
0-1 diagnostics characterize finite trajectories or time series. None of these
operations alone proves hiddenness.

A hiddenness label requires the declared sampled-neighborhood or basin contract,
including all relevant equilibria, solver settings, classifier thresholds, and
reproducible outputs. The resulting statement remains finite numerical evidence,
not a global mathematical proof.

## Authorship and provenance

Maria Fernanda Moreno Lopez is the principal author and maintainer. Esteban
Tlelo Cuautle supervised the research; Oscar Martinez-Fuentes reviewed the
fractional-calculus methodology; Luis Gerardo de la Fraga provided code
provenance for EFORK and the integer-order Lyapunov algorithm.
