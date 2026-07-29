# Installation

## PyPI

```bash
python -m pip install hidden-attractors-fo
```

Install optional time-series and complexity diagnostics with:

```bash
python -m pip install "hidden-attractors-fo[analysis]"
```

Verify the library and CLI:

```bash
python -c "import hidden_attractors; print(hidden_attractors.__name__)"
hidden-attractors --help
hidden-attractors inspect systems
```

## Development checkout

From `version_2/`:

```bash
python -m pip install -e ".[dev,analysis,docs]"
python -m pytest -q -m "hygiene or release_readiness"
```

Python 3.11, 3.12, and 3.13 are supported by the release CI matrix.

## Runtime locations

Generated outputs default to `./outputs`. Override output and cache locations
with:

```bash
HIDDEN_ATTRACTORS_OUTPUT_DIR=/path/to/output
HIDDEN_ATTRACTORS_CACHE_DIR=/path/to/cache
```

Native compilation products are stored in the user cache. The installed
package directory is read-only from the library's point of view.

## Native backends

Native C backends require a compatible C compiler. Python reference backends
remain available when native compilation is unavailable. Any reported result
must record the selected backend and numerical contract.
