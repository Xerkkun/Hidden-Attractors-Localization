# Installation

## Editable Install

From the repository root:

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

For the historical scripts in `tools/legacy/`:

```bash
python -m pip install -e ".[legacy]"
```

Those scripts remain outside the public API, but some reproducibility workflows
still need SciPy and PyYAML.

## Native Backends

The package contains C sources in `hidden_attractors/native/csrc/` and compiles
shared libraries into `.runtime_native/` when a native workflow is executed.

Requirements:

- Python 3.10 or newer.
- `numpy` and `matplotlib`.
- A C compiler for native workflows.
- On macOS, Homebrew `libomp` may be required for OpenMP-enabled builds.

## GitHub Install Shape

Once published, the expected install command is:

```bash
python -m pip install "git+https://github.com/Xerkkun/Hidden-Attractors-Localization.git#subdirectory=version_2"
```
