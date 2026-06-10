# Installation

## Editable Install

Depending on your current working directory, run one of the following commands:

### A. From the Repository Root Directory
```bash
python -m pip install -e version_2
```

### B. From the `version_2/` Subdirectory
```bash
python -m pip install -e .
```

---

## Recommended Install for Validation and Development

To run unit tests, validation contracts, and chaos analysis:

### A. From the Repository Root Directory
```bash
python -m pip install -e "version_2[dev,analysis,legacy]"
```

### B. From the `version_2/` Subdirectory
```bash
python -m pip install -e ".[dev,analysis,legacy]"
```

---

## Supported Environments

- **Python Version**: Requires `Python >= 3.11`.
- **CI Matrix**: Versions `3.11`, `3.12`, and `3.13` are fully tested in the automatic CI pipeline.
- For detailed package support boundaries and rolling support guidelines, refer to the [Dependency Policy](dependency_policy.md).

---

## Verification and Smoke Checks

After installation, verify that the unifed CLI command is registered and running correctly:

```bash
hidden-attractors --help
hidden-attractors inspect systems
hidden-attractors validate contract --allow-pending
```

---

## Native C-solvers Compilation

The library contains native high-performance C solvers under `hidden_attractors/native/csrc/`. These compile dynamically into `.runtime_native/` on demand.
- **Windows**: Requires a GCC compiler (e.g. MinGW) on your system `PATH`.
- **Linux**: Requires `build-essential`.
- **macOS**: Requires Xcode Command Line Tools. OpenMP parallelism can be supported by running `brew install libomp`. If OpenMP is missing or disabled, compile by passing the environment variable `ALLOW_NO_OPENMP=1`.
