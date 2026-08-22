# Dependency Policy

This project follows the spirit of the [SPEC-0][spec0] rolling-window support policy adopted by the Scientific Python ecosystem (NumPy, SciPy, Matplotlib, etc.). However, to prioritize stability and user convenience across older computing environments (such as legacy HPC clusters), the project adopts a **pragmatic extended support policy** rather than strict, literal compliance with the SPEC-0 rolling drop dates.

---

## Python version support

This project follows the spirit of SPEC-0 but intentionally keeps an extended-support window for thesis reproducibility and older scientific environments. The automatic CI matrix currently tests Python 3.11, 3.12, and 3.13. Python 3.14 was validated locally on Windows but is not yet tested in CI. Python 3.15 and newer are outside the declared compatibility range.

| Component | Supported/Tested | Policy |
| :--- | :--- | :--- |
| Python 3.11 | tested in CI | extended support |
| Python 3.12 | tested in CI | standard support |
| Python 3.13 | tested in CI | standard/current support |
| Python 3.14 | validated locally; not tested in CI | current/experimental support |
| NumPy >=1.26 | supported | extended lower bound |
| SciPy >=1.12 | supported | extended lower bound |
| Matplotlib >=3.8 | supported | extended lower bound |

---

## Core dependencies

Core dependencies are pinned with a **lower bound only**. Upper bounds are intentionally omitted to avoid unnecessary conflicts in user environments.

| Package | Current lower bound | Released | SPEC-0 drop date | Note |
| --- | --- | --- | --- | --- |
| `numpy` | `>=1.26` | 2023-06 | 2025-06 | Extended support |
| `matplotlib` | `>=3.8` | 2023-09 | 2025-09 | Extended support |
| `scipy` | `>=1.12` | 2024-01 | 2026-01 | Extended support |
| `packaging` | `>=24.0` | 2024-03 | not SPEC-0 governed | PEP 440 version and Python-compatibility checks for safe CLI updates |

> [!NOTE]
> Lower bounds are maintained beyond their strict SPEC-0 drop dates under our extended support model. They will be bumped pragmatically on minor releases of this package if required by new API features or when dropping older Python minor versions.

---

## Optional extras

| Extra | Purpose | Key packages |
| --- | --- | --- |
| `dev` | Test suite | `pytest>=8.0`, `pytest-cov>=5.0` |
| `analysis` | Nonlinear time-series metrics | `antropy>=0.1.6`, `nolds>=0.6.3,<0.7`, `scipy>=1.12` |
| `docs` | Documentation build | `mkdocs>=1.6`, `mkdocs-material>=9.5`, `mkdocstrings[python]>=0.25` |
| `pydstool` | Numerical continuation | `PyDSTool` (no stable PyPI release, no pin) |

> [!WARNING]
> `pydstool` has no stable PyPI release as of 2026. Install manually from
> source if needed. It is not part of the standard CI matrix.

---

## Compatibility bounds

The declared bounds describe the versions supported by this release.
Upper bounds are used only for a documented incompatibility; exact
environment snapshots belong in validation records rather than the public
runtime requirements.

---

## Pinning policy for reproducibility

Exact pins for **reproducible environments** are maintained in a separate
lockfile (not committed). Collaborators should generate one locally:

```bash
pip install -e ".[dev,analysis,docs]"
pip freeze > requirements-lock.txt   # for your own reference only
```

Do **not** commit `requirements-lock.txt` to the repository — the
`pyproject.toml` lower bounds are sufficient for library use.

---

## C extension compatibility

The native backend (`hidden_attractors/native/`) builds a plain C shared
library and loads it through `ctypes`; it is not a CPython extension. Binary
compatibility therefore depends on the C ABI, platform, architecture, compiler
runtime, and the declared `ctypes` layouts. Platform-specific notes:

| Platform | Requirement |
| --- | --- |
| Linux / macOS | `gcc` or `clang` with OpenMP (`libomp`) |
| Windows | `gcc` in `PATH` (e.g. via MSYS2/UCRT64); OpenMP via `libgomp` |

Set `ALLOW_NO_OPENMP=1` to build without OpenMP (single-threaded fallback).
The pure-Python EFORK solver is always available as a fallback and requires
no C toolchain.

---

## References

- [SPEC-0 — Minimum Supported Versions][spec0]
- [NumPy release schedule](https://numpy.org/doc/stable/release/index.html)
- [Matplotlib release schedule](https://matplotlib.org/stable/devel/release_guide.html)

[spec0]: https://scientific-python.org/specs/spec-0000/
