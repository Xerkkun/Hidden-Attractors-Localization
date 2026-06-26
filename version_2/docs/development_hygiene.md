# Development Hygiene Policies

This document outlines the coding standards and file system rules established for the `thesis-freeze` release to ensure maximum reproducibility, organization, and a clean public API surface.

## 1. Official Entry Points

To maintain a clean and uncluttered workspace, all user-facing execution commands are unified under the primary command-line tool:

* **`hidden-attractors`**: The primary CLI entry point defined in `pyproject.toml` pointing to `hidden_attractors.cli.main:main`. It groups all research and validation workflows as subcommands (e.g., `hidden-attractors protocol`, `hidden-attractors basin`, etc.).
* **Official Examples**: Runnable examples meant for users reside in the `examples/` directory (e.g., `examples/chua_nonsmooth_biased_hidden_attractor/run_example.py`).

No other standalone executable commands should be installed as public entry points.

## 2. Prohibited Script Patterns in Active Layers

To prevent "polluting" the active library directories with exploratory, backup, or temporary scripts, files matching the following patterns are **strictly prohibited** in active directories (including the root `version_2/`, `examples/`, `hidden_attractors/`, `tools/` (except `tools/legacy/`), and `tests/`):

* `scratch_*.py`
* `step*.py`
* `generate_*plots*.py` / `generate_*_plots*.py`
* `generate_*figures*.py`
* `search_*candidates*.py` / `search_*_candidates*.py`
* `compare_*solvers*.py` / `compare_solvers_*.py`
* `*_old.py`
* `*_backup.py`
* `*_tmp.py`
* `test_manual_*.py`

### Exceptions

The following files are explicitly exempted or naturally permitted:

* Files within explicitly designated legacy/archive directories (such as `tools/legacy/` or `_archive/`).
* `hidden_attractors/plotting/generate_publication_figures.py` (which is a core library module defining plotting logic, not an exploratory script).

## 3. Legacy and Historical Scripts

Any exploratory, historical, or transition script must be moved to:

* **`tools/legacy/`** (or a directory named `_archive/`).

### Legacy Policy Requirements

1. **API Disclaimer**: Any script placed in these legacy directories must begin with a header comment clarifying its legacy status:

   ```python
   # Legacy exploratory script. Not part of the public API or promoted validation evidence.
   ```

2. **No Command Recommendations**: Legacy scripts must **not** be recommended as runnable command-line instructions in the documentation (`README.md`, `quick_start.md`, `REFERENCE_GUIDE.md`, etc.).
3. **Internal Only**: These scripts do not form part of the stable public API. They are preserved purely for traceability and historical reproducibility.

## 4. Enforcement

These rules are programmatically enforced by the following test suites:

* `tests/test_no_loose_active_scripts.py`
* `tests/test_no_loose_figure_scripts.py`
* `tests/test_cli_no_redundant_public_scripts.py`
