# Test Suite Refactor and Audit Report

This report summarizes the audit, classification, cleaning, and refactoring performed on the test suite of the `Hidden-Attractors-Localization` repository.

## Summary of Changes

### Conserved Tests
All 117+ original test files protecting scientific contracts, local and global solvers, fractional continuation, describing function seed generation, Weyl-Caputo bridge logic, Matignon criteria, and Lyapunov exponent calculations have been fully preserved to guarantee scientific reproducibility.

### Merged and Eliminated Tests
- **Merged**: `tests/test_cli_grouped_commands.py` was consolidated into [test_cli_smoke.py](../tests/test_cli_smoke.py) using a parameterized test framework.
- **Eliminated**: `tests/test_cli_grouped_commands.py` was deleted from the repository.
  - *Justification*: Redundant `--help` and subcommand smoke checks were replaced by a parameterized test suite `test_grouped_cli_help` which tests all commands and aliases programmatically.
  - *Risk*: Negligible, as all code paths are covered by the consolidated parameterized test.

### Refactored for `tmp_path` (Hygienic Execution)
The following test suites were refactored to use `tmp_path` and `monkeypatch` to prevent un-mocked writes to real folders (like `validation/`, `outputs/`, `library_figures/`, etc.) during test runs:
- [test_figure_manifest.py](../tests/test_figure_manifest.py)
- [test_biased_figure_manifest.py](../tests/test_biased_figure_manifest.py)
- [test_published_case_reproduction.py](../tests/test_published_case_reproduction.py)
- [test_official_wolfram_artifacts.py](../tests/test_official_wolfram_artifacts.py)
- [test_wolfram_validations.py](../tests/test_wolfram_validations.py)
- [test_integer_lure_workflow.py](../tests/test_integer_lure_workflow.py)
- [test_lyapunov_promotion.py](../tests/test_lyapunov_promotion.py)

### Migration Checks Converted to Invariants
- [test_no_loose_figure_scripts.py](../tests/test_no_loose_figure_scripts.py) was generalized from a hardcoded list to dynamically scan all active paths (e.g., `.`, `version_2/examples`, `version_2/tools/cli`) for obsolete script pattern matches (e.g., `scratch_*.py`, `step[0-9]_*.py`, etc.), while ignoring archived/legacy subfolders.

### New Hygiene & Consistency Tests
- [test_test_suite_classification.py](../tests/test_test_suite_classification.py): Asserts that all files on disk are documented in the inventory, and all markers used in codebase are registered in `pyproject.toml`.
- [test_tests_do_not_write_real_artifacts.py](../tests/hygiene/test_tests_do_not_write_real_artifacts.py): Scans for file-writing calls in tests targeting real repository paths to prevent leakage.

## Validation Execution and State
All fast unit, CLI, and contract tests pass successfully:
- Quick suite: `pytest -m "not slow" -v` -> PASSED
- Scientific contracts: `pytest -m "scientific_contract" -v` -> PASSED
- CLI: `pytest -m "cli" -v` -> PASSED
- Plotting: `pytest -m "plotting" -v` -> PASSED
