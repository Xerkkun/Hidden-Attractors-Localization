"""
tests/test_wolfram_validations.py
==================================
Pytest test suite for Wolfram Language algebraic/numerical validation cases.

Each test is marked with @pytest.mark.wolfram.
If wolframscript is not in PATH, all tests are automatically skipped.

Run only these tests with:
    pytest -m wolfram -v

Skip Wolfram tests (for CI without Wolfram Engine):
    pytest -m "not wolfram"

Note: These tests do NOT certify hidden attractor existence.
They only certify algebraic formulas, Lur'e form, equilibria,
Jacobians, Matignon criterion data, transfer function, and seed data.
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make the validation/python directory importable during pytest collection.
# This avoids requiring the user to install validation scripts as a package.
# ---------------------------------------------------------------------------
_VALIDATION_PYTHON_DIR = Path(__file__).resolve().parents[1] / "validation" / "python"
if str(_VALIDATION_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_PYTHON_DIR))

from run_wolfram_validations import (  # noqa: E402
    DEFAULT_CASES,
    repo_root,
    run_case,
    find_wolframscript,
)


# ---------------------------------------------------------------------------
# Shared skip condition
# ---------------------------------------------------------------------------

def _wolframscript_available() -> bool:
    return find_wolframscript() is not None


_SKIP_NO_WOLFRAM = pytest.mark.skipif(
    not _wolframscript_available(),
    reason="wolframscript not found in PATH — install Wolfram Engine to run these tests",
)


# ---------------------------------------------------------------------------
# Individual case tests
# ---------------------------------------------------------------------------

@pytest.mark.wolfram
@_SKIP_NO_WOLFRAM
@pytest.mark.parametrize(
    "case_relpath",
    DEFAULT_CASES,
    ids=[Path(c).stem for c in DEFAULT_CASES],
)
def test_wolfram_case_validation(case_relpath: str, tmp_path) -> None:
    """Run a Wolfram validation script and assert that passed=true in the summary JSON.

    The output is written to a temporary directory so tests do not pollute the
    repository.  For persistent outputs use run_wolfram_validations.py directly.
    """
    root = repo_root()
    case_path = root / case_relpath
    out_dir = tmp_path / f"wolfram_{uuid.uuid4().hex}" / Path(case_relpath).stem
    try:
        result = run_case(case_path, out_dir)
        assert result["summary"]["passed"] is True, (
            f"Wolfram validation failed for {case_relpath}:\n"
            f"{result['summary']}"
        )
    finally:
        shutil.rmtree(out_dir.parent, ignore_errors=True)


# ---------------------------------------------------------------------------
# Existence / smoke tests (no wolframscript required)
# ---------------------------------------------------------------------------

@pytest.mark.hygiene
def test_wolfram_case_files_exist() -> None:
    """All DEFAULT_CASES point to files that are actually present in the repo."""
    root = repo_root()
    for relpath in DEFAULT_CASES:
        p = root / relpath
        assert p.exists(), f"Expected Wolfram case not found: {p}"


@pytest.mark.hygiene
def test_common_wolfram_files_exist() -> None:
    """The common Wolfram helper scripts are present."""
    root = repo_root()
    common_dir = root / "validation" / "wolfram" / "common"
    expected = [
        "ha_validation_common.wl",
        "chua_saturation_validation.wl",
        "chua_arctan_validation.wl",
    ]
    for name in expected:
        p = common_dir / name
        assert p.exists(), f"Common Wolfram file missing: {p}"


@pytest.mark.hygiene
def test_template_wolfram_file_exists() -> None:
    """The new_lure_system_template.wl is present in validation/wolfram/template/."""
    root = repo_root()
    p = root / "validation" / "wolfram" / "template" / "new_lure_system_template.wl"
    assert p.exists(), f"Template file missing: {p}"


@pytest.mark.hygiene
def test_runner_script_exists() -> None:
    """validation/python/run_wolfram_validations.py is present."""
    root = repo_root()
    p = root / "validation" / "python" / "run_wolfram_validations.py"
    assert p.exists(), f"Runner script missing: {p}"


@pytest.mark.hygiene
def test_compare_module_exists() -> None:
    """validation/python/compare_with_library.py is present."""
    root = repo_root()
    p = root / "validation" / "python" / "compare_with_library.py"
    assert p.exists(), f"Compare module missing: {p}"


@pytest.mark.hygiene
def test_wolfram_validation_readme_exists() -> None:
    """validation/wolfram/README.md is present."""
    root = repo_root()
    p = root / "validation" / "wolfram" / "README.md"
    assert p.exists(), f"Wolfram validation README missing: {p}"
