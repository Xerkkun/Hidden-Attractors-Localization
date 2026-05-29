"""
tests/test_wolfram_python_consistency.py
=========================================
Tests that verify consistency between Wolfram validation outputs and
the Python library for the cases where Wolfram outputs are already present.

These tests are marked @pytest.mark.wolfram and will skip if either
wolframscript is unavailable or if Wolfram outputs have not been generated
yet (the output directory is empty).

They test the following quantities against tolerances specified in the task:
    omega0   : 1e-8
    k        : 1e-8
    DF residual |N(a0)-k| : 1e-8
    X_seed   : 1e-7 per component
    W(z)     : 1e-8

Mathematical constraints enforced:
  * W_hat_q(z) evaluated at z = (j omega)^q, never z = j omega for q != 1.
  * S built via P0 S = S Hq, not from eigenvectors.
  * X_seed = a0 * S[:, 0].
  * Describing function treated as first harmonic approximation only.
    It does NOT prove limit cycle existence or attractor hiddenness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_VALIDATION_PYTHON_DIR = Path(__file__).resolve().parents[1] / "validation" / "python"
if str(_VALIDATION_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_PYTHON_DIR))

from run_wolfram_validations import find_wolframscript, repo_root  # noqa: E402

# ---------------------------------------------------------------------------
# Tolerances (from specification)
# ---------------------------------------------------------------------------
TOL_OMEGA0 = 1e-8
TOL_K = 1e-8
TOL_A0 = 1e-8       # used as: N(a0_wolfram) should ≈ k_wolfram within this
TOL_SEED = 1e-7     # per component
TOL_W = 1e-8


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

def _has_wolframscript() -> bool:
    return find_wolframscript() is not None


def _wolfram_outputs_exist(system_id: str) -> bool:
    out = (
        repo_root()
        / "validation"
        / "outputs"
        / "wolfram"
        / system_id
    )
    return out.is_dir() and any(out.glob("*_seed_data.json"))


_WOLFRAM_CASES = [
    "chua_integer_saturation",
    "chua_fractional_saturation",
    "chua_fractional_arctan",
]

# ---------------------------------------------------------------------------
# Lazy library import helper
# ---------------------------------------------------------------------------

def _try_import_library():
    try:
        from compare_with_library import (  # noqa: F401
            compare_seed_data,
            compare_matrix_data,
            compare_eigenvalues,
            _LIBRARY_AVAILABLE,
        )
        return True, None
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.wolfram
@pytest.mark.parametrize("system_id", _WOLFRAM_CASES)
def test_wolfram_python_seed_consistency(system_id: str) -> None:
    """Transfer function and seed values from Wolfram must match Python within tolerance.

    This test reads pre-generated Wolfram outputs.  It does NOT call wolframscript.
    If outputs are absent, the test is skipped.

    Note: These results certify algebraic formulas and seed construction.
    They do NOT declare the existence of a hidden attractor.
    """
    if not _wolfram_outputs_exist(system_id):
        pytest.skip(
            f"No Wolfram outputs found for '{system_id}' — "
            "run: python validation/python/run_wolfram_validations.py --all"
        )

    lib_ok, lib_err = _try_import_library()
    if not lib_ok:
        pytest.skip(f"Library comparison not available: {lib_err}")

    from compare_with_library import compare_seed_data  # noqa: F811

    out_dir = repo_root() / "validation" / "outputs" / "wolfram" / system_id
    seed_json = out_dir / f"{system_id}_seed_data.json"

    results = compare_seed_data(
        seed_json,
        system_id,
        tol_scalar=TOL_K,
        tol_vector=TOL_SEED,
    )

    ok_rows = [r for r in results if r.get("status", "ok") == "ok" or "k_diff" in r]
    assert len(ok_rows) > 0, (
        f"No valid (status=ok) seed rows found in {seed_json}. "
        "Check that the Wolfram script produced output."
    )

    failures = [r for r in ok_rows if not r.get("passed", True)]
    assert not failures, (
        f"{len(failures)} seed row(s) exceeded tolerance for '{system_id}':\n"
        + json.dumps(failures, indent=2, ensure_ascii=False)
    )


@pytest.mark.wolfram
def test_transfer_function_fractional_formula() -> None:
    """Verify that the library evaluates W_hat_q(z) at z=(j omega)^q, not z=j*omega.

    This is a pure-Python test and does NOT require wolframscript.
    It checks that W_eval with transfer_mode='fractional' and transfer_mode='integer'
    give different results for q != 1, confirming the correct formula is used.
    """
    lib_ok, lib_err = _try_import_library()
    if not lib_ok:
        pytest.skip(f"Library not importable: {lib_err}")

    try:
        from hidden_attractors.systems.builtins import chua_system
        from hidden_attractors.lure.transfer import W_eval
    except ImportError as e:
        pytest.skip(f"Library not installed: {e}")

    sys_obj = chua_system("nonsmooth")
    P = sys_obj.lure.matrix
    b = sys_obj.lure.input_vector
    r = sys_obj.lure.output_vector
    omega0 = 2.0

    # For q=1 the two modes must agree
    W_int = W_eval(omega0, 1.0, "integer", P, b, r)
    W_frac_q1 = W_eval(omega0, 1.0, "fractional", P, b, r)
    assert abs(W_int - W_frac_q1) < 1e-12, (
        f"For q=1, integer and fractional modes must agree; got diff={abs(W_int-W_frac_q1)}"
    )

    # For q < 1 the two modes must differ
    q = 0.9998
    W_int_q = W_eval(omega0, q, "integer", P, b, r)
    W_frac_q = W_eval(omega0, q, "fractional", P, b, r)
    assert abs(W_int_q - W_frac_q) > 1e-6, (
        f"For q={q}, integer and fractional modes should differ significantly; "
        f"got diff={abs(W_int_q-W_frac_q):.3e}. "
        "This suggests the fractional formula z=(j omega)^q is not being used."
    )


@pytest.mark.wolfram
@pytest.mark.parametrize("system_id", _WOLFRAM_CASES)
def test_wolfram_summary_passed(system_id: str) -> None:
    """If Wolfram outputs exist, the validation_summary.json must report passed=true."""
    if not _wolfram_outputs_exist(system_id):
        pytest.skip(
            f"No Wolfram outputs for '{system_id}' — "
            "run: python validation/python/run_wolfram_validations.py --all"
        )
    out_dir = repo_root() / "validation" / "outputs" / "wolfram" / system_id
    summaries = sorted(out_dir.glob("*_validation_summary.json"))
    assert summaries, f"No *_validation_summary.json in {out_dir}"
    summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
    assert summary.get("passed") is True, (
        f"Wolfram summary reports failure for '{system_id}':\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
    )


def test_no_hidden_attractor_claim_in_wolfram_outputs() -> None:
    """Wolfram validation outputs must NOT contain 'hidden_verified' claims.

    These scripts certify algebraic formulas and seeds only.
    Claiming a verified hidden attractor from symbolic algebra alone is
    mathematically incorrect.
    """
    out_root = repo_root() / "validation" / "outputs" / "wolfram"
    if not out_root.exists():
        pytest.skip("No Wolfram outputs directory yet.")

    for json_file in out_root.rglob("*.json"):
        content = json_file.read_text(encoding="utf-8").lower()
        assert "hidden_verified" not in content, (
            f"Found 'hidden_verified' in Wolfram output {json_file}. "
            "Wolfram algebraic validation must not claim attractor verification."
        )


def test_build_S_from_similarity_no_eigenvectors() -> None:
    """Verify that build_S_from_similarity satisfies the P0 S = S Hq relation
    and normalisation constraints r^T S = {1, 0, -h} WITHOUT using eigenvectors.
    """
    import math
    from unittest.mock import patch
    from compare_with_library import build_S_from_similarity

    alpha = 8.4562
    beta = 12.0732
    gamma = 0.0052
    m1 = -1.1468
    k = 0.2098673545150838
    omega0 = 2.039186939959001
    d = 1.538510163250452
    h = 16.71582245895634
    q = 1.0
    r = np.array([1.0, 0.0, 0.0])

    P = np.array([
        [-alpha * (1.0 + m1), alpha, 0.0],
        [1.0, -1.0, 1.0],
        [0.0, -beta, -gamma],
    ])
    b = np.array([-alpha, 0.0, 0.0])
    P0 = P + k * np.outer(b, r)

    with patch("numpy.linalg.eig") as mock_eig, patch("numpy.linalg.eigh") as mock_eigh:
        mock_eig.side_effect = RuntimeError("np.linalg.eig is forbidden")
        mock_eigh.side_effect = RuntimeError("np.linalg.eigh is forbidden")

        S = build_S_from_similarity(P0, omega0, q, d, h, r)

        # 1. Check similarity relation
        zr = (omega0 ** q) * math.cos(q * math.pi / 2.0)
        zi = (omega0 ** q) * math.sin(q * math.pi / 2.0)
        Hq = np.array([
            [zr, -zi, 0.0],
            [zi,  zr, 0.0],
            [0.0, 0.0, -d],
        ])

        assert np.linalg.norm(P0 @ S - S @ Hq) < 1e-8
        # 2. Check normalization constraints
        assert abs(r @ S[:, 0] - 1.0) < 1e-8
        assert abs(r @ S[:, 1]) < 1e-8
        assert abs(r @ S[:, 2] + h) < 1e-8


@pytest.mark.wolfram
@pytest.mark.parametrize("system_id", _WOLFRAM_CASES)
def test_compare_seed_data_failures(system_id: str) -> None:
    """Verify that compare_seed_data fails when tolerances are exceeded."""
    if not _wolfram_outputs_exist(system_id):
        pytest.skip(f"No Wolfram outputs found for '{system_id}'")

    lib_ok, lib_err = _try_import_library()
    if not lib_ok:
        pytest.skip(f"Library not importable: {lib_err}")

    from compare_with_library import compare_seed_data

    out_dir = repo_root() / "validation" / "outputs" / "wolfram" / system_id
    seed_json = out_dir / f"{system_id}_seed_data.json"

    # Extremely strict tolerances to trigger failure
    with pytest.raises(AssertionError):
        compare_seed_data(seed_json, system_id, tol_scalar=1e-30, tol_vector=1e-30)


@pytest.mark.wolfram
@pytest.mark.parametrize("system_id", _WOLFRAM_CASES)
def test_compare_matrix_data_consistency(system_id: str) -> None:
    """Verify numeric consistency of Lur'e matrices between Wolfram and Python."""
    if not _wolfram_outputs_exist(system_id):
        pytest.skip(f"No Wolfram outputs found for '{system_id}'")

    lib_ok, lib_err = _try_import_library()
    if not lib_ok:
        pytest.skip(f"Library not importable: {lib_err}")

    from compare_with_library import compare_matrix_data

    out_dir = repo_root() / "validation" / "outputs" / "wolfram" / system_id
    sym_json = out_dir / f"{system_id}_symbolic_summary.json"

    res = compare_matrix_data(sym_json, system_id)
    assert res["passed"] is True
    assert res["P_diff"] < 1e-12
    assert res["b_diff"] < 1e-12
    assert res["r_diff"] < 1e-12


@pytest.mark.wolfram
@pytest.mark.parametrize("system_id", _WOLFRAM_CASES)
def test_compare_equilibria_consistency(system_id: str) -> None:
    """Verify consistency of equilibria between Wolfram and Python."""
    if not _wolfram_outputs_exist(system_id):
        pytest.skip(f"No Wolfram outputs found for '{system_id}'")

    lib_ok, lib_err = _try_import_library()
    if not lib_ok:
        pytest.skip(f"Library not importable: {lib_err}")

    from compare_with_library import compare_equilibria

    out_dir = repo_root() / "validation" / "outputs" / "wolfram" / system_id
    eq_csv = out_dir / f"{system_id}_equilibria_residuals.csv"

    res = compare_equilibria(eq_csv, system_id, tol=1e-8)
    assert res["passed"]
    assert res["max_distance"] < 1e-8


@pytest.mark.wolfram
@pytest.mark.parametrize("system_id", _WOLFRAM_CASES)
def test_compare_eigenvalues_consistency(system_id: str) -> None:
    """Verify consistency of regional/equilibria eigenvalues between Wolfram and Python."""
    if not _wolfram_outputs_exist(system_id):
        pytest.skip(f"No Wolfram outputs found for '{system_id}'")

    lib_ok, lib_err = _try_import_library()
    if not lib_ok:
        pytest.skip(f"Library not importable: {lib_err}")

    from compare_with_library import compare_eigenvalues

    out_dir = repo_root() / "validation" / "outputs" / "wolfram" / system_id
    eig_csv = out_dir / f"{system_id}_eigenvalues_matignon.csv"

    res = compare_eigenvalues(eig_csv, system_id, tol=1e-7)
    for group in res:
        assert group["passed"]
        assert group["max_eigenvalue_diff"] < 1e-7


def test_w_eval_direct_fractional() -> None:
    """Directly verify W_eval for q=1 and q<1 fractional transfer function formula."""
    lib_ok, lib_err = _try_import_library()
    if not lib_ok:
        pytest.skip(f"Library not importable: {lib_err}")

    try:
        from hidden_attractors.lure.transfer import W_eval
    except ImportError as e:
        pytest.skip(f"Library not installed: {e}")

    # Define some test parameters
    alpha, beta, gamma = 8.0, 12.0, 0.005
    m0, m1 = -1.1, -0.6
    P = np.array([
        [-alpha * (1.0 + m1), alpha, 0.0],
        [1.0, -1.0, 1.0],
        [0.0, -beta, -gamma],
    ])
    b = np.array([-alpha, 0.0, 0.0])
    r = np.array([1.0, 0.0, 0.0])

    # For q = 1.0
    omega = 2.5
    q = 1.0
    W_val_py = W_eval(omega, q, "fractional", P, b, r)

    # Analytical formula evaluation:
    z = (1j * omega) ** q
    W_ana = r @ np.linalg.solve(z * np.eye(3) - P, b)
    assert abs(W_val_py - W_ana) < 1e-12

    # For q = 0.9998
    q = 0.9998
    W_val_py_q = W_eval(omega, q, "fractional", P, b, r)
    z_q = (omega ** q) * np.exp(1j * q * np.pi / 2.0)
    W_ana_q = r @ np.linalg.solve(z_q * np.eye(3) - P, b)
    assert abs(W_val_py_q - W_ana_q) < 1e-12


@pytest.mark.wolfram
@pytest.mark.parametrize("system_id", _WOLFRAM_CASES)
def test_compare_all_summary(system_id: str) -> None:
    """Verify that compare_all executes correctly and writes a complete summary JSON."""
    if not _wolfram_outputs_exist(system_id):
        pytest.skip(f"No Wolfram outputs found for '{system_id}'")

    lib_ok, lib_err = _try_import_library()
    if not lib_ok:
        pytest.skip(f"Library not importable: {lib_err}")

    from compare_with_library import compare_all

    out_dir = repo_root() / "validation" / "outputs" / "wolfram" / system_id
    res = compare_all(out_dir, system_id)

    assert res["passed"]
    assert res["system_id"] == system_id

    summary_path = out_dir / f"{system_id}_python_consistency_summary.json"
    assert summary_path.exists()

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["system_id"] == system_id
    assert "output_dir" in summary
    assert summary["passed"]
    assert "checks" in summary
    assert "comparisons" in summary
    assert "missing_comparisons" in summary


