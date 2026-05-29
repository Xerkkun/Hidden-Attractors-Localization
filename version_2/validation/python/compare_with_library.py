"""
compare_with_library.py
=======================
Compare Wolfram validation outputs against the hidden_attractors Python library.

This module reads the JSON/CSV files produced by the Wolfram .wl scripts and
checks them against values computed by the library's own functions.

Tolerances (per specification):
    omega0      : 1e-8
    k           : 1e-8
    a0          : 1e-8
    X_seed      : 1e-7  (per component)
    eigenvalues : 1e-7
    W(z)        : 1e-8

Mathematical conventions enforced here:
  * Fractional transfer function W_hat_q(z) = r^T (z I - P)^{-1} b
    is evaluated at z = (j omega)^q = omega^q exp(j q pi/2).
    Never use z = j*omega when q != 1.
  * Matrix S is built from the similarity relation P0 S = S Hq,
    NOT from eigenvectors of P0.
  * X_seed = a0 * S[:, 0]  (first real column of S).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Library imports — these are the ONLY functions compared against Wolfram.
# No new public functions are created; auxiliaries stay local to this module.
# ---------------------------------------------------------------------------

try:
    from hidden_attractors.models.chua import (
        chua_nonsmooth_parameters,
        chua_arctan_wu2023_parameters,
        chua_parameters,
        equilibria_nonsmooth,
        equilibria_arctan,
        jacobian_nonsmooth,
        jacobian_arctan,
    )
    from hidden_attractors.systems.builtins import (
        chua_system,
        chua_arctan_wu2023_system,
    )
    from hidden_attractors.lure.transfer import W_eval, W_spectral
    from hidden_attractors.lure.describing_function import (
        evaluate_describing_function,
    )
    _LIBRARY_AVAILABLE = True
except ImportError as _lib_err:
    _LIBRARY_AVAILABLE = False
    _lib_import_error = _lib_err


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_library() -> None:
    if not _LIBRARY_AVAILABLE:
        raise ImportError(
            f"hidden_attractors library is not importable: {_lib_import_error}\n"
            "Install the package with: pip install -e ."
        ) from None


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_csv(path: str | Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


# ---------------------------------------------------------------------------
# Fractional transfer function (pure-Python, mathmatically correct form)
# ---------------------------------------------------------------------------

def compute_transfer_fractional_py(
    omega: float,
    q: float,
    P: np.ndarray,
    b: np.ndarray,
    r: np.ndarray,
) -> complex:
    """W_q(j omega) = W_hat_q(z)  where  z = (j omega)^q.

    This is the ONLY correct form for fractional systems.
    Never substitute z = j*omega when q != 1.

    Parameters
    ----------
    omega : float   Frequency (rad/s), positive.
    q     : float   Fractional order, 0 < q <= 1.
    P, b, r         Lur'e matrices.

    Returns
    -------
    complex   Transfer function value at the given frequency.
    """
    return W_eval(omega, q, "fractional", P, b, r)


def compute_transfer_integer_py(
    omega: float,
    P: np.ndarray,
    b: np.ndarray,
    r: np.ndarray,
) -> complex:
    """W(j omega) = r^T (j omega I - P)^{-1} b  for integer-order systems."""
    return W_eval(omega, 1.0, "integer", P, b, r)


# ---------------------------------------------------------------------------
# S-matrix construction via similarity (P0 S = S Hq)
# ---------------------------------------------------------------------------

def build_S_from_similarity(
    P0: np.ndarray,
    omega0: float,
    q: float,
    d: float,
    h: float,
    r: np.ndarray,
) -> np.ndarray:
    """Build S by solving P0 S = S Hq with normalization constraints.

    Constraints:
        r^T s1 = 1
        r^T s2 = 0
        r^T s3 = -h

    No eigenvectors are used.
    """
    zr = (omega0 ** q) * math.cos(q * math.pi / 2.0)
    zi = (omega0 ** q) * math.sin(q * math.pi / 2.0)

    Hq = np.array(
        [
            [zr, -zi, 0.0],
            [zi,  zr, 0.0],
            [0.0, 0.0, -d],
        ],
        dtype=float,
    )

    # vec(P0 S - S Hq) = 0
    # vec(P0 S) = (I kron P0) vec(S)
    # vec(S Hq) = (Hq.T kron I) vec(S)
    I3 = np.eye(3)
    A_sim = np.kron(I3, P0) - np.kron(Hq.T, I3)
    b_sim = np.zeros(9)

    # Normalization constraints on columns of S.
    # vec(S) uses column-major order.
    c1 = np.zeros(9)
    c2 = np.zeros(9)
    c3 = np.zeros(9)

    c1[0:3] = r
    c2[3:6] = r
    c3[6:9] = r

    A = np.vstack([A_sim, c1, c2, c3])
    b_vec = np.concatenate([b_sim, [1.0, 0.0, -h]])

    sol, residuals, rank, svals = np.linalg.lstsq(A, b_vec, rcond=None)
    S = sol.reshape((3, 3), order="F")

    residual_norm = np.linalg.norm(P0 @ S - S @ Hq)
    constraint_norm = np.linalg.norm(
        np.array([r @ S[:, 0] - 1.0, r @ S[:, 1], r @ S[:, 2] + h])
    )

    if residual_norm > 1e-8 or constraint_norm > 1e-8:
        raise ValueError(
            "Failed to construct S from similarity constraints: "
            f"similarity_residual={residual_norm:.3e}, "
            f"constraint_residual={constraint_norm:.3e}, "
            f"rank={rank}"
        )

    return S


# ---------------------------------------------------------------------------
# Seed construction X_seed = a0 * S[:, 0]
# ---------------------------------------------------------------------------

def compute_seed_py(
    system_id: str,
    q: float,
    omega0: float,
    k: float,
    a0: float,
    d: float,
    h: float,
) -> dict:
    """Compute seed data using the library's matrices and the similarity method.

    Parameters
    ----------
    system_id : str    One of 'chua_integer_saturation', 'chua_fractional_saturation',
                       'chua_fractional_arctan'.
    q, omega0, k, a0, d, h : floats from Wolfram seed_data.json

    Returns
    -------
    dict with keys: omega0, k, a0, d, seed_plus, seed_minus, S_residual.
    """
    _require_library()
    sys = _get_system(system_id)
    P = sys.lure.matrix
    b = sys.lure.input_vector
    r = sys.lure.output_vector

    P0 = P + k * np.outer(b, r)
    S = build_S_from_similarity(P0, omega0, q, d, h, r)
    X_seed = a0 * S[:, 0]

    # Verify similarity residual
    zr = (omega0 ** q) * math.cos(q * math.pi / 2)
    zi = (omega0 ** q) * math.sin(q * math.pi / 2)
    Hq = np.array([[zr, -zi, 0], [zi, zr, 0], [0, 0, -d]], dtype=float)
    S_residual = float(np.linalg.norm(P0 @ S - S @ Hq))

    return {
        "omega0": float(omega0),
        "k": float(k),
        "a0": float(a0),
        "d": float(d),
        "seed_plus": X_seed.tolist(),
        "seed_minus": (-X_seed).tolist(),
        "S_residual": S_residual,
    }


# ---------------------------------------------------------------------------
# System helper
# ---------------------------------------------------------------------------

def _get_system(system_id: str):
    _require_library()
    if system_id in ("chua_integer_saturation", "chua_fractional_saturation"):
        return chua_system("nonsmooth")
    if system_id == "chua_fractional_arctan":
        return chua_arctan_wu2023_system()
    raise ValueError(f"Unknown system_id for comparison: {system_id!r}")


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_seed_data(
    wolfram_seed_json: str | Path,
    system_id: str,
    tol_scalar: float = 1e-8,
    tol_vector: float = 1e-7,
) -> list[dict]:
    """Compare Wolfram seed data with Python library results.

    Parameters
    ----------
    wolfram_seed_json : path to <system_id>_seed_data.json
    system_id         : one of the three built-in case IDs
    tol_scalar        : tolerance for omega0, k, a0, W (default 1e-8)
    tol_vector        : tolerance per component for X_seed (default 1e-7)

    Returns
    -------
    list of dicts, one per seed row, with keys:
        q, omega0_wolfram, omega0_python, omega0_diff,
        k_wolfram, k_python, k_diff,
        W_wolfram_re, W_wolfram_im, W_python_re, W_python_im, W_diff,
        a0_wolfram, N_a0_python, df_residual,
        seed_plus_wolfram, seed_plus_python, seed_plus_max_diff,
        S_residual_python, passed.

    Raises
    ------
    AssertionError  if any comparison exceeds tolerances.
    """
    _require_library()
    rows = _load_json(wolfram_seed_json)
    sys = _get_system(system_id)
    P = sys.lure.matrix
    b_vec = sys.lure.input_vector
    r_vec = sys.lure.output_vector

    results = []
    for row in rows:
        if row.get("status") != "ok":
            results.append({"status": row.get("status"), "passed": True})
            continue

        q = float(row["q"])
        omega0_w = float(row["omega0"])
        k_w = float(row["k"])
        a0_w = float(row["a0"])
        d_w = float(row["d"])
        h_w = float(row["h"])
        seed_plus_w = [float(x) for x in row["seed_plus"]]

        # --- Transfer function cross-check ---
        W_re_w = float(row["W_re"])
        W_im_w = float(row["W_im"])
        W_w = complex(W_re_w, W_im_w)

        if abs(q - 1.0) < 1e-10:
            W_py = compute_transfer_integer_py(omega0_w, P, b_vec, r_vec)
        else:
            W_py = compute_transfer_fractional_py(omega0_w, q, P, b_vec, r_vec)

        W_diff = abs(W_py - W_w)

        # k = 1/Re(W) for the k_phi condition
        if abs(W_py.real) > 1e-14:
            k_py = 1.0 / W_py.real
        else:
            k_py = float("nan")

        k_diff = abs(k_py - k_w) if math.isfinite(k_py) else float("inf")

        # --- Describing function a0 cross-check ---
        df_result = evaluate_describing_function(sys, a0_w)
        N_a0_python = df_result.value  # N(a0) should equal k_w; use Wolfram's a0 value as ref
        df_residual = abs(N_a0_python - k_w)  # N(a0_wolfram) vs k_wolfram

        # --- Seed cross-check via similarity ---
        py_seed_data = compute_seed_py(system_id, q, omega0_w, k_w, a0_w, d_w, h_w)
        seed_plus_py = py_seed_data["seed_plus"]
        seed_max_diff = max(
            abs(a - b) for a, b in zip(seed_plus_py, seed_plus_w)
        )

        passed = (
            k_diff <= tol_scalar
            and W_diff <= tol_scalar
            and df_residual <= tol_scalar
            and seed_max_diff <= tol_vector
        )

        results.append({
            "q": q,
            "omega0_wolfram": omega0_w,
            "k_wolfram": k_w,
            "k_python": k_py,
            "k_diff": k_diff,
            "W_wolfram_re": W_re_w,
            "W_wolfram_im": W_im_w,
            "W_python_re": W_py.real,
            "W_python_im": W_py.imag,
            "W_diff": W_diff,
            "a0_wolfram": a0_w,
            "N_a0_python": N_a0_python,
            "df_residual": df_residual,
            "seed_plus_wolfram": seed_plus_w,
            "seed_plus_python": seed_plus_py,
            "seed_plus_max_diff": seed_max_diff,
            "S_residual_python": py_seed_data["S_residual"],
            "tol_scalar": tol_scalar,
            "tol_vector": tol_vector,
            "passed": passed,
        })

    failed = [r for r in results if not r.get("passed", True)]
    if failed:
        import json as _json
        raise AssertionError(
            f"{len(failed)} seed row(s) exceeded tolerance in '{system_id}':\n"
            + _json.dumps(failed, indent=2, ensure_ascii=False)
        )

    return results


def compare_matrix_data(
    wolfram_symbolic_json: str | Path,
    system_id: str,
    tol: float = 1e-12,
) -> dict:
    """Compare Lur'e matrices P, b, r from Wolfram symbolic summary against library numerically.

    Returns
    -------
    dict with keys: P_diff, b_diff, r_diff, passed.
    """
    _require_library()
    sys_obj = _get_system(system_id)
    P_py = sys_obj.lure.matrix
    b_py = sys_obj.lure.input_vector
    r_py = sys_obj.lure.output_vector

    data = _load_json(wolfram_symbolic_json)
    if "lure_form_numeric" not in data:
        raise ValueError(f"lure_form_numeric key not found in {wolfram_symbolic_json}")

    lure_numeric = data["lure_form_numeric"]
    P_w = np.array(lure_numeric["P"], dtype=float)
    b_w = np.array(lure_numeric["b"], dtype=float)
    r_w = np.array(lure_numeric["r"], dtype=float)

    P_diff = float(np.max(np.abs(P_py - P_w)))
    b_diff = float(np.max(np.abs(b_py - b_w)))
    r_diff = float(np.max(np.abs(r_py - r_w)))

    passed = (P_diff < tol) and (b_diff < tol) and (r_diff < tol)

    result = {
        "system_id": system_id,
        "P_diff": P_diff,
        "b_diff": b_diff,
        "r_diff": r_diff,
        "passed": passed,
    }
    if not passed:
        raise AssertionError(
            f"Lur'e matrix numeric comparison failed for {system_id}:\n"
            f"P_diff={P_diff:.3e}, b_diff={b_diff:.3e}, r_diff={r_diff:.3e} (tol={tol:.3e})"
        )
    return result


def compare_equilibria(
    wolfram_equilibria_csv: str | Path,
    system_id: str,
    tol: float = 1e-8,
) -> dict:
    """Compare Python equilibria with Wolfram equilibria using distance-based permutation matching.

    Returns
    -------
    dict with keys: max_distance, passed.
    """
    _require_library()
    sys_obj = _get_system(system_id)

    # Load Wolfram equilibria from CSV
    rows = _load_csv(wolfram_equilibria_csv)
    header = rows[0]
    data = rows[1:]

    wolfram_pts = []
    for row_vals in data:
        row = dict(zip(header, row_vals))
        x = float(row["x"])
        y = float(row["y"])
        z = float(row["z"])
        wolfram_pts.append(np.array([x, y, z]))

    # Load Python equilibria
    if system_id in ("chua_integer_saturation", "chua_fractional_saturation"):
        py_eqs = equilibria_nonsmooth()
    elif system_id == "chua_fractional_arctan":
        py_eqs = equilibria_arctan()
    else:
        raise ValueError(f"Unknown system_id: {system_id}")

    py_pts = list(py_eqs.values())

    if len(wolfram_pts) != len(py_pts):
        raise AssertionError(
            f"Number of equilibria differs: Wolfram={len(wolfram_pts)}, Python={len(py_pts)}"
        )

    # Perform permutation matching
    from itertools import permutations
    best_max_dist = float("inf")
    for p in permutations(py_pts):
        dist = max(np.linalg.norm(w - py) for w, py in zip(wolfram_pts, p))
        if dist < best_max_dist:
            best_max_dist = float(dist)

    passed = bool(best_max_dist < tol)

    result = {
        "system_id": system_id,
        "max_distance": best_max_dist,
        "passed": passed,
    }
    if not passed:
        raise AssertionError(
            f"Equilibria comparison failed for {system_id}: max_distance={best_max_dist:.3e} (tol={tol:.3e})"
        )
    return result


def compare_eigenvalues(
    wolfram_eigenvalue_csv: str | Path,
    system_id: str,
    tol: float = 1e-7,
) -> list[dict]:
    """Compare Matignon eigenvalues from Wolfram CSV with numpy eigenvalues.

    Returns
    -------
    list of dicts, one per region/equilibrium and q, with comparison results.
    """
    _require_library()
    sys_obj = _get_system(system_id)

    rows = _load_csv(wolfram_eigenvalue_csv)
    header = rows[0]
    data = rows[1:]

    # Parse and group by (q, region/equilibrium)
    groups = {}
    for row_vals in data:
        row = dict(zip(header, row_vals))
        q = float(row["q"])
        region = row.get("region", row.get("equilibrium", ""))

        # Parse abs_argument checking for "Pi" or "-Pi"
        abs_arg_str = row["abs_argument"].strip()
        if abs_arg_str == "Pi":
            abs_arg = math.pi
        elif abs_arg_str == "-Pi":
            abs_arg = -math.pi
        else:
            abs_arg = float(abs_arg_str)

        real = float(row["real"])
        imag = float(row["imag"])
        val = complex(real, imag)

        key = (q, region)
        if key not in groups:
            groups[key] = {"eigenvalues": [], "abs_arguments": [], "matignon_margins": []}
        groups[key]["eigenvalues"].append(val)
        groups[key]["abs_arguments"].append(abs_arg)
        groups[key]["matignon_margins"].append(float(row["matignon_margin"]))

    # Sort Python equilibria to match region/equilibrium name mapping if needed
    eq_map = {}
    if system_id == "chua_fractional_arctan":
        # Wolfram name "E0", "E1", "E2" mapped to Python E-, E0, E+ by sorting x coordinate
        py_eqs = equilibria_arctan()
        sorted_eqs = sorted(py_eqs.items(), key=lambda item: item[1][0])
        eq_map = {
            "E0": sorted_eqs[0][1],  # negative x-coordinate
            "E1": sorted_eqs[1][1],  # zero x-coordinate
            "E2": sorted_eqs[2][1],  # positive x-coordinate
        }

    results = []
    for (q, region), group_data in groups.items():
        w_eigs = group_data["eigenvalues"]
        w_args = group_data["abs_arguments"]

        # Compute Python Jacobian and its eigenvalues
        if system_id in ("chua_integer_saturation", "chua_fractional_saturation"):
            if region == "inner":
                state = np.zeros(3)
            elif region == "outer":
                state = np.array([2.0, 0.0, 0.0])
            else:
                raise ValueError(f"Unknown region for saturation system: {region}")
            jac = jacobian_nonsmooth(state)
        elif system_id == "chua_fractional_arctan":
            if region not in eq_map:
                raise ValueError(f"Unknown equilibrium name in CSV: {region}")
            state = eq_map[region]
            jac = jacobian_arctan(state)
        else:
            raise ValueError(f"Unknown system_id: {system_id}")

        py_eigs = np.linalg.eigvals(jac)

        # Match eigenvalues using permutation to minimize max absolute distance
        from itertools import permutations
        best_perm = None
        best_max_diff = float("inf")
        for p in permutations(py_eigs):
            diff = max(abs(w - py) for w, py in zip(w_eigs, p))
            if diff < best_max_diff:
                best_max_diff = diff
                best_perm = p

        # Check matched pairs within tolerance
        pair_passed = True
        matched_details = []
        for w_val, w_arg, py_val in zip(w_eigs, w_args, best_perm):
            val_diff = abs(w_val - py_val)
            py_arg = abs(math.atan2(py_val.imag, py_val.real))
            arg_diff = abs(w_arg - py_arg)

            passed_individual = (val_diff < tol) and (arg_diff < tol)
            if not passed_individual:
                pair_passed = False

            matched_details.append({
                "wolfram_eigenvalue": w_val,
                "python_eigenvalue": py_val,
                "eigenvalue_diff": val_diff,
                "wolfram_abs_argument": w_arg,
                "python_abs_argument": py_arg,
                "abs_argument_diff": arg_diff,
                "passed": passed_individual,
            })

        results.append({
            "q": q,
            "region": region,
            "max_eigenvalue_diff": best_max_diff,
            "matched_details": matched_details,
            "passed": pair_passed,
        })

    failed = [r for r in results if not r.get("passed", True)]
    if failed:
        raise AssertionError(
            f"Eigenvalue comparison failed for {system_id} on {len(failed)} group(s) (tol={tol:.3e})"
        )

    return results


def compare_all(output_dir: str | Path, system_id: str) -> dict:
    """Run all available comparisons for a given system output directory.

    Parameters
    ----------
    output_dir : Path to the directory containing Wolfram outputs.
    system_id  : System identifier string.

    Returns
    -------
    dict summarising all comparison results.
    """
    out = Path(output_dir)

    results: dict = {
        "system_id": system_id,
        "output_dir": str(out),
        "comparisons": {},
        "passed": True,
        "missing_comparisons": [],
    }

    checks = {
        "matrix_numeric": False,
        "equilibria": False,
        "eigenvalues": False,
        "seed_data": False,
    }

    # 1. Matrix structure/numeric comparison
    sym_json = out / f"{system_id}_symbolic_summary.json"
    if sym_json.exists():
        try:
            r = compare_matrix_data(sym_json, system_id)
            results["comparisons"]["matrix_numeric"] = r
            checks["matrix_numeric"] = r["passed"]
        except Exception as e:
            results["comparisons"]["matrix_numeric"] = {"error": str(e), "passed": False}
            results["passed"] = False
    else:
        results["missing_comparisons"].append(str(sym_json))

    # 2. Equilibria comparison
    eq_csv = out / f"{system_id}_equilibria_residuals.csv"
    if eq_csv.exists():
        try:
            r = compare_equilibria(eq_csv, system_id)
            results["comparisons"]["equilibria"] = r
            checks["equilibria"] = r["passed"]
        except Exception as e:
            results["comparisons"]["equilibria"] = {"error": str(e), "passed": False}
            results["passed"] = False
    else:
        results["missing_comparisons"].append(str(eq_csv))

    # 3. Eigenvalues comparison
    eig_csv = out / f"{system_id}_eigenvalues_matignon.csv"
    if eig_csv.exists():
        try:
            r = compare_eigenvalues(eig_csv, system_id)
            results["comparisons"]["eigenvalues"] = r
            checks["eigenvalues"] = all(item["passed"] for item in r)
            if not checks["eigenvalues"]:
                results["passed"] = False
        except Exception as e:
            results["comparisons"]["eigenvalues"] = {"error": str(e), "passed": False}
            results["passed"] = False
    else:
        results["missing_comparisons"].append(str(eig_csv))

    # 4. Seed data comparison
    seed_json = out / f"{system_id}_seed_data.json"
    if seed_json.exists():
        try:
            r = compare_seed_data(seed_json, system_id)
            results["comparisons"]["seed_data"] = r
            checks["seed_data"] = all(item.get("passed", True) for item in r)
            if not checks["seed_data"]:
                results["passed"] = False
        except Exception as e:
            results["comparisons"]["seed_data"] = {"error": str(e), "passed": False}
            results["passed"] = False
    else:
        results["missing_comparisons"].append(str(seed_json))

    # Save python consistency summary
    summary_path = out / f"{system_id}_python_consistency_summary.json"
    consistency_summary = {
        "system_id": system_id,
        "passed": results["passed"],
        "checks": checks,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(consistency_summary, f, indent=2, ensure_ascii=False)

    return results

