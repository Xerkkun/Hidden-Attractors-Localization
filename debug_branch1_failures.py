from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

import chua_initial_cond as chua
from equilibria_analysis import local_jacobian, solve_equilibria
from extended_search_utils import chua_ic_params, json_safe, load_config, write_csv


CSV_FIELDS = [
    "candidate_id",
    "branch",
    "mu",
    "theta",
    "q",
    "A",
    "omega",
    "phase",
    "seed_x",
    "seed_y",
    "seed_z",
    "N_re",
    "N_im",
    "abs_N",
    "arg_N",
    "logN_re",
    "logN_im",
    "N_mu_re",
    "N_mu_im",
    "abs_N_mu",
    "arg_N_mu",
    "lambda_re",
    "lambda_im",
    "W_re",
    "W_im",
    "residual_abs",
    "has_nan_anywhere",
    "has_inf_anywhere",
    "failed_stage",
    "failed_function",
    "failed_matrix_name",
    "condition_number_max",
    "eig_solver_used",
    "eig_success",
    "eig_error_message",
    "scipy_available",
    "schur_success",
    "diagnosis_label",
    "recommended_fix",
]


def cpair(z: complex) -> List[float]:
    z = complex(z)
    return [float(np.real(z)), float(np.imag(z))]


def stringify_eigs(values: Sequence[complex] | None) -> str:
    if values is None:
        return ""
    return ";".join(f"{complex(v).real:.16g}{complex(v).imag:+.16g}j" for v in values)


def matrix_entries(M: np.ndarray) -> List[List[List[float]]]:
    A = np.asarray(M)
    if A.ndim != 2:
        return []
    return [[cpair(A[i, j]) for j in range(A.shape[1])] for i in range(A.shape[0])]


def validate_matrix_for_eigs(M: Any, name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "matrix_name": name,
        "valid_for_eigs": False,
        "validation_error": "",
        "has_nan": False,
        "has_inf": False,
        "matrix_entries": "",
        "matrix_norm_2": float("nan"),
        "matrix_norm_fro": float("nan"),
        "condition_number": float("nan"),
        "determinant": "",
        "min_abs_entry": float("nan"),
        "max_abs_entry": float("nan"),
        "rank_estimate": "",
        "singular_values": "",
    }
    try:
        A = np.asarray(M)
    except Exception as exc:
        out["validation_error"] = f"array_conversion_failed: {exc}"
        return out
    out["shape"] = list(A.shape)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        out["validation_error"] = "matrix_not_square"
        return out
    if A.dtype.kind not in {"f", "c", "i", "u"}:
        out["validation_error"] = f"invalid_dtype: {A.dtype}"
        return out
    A = A.astype(np.complex128 if np.iscomplexobj(A) else np.float64)
    out["matrix_entries"] = matrix_entries(A)
    out["has_nan"] = bool(np.isnan(A).any())
    out["has_inf"] = bool(np.isinf(A).any())
    if out["has_nan"] or out["has_inf"]:
        out["validation_error"] = "matrix_contains_nan_or_inf"
        return out
    try:
        abs_A = np.abs(A)
        out["min_abs_entry"] = float(np.min(abs_A))
        out["max_abs_entry"] = float(np.max(abs_A))
        out["matrix_norm_fro"] = float(np.linalg.norm(A, ord="fro"))
        out["matrix_norm_2"] = float(np.linalg.norm(A, ord=2))
        s = np.linalg.svd(A, compute_uv=False)
        out["singular_values"] = [float(v) for v in s]
        out["rank_estimate"] = int(np.linalg.matrix_rank(A))
        out["condition_number"] = float(np.linalg.cond(A))
        if A.shape[0] <= 6:
            out["determinant"] = cpair(np.linalg.det(A))
    except Exception as exc:
        out["validation_error"] = f"matrix_diagnostics_failed: {exc}"
        return out
    if not np.isfinite(out["matrix_norm_2"]):
        out["validation_error"] = "matrix_norm_not_finite"
        return out
    out["valid_for_eigs"] = True
    return out


def safe_eig(M: Any, name: str) -> Dict[str, Any]:
    validation = validate_matrix_for_eigs(M, name)
    out: Dict[str, Any] = {
        **validation,
        "eig_solver_used": "",
        "eig_success": False,
        "eig_error_message": "",
        "eig_values_if_success": "",
        "numpy_eig_success_if_available": False,
        "scipy_available": False,
        "scipy_eig_success_if_available": False,
        "schur_success": False,
        "schur_values_if_success": "",
        "attempts": [],
    }
    if not validation["valid_for_eigs"]:
        out["eig_error_message"] = "eig_skipped_invalid_matrix: " + str(validation["validation_error"])
        return out
    A = np.asarray(M, dtype=np.complex128 if np.iscomplexobj(M) else np.float64)
    try:
        vals = np.linalg.eigvals(A)
        out["numpy_eig_success_if_available"] = True
        out["eig_success"] = True
        out["eig_solver_used"] = "numpy.linalg.eigvals"
        out["eig_values_if_success"] = stringify_eigs(vals)
        out["attempts"].append({"solver": "numpy.linalg.eigvals", "success": True, "values": [cpair(v) for v in vals]})
        return out
    except Exception as exc:
        out["attempts"].append({"solver": "numpy.linalg.eigvals", "success": False, "error": str(exc)})
        out["eig_error_message"] = str(exc)
    try:
        import scipy.linalg as sla  # type: ignore

        out["scipy_available"] = True
        try:
            vals = sla.eigvals(A)
            out["scipy_eig_success_if_available"] = True
            out["eig_success"] = True
            out["eig_solver_used"] = "scipy.linalg.eigvals"
            out["eig_values_if_success"] = stringify_eigs(vals)
            out["attempts"].append({"solver": "scipy.linalg.eigvals", "success": True, "values": [cpair(v) for v in vals]})
            return out
        except Exception as exc:
            out["attempts"].append({"solver": "scipy.linalg.eigvals", "success": False, "error": str(exc)})
            out["eig_error_message"] += f"; scipy.linalg.eigvals: {exc}"
        try:
            _T, Z = sla.schur(A, output="complex")
            vals = np.diag(_T)
            out["schur_success"] = True
            out["eig_success"] = True
            out["eig_solver_used"] = "scipy.linalg.schur"
            out["eig_values_if_success"] = stringify_eigs(vals)
            out["schur_values_if_success"] = stringify_eigs(vals)
            out["attempts"].append({"solver": "scipy.linalg.schur", "success": True, "values": [cpair(v) for v in vals]})
            _ = Z
            return out
        except Exception as exc:
            out["attempts"].append({"solver": "scipy.linalg.schur", "success": False, "error": str(exc)})
            out["eig_error_message"] += f"; scipy.linalg.schur: {exc}"
    except Exception as exc:
        out["scipy_available"] = False
        out["attempts"].append({"solver": "scipy import", "success": False, "error": str(exc)})
    return out


def detect_transfer_convention(p: Dict[str, Any], q: float, omega: float) -> Dict[str, Any]:
    P, b, r = chua.chua_matrices(p)
    lam = (1j * float(omega)) ** float(q)
    M_code = P.astype(np.complex128) - lam * np.eye(3, dtype=np.complex128)
    M_report = lam * np.eye(3, dtype=np.complex128) - P.astype(np.complex128)
    W_repo = complex(chua.W_frac(omega, q, p))
    W_code = complex((r.reshape(1, -1).astype(np.complex128) @ np.linalg.inv(M_code) @ b.reshape(-1, 1).astype(np.complex128))[0, 0])
    W_report = complex((r.reshape(1, -1).astype(np.complex128) @ np.linalg.inv(M_report) @ b.reshape(-1, 1).astype(np.complex128))[0, 0])
    d_code = abs(W_repo - W_code)
    d_report = abs(W_repo - W_report)
    if d_code <= d_report:
        convention = "repo_uses_P_minus_lambda_I"
        formula = "W_code(lambda)=r^T(P-lambda I)^(-1)b"
    else:
        convention = "repo_uses_lambda_I_minus_P"
        formula = "W_report(lambda)=r^T(lambda I-P)^(-1)b"
    return {
        "transfer_sign_convention": convention,
        "transfer_formula_used": formula,
        "W_repo": W_repo,
        "W_code_difference_abs": float(d_code),
        "W_report_difference_abs": float(d_report),
    }


def load_summary_records(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for rec in data.get("records", []):
        if int(rec.get("branch_index", -1)) == 1:
            out[str(rec.get("slug", ""))] = rec
    return out


def phase_tag(theta: float) -> str:
    return f"{float(theta):.5f}".replace("-", "m").replace(".", "p")


def mu_tag(mu: float) -> str:
    return f"{float(mu):.5f}".replace("-", "m").replace(".", "p")


def trajectory_csv_for_candidate(summary_record: Dict[str, Any], output_root: Path, slug: str) -> Path:
    outdir = summary_record.get("output_dir")
    if outdir:
        candidate_dir = Path(outdir)
    else:
        parts = slug.split("_")
        branch = parts[1]
        mu = parts[3]
        theta = parts[5]
        candidate_dir = output_root / f"branch_{branch}" / f"mu_{mu}" / f"theta_{theta}"
    return candidate_dir / f"final_attractor_{slug}.csv"


def covariance_diagnostic(csv_path: Path) -> Dict[str, Any]:
    if not csv_path.exists():
        return {"available": False, "error": "trajectory_csv_not_found"}
    try:
        traj = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        X = np.asarray(traj[:, 1:4], dtype=float)
        centered = X - np.mean(X, axis=0)
        cov = np.cov(centered.T)
        return {
            "available": True,
            "trajectory_has_nan": bool(np.isnan(X).any()),
            "trajectory_has_inf": bool(np.isinf(X).any()),
            "trajectory_max_abs": float(np.nanmax(np.abs(X))),
            "trajectory_rows": int(X.shape[0]),
            "matrix": cov,
            "safe_eig": safe_eig(cov, "trajectory_covariance"),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def candidate_grid(cfg: Dict[str, Any], summary_records: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    dbg = cfg.get("debug_branch1_failures", {})
    branch = int(dbg.get("branches", [1])[0])
    mus = [float(v) for v in dbg.get("mu_values", [0.25, 0.5, 1.0, 2.0, 4.0])]
    thetas = [float(v) for v in dbg.get("theta_values", [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])]
    out = []
    for mu in mus:
        for theta in thetas:
            slug = f"branch_{branch}_mu_{mu_tag(mu)}_theta_{phase_tag(theta)}"
            rec = dict(summary_records.get(slug, {}))
            out.append({"branch": branch, "mu": mu, "theta": theta, "slug": slug, "summary_record": rec})
    return out


def compute_candidate_debug(
    cand: Dict[str, Any],
    cfg: Dict[str, Any],
    p: Dict[str, Any],
    output_root: Path,
    summary_output_root: Path,
) -> Dict[str, Any]:
    q = float(cfg.get("q", cfg.get("frac_order", 0.9998)))
    branch = int(cand["branch"])
    mu = float(cand["mu"])
    theta = float(cand["theta"])
    slug = str(cand["slug"])
    matrices: List[Dict[str, Any]] = []
    spectral: List[Dict[str, Any]] = []
    has_nan_anywhere = False
    has_inf_anywhere = False
    failed_stage = ""
    failed_function = ""
    failed_matrix_name = ""
    recommended_fix = ""
    diagnosis_label = "ok_no_failure"

    raw_pairs = chua.find_omega_k_candidates(q, p, wmin=chua.WMIN, wmax=chua.WMAX, nscan=chua.NSCAN)
    omega, k = [float(v) for v in raw_pairs[branch][:2]]
    lambda_pow = (1j * omega) ** q
    lambda_explicit = omega ** q * np.exp(1j * q * np.pi / 2.0)
    lambda_difference_abs = float(abs(lambda_pow - lambda_explicit))
    convention = detect_transfer_convention(p, q, omega)
    W = convention["W_repo"]
    P, b, r = chua.chua_matrices(p)
    lam = lambda_pow

    def add_matrix(name: str, M: Any) -> Dict[str, Any]:
        safe = safe_eig(M, name)
        matrices.append(safe)
        spectral.append({k: safe.get(k) for k in [
            "matrix_name", "eig_solver_used", "eig_success", "eig_error_message",
            "eig_values_if_success", "schur_success", "schur_values_if_success",
            "scipy_eig_success_if_available", "numpy_eig_success_if_available",
        ]})
        return safe

    transfer_matrix = P.astype(np.complex128) - lam * np.eye(3, dtype=np.complex128)
    report_matrix = lam * np.eye(3, dtype=np.complex128) - P.astype(np.complex128)
    add_matrix("transfer_P_minus_lambda_I", transfer_matrix)
    add_matrix("transfer_lambda_I_minus_P", report_matrix)
    N = complex(np.nan, np.nan)
    logN = complex(np.nan, np.nan)
    N_mu = complex(np.nan, np.nan)
    A = float("nan")
    seed = np.array([np.nan, np.nan, np.nan], dtype=float)
    residual = complex(np.nan, np.nan)

    try:
        A = float(chua.solve_machado_amplitude_from_k(k, p, mu))
        N = complex(chua.N_sat(A, p), 0.0)
        if abs(N) < 1e-10:
            diagnosis_label = "invalid_near_zero_N"
            failed_stage = "machado_power"
            failed_function = "machado_complex_power"
            recommended_fix = "regularize_or_skip_near_zero_N_before_fractional_power"
        else:
            logN = np.log(abs(N)) + 1j * np.angle(N)
            N_mu = complex(chua.machado_complex_power(N, mu, branch=int(cfg.get("machado", {}).get("branch", 0))))
            residual = 1.0 + W * N_mu
        P0 = chua.build_P0(p, k)
        add_matrix("P0_linearized_for_seed", P0)
        add_matrix("P0_minus_lambda_I", P0.astype(np.complex128) - lam * np.eye(3, dtype=np.complex128))
        seed, _v, _eig_match = chua.build_fractional_seed(q, p, omega, k, A, theta=theta)
    except Exception as exc:
        if not failed_stage:
            failed_stage = "amplitude_or_seed_reconstruction"
            failed_function = "solve_machado_amplitude_from_k/build_fractional_seed"
        seed_error = str(exc)
        if "ganancia" in seed_error.lower() and "compatible" in seed_error.lower():
            diagnosis_label = "invalid_candidate_range"
            failed_function = "solve_machado_amplitude_from_k"
            recommended_fix = "skip_mu_for_branch_1_when_gain_condition_0_lt_k_lt_Delta_mu_fails"
        if diagnosis_label == "ok_no_failure":
            diagnosis_label = "unknown_failure"
        recommended_fix = recommended_fix or "inspect_gain_admissibility_and_seed_reconstruction_for_branch_1"
    else:
        seed_error = ""

    cov = covariance_diagnostic(trajectory_csv_for_candidate(cand["summary_record"], summary_output_root, slug))
    if cov.get("available"):
        cov_safe = cov["safe_eig"]
        matrices.append(cov_safe)
        spectral.append({k: cov_safe.get(k) for k in [
            "matrix_name", "eig_solver_used", "eig_success", "eig_error_message",
            "eig_values_if_success", "schur_success", "schur_values_if_success",
            "scipy_eig_success_if_available", "numpy_eig_success_if_available",
        ]})
        if cov_safe.get("has_nan") or cov.get("trajectory_has_nan"):
            has_nan_anywhere = True
        if cov_safe.get("has_inf") or cov.get("trajectory_has_inf"):
            has_inf_anywhere = True
        if not cov_safe.get("valid_for_eigs", False):
            failed_stage = "trajectory_shape_diagnostics"
            failed_function = "np.linalg.eigvalsh(covariance)"
            failed_matrix_name = "trajectory_covariance"
            if cov_safe.get("has_inf") or cov_safe.get("has_nan"):
                diagnosis_label = "invalid_nan_or_inf"
                recommended_fix = "classify_divergent_or_rescale_before_covariance_eig; do_not_call_eig_on_inf_covariance"
            else:
                diagnosis_label = "eig_solver_failure_finite_matrix"
                recommended_fix = "use_safe_eig_diagnostics_or_svd_for_covariance_shape"

    for safe in matrices:
        has_nan_anywhere = has_nan_anywhere or bool(safe.get("has_nan"))
        has_inf_anywhere = has_inf_anywhere or bool(safe.get("has_inf"))
    condition_values = [float(m.get("condition_number", np.nan)) for m in matrices]
    finite_conditions = [v for v in condition_values if np.isfinite(v)]
    condition_number_max = float(max(finite_conditions)) if finite_conditions else float("nan")
    if lambda_difference_abs > 1e-10:
        diagnosis_label = "inconsistent_fractional_lambda"
        failed_stage = failed_stage or "fractional_lambda"
        recommended_fix = "use_explicit_lambda_omega_q_exp_iqpi2_consistently"
    if not failed_stage and any((not m.get("eig_success", False)) and m.get("valid_for_eigs", False) for m in matrices):
        diagnosis_label = "eig_solver_failure_finite_matrix"
        failed_stage = "spectral_diagnostic"
        recommended_fix = "try_scipy_schur_or_svd_based_metric"
    if not failed_stage:
        failed_stage = "none"
        failed_function = "none"
        recommended_fix = "no_fix_needed_for_diagnostic_subset"
    if has_nan_anywhere and diagnosis_label == "ok_no_failure":
        diagnosis_label = "invalid_nan_or_inf"
    if has_inf_anywhere and diagnosis_label == "ok_no_failure":
        diagnosis_label = "invalid_nan_or_inf"

    jacobian_rows = []
    for eq_id, eq in solve_equilibria(p).items():
        J = local_jacobian(p, eq)
        jsafe = safe_eig(J, f"jacobian_{eq_id}")
        eigvals = jsafe.get("eig_values_if_success", "")
        margins = []
        if jsafe.get("eig_success") and eigvals:
            # Recompute from numpy when safe; this is diagnostic only.
            vals = np.linalg.eigvals(J)
            margins = [float(abs(np.angle(v)) - q * np.pi / 2.0) for v in vals]
        jacobian_rows.append({
            "equilibrium_id": eq_id,
            "jacobian_matrix": matrix_entries(J),
            "jacobian_has_nan": bool(jsafe.get("has_nan")),
            "jacobian_has_inf": bool(jsafe.get("has_inf")),
            "jacobian_condition_number": jsafe.get("condition_number"),
            "jacobian_eigenvalues": eigvals,
            "matignon_margin": min(margins) if margins else "",
            "matignon_status": bool(margins and min(margins) > 0.0),
        })

    eig_success = bool(
        matrices
        and all((not m.get("valid_for_eigs", False)) or m.get("eig_success", False) for m in matrices)
        and not any((not m.get("valid_for_eigs", False)) and m.get("validation_error") for m in matrices)
    )
    eig_error_message = " | ".join(str(m.get("eig_error_message", "")) for m in matrices if m.get("eig_error_message"))
    solver_used = ";".join(str(m.get("eig_solver_used", "")) for m in matrices if m.get("eig_solver_used"))
    schur_success = bool(any(m.get("schur_success", False) for m in matrices))
    scipy_available = bool(any(m.get("scipy_available", False) for m in matrices))

    row = {
        "candidate_id": slug,
        "branch": branch,
        "mu": mu,
        "theta": theta,
        "q": q,
        "A": A,
        "omega": omega,
        "phase": theta,
        "sigma0": 0.0,
        "seed_x": float(seed[0]),
        "seed_y": float(seed[1]),
        "seed_z": float(seed[2]),
        "N_re": float(np.real(N)),
        "N_im": float(np.imag(N)),
        "abs_N": float(abs(N)),
        "arg_N": float(np.angle(N)) if np.isfinite(abs(N)) else float("nan"),
        "logN_re": float(np.real(logN)),
        "logN_im": float(np.imag(logN)),
        "log_branch_used": int(cfg.get("machado", {}).get("branch", 0)),
        "N_mu_re": float(np.real(N_mu)),
        "N_mu_im": float(np.imag(N_mu)),
        "abs_N_mu": float(abs(N_mu)),
        "arg_N_mu": float(np.angle(N_mu)) if np.isfinite(abs(N_mu)) else float("nan"),
        "lambda_re": float(np.real(lambda_pow)),
        "lambda_im": float(np.imag(lambda_pow)),
        "lambda_explicit_re": float(np.real(lambda_explicit)),
        "lambda_explicit_im": float(np.imag(lambda_explicit)),
        "lambda_difference_abs": lambda_difference_abs,
        "W_re": float(np.real(W)),
        "W_im": float(np.imag(W)),
        "abs_W": float(abs(W)),
        "arg_W": float(np.angle(W)),
        "transfer_sign_convention": convention["transfer_sign_convention"],
        "residual_re": float(np.real(residual)),
        "residual_im": float(np.imag(residual)),
        "residual_abs": float(abs(residual)),
        "residual_formula_used": "1 + W_code(lambda) * N_mu",
        "has_nan_anywhere": has_nan_anywhere,
        "has_inf_anywhere": has_inf_anywhere,
        "failed_stage": failed_stage,
        "failed_function": failed_function or ("none" if failed_stage == "none" else ""),
        "failed_matrix_name": failed_matrix_name,
        "condition_number_max": condition_number_max,
        "eig_solver_used": solver_used,
        "eig_success": eig_success,
        "eig_error_message": eig_error_message or seed_error,
        "scipy_available": scipy_available,
        "schur_success": schur_success,
        "diagnosis_label": diagnosis_label,
        "recommended_fix": recommended_fix,
    }
    raw = {
        "candidate": row,
        "summary_record": cand["summary_record"],
        "raw_pairs": [{"branch": i, "omega": float(v[0]), "k": float(v[1])} for i, v in enumerate(raw_pairs)],
        "lambda": {"pow": cpair(lambda_pow), "explicit": cpair(lambda_explicit), "difference_abs": lambda_difference_abs},
        "transfer": convention,
        "matrices": matrices,
        "spectral": spectral,
        "trajectory_covariance": {k: v for k, v in cov.items() if k not in {"matrix", "safe_eig"}},
        "jacobians": jacobian_rows,
    }
    return {"row": row, "raw": raw, "matrices": matrices}


def write_debug_report(rows: List[Dict[str, Any]], outdir: Path, convention: str) -> None:
    def count(pred):
        return sum(1 for r in rows if pred(r))

    def finite_or(row: Dict[str, Any], key: str, default: float) -> float:
        try:
            value = float(row.get(key, default))
        except Exception:
            return default
        return value if np.isfinite(value) else default

    sorted_problem = sorted(
        rows,
        key=lambda r: (
            not bool(r.get("has_nan_anywhere")),
            not bool(r.get("has_inf_anywhere")),
            -finite_or(r, "condition_number_max", 0.0),
            finite_or(r, "abs_N", np.inf),
            -finite_or(r, "residual_abs", 0.0),
        ),
    )[:20]
    labels = {}
    for r in rows:
        labels[str(r["diagnosis_label"])] = labels.get(str(r["diagnosis_label"]), 0) + 1
    lines = [
        "# Corrida 0: diagnostico de fallos en rama 1 Machado",
        "",
        f"Convencion detectada: `{convention}`.",
        "La evaluacion fraccionaria usa `lambda=(j omega)^q=omega^q exp(j q pi/2)`.",
        "",
        "## Resumen numerico",
        "",
        f"- candidatos revisados: `{len(rows)}`",
        f"- con NaN: `{count(lambda r: bool(r.get('has_nan_anywhere')))}`",
        f"- con Inf: `{count(lambda r: bool(r.get('has_inf_anywhere')))}`",
        f"- con abs(N) < 1e-10: `{count(lambda r: float(r.get('abs_N', np.inf)) < 1e-10)}`",
        f"- matrices mal condicionadas cond>1e12: `{count(lambda r: np.isfinite(float(r.get('condition_number_max', np.nan))) and float(r.get('condition_number_max')) > 1e12)}`",
        f"- eig falla aunque matriz finita: `{count(lambda r: r.get('diagnosis_label') == 'eig_solver_failure_finite_matrix')}`",
        f"- scipy/schur logra eigenvalores: `{count(lambda r: bool(r.get('schur_success')))}`",
        "",
        "Etiquetas:",
    ]
    for label, n in sorted(labels.items()):
        lines.append(f"- `{label}`: `{n}`")
    lines.extend([
        "",
        "## 20 candidatos mas problematicos",
        "",
        "| candidate_id | mu | theta | failed_stage | label | cond_max | abs_N | residual | fix |",
        "|---|---:|---:|---|---|---:|---:|---:|---|",
    ])
    for r in sorted_problem:
        lines.append(
            f"| `{r['candidate_id']}` | {r['mu']} | {r['theta']:.6g} | {r['failed_stage']} | "
            f"{r['diagnosis_label']} | {r['condition_number_max']:.6g} | {r['abs_N']:.6g} | "
            f"{r['residual_abs']:.6g} | {r['recommended_fix']} |"
        )
    lines.extend([
        "",
        "## Diagnostico textual",
        "",
        "La causa mas probable del mensaje `Eigenvalues did not converge` no es la transferencia fraccionaria ni la potencia Machado. En los candidatos con trayectoria disponible, la integracion produce estados finitos pero de magnitud enorme; al calcular la covarianza para `trajectory_shape_diagnostics`, `np.cov` desborda a `Inf` y luego `np.linalg.eigvalsh(cov)` falla.",
        "",
        "La rama 1 no debe declararse inutil solo por este error algebraico, pero si debe tratarse como dinamicamente divergente o no acotada antes de cualquier diagnostico de forma basado en covarianza. La reparacion minima posterior conviene hacerla en `trajectory_shape_diagnostics`: validar finitud y escala de la covarianza antes de llamar a `eigvalsh`, y clasificar la trayectoria como divergente/no confiable si excede el umbral.",
    ])
    (outdir / "debug_branch1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_debug_branch1_failures(config_path: str | Path = "configs/chua_fractional_nonsmooth.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    outdir = Path(cfg.get("debug_branch1_failures", {}).get("output_dir", "outputs/extended_search"))
    outdir.mkdir(parents=True, exist_ok=True)
    matrices_dir = outdir / "debug_branch1_matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(cfg.get("debug_branch1_failures", {}).get(
        "source_summary",
        "runs_machado_sweep_fast/chua_piecewise/machado_sweep/machado_sweep_summary.json",
    ))
    summary_records = load_summary_records(summary_path)
    summary_output_root = summary_path.parent if summary_path.exists() else Path("runs_machado_sweep_fast/chua_piecewise/machado_sweep")
    p = chua_ic_params(cfg)
    chua.PARAMS = p
    chua.QORD = np.float64(float(cfg.get("q", 0.9998)))
    rows: List[Dict[str, Any]] = []
    raw_path = outdir / "debug_branch1_candidates_raw.jsonl"
    matrices_path = matrices_dir / "matrices.jsonl"
    if raw_path.exists():
        raw_path.unlink()
    if matrices_path.exists():
        matrices_path.unlink()
    first_convention = ""
    for cand in candidate_grid(cfg, summary_records):
        result = compute_candidate_debug(cand, cfg, p, outdir, summary_output_root)
        rows.append(result["row"])
        first_convention = first_convention or str(result["row"].get("transfer_sign_convention", ""))
        with raw_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(result["raw"]), ensure_ascii=False) + "\n")
        with matrices_path.open("a", encoding="utf-8") as f:
            for m in result["matrices"]:
                f.write(json.dumps(json_safe({"candidate_id": result["row"]["candidate_id"], **m}), ensure_ascii=False) + "\n")

    write_csv(outdir / "debug_branch1_failures.csv", rows, CSV_FIELDS)
    summary = {
        "config_path": str(config_path),
        "source_summary": str(summary_path),
        "candidates_reviewed": len(rows),
        "transfer_sign_convention": first_convention,
        "counts_by_label": {},
        "counts_by_failed_stage": {},
        "outputs": {
            "csv": str(outdir / "debug_branch1_failures.csv"),
            "json": str(outdir / "debug_branch1_summary.json"),
            "report": str(outdir / "debug_branch1_report.md"),
            "raw_jsonl": str(raw_path),
            "matrices_jsonl": str(matrices_path),
        },
    }
    for key in ["diagnosis_label", "failed_stage"]:
        target = summary["counts_by_label"] if key == "diagnosis_label" else summary["counts_by_failed_stage"]
        for r in rows:
            label = str(r.get(key, ""))
            target[label] = target.get(label, 0) + 1
    (outdir / "debug_branch1_summary.json").write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    write_debug_report(rows, outdir, first_convention)

    print("candidate_id,mu,theta,failed_stage,diagnosis_label,condition_number_max,abs_N,eig_success,recommended_fix", flush=True)
    for r in rows:
        print(
            f"{r['candidate_id']},{r['mu']},{r['theta']},{r['failed_stage']},{r['diagnosis_label']},"
            f"{r['condition_number_max']},{r['abs_N']},{r['eig_success']},{r['recommended_fix']}",
            flush=True,
        )
    return summary
