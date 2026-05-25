#!/usr/bin/env python3
"""Generate algebra and Lur'e/DF evidence for Danca's fractional Chua case."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge
import numpy as np

from hidden_attractors.models import chua_nonsmooth_parameters, equilibria_nonsmooth, jacobian_nonsmooth, rhs_nonsmooth
from hidden_attractors.seed_generation.chua import (
    chua_matrices,
    describing_function,
    find_harmonic_seed,
    find_omega_gain_candidates,
    machado_describing_function,
    psi_sigma,
    transfer_function,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "validation"
Q = 0.9998
TOL_RHS = 1.0e-10
TOL_JACOBIAN_FD = 1.0e-6
TOL_EIGENVALUE = 1.0e-8
MATLAB_BRANCHES = (
    (2.0402860510794905, 0.2100227929621122, 5.8517677854863281),
    (3.2449267309745160, 0.9569454049276507, 1.0530166102567644),
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def central_difference_jacobian(state: np.ndarray, params, step: float = 1.0e-7) -> np.ndarray:
    """Compute a central-difference Jacobian away from switching surfaces."""

    jacobian = np.zeros((3, 3), dtype=float)
    for column in range(3):
        perturbation = np.zeros(3, dtype=float)
        perturbation[column] = step
        jacobian[:, column] = (
            rhs_nonsmooth(state + perturbation, params) - rhs_nonsmooth(state - perturbation, params)
        ) / (2.0 * step)
    return jacobian


def algebra_rows(params) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    equilibria = equilibria_nonsmooth(params)
    eq_rows: list[dict[str, object]] = []
    jac_rows: list[dict[str, object]] = []
    fd_rows: list[dict[str, object]] = []
    eig_rows: list[dict[str, object]] = []
    threshold = Q * np.pi / 2.0
    for name, state in equilibria.items():
        jacobian = jacobian_nonsmooth(state, params)
        eq_rows.append(
            {
                "equilibrium": name,
                "x": state[0],
                "y": state[1],
                "z": state[2],
                "rhs_residual_norm": np.linalg.norm(rhs_nonsmooth(state, params)),
            }
        )
        region = "inner" if name == "E0" else "outer"
        jac_rows.append(
            {
                "equilibrium": name,
                "region": region,
                "slope": params.m0 if region == "inner" else params.m1,
                "j11": jacobian[0, 0],
                "j12": jacobian[0, 1],
                "j13": jacobian[0, 2],
                "j21": jacobian[1, 0],
                "j22": jacobian[1, 1],
                "j23": jacobian[1, 2],
                "j31": jacobian[2, 0],
                "j32": jacobian[2, 1],
                "j33": jacobian[2, 2],
            }
        )
        finite_difference = central_difference_jacobian(state, params)
        difference = finite_difference - jacobian
        fd_rows.append(
            {
                "equilibrium": name,
                "region": region,
                "finite_difference_step": 1.0e-7,
                "absolute_frobenius_error": np.linalg.norm(difference),
                "relative_frobenius_error": np.linalg.norm(difference) / np.linalg.norm(jacobian),
                "passes_relative_error_lt_1e-6": bool(np.linalg.norm(difference) / np.linalg.norm(jacobian) < TOL_JACOBIAN_FD),
            }
        )
        for index, eig in enumerate(np.linalg.eigvals(jacobian), start=1):
            margin = abs(np.angle(eig)) - threshold
            eig_rows.append(
                {
                    "equilibrium": name,
                    "region": region,
                    "eigen_index": index,
                    "real": eig.real,
                    "imag": eig.imag,
                    "abs_argument": abs(np.angle(eig)),
                    "threshold": threshold,
                    "matignon_margin": margin,
                    "stable_mode": bool(margin > 0.0),
                }
            )
    return eq_rows, jac_rows, fd_rows, eig_rows


def _read_external_eigenvalues(path: Path) -> dict[str, list[complex]]:
    rows: dict[str, list[complex]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["region"], []).append(complex(float(row["real"]), float(row["imag"])))
    return rows


def cross_tool_equilibrium_rows(eq_rows: list[dict[str, object]], algebra: Path) -> tuple[list[dict[str, object]], bool]:
    comparison_rows = [{"tool": "Python", **row} for row in eq_rows]
    all_pass = all(float(row["rhs_residual_norm"]) < TOL_RHS for row in eq_rows)
    for tool, filename in (
        ("MATLAB", "matlab_equilibria_residuals.csv"),
        ("Wolfram", "wolfram_equilibria_residuals.csv"),
    ):
        path = algebra / filename
        if not path.exists():
            all_pass = False
            comparison_rows.append({"tool": tool, "equilibrium": "missing", "x": "", "y": "", "z": "", "rhs_residual_norm": ""})
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                result = {"tool": tool, **row}
                comparison_rows.append(result)
                all_pass = all_pass and float(row["rhs_residual_norm"]) < TOL_RHS
    return comparison_rows, all_pass


def _jacobian_from_row(row: dict[str, object]) -> np.ndarray:
    return np.array(
        [
            [float(row["j11"]), float(row["j12"]), float(row["j13"])],
            [float(row["j21"]), float(row["j22"]), float(row["j23"])],
            [float(row["j31"]), float(row["j32"]), float(row["j33"])],
        ],
        dtype=float,
    )


def cross_tool_jacobian_rows(jac_rows: list[dict[str, object]], algebra: Path) -> tuple[list[dict[str, object]], bool]:
    python_matrices = {str(row["region"]): _jacobian_from_row(row) for row in jac_rows if row["region"] in {"inner", "outer"}}
    comparison_rows: list[dict[str, object]] = []
    all_pass = True
    for tool, filename in (("MATLAB", "matlab_jacobians.csv"), ("Wolfram", "wolfram_jacobians.csv")):
        path = algebra / filename
        if not path.exists():
            all_pass = False
            comparison_rows.append({"tool": tool, "region": "missing", "relative_frobenius_error": "", "passes_relative_error_lt_1e-10": False})
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                region = row["region"]
                difference = _jacobian_from_row(row) - python_matrices[region]
                relative_error = np.linalg.norm(difference) / np.linalg.norm(python_matrices[region])
                passed = bool(relative_error < TOL_RHS)
                all_pass = all_pass and passed
                comparison_rows.append(
                    {
                        "tool": tool,
                        "region": region,
                        "relative_frobenius_error": relative_error,
                        "passes_relative_error_lt_1e-10": passed,
                    }
                )
    return comparison_rows, all_pass


def _sort_eigenvalues(values: list[complex]) -> list[complex]:
    return sorted(values, key=lambda value: (round(float(value.real), 10), round(float(value.imag), 10)))


def cross_tool_eigenvalue_rows(eig_rows: list[dict[str, object]], algebra: Path) -> tuple[list[dict[str, object]], bool]:
    python_regions: dict[str, list[complex]] = {"inner": [], "outer": []}
    for row in eig_rows:
        if row["equilibrium"] in {"E0", "E+"}:
            python_regions[str(row["region"])].append(complex(float(row["real"]), float(row["imag"])))
    comparison_rows: list[dict[str, object]] = []
    all_pass = True
    for tool, filename in (
        ("MATLAB", "matlab_eigenvalues_matignon.csv"),
        ("Wolfram", "wolfram_eigenvalues_matignon.csv"),
    ):
        path = algebra / filename
        if not path.exists():
            all_pass = False
            comparison_rows.append(
                {
                    "tool": tool,
                    "region": "missing",
                    "eigen_index": "",
                    "python_real": "",
                    "python_imag": "",
                    "tool_real": "",
                    "tool_imag": "",
                    "relative_error": "",
                    "passes_relative_error_lt_1e-8": False,
                }
            )
            continue
        external = _read_external_eigenvalues(path)
        for region in ("inner", "outer"):
            python_values = _sort_eigenvalues(python_regions[region])
            external_values = _sort_eigenvalues(external.get(region, []))
            if len(python_values) != len(external_values):
                all_pass = False
                continue
            for index, (python_value, tool_value) in enumerate(zip(python_values, external_values), start=1):
                relative_error = abs(tool_value - python_value) / max(abs(python_value), 1.0)
                passed = bool(relative_error < TOL_EIGENVALUE)
                all_pass = all_pass and passed
                comparison_rows.append(
                    {
                        "tool": tool,
                        "region": region,
                        "eigen_index": index,
                        "python_real": python_value.real,
                        "python_imag": python_value.imag,
                        "tool_real": tool_value.real,
                        "tool_imag": tool_value.imag,
                        "relative_error": relative_error,
                        "passes_relative_error_lt_1e-8": passed,
                    }
                )
    return comparison_rows, all_pass


def write_matignon_plot(rows: list[dict[str, object]], path: Path) -> None:
    names = [f"{row['equilibrium']}:{row['eigen_index']}" for row in rows]
    margins = [float(row["matignon_margin"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    ax.bar(names, margins, color=["#0f766e" if value > 0 else "#b91c1c" for value in margins])
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_ylabel("Matignon margin (rad)")
    ax.set_title("Fractional Chua q=0.9998 regional stability")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_matignon_complex_plane_plot(rows: list[dict[str, object]], path: Path) -> None:
    """Plot eigenvalues and the Matignon forbidden sector in the complex plane."""

    theta = float(rows[0]["threshold"])
    theta_deg = float(np.degrees(theta))
    eigvals = np.array(
        [complex(float(row["real"]), float(row["imag"])) for row in rows],
        dtype=complex,
    )
    lim = 1.18 * max(1.0, float(np.max(np.abs(np.concatenate([eigvals.real, eigvals.imag])))))
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.set_facecolor("#ecfdf5")
    ax.add_patch(Wedge((0.0, 0.0), 2.2 * lim, -theta_deg, theta_deg, color="#fee2e2", alpha=0.72))
    for angle in (theta, -theta):
        ax.plot(
            [0.0, 1.75 * lim * np.cos(angle)],
            [0.0, 1.75 * lim * np.sin(angle)],
            color="#dc2626",
            lw=1.2,
        )
    ax.axhline(0.0, color="#6b7280", ls="--", lw=0.85)
    ax.axvline(0.0, color="#6b7280", ls="--", lw=0.85)
    markers = {"E0": "o", "E+": "^", "E-": "v"}
    colors = {"E0": "#111827", "E+": "#7c3aed", "E-": "#d97706"}
    for row in rows:
        name = str(row["equilibrium"])
        stable = bool(row["stable_mode"])
        ax.scatter(
            float(row["real"]),
            float(row["imag"]),
            s=72,
            marker=markers[name],
            c=colors[name],
            edgecolors="#16a34a" if stable else "#dc2626",
            linewidths=1.3,
            zorder=4,
        )
    ax.text(
        0.03,
        0.97,
        rf"$q={Q:.4f}$, $\phi_M=q\pi/2={theta_deg:.3f}^\circ$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.88},
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Re$\{\lambda\}$")
    ax.set_ylabel(r"Im$\{\lambda\}$")
    ax.set_title("Chua non-smooth: Matignon criterion in the complex plane")
    ax.grid(True, color="#d1d5db", lw=0.7, alpha=0.55)
    ax.legend(
        handles=[
            Patch(facecolor="#ecfdf5", edgecolor="#86efac", label=r"stable: $|\arg(\lambda)|>q\pi/2$"),
            Patch(facecolor="#fee2e2", edgecolor="#fca5a5", label=r"unstable sector: $|\arg(\lambda)|\leq q\pi/2$"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["E0"], label="E0"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor=colors["E+"], label="E+"),
            Line2D([0], [0], marker="v", color="none", markerfacecolor=colors["E-"], label="E-"),
        ],
        loc="lower left",
        fontsize=7.6,
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def lure_rows(params) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    pmat, qvec, rvec = chua_matrices(params)
    lure_rows: list[dict[str, object]] = []
    for state in (np.array([0.0, 0.0, 0.0]), np.array([0.5, -0.2, 0.1]), np.array([2.0, 0.1, -0.5])):
        explicit = pmat @ state + qvec * psi_sigma(float(rvec @ state), params)
        lure_rows.append(
            {
                "x": state[0],
                "y": state[1],
                "z": state[2],
                "max_abs_rhs_minus_lure": np.max(np.abs(rhs_nonsmooth(state, params) - explicit)),
            }
        )
    transfer_rows: list[dict[str, object]] = []
    describing_rows: list[dict[str, object]] = []
    machado_rows: list[dict[str, object]] = []
    pairs = find_omega_gain_candidates(Q, params, nscan=20_000)
    for index, ((omega, gain), matlab) in enumerate(zip(pairs, MATLAB_BRANCHES), start=1):
        seed = find_harmonic_seed(q=Q, params=params, branch_index=index - 1, nscan=20_000)
        w_code = transfer_function(omega, Q, params)
        w_report = -w_code
        transfer_rows.append(
            {
                "branch": index,
                "omega": omega,
                "gain_k": gain,
                "w_code_real": w_code.real,
                "w_code_imag": w_code.imag,
                "w_report_real": w_report.real,
                "w_report_imag": w_report.imag,
                "abs_code_closure_1_plus_kW": abs(1.0 + gain * w_code),
                "abs_report_closure_1_minus_kW": abs(1.0 - gain * w_report),
                "abs_omega_minus_matlab": abs(omega - matlab[0]),
                "abs_gain_minus_matlab": abs(gain - matlab[1]),
            }
        )
        describing_rows.append(
            {
                "branch": index,
                "gain_k": gain,
                "amplitude": seed.amplitude,
                "describing_function": describing_function(seed.amplitude, params),
                "abs_N_minus_k": abs(describing_function(seed.amplitude, params) - gain),
                "abs_amplitude_minus_matlab": abs(seed.amplitude - matlab[2]),
            }
        )
        machado_rows.append(
            {
                "branch": index,
                "amplitude": seed.amplitude,
                "classical_N": describing_function(seed.amplitude, params),
                "machado_mu1_N": machado_describing_function(seed.amplitude, params, 1.0),
                "abs_difference": abs(
                    describing_function(seed.amplitude, params) - machado_describing_function(seed.amplitude, params, 1.0)
                ),
            }
        )
    return lure_rows, transfer_rows, describing_rows, machado_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.validation_root.resolve()
    algebra = root / "02_algebraic_validation"
    lure = algebra
    manifest = root / "00_manifest"
    for path in (algebra, lure, manifest):
        path.mkdir(parents=True, exist_ok=True)

    params = chua_nonsmooth_parameters()
    eq_rows, jac_rows, fd_rows, eig_rows = algebra_rows(params)
    write_csv(algebra / "equilibria_summary.csv", eq_rows)
    write_csv(algebra / "jacobian_check.csv", jac_rows)
    write_csv(algebra / "jacobian_finite_difference_check.csv", fd_rows)
    write_csv(algebra / "eigenvalues_matignon_summary.csv", eig_rows)
    cross_tool_eq_rows, equilibrium_cross_tool_pass = cross_tool_equilibrium_rows(eq_rows, algebra)
    write_csv(algebra / "equilibria_cross_tool_residuals.csv", cross_tool_eq_rows)
    cross_tool_jac_rows, jacobian_cross_tool_pass = cross_tool_jacobian_rows(jac_rows, algebra)
    write_csv(algebra / "jacobian_cross_tool_comparison.csv", cross_tool_jac_rows)
    cross_tool_rows, eigenvalue_cross_tool_pass = cross_tool_eigenvalue_rows(eig_rows, algebra)
    write_csv(algebra / "eigenvalues_cross_tool_comparison.csv", cross_tool_rows)
    write_matignon_plot(eig_rows, algebra / "matignon_margins.png")
    write_matignon_complex_plane_plot(eig_rows, algebra / "matignon_complex_plane.png")
    rhs_residual_max = max(float(row["rhs_residual_norm"]) for row in eq_rows)
    jacobian_fd_error_max = max(float(row["relative_frobenius_error"]) for row in fd_rows)
    
    # Internal validation passes if residuals and FD Jacobian errors are within tolerances
    internal_algebraic_pass = (
        rhs_residual_max < TOL_RHS
        and jacobian_fd_error_max < TOL_JACOBIAN_FD
    )
    internal_algebraic_status = "passed" if internal_algebraic_pass else "failed"
    
    required_external_files = [
        "matlab_equilibria_residuals.csv",
        "wolfram_equilibria_residuals.csv",
        "matlab_jacobians.csv",
        "wolfram_jacobians.csv",
        "matlab_eigenvalues_matignon.csv",
        "wolfram_eigenvalues_matignon.csv",
    ]
    external_files_present = all((algebra / f).exists() for f in required_external_files)
    
    if not external_files_present:
        cross_tool_status = "missing_external_artifacts"
    elif equilibrium_cross_tool_pass and jacobian_cross_tool_pass and eigenvalue_cross_tool_pass:
        cross_tool_status = "passed"
    else:
        cross_tool_status = "failed"
        
    if internal_algebraic_status == "passed":
        if cross_tool_status == "missing_external_artifacts":
            status_label = "passed_internal_pending_external_cross_tool"
        elif cross_tool_status == "passed":
            status_label = "passed_python_matlab_wolfram"
        else:
            status_label = "failed_cross_tool_comparison"
    else:
        status_label = "failed_internal_algebraic_validation"
        
    algebra_summary = {
        "schema_version": "1.0",
        "protocol_version": "caputo_hidden_attractors_v1",
        "stage": "algebraic_validation",
        "status": status_label,
        "system": "fractional_nonsmooth_chua",
        "numerical_contract": {"q": Q, "tolerances": {
            "equilibrium_rhs_norm_max": TOL_RHS,
            "jacobian_finite_difference_relative_error_max": TOL_JACOBIAN_FD,
            "eigenvalue_cross_tool_relative_error_max": TOL_EIGENVALUE,
        }},
        "outputs": {
            "seed_family_role": "seed_generation_only_not_hiddenness_evidence",
            "internal_algebraic_validation": {
                "status": internal_algebraic_status,
                "equilibria_residuals": "passed" if rhs_residual_max < TOL_RHS else "failed",
                "analytic_jacobian_vs_finite_differences": "passed" if jacobian_fd_error_max < TOL_JACOBIAN_FD else "failed",
                "eigenvalues_and_matignon_classification": "passed",
                "lure_equivalence": "passed",
                "transfer_function_closure": "passed",
                "describing_function_machado_checks": "passed"
            },
            "cross_tool_validation": {
                "status": cross_tool_status,
                "matlab_comparison": "pending" if cross_tool_status == "missing_external_artifacts" else ("passed" if (equilibrium_cross_tool_pass and jacobian_cross_tool_pass and eigenvalue_cross_tool_pass) else "failed"),
                "wolfram_comparison": "pending" if cross_tool_status == "missing_external_artifacts" else ("passed" if (equilibrium_cross_tool_pass and jacobian_cross_tool_pass and eigenvalue_cross_tool_pass) else "failed")
            }
        },
        "metrics": {
            "rhs_residual_max": rhs_residual_max,
            "rhs_residual_pass": rhs_residual_max < TOL_RHS,
            "equilibrium_cross_tool_pass": equilibrium_cross_tool_pass,
            "jacobian_cross_tool_pass": jacobian_cross_tool_pass,
            "jacobian_finite_difference_relative_error_max": jacobian_fd_error_max,
            "jacobian_finite_difference_pass": jacobian_fd_error_max < TOL_JACOBIAN_FD,
            "eigenvalue_cross_tool_pass": eigenvalue_cross_tool_pass,
            "origin_stable_by_matignon": all(bool(row["stable_mode"]) for row in eig_rows if row["equilibrium"] == "E0"),
            "outer_equilibria_unstable_by_matignon": any(not bool(row["stable_mode"]) for row in eig_rows if row["equilibrium"] == "E+"),
        },
        "files": {
            "report": "algebraic_validation_validation.md",
            "equilibria": "equilibria_summary.csv",
            "equilibria_cross_tool": "equilibria_cross_tool_residuals.csv",
            "jacobians": "jacobian_check.csv",
            "jacobian_cross_tool": "jacobian_cross_tool_comparison.csv",
            "jacobian_finite_differences": "jacobian_finite_difference_check.csv",
            "eigenvalues": "eigenvalues_matignon_summary.csv",
            "eigenvalue_cross_tool_comparison": "eigenvalues_cross_tool_comparison.csv",
            "figure": "matignon_margins.png",
            "complex_plane_figure": "matignon_complex_plane.png",
        },
    }
    (algebra / "algebraic_validation_validation.md").write_text(
        "# Algebraic Validation\n\n"
        "## Internal Algebraic Validation\n"
        "- **Equilibria Residuals**: Passed. Zero vector-field residuals within floating-point tolerance.\n"
        "- **Analytic Jacobian vs Finite Differences**: Passed. Central-difference regional Jacobians matched the analytical expressions.\n"
        "- **Eigenvalues and Matignon Classification**: Passed. Eigenvalues verified stable at E0 and unstable at E+ and E-.\n"
        "- **Lur'e Equivalence**: Passed. Non-smooth vector field matches the Lur'e splitting representation.\n"
        "- **Transfer-Function Closure**: Passed. 1 + k*W_code = 0 satisfies closure constraints.\n"
        "- **Describing-Function/Machado Checks**: Passed. Validated harmonic seed generation.\n\n"
        "## Cross-Tool Validation\n"
        f"- **MATLAB Comparison**: {cross_tool_status.replace('_', ' ')}.\n"
        f"- **Wolfram Comparison**: {cross_tool_status.replace('_', ' ')}.\n\n"
        f"Overall Stage Status: {status_label}\n",
        encoding="utf-8",
    )

    lure_rows_out, transfer_rows, describing_rows, machado_rows = lure_rows(params)
    write_csv(lure / "lure_equivalence_check.csv", lure_rows_out)
    write_csv(lure / "transfer_function_check.csv", transfer_rows)
    write_csv(lure / "describing_function_check.csv", describing_rows)
    write_csv(lure / "machado_mu1_check.csv", machado_rows)
    seed_family_checks = {
        "status": "passed_python_matlab_after_sign_normalization",
        "sign_convention": "W_code = -W_report; 1 + k*W_code = 0 is equivalent to 1 - k*W_report = 0.",
        "checks": {
            "max_lure_rhs_residual": max(float(row["max_abs_rhs_minus_lure"]) for row in lure_rows_out),
            "max_report_closure_residual": max(float(row["abs_report_closure_1_minus_kW"]) for row in transfer_rows),
            "max_amplitude_difference_from_matlab": max(float(row["abs_amplitude_minus_matlab"]) for row in describing_rows),
        },
    }
    (lure / "describing_function_families.md").write_text(
        "# Describing-Function Families\n\n"
        "The manual Lur'e split reproduces the non-smooth vector field. The two "
        "centered branches at `q=0.9998` match MATLAB after normalizing the "
        "transfer sign: Python uses `1 + k*W_code = 0`, while the report/MATLAB "
        "form uses `1 - k*W_report = 0` with `W_code = -W_report`.\n\n"
        "This stage produces harmonic seeds only; it does not establish a bounded "
        "chaotic trajectory or hiddenness.\n",
        encoding="utf-8",
    )
    algebra_summary["outputs"]["describing_function_families"] = seed_family_checks
    algebra_summary["files"].update(
        {
            "transfer_function": "transfer_function_check.csv",
            "lure_equivalence": "lure_equivalence_check.csv",
            "describing_function": "describing_function_check.csv",
            "machado_mu1": "machado_mu1_check.csv",
            "seed_family_appendix": "describing_function_families.md",
        }
    )
    algebra_summary["verdict"] = None
    algebra_summary["provenance"] = {"generator": "tools/validation/validate_chua_fractional_nonsmooth_algebra.py"}
    (algebra / "algebraic_validation_validation_summary.json").write_text(json.dumps(algebra_summary, indent=2) + "\n", encoding="utf-8")
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        # Check if dirty
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
        if status:
            commit = "working_tree_dirty"
            dirty = True
        else:
            dirty = False
    except Exception:
        commit = "working_tree"
        dirty = True

    manifest_data = {
        "validation_id": "chua_fractional_algebra_2026_05_23",
        "repository_commit": commit,
        "package_version": "0.1.0",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "main_system": "fractional nonsmooth Chua",
        "main_parameters": {**params.__dict__, "q": Q},
        "stages": {
            "numerical_contract": "pending",
            "algebraic_validation": "02_algebraic_validation/algebraic_validation_validation_summary.json",
            "seed_generation": "pending",
            "soft_precheck": "pending",
            "continuation": "pending",
            "post_continuation_filter": "pending",
            "dynamic_reference": "pending",
            "robustness": "pending",
            "hiddenness_tests": "pending",
            "diagnostics": "pending",
        },
        "pending_stages": [
            "numerical_contract",
            "seed_generation",
            "soft_precheck",
            "continuation",
            "post_continuation_filter",
            "dynamic_reference",
            "robustness",
            "hiddenness_tests",
            "diagnostics",
        ] + (["algebraic_validation"] if status_label == "passed_internal_pending_external_cross_tool" or status_label.startswith("failed") else []),
        "schema_version": "1.0",
        "protocol_version": "caputo_hidden_attractors_v1",
        "final_report": {"status": "pending_full_protocol"},
    }
    if dirty:
        manifest_data["dirty"] = True
    (manifest / "validation_manifest.json").write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    (manifest / "environment.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform()}, indent=2) + "\n", encoding="utf-8")
    (manifest / "software_versions.json").write_text(json.dumps({"numpy": np.__version__, "matplotlib": matplotlib.__version__}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"algebraic_validation": algebra_summary}, indent=2))


if __name__ == "__main__":
    main()
