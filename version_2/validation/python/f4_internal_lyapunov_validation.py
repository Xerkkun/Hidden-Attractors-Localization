#!/usr/bin/env python3
"""Assemble conservative F4 internal Lyapunov validation evidence.

F4 is an internal consistency layer. It records controlled numerical checks
and reuses existing published-reference artifacts without promoting any
fractional method or certifying chaos or hiddenness.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import yaml

from hidden_attractors.analysis.lyapunov import integer_qr_benettin_lyapunov_exponents
from hidden_attractors.analysis.lyapunov_cloned import compute_cloned_dynamics_spectrum
from hidden_attractors.analysis.lyapunov_fractional import fractional_variational_abm_qr
from hidden_attractors.models.chua import (
    chua_nonsmooth_parameters,
    jacobian_nonsmooth,
    rhs_nonsmooth,
)
from hidden_attractors.solvers import efork_q1_integrate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
F4_ROOT = (
    PROJECT_ROOT
    / "validation"
    / "chaos_validation"
    / "lyapunov_methods"
    / "F4_internal_validation"
)
STAGE_DIRS = {
    "F4_1": F4_ROOT / "F4_1_integer_linear",
    "F4_2": F4_ROOT / "F4_2_integer_chua_q1",
    "F4_3": F4_ROOT / "F4_3_fractional_published_dk2018",
    "F4_4": F4_ROOT / "F4_4_cloned_dynamics_fischer2020",
}
SUMMARY_DIR = F4_ROOT / "summaries"
GLOBAL_SUMMARY = F4_ROOT / "f4_internal_validation_summary.json"
DK2018_SUMMARY = (
    PROJECT_ROOT
    / "validation"
    / "chaos_validation"
    / "lyapunov_methods"
    / "fractional_variational_dk2018_block_restart_abm_gs_published"
    / "validation_summary.json"
)
FISCHER2020_SUMMARY = (
    PROJECT_ROOT
    / "validation"
    / "chaos_validation"
    / "lyapunov_methods"
    / "fractional_cloned_dynamics_abm_gs_published"
    / "validation_summary.json"
)
FISCHER2020_SENSITIVITY = (
    FISCHER2020_SUMMARY.parent / "discrepancy_diagnostics" / "sensitivity_summary.json"
)
FISCHER2020_CLASSIFICATION = (
    FISCHER2020_SUMMARY.parent
    / "discrepancy_diagnostics"
    / "fischer2020_row_classification.csv"
)
DK2018_BENCHMARKS = (
    PROJECT_ROOT
    / "validation"
    / "lyapunov_benchmarks"
    / "fractional_variational_abm_qr"
)
DK2018_CASE_FILES = (
    "published_dk2018_lorenz_q0985.yaml",
    "published_dk2018_rabinovich_fabrikant_q0999.yaml",
)

NO_FALSE_CERTIFICATION = {
    "chaos_certified_by_this_pipeline": False,
    "hiddenness_certified_by_this_pipeline": False,
    "fractional_lyapunov_validated_by_f4": False,
    "caputo_lyapunov_validated_by_f4": False,
    "published_decimal_reproduction_implies_method_validation": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), separators=(",", ":"))
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(row.get(key, "")) for key in fieldnames}
            for row in rows
        )


def _write_readme(path: Path, title: str, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([f"# {title}", "", *lines, ""]),
        encoding="utf-8",
    )


def _sign_pattern(values: np.ndarray, tolerance: float = 1.0e-8) -> list[str]:
    out = []
    for value in np.asarray(values, dtype=float):
        if abs(float(value)) <= tolerance:
            out.append("zero_like")
        elif value > 0.0:
            out.append("positive")
        else:
            out.append("negative")
    return out


def _linear_rhs(diagonal: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return lambda state: diagonal * np.asarray(state, dtype=float)


def _linear_jacobian(diagonal: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return lambda state: np.diag(diagonal)  # noqa: ARG005


def _integer_linear_row(
    case_id: str,
    diagonal: list[float],
    *,
    h: float,
    t_final: float,
    reorthonormalize_every: int,
) -> dict[str, Any]:
    expected = np.asarray(diagonal, dtype=float)
    result = integer_qr_benettin_lyapunov_exponents(
        _linear_rhs(expected),
        _linear_jacobian(expected),
        np.ones(expected.size),
        h=h,
        t_final=t_final,
        reorthonormalize_every=reorthonormalize_every,
    )
    error = np.abs(np.asarray(result.exponents) - expected)
    return {
        "case_id": case_id,
        "method_id": result.method_id,
        "dimension": expected.size,
        "h": h,
        "t_final": t_final,
        "reorthonormalize_every": reorthonormalize_every,
        "computed_exponents": result.exponents.tolist(),
        "expected_exponents": expected.tolist(),
        "absolute_error": error.tolist(),
        "max_absolute_error": float(np.max(error)),
        "sign_pattern": _sign_pattern(result.exponents),
        "status": result.status,
        "controlled_check_passed": bool(result.status == "ok" and np.max(error) < 0.03),
    }


def _cloned_linear_row(
    method: str,
    *,
    delta: float,
    h: float = 0.01,
    t_clone: float = 0.5,
    k_blocks: int = 8,
) -> dict[str, Any]:
    diagonal = np.asarray([0.1, -0.2, -0.5], dtype=float)
    result = compute_cloned_dynamics_spectrum(
        _linear_rhs(diagonal),
        np.ones(3),
        orders=[1.0],
        h=h,
        t_clone=t_clone,
        n_clones=3,
        k_blocks=k_blocks,
        delta=delta,
        method=method,
        integration_mode="integer_rk4_reference",
    )
    error = np.abs(np.asarray(result.exponents) - diagonal)
    return {
        "case_id": f"linear_3d_cloned_{method}_delta_{delta:g}",
        "method_id": result.method_id,
        "orthonormalization": method,
        "delta": delta,
        "h": h,
        "t_clone": t_clone,
        "k_blocks": k_blocks,
        "computed_exponents": result.exponents.tolist(),
        "expected_exponents": diagonal.tolist(),
        "absolute_error": error.tolist(),
        "max_absolute_error": float(np.max(error)),
        "sign_pattern": _sign_pattern(result.exponents),
        "status": result.status,
        "controlled_check_passed": bool(result.status == "ok" and np.max(error) < 0.05),
    }


def run_f4_1(*, fast: bool) -> dict[str, Any]:
    """Run exact diagonal controls for integer and q=1 cloned methods."""

    stage_dir = STAGE_DIRS["F4_1"]
    results = [
        _integer_linear_row(
            "linear_diagonal_3d",
            [0.1, -0.2, -0.5],
            h=0.01,
            t_final=40.0 if fast else 200.0,
            reorthonormalize_every=10,
        ),
        _integer_linear_row(
            "linear_diagonal_4d",
            [0.2, 0.0, -0.1, -0.7],
            h=0.01,
            t_final=40.0 if fast else 200.0,
            reorthonormalize_every=10,
        ),
        _cloned_linear_row("gs_modified", delta=1.0e-3),
        _cloned_linear_row("qr", delta=1.0e-3),
    ]
    sensitivity = [
        _integer_linear_row(
            f"linear_diagonal_3d_h_{h:g}",
            [0.1, -0.2, -0.5],
            h=h,
            t_final=30.0 if fast else 100.0,
            reorthonormalize_every=max(1, round(0.1 / h)),
        )
        for h in (0.02, 0.01, 0.005)
    ]
    sensitivity += [
        _cloned_linear_row("gs_modified", delta=delta)
        for delta in (1.0e-2, 1.0e-3, 1.0e-4)
    ]
    results_csv = stage_dir / "integer_linear_results.csv"
    sensitivity_csv = stage_dir / "integer_linear_sensitivity.csv"
    _write_csv(results_csv, results)
    _write_csv(sensitivity_csv, sensitivity)
    summary = {
        "schema_version": "1.0",
        "stage": "F4_1_integer_linear",
        "status": "controlled_linear_checks_passed"
        if all(row["controlled_check_passed"] for row in results)
        else "controlled_linear_checks_failed",
        "fast_smoke_only": fast,
        "methods": ["integer_qr_benettin", "fractional_cloned_dynamics_abm_gs_published", "fractional_cloned_dynamics_abm_qr"],
        "checks": {
            "exact_diagonal_controls": True,
            "integer_h_sensitivity": True,
            "cloned_delta_sensitivity": True,
            "all_controlled_rows_passed": all(row["controlled_check_passed"] for row in results),
        },
        "files": {
            "results_csv": _relative(results_csv),
            "sensitivity_csv": _relative(sensitivity_csv),
        },
        "certifications": NO_FALSE_CERTIFICATION,
    }
    _write_json(stage_dir / "summary.json", summary)
    _write_readme(
        stage_dir / "README.md",
        "F4.1 integer linear controls",
        [
            "This stage checks exact diagonal systems before using nonlinear examples.",
            "The q=1 cloned-dynamics rows are internal consistency controls only.",
            f"`fast_smoke_only={str(fast).lower()}`.",
            "These checks do not certify chaos, hiddenness, or Caputo method validity.",
        ],
    )
    return summary


CHUA_REFERENCE = {
    "omega0": 2.039186939959001,
    "k": 0.209867354515084,
    "a0": 5.856145086257356,
    "x0": [5.856145086257356, 0.369331578246782, -8.366536168331880],
}


def _integer_chua_row(*, h: float, t_final: float, t_burn: float) -> dict[str, Any]:
    params = chua_nonsmooth_parameters()
    x0 = np.asarray(CHUA_REFERENCE["x0"], dtype=float)
    rhs = lambda state: rhs_nonsmooth(state, params)
    jacobian = lambda state: jacobian_nonsmooth(state, params)
    result = integer_qr_benettin_lyapunov_exponents(
        rhs,
        jacobian,
        x0,
        h=h,
        t_final=t_final,
        t_burn=t_burn,
        reorthonormalize_every=max(1, round(0.1 / h)),
        div_threshold=1.0e6,
    )
    trajectory, trajectory_status = efork_q1_integrate(
        rhs,
        x0,
        h=h,
        t_final=t_final + t_burn,
        div_threshold=1.0e6,
    )
    states = trajectory[:, 1:]
    return {
        "case_id": f"integer_chua_q1_h_{h:g}",
        "method_id": result.method_id,
        "h": h,
        "t_final": t_final,
        "t_burn": t_burn,
        "computed_exponents": result.exponents.tolist(),
        "sign_pattern": _sign_pattern(result.exponents),
        "status": result.status,
        "trajectory_status": trajectory_status,
        "trajectory_bounded": bool(
            trajectory_status == "ok"
            and np.all(np.isfinite(states))
            and float(np.max(np.linalg.norm(states, axis=1))) < 1.0e6
        ),
        "max_state_norm": float(np.max(np.linalg.norm(states, axis=1))),
    }


def _cloned_chua_row(method: str, *, delta: float, h: float, k_blocks: int) -> dict[str, Any]:
    params = chua_nonsmooth_parameters()
    result = compute_cloned_dynamics_spectrum(
        lambda state: rhs_nonsmooth(state, params),
        np.asarray(CHUA_REFERENCE["x0"], dtype=float),
        orders=[1.0],
        h=h,
        t_clone=0.5,
        n_clones=3,
        k_blocks=k_blocks,
        delta=delta,
        method=method,
        integration_mode="integer_rk4_reference",
        divergence_norm=1.0e6,
        system_id="chua_nonsmooth_q1_reference",
    )
    return {
        "case_id": f"integer_chua_q1_cloned_{method}_delta_{delta:g}",
        "method_id": result.method_id,
        "orthonormalization": method,
        "delta": delta,
        "h": h,
        "t_clone": 0.5,
        "k_blocks": k_blocks,
        "computed_exponents": result.exponents.tolist(),
        "sign_pattern": _sign_pattern(result.exponents),
        "status": result.status,
        "trajectory_bounded": result.bounded_trajectory,
    }


def run_f4_2(*, fast: bool) -> dict[str, Any]:
    """Compare q=1 methods on the promoted integer Chua seed."""

    stage_dir = STAGE_DIRS["F4_2"]
    horizon = 25.0 if fast else 100.0
    burn = 5.0 if fast else 25.0
    results = [
        _integer_chua_row(h=0.01, t_final=horizon, t_burn=burn),
        _cloned_chua_row("gs_modified", delta=1.0e-3, h=0.01, k_blocks=12 if fast else 80),
        _cloned_chua_row("qr", delta=1.0e-3, h=0.01, k_blocks=12 if fast else 80),
    ]
    sensitivity = [
        _integer_chua_row(h=h, t_final=15.0 if fast else 60.0, t_burn=3.0 if fast else 15.0)
        for h in (0.02, 0.01)
    ]
    sensitivity += [
        _cloned_chua_row("gs_modified", delta=delta, h=0.01, k_blocks=8 if fast else 40)
        for delta in (1.0e-2, 1.0e-3)
    ]
    finite = all(np.all(np.isfinite(row["computed_exponents"])) for row in results)
    bounded = all(bool(row["trajectory_bounded"]) for row in results)
    sign_patterns = {tuple(row["sign_pattern"]) for row in results}
    results_csv = stage_dir / "integer_chua_q1_results.csv"
    sensitivity_csv = stage_dir / "integer_chua_q1_sensitivity.csv"
    _write_csv(results_csv, results)
    _write_csv(sensitivity_csv, sensitivity)
    summary = {
        "schema_version": "1.0",
        "stage": "F4_2_integer_chua_q1",
        "status": "finite_bounded_with_method_differences_documented"
        if finite and bounded
        else "controlled_chua_q1_inconclusive",
        "fast_smoke_only": fast,
        "published_spectrum_claimed": False,
        "reference_seed": CHUA_REFERENCE,
        "checks": {
            "all_spectra_finite": finite,
            "all_trajectories_bounded": bounded,
            "strict_sign_pattern_agreement": len(sign_patterns) == 1,
            "sign_patterns_observed": [list(pattern) for pattern in sorted(sign_patterns)],
            "interpretation": "Loose finite-time consistency diagnostic; no published integer Chua spectrum is asserted.",
        },
        "files": {
            "results_csv": _relative(results_csv),
            "sensitivity_csv": _relative(sensitivity_csv),
        },
        "certifications": NO_FALSE_CERTIFICATION,
    }
    _write_json(stage_dir / "summary.json", summary)
    _write_readme(
        stage_dir / "README.md",
        "F4.2 integer Chua q=1 consistency",
        [
            "This stage uses the promoted q=1 reference seed and compares finite-time indicators.",
            "No published Lyapunov spectrum is invented for the integer Chua reference.",
            "Method differences are recorded as diagnostics rather than smoothed into a pass.",
            "These outputs do not certify chaos or hiddenness.",
        ],
    )
    return summary


def _full_history_qr_synthetic_row(*, h: float) -> dict[str, Any]:
    result = fractional_variational_abm_qr(
        lambda state: -np.asarray(state, dtype=float),
        lambda state: np.asarray([[-1.0]]),  # noqa: ARG005
        np.ones(1),
        q=0.9,
        h=h,
        t_final=0.3,
        reorthonormalization_time=0.1,
        history_aware_qr=True,
    )
    return {
        "case_id": f"full_history_qr_synthetic_stable_h_{h:g}",
        "method_id": "fractional_variational_abm_qr",
        "q": 0.9,
        "h": h,
        "t_final": 0.3,
        "history_aware_qr": True,
        "computed_exponents": result.exponents.tolist(),
        "status": result.status,
        "all_finite": bool(np.all(np.isfinite(result.exponents))),
        "all_nonpositive": bool(np.all(np.asarray(result.exponents) <= 0.0)),
    }


def run_f4_3(*, fast: bool, use_existing: bool) -> dict[str, Any]:
    """Reuse DK2018 artifacts and keep the local full-history QR lane separate."""

    stage_dir = STAGE_DIRS["F4_3"]
    if not use_existing:
        raise RuntimeError(
            "F4.3 long DK2018 reproduction is intentionally not run here. "
            "Use the dedicated RUN_PUBLISHED_LYAPUNOV=1 runner, then rerun F4 with --use-existing."
        )
    if not DK2018_SUMMARY.exists():
        raise FileNotFoundError(f"Missing existing DK2018 artifact: {DK2018_SUMMARY}")
    official = _read_json(DK2018_SUMMARY)
    published_rows = []
    for row in official["published_case_verdicts"]:
        published_rows.append(
            {
                "case_id": row["case_id"],
                "method_id": official["method_id"],
                "status": row["status"],
                "computed_exponents": row["computed_exponents"],
                "expected_exponents": row["expected_exponents"],
                "absolute_differences": row["absolute_differences"],
                "failing_components": row["failing_components"],
                "validation_run_class": official["validation_run_class"],
                "reused_existing_artifact": True,
            }
        )
    local_qr_rows = [_full_history_qr_synthetic_row(h=h) for h in (0.02, 0.01)]
    published_contract_rows = []
    for filename in DK2018_CASE_FILES:
        case = _read_yaml(DK2018_BENCHMARKS / filename)
        integration = case["integration"]
        published_contract_rows.append(
            {
                "case_id": case["case_id"],
                "method_id": case["method_id"],
                "q": case["system"]["q"],
                "h": integration["h"],
                "t_final": integration["t_final"],
                "t_burn": integration["t_burn"],
                "reorthonormalization_time": integration["reorthonormalization_time"],
                "memory_mode": integration["memory_mode"],
                "execution_contract": case["execution"]["execution_contract"],
                "sensitivity_interpretation": "fixed_published_contract_reused_not_parameter_sweep",
            }
        )
    results_csv = stage_dir / "dk2018_published_results.csv"
    published_contract_csv = stage_dir / "dk2018_published_contract.csv"
    sensitivity_csv = stage_dir / "fractional_method_sensitivity.csv"
    _write_csv(results_csv, published_rows)
    _write_csv(published_contract_csv, published_contract_rows)
    _write_csv(sensitivity_csv, local_qr_rows)
    summary = {
        "schema_version": "1.0",
        "stage": "F4_3_fractional_published_dk2018",
        "status": "controlled_benchmark_with_documented_rf_lambda_3_discrepancy",
        "fast_smoke_only": fast,
        "uses_existing_published_long_evidence": True,
        "published_lane": {
            "method_id": official["method_id"],
            "source_summary": _relative(DK2018_SUMMARY),
            "source_status": official["status"],
            "lorenz_status": published_rows[0]["status"],
            "rabinovich_fabrikant_status": published_rows[1]["status"],
            "documented_discrepancy": "RF lambda_3 exceeds the absolute tolerance.",
        },
        "local_full_history_qr_lane": {
            "method_id": "fractional_variational_abm_qr",
            "status": "separate_internal_synthetic_control_only",
            "validated_against_dk2018_published_lane": False,
            "sensitivity_axis": "h",
        },
        "files": {
            "results_csv": _relative(results_csv),
            "published_contract_csv": _relative(published_contract_csv),
            "sensitivity_csv": _relative(sensitivity_csv),
        },
        "certifications": NO_FALSE_CERTIFICATION,
    }
    _write_json(stage_dir / "summary.json", summary)
    _write_readme(
        stage_dir / "README.md",
        "F4.3 DK2018 fractional published-reference lane",
        [
            "This stage reuses the existing long DK2018 artifact; it does not rerun the published benchmark.",
            "Lorenz passes quantitatively. RF remains pending because lambda_3 exceeds tolerance.",
            "`fractional_variational_abm_qr` remains a separate full-history QR contract.",
            "Passing the DK2018 block-restart lane does not validate the local full-history QR lane.",
        ],
    )
    return summary


def run_f4_4(*, fast: bool, use_existing: bool) -> dict[str, Any]:
    """Reuse Fischer 2020 rows and the existing bounded sensitivity audit."""

    stage_dir = STAGE_DIRS["F4_4"]
    if not use_existing:
        raise RuntimeError(
            "F4.4 long Fischer reproduction is intentionally not run here. "
            "Use the dedicated RUN_PUBLISHED_CLONED=1 runner, then rerun F4 with --use-existing."
        )
    for path in (FISCHER2020_SUMMARY, FISCHER2020_SENSITIVITY, FISCHER2020_CLASSIFICATION):
        if not path.exists():
            raise FileNotFoundError(f"Missing existing Fischer artifact: {path}")
    official = _read_json(FISCHER2020_SUMMARY)
    sensitivity = _read_json(FISCHER2020_SENSITIVITY)
    results_rows = []
    for row in official["results"]:
        results_rows.append(
            {
                "case_file": row["case_file"],
                "system": row["system"],
                "row_index": row["row_index"],
                "orders": row["orders"],
                "computed_exponents": row["computed_LE"],
                "published_exponents": row["published_LE"],
                "absolute_error": row["abs_error"],
                "status": row["status"],
                "sign_match": row["sign_match"],
                "reused_existing_artifact": True,
            }
        )
    sensitivity_rows = [
        {
            "axis": row["axis"],
            "runs": row["runs"],
            "best_row": row["best_row"],
            "best_max_abs_error": row["best_max_abs_error"],
            "improved_rows": row["improved_rows"],
            "degraded_rows": row["degraded_rows"],
            "source_runs_total": sensitivity["runs_total"],
            "reused_existing_artifact": True,
        }
        for row in sensitivity["axis_summary"]
    ]
    results_csv = stage_dir / "fischer2020_f4_results.csv"
    sensitivity_csv = stage_dir / "fischer2020_f4_sensitivity.csv"
    _write_csv(results_csv, results_rows)
    _write_csv(sensitivity_csv, sensitivity_rows)
    summary = {
        "schema_version": "1.0",
        "stage": "F4_4_cloned_dynamics_fischer2020",
        "status": "controlled_benchmark_with_documented_discrepancies",
        "fast_smoke_only": fast,
        "uses_existing_published_long_evidence": True,
        "method_id": official["method_id"],
        "source_summary": _relative(FISCHER2020_SUMMARY),
        "source_status": official["status"],
        "validated": False,
        "rows_total": official["rows_total"],
        "rows_passed_quantitative": official["rows_passed_quantitative"],
        "rows_passed_sign_pattern": official["rows_passed_sign_pattern"],
        "rows_failed": official["rows_failed"],
        "sensitivity_runs_reused": sensitivity["runs_total"],
        "sensitivity_axes_reused": sensitivity["axes_executed"],
        "files": {
            "results_csv": _relative(results_csv),
            "sensitivity_csv": _relative(sensitivity_csv),
            "source_classification_csv": _relative(FISCHER2020_CLASSIFICATION),
        },
        "certifications": NO_FALSE_CERTIFICATION,
    }
    _write_json(stage_dir / "summary.json", summary)
    _write_readme(
        stage_dir / "README.md",
        "F4.4 Fischer 2020 cloned-dynamics lane",
        [
            "This stage reuses the existing 24-row published-reference artifact and 164 diagnostic sensitivity runs.",
            "The status is `controlled_benchmark_with_documented_discrepancies`.",
            "The GS lane remains `validated=false`; the QR variant remains internal and experimental.",
            "Fischer decimal comparisons do not validate the local full-history Caputo QR lane.",
        ],
    )
    return summary


def build_global_summary(stage_summaries: dict[str, dict[str, Any]], *, fast: bool) -> dict[str, Any]:
    """Write the F4 closure summary without promoting method validation."""

    expected = {"F4_1", "F4_2", "F4_3", "F4_4"}
    missing = sorted(expected - set(stage_summaries))
    completed = not missing
    method_rows = [
        {
            "method_id": "integer_qr_benettin",
            "controlled_benchmark": "F4_1_integer_linear and F4_2_integer_chua_q1",
            "sensitivity_reference": _relative(STAGE_DIRS["F4_1"] / "integer_linear_sensitivity.csv"),
            "bibliographic_or_internal_reference": "Benettin et al. 1980; Wolf et al. 1985",
            "status": "controlled_q1_checks_recorded",
            "validated_by_f4": False,
        },
        {
            "method_id": "fractional_variational_abm_qr",
            "controlled_benchmark": "F4_3 local stable synthetic full-history QR rows",
            "sensitivity_reference": _relative(STAGE_DIRS["F4_3"] / "fractional_method_sensitivity.csv"),
            "bibliographic_or_internal_reference": "Local fixed-lower-limit history-aware QR implementation contract",
            "status": "separate_internal_control_only_published_validation_pending",
            "validated_by_f4": False,
        },
        {
            "method_id": "fractional_variational_dk2018_block_restart_abm_gs",
            "controlled_benchmark": "F4_3 reused DK2018 Lorenz and RF artifact",
            "sensitivity_reference": _relative(STAGE_DIRS["F4_3"] / "dk2018_published_contract.csv"),
            "bibliographic_or_internal_reference": "Danca and Kuznetsov 2018 DOI 10.1142/S0218127418500670",
            "status": "published_benchmarks_pending_reproduced_discrepancy",
            "validated_by_f4": False,
        },
        {
            "method_id": "fractional_cloned_dynamics_abm_gs_published",
            "controlled_benchmark": "F4_4 reused Fischer 2020 artifact",
            "sensitivity_reference": _relative(STAGE_DIRS["F4_4"] / "fischer2020_f4_sensitivity.csv"),
            "bibliographic_or_internal_reference": "Fischer et al. 2020 DOI 10.1016/j.apnum.2020.03.027",
            "status": "controlled_benchmark_with_documented_discrepancies",
            "validated_by_f4": False,
        },
        {
            "method_id": "fractional_cloned_dynamics_abm_qr",
            "controlled_benchmark": "F4_1 q=1 diagonal QR cloned control",
            "sensitivity_reference": _relative(STAGE_DIRS["F4_1"] / "integer_linear_sensitivity.csv"),
            "bibliographic_or_internal_reference": "Internal QR comparison variant based on Fischer et al. 2020",
            "status": "internal_experimental_variant_controlled_only",
            "validated_by_f4": False,
        },
    ]
    summary = {
        "schema_version": "1.0",
        "phase": "F4_internal_validation",
        "generated_at_utc": _utc_now(),
        "status": "f4_complete_with_documented_discrepancies" if completed else "f4_partial",
        "fast_smoke_only": fast,
        "uses_existing_published_long_evidence": completed,
        "missing_stages": missing,
        "stage_summaries": {
            name: _relative(STAGE_DIRS[name] / "summary.json")
            for name in sorted(stage_summaries)
        },
        "method_controls": method_rows if completed else [],
        "closure_checks": {
            "each_method_has_controlled_benchmark": completed,
            "each_method_has_sensitivity_reference": completed,
            "each_method_has_bibliographic_or_internal_reference": completed,
            "discrepancies_preserved": completed,
            "validation_state_promoted": False,
        },
        "certifications": NO_FALSE_CERTIFICATION,
        "notes": [
            "F4 is an internal consistency layer, not a published validation promotion.",
            "Fast smoke controls do not replace long published quantitative runs.",
            "DK2018 RF lambda_3 and Fischer 2020 discrepancies remain explicit.",
            "The local full-history QR and DK2018 block-restart lanes remain separate.",
        ],
    }
    _write_json(GLOBAL_SUMMARY, summary)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(SUMMARY_DIR / "method_control_matrix.json", {"method_controls": summary["method_controls"]})
    _write_readme(
        F4_ROOT / "README.md",
        "F4 internal Lyapunov validation",
        [
            "F4 records controlled internal consistency evidence for each implemented Lyapunov family.",
            "It reuses existing published-reference artifacts and does not run long sweeps by default.",
            "The closure state is `f4_complete_with_documented_discrepancies` when every method has a control, a sensitivity reference, and a bibliographic or internal reference.",
            "This state does not certify chaos, hiddenness, or fractional method validity.",
        ],
    )
    return summary


def run_selected_stages(
    stages: Iterable[str],
    *,
    fast: bool,
    use_existing: bool,
) -> dict[str, Any]:
    """Run selected stages and write a global summary for the executed set."""

    selected = list(dict.fromkeys(stages))
    runners = {
        "F4_1": lambda: run_f4_1(fast=fast),
        "F4_2": lambda: run_f4_2(fast=fast),
        "F4_3": lambda: run_f4_3(fast=fast, use_existing=use_existing),
        "F4_4": lambda: run_f4_4(fast=fast, use_existing=use_existing),
    }
    summaries = {stage: runners[stage]() for stage in selected}
    for stage, stage_dir in STAGE_DIRS.items():
        summary_path = stage_dir / "summary.json"
        if stage not in summaries and summary_path.exists():
            summaries[stage] = _read_json(summary_path)
    return build_global_summary(summaries, fast=fast)


__all__ = [
    "F4_ROOT",
    "GLOBAL_SUMMARY",
    "NO_FALSE_CERTIFICATION",
    "STAGE_DIRS",
    "build_global_summary",
    "run_f4_1",
    "run_f4_2",
    "run_f4_3",
    "run_f4_4",
    "run_selected_stages",
]
