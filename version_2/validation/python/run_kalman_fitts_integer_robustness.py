#!/usr/bin/env python3
"""Finite-time integrator robustness matrix for the Kalman--Fitts q=1 cycle.

The audit starts every integration from the canonical ``target_seed`` stored in
``05_verification_summary.json`` and compares its post-transient cloud with the
maintained ``04_final_attractor.csv``.  It does not repeat the continuation or
the equilibrium-neighbourhood campaign and therefore cannot certify hiddenness.

Default matrix
--------------
* DOP853, standard tolerances and ``max_step=0.05``;
* DOP853, tight tolerances and ``max_step=0.01``;
* classical RK4 with ``h=0.01``;
* classical RK4 with ``h=0.005``;
* horizons ``T=300`` and ``T=600`` for every configuration.

Outputs are written beside the maintained reference case:
``06_integrator_robustness_matrix.csv`` and
``06_integrator_robustness.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy


VERSION2 = Path(__file__).resolve().parents[2]
if str(VERSION2) not in sys.path:
    sys.path.insert(0, str(VERSION2))

from hidden_attractors import get_system
from hidden_attractors.analysis.boundedness import compute_boundedness_metrics
from hidden_attractors.analysis.trajectory import cloud_median_distance, sample_rows
from hidden_attractors.integrations.rk4 import rk4_integrate
from hidden_attractors.solvers.integer import dop853_q1_integrate


DEFAULT_CASE_DIR = (
    VERSION2 / "validation" / "reference_cases" / "kalman_fitts_integer_q1"
)
SYSTEM_ID = "kalman-fitts-2019"
DIV_THRESHOLD = 100.0
ANALYSIS_START_FRACTION = 0.5
MAX_CLOUD_POINTS = 800

# These finite-time consistency limits are fixed before inspecting this matrix.
THRESHOLDS: dict[str, float] = {
    "period_relative_error_max": 5.0e-3,
    "period_cv_max": 5.0e-3,
    "return_state_error_max": 2.0e-2,
    "cloud_distance_normalized_max": 2.0e-2,
}

CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {
        "config_id": "dop853_standard",
        "integrator": "DOP853",
        "output_h": 0.01,
        "rtol": 1.0e-9,
        "atol": 1.0e-12,
        "max_step": 0.05,
    },
    {
        "config_id": "dop853_tight",
        "integrator": "DOP853",
        "output_h": 0.01,
        "rtol": 1.0e-11,
        "atol": 1.0e-13,
        "max_step": 0.01,
    },
    {
        "config_id": "rk4_h_0p01",
        "integrator": "RK4",
        "h": 0.01,
    },
    {
        "config_id": "rk4_h_0p005",
        "integrator": "RK4",
        "h": 0.005,
    },
)

CSV_FIELDS = (
    "case_id",
    "config_id",
    "integrator",
    "horizon",
    "output_h",
    "internal_h",
    "rtol",
    "atol",
    "max_step",
    "status",
    "n_output_rows",
    "t_reached",
    "finite_fraction",
    "boundedness_status",
    "max_norm",
    "norm_growth_ratio",
    "n_returns",
    "period_mean",
    "period_std",
    "period_cv",
    "period_relative_error",
    "return_state_error_max",
    "return_state_error_median",
    "cloud_distance",
    "cloud_distance_normalized",
    "bounded_finite_time",
    "period_consistent",
    "return_consistent",
    "cloud_consistent",
    "configuration_consistent",
    "elapsed_seconds",
    "discrepancy_reason",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(VERSION2.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _clean_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_clean_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    return value


def _same_direction_crossings(
    trajectory: np.ndarray,
    *,
    t_start: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate upward crossings of the canonical section ``x3=0``."""

    data = np.asarray(trajectory, dtype=float)
    if data.ndim != 2 or data.shape[1] != 5 or data.shape[0] < 2:
        return np.empty(0, dtype=float), np.empty((0, 4), dtype=float)
    section_values = data[:, 3]
    indices = np.where((section_values[:-1] < 0.0) & (section_values[1:] >= 0.0))[0]
    times: list[float] = []
    states: list[np.ndarray] = []
    for index in indices:
        denominator = section_values[index + 1] - section_values[index]
        if denominator == 0.0:
            continue
        fraction = -section_values[index] / denominator
        row = data[index] + fraction * (data[index + 1] - data[index])
        if float(row[0]) < float(t_start):
            continue
        times.append(float(row[0]))
        states.append(np.asarray(row[1:], dtype=float))
    if not states:
        return np.asarray(times, dtype=float), np.empty((0, 4), dtype=float)
    return np.asarray(times, dtype=float), np.vstack(states)


def _integrate(
    system: Any,
    seed: np.ndarray,
    configuration: dict[str, Any],
    horizon: float,
) -> tuple[np.ndarray, str]:
    # ``ChaoticSystem.evaluate`` also exposes an optional parameters argument.
    # The one-argument wrapper makes the autonomous signature unambiguous to
    # generic integrator adapters that distinguish rhs(x) from rhs(t, x).
    def autonomous_rhs(state: np.ndarray) -> np.ndarray:
        return np.asarray(system.evaluate(state), dtype=float)

    if configuration["integrator"] == "DOP853":
        return dop853_q1_integrate(
            autonomous_rhs,
            seed,
            t_final=horizon,
            h=float(configuration["output_h"]),
            rtol=float(configuration["rtol"]),
            atol=float(configuration["atol"]),
            max_step=float(configuration["max_step"]),
            div_threshold=DIV_THRESHOLD,
        )
    h = float(configuration["h"])
    exact_steps = horizon / h
    steps = int(round(exact_steps))
    if not math.isclose(steps * h, horizon, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"horizon {horizon} is not an integer multiple of h={h}")
    times, states, status, _info = rk4_integrate(
        autonomous_rhs,
        seed,
        h=h,
        N=steps,
        divergence_norm=DIV_THRESHOLD,
    )
    return np.column_stack((times, states)), status


def _empty_row(
    configuration: dict[str, Any],
    horizon: float,
    *,
    status: str,
    elapsed_seconds: float,
    discrepancy_reason: str,
) -> dict[str, Any]:
    return {
        "case_id": "kalman_fitts_integer_q1",
        "config_id": configuration["config_id"],
        "integrator": configuration["integrator"],
        "horizon": float(horizon),
        "output_h": configuration.get("output_h", configuration.get("h")),
        "internal_h": configuration.get("h"),
        "rtol": configuration.get("rtol"),
        "atol": configuration.get("atol"),
        "max_step": configuration.get("max_step"),
        "status": status,
        "n_output_rows": 0,
        "t_reached": None,
        "finite_fraction": None,
        "boundedness_status": "not_evaluated",
        "max_norm": None,
        "norm_growth_ratio": None,
        "n_returns": 0,
        "period_mean": None,
        "period_std": None,
        "period_cv": None,
        "period_relative_error": None,
        "return_state_error_max": None,
        "return_state_error_median": None,
        "cloud_distance": None,
        "cloud_distance_normalized": None,
        "bounded_finite_time": False,
        "period_consistent": False,
        "return_consistent": False,
        "cloud_consistent": False,
        "configuration_consistent": False,
        "elapsed_seconds": float(elapsed_seconds),
        "discrepancy_reason": discrepancy_reason,
    }


def _evaluate(
    trajectory: np.ndarray,
    *,
    status: str,
    configuration: dict[str, Any],
    horizon: float,
    elapsed_seconds: float,
    canonical_cloud: np.ndarray,
    canonical_scale: float,
    canonical_period: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = np.asarray(trajectory, dtype=float)
    if data.ndim != 2 or data.shape[1] != 5 or data.shape[0] == 0:
        row = _empty_row(
            configuration,
            horizon,
            status=status,
            elapsed_seconds=elapsed_seconds,
            discrepancy_reason="trajectory_shape_invalid",
        )
        return row, {"boundedness": None, "coordinate_extrema": None}

    analysis_start = ANALYSIS_START_FRACTION * horizon
    times = data[:, 0]
    states = data[:, 1:]
    boundedness = compute_boundedness_metrics(
        times,
        states,
        burn_time=analysis_start,
        divergence_radius=DIV_THRESHOLD,
    )
    boundedness.pop("norm_timeseries", None)
    full_finite_fraction = float(np.count_nonzero(np.isfinite(states)) / states.size)

    crossing_times, crossing_states = _same_direction_crossings(
        data,
        t_start=analysis_start,
    )
    selected_times = crossing_times[-20:]
    selected_states = crossing_states[-20:]
    periods = np.diff(selected_times)
    recurrence = (
        np.linalg.norm(np.diff(selected_states, axis=0), axis=1)
        if selected_states.shape[0] >= 2
        else np.empty(0, dtype=float)
    )
    period_mean = float(np.mean(periods)) if periods.size else float("nan")
    period_std = float(np.std(periods)) if periods.size else float("nan")
    period_cv = (
        period_std / abs(period_mean)
        if math.isfinite(period_mean) and abs(period_mean) > 0.0
        else float("nan")
    )
    period_relative_error = (
        abs(period_mean - canonical_period) / abs(canonical_period)
        if math.isfinite(period_mean)
        else float("nan")
    )
    return_error_max = float(np.max(recurrence)) if recurrence.size else float("nan")
    return_error_median = float(np.median(recurrence)) if recurrence.size else float("nan")

    post_states = states[times >= analysis_start]
    candidate_cloud = sample_rows(post_states, MAX_CLOUD_POINTS)
    cloud_distance = cloud_median_distance(candidate_cloud, canonical_cloud)
    cloud_distance_normalized = (
        cloud_distance / canonical_scale
        if math.isfinite(cloud_distance) and canonical_scale > 0.0
        else float("nan")
    )

    reached_horizon = math.isclose(
        float(times[-1]),
        horizon,
        rel_tol=0.0,
        abs_tol=max(1.0e-10, 1.0e-10 * horizon),
    )
    bounded_finite_time = bool(
        status == "ok"
        and reached_horizon
        and full_finite_fraction == 1.0
        and boundedness.get("boundedness_status") == "bounded_candidate"
    )
    period_consistent = bool(
        periods.size >= 4
        and math.isfinite(period_relative_error)
        and period_relative_error <= THRESHOLDS["period_relative_error_max"]
        and math.isfinite(period_cv)
        and period_cv <= THRESHOLDS["period_cv_max"]
    )
    return_consistent = bool(
        recurrence.size >= 4
        and math.isfinite(return_error_max)
        and return_error_max <= THRESHOLDS["return_state_error_max"]
    )
    cloud_consistent = bool(
        math.isfinite(cloud_distance_normalized)
        and cloud_distance_normalized <= THRESHOLDS["cloud_distance_normalized_max"]
    )
    configuration_consistent = bool(
        bounded_finite_time
        and period_consistent
        and return_consistent
        and cloud_consistent
    )

    reasons: list[str] = []
    if status != "ok":
        reasons.append(f"solver_status={status}")
    if not reached_horizon:
        reasons.append("horizon_not_reached")
    if not bounded_finite_time:
        reasons.append("finite_time_boundedness_not_confirmed")
    if not period_consistent:
        reasons.append("period_not_consistent")
    if not return_consistent:
        reasons.append("return_not_consistent")
    if not cloud_consistent:
        reasons.append("canonical_cloud_not_consistent")

    row = {
        "case_id": "kalman_fitts_integer_q1",
        "config_id": configuration["config_id"],
        "integrator": configuration["integrator"],
        "horizon": float(horizon),
        "output_h": configuration.get("output_h", configuration.get("h")),
        "internal_h": configuration.get("h"),
        "rtol": configuration.get("rtol"),
        "atol": configuration.get("atol"),
        "max_step": configuration.get("max_step"),
        "status": status,
        "n_output_rows": int(data.shape[0]),
        "t_reached": float(times[-1]),
        "finite_fraction": full_finite_fraction,
        "boundedness_status": boundedness.get("boundedness_status"),
        "max_norm": boundedness.get("max_norm"),
        "norm_growth_ratio": boundedness.get("norm_growth_ratio"),
        "n_returns": int(crossing_times.size),
        "period_mean": period_mean,
        "period_std": period_std,
        "period_cv": period_cv,
        "period_relative_error": period_relative_error,
        "return_state_error_max": return_error_max,
        "return_state_error_median": return_error_median,
        "cloud_distance": cloud_distance,
        "cloud_distance_normalized": cloud_distance_normalized,
        "bounded_finite_time": bounded_finite_time,
        "period_consistent": period_consistent,
        "return_consistent": return_consistent,
        "cloud_consistent": cloud_consistent,
        "configuration_consistent": configuration_consistent,
        "elapsed_seconds": float(elapsed_seconds),
        "discrepancy_reason": ";".join(reasons),
    }
    detail = {
        "analysis_start": analysis_start,
        "boundedness": boundedness,
        "coordinate_extrema": {
            "minimum": np.min(post_states, axis=0) if post_states.size else None,
            "maximum": np.max(post_states, axis=0) if post_states.size else None,
            "span": np.ptp(post_states, axis=0) if post_states.size else None,
        },
        "return_states_tail": selected_states[-5:] if selected_states.size else [],
    }
    return row, detail


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            cleaned = _clean_json(row)
            writer.writerow({field: cleaned.get(field) for field in CSV_FIELDS})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the Kalman--Fitts q=1 cycle across DOP853 and RK4.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=DEFAULT_CASE_DIR,
        help="Reference-case directory (default: maintained Kalman--Fitts case).",
    )
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=(300.0, 600.0),
        help="Positive integration horizons (default: 300 600).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    case_dir = args.case_dir
    if not case_dir.is_absolute():
        case_dir = VERSION2 / case_dir
    horizons = tuple(float(value) for value in args.horizons)
    if not horizons or any(not math.isfinite(value) or value <= 0.0 for value in horizons):
        raise ValueError("all horizons must be finite and positive")

    summary_path = case_dir / "05_verification_summary.json"
    cloud_path = case_dir / "04_final_attractor.csv"
    if not summary_path.is_file() or not cloud_path.is_file():
        raise FileNotFoundError("the canonical summary and trajectory are required")
    canonical_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed = np.asarray(canonical_summary["target_seed"], dtype=float)
    canonical_period = float(canonical_summary["poincare"]["period_mean"])
    canonical_trajectory = np.loadtxt(cloud_path, delimiter=",", skiprows=1)
    if canonical_trajectory.ndim != 2 or canonical_trajectory.shape[1] != 5:
        raise ValueError("04_final_attractor.csv must have columns t,x1,x2,x3,x4")
    seed_residual = float(np.linalg.norm(canonical_trajectory[0, 1:] - seed))
    if seed_residual > 1.0e-12:
        raise ValueError("the canonical trajectory does not start at target_seed")
    canonical_cloud = sample_rows(canonical_trajectory[:, 1:], MAX_CLOUD_POINTS)
    canonical_scale = float(np.linalg.norm(np.ptp(canonical_cloud, axis=0)))
    if not math.isfinite(canonical_scale) or canonical_scale <= 0.0:
        raise ValueError("the canonical cloud has zero or invalid scale")

    system = get_system(SYSTEM_ID)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        for horizon in horizons:
            started = time.perf_counter()
            try:
                trajectory, status = _integrate(system, seed, configuration, horizon)
                elapsed = time.perf_counter() - started
                row, detail = _evaluate(
                    trajectory,
                    status=status,
                    configuration=configuration,
                    horizon=horizon,
                    elapsed_seconds=elapsed,
                    canonical_cloud=canonical_cloud,
                    canonical_scale=canonical_scale,
                    canonical_period=canonical_period,
                )
            except Exception as exc:  # keep configuration failures as evidence
                elapsed = time.perf_counter() - started
                status = f"exception:{type(exc).__name__}"
                row = _empty_row(
                    configuration,
                    horizon,
                    status=status,
                    elapsed_seconds=elapsed,
                    discrepancy_reason=f"{type(exc).__name__}: {exc}",
                )
                detail = {"exception": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            details.append(
                {
                    "config_id": configuration["config_id"],
                    "horizon": horizon,
                    **detail,
                }
            )
            print(
                f"{configuration['config_id']:18s} T={horizon:7g} "
                f"status={row['status']} consistent={row['configuration_consistent']}"
            )

    discrepancies = [
        {
            "config_id": row["config_id"],
            "horizon": row["horizon"],
            "status": row["status"],
            "reason": row["discrepancy_reason"],
        }
        for row in rows
        if not row["configuration_consistent"]
    ]
    payload = {
        "schema_version": 1,
        "case_id": "kalman_fitts_integer_q1",
        "system_id": SYSTEM_ID,
        "purpose": "finite_time_integrator_step_and_horizon_robustness",
        "evidence_scope": (
            "boundedness, Poincare period/return consistency, and geometric cloud "
            "agreement only; not a proof of periodicity, attraction, hiddenness, or "
            "global basin separation"
        ),
        "inputs": {
            "target_seed": seed,
            "target_seed_source": _relative_path(summary_path),
            "target_seed_source_sha256": _sha256(summary_path),
            "canonical_cloud_source": _relative_path(cloud_path),
            "canonical_cloud_source_sha256": _sha256(cloud_path),
            "canonical_cloud_points_used": int(canonical_cloud.shape[0]),
            "canonical_cloud_scale": canonical_scale,
            "canonical_period": canonical_period,
            "seed_to_cloud_first_row_residual": seed_residual,
            "system_parameters": dict(system.parameters),
            "system_reference": dict(system.reference),
        },
        "matrix_contract": {
            "configurations": CONFIGURATIONS,
            "horizons": horizons,
            "horizon_rationale": (
                "T=300 matches the maintained final retained window; T=600 doubles it "
                "and matches the maintained hiddenness integration horizon"
            ),
            "analysis_start_fraction": ANALYSIS_START_FRACTION,
            "poincare_section": "upward crossings of x3=0 with linear interpolation",
            "return_window": "last 20 crossings after the analysis start",
            "cloud_metric": "symmetric median nearest-neighbour distance",
            "cloud_normalization": "Euclidean norm of canonical coordinate spans",
            "max_cloud_points": MAX_CLOUD_POINTS,
            "divergence_norm": DIV_THRESHOLD,
            "thresholds": THRESHOLDS,
        },
        "rows": rows,
        "details": details,
        "summary": {
            "n_rows": len(rows),
            "n_consistent": sum(bool(row["configuration_consistent"]) for row in rows),
            "n_discrepant": len(discrepancies),
            "all_configurations_consistent": not discrepancies,
            "discrepancies": discrepancies,
            "interpretation": (
                "A consistent row supports finite-time numerical persistence of the "
                "maintained cycle under that solver contract only."
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }

    csv_path = case_dir / "06_integrator_robustness_matrix.csv"
    json_path = case_dir / "06_integrator_robustness.json"
    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(_clean_json(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"CSV: {_relative_path(csv_path)}")
    print(f"JSON: {_relative_path(json_path)}")
    print(
        f"consistent={payload['summary']['n_consistent']}/{payload['summary']['n_rows']} "
        f"discrepant={payload['summary']['n_discrepant']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
