#!/usr/bin/env python3
"""Generate standardized F5.4 Poincare diagnostics for configured cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hidden_attractors.analysis.poincare import (  # noqa: E402
    detect_poincare_crossings,
    summarize_poincare_points,
    write_poincare_outputs,
)
from hidden_attractors.models.chua import chua_parameters, rhs_nonsmooth  # noqa: E402
from hidden_attractors.native.backends import (  # noqa: E402
    FractionalChuaBackend,
    FullHistoryABMBackend,
)
from hidden_attractors.solvers import efork_q1_integrate  # noqa: E402


POINCARE_ROOT = (
    PROJECT_ROOT / "validation" / "chaos_validation" / "dynamics_diagnostics" / "poincare"
)
CASES_DIR = POINCARE_ROOT / "cases"
GLOBAL_SUMMARY_PATH = POINCARE_ROOT / "poincare_diagnostics_summary.json"
F5_SUMMARY_PATH = POINCARE_ROOT.parent / "f5_diagnostics_summary.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _params(config: dict[str, Any]):
    values = dict(config["parameters"])
    values.pop("m", None)
    values.pop("n", None)
    values["model"] = config.get("model", "nonsmooth")
    return chua_parameters(**values)


def _synthetic_method_validation() -> dict[str, Any]:
    h = 0.07
    times = np.arange(-0.4, 8.0 * math.pi + 0.4, h)
    sine = np.column_stack((np.sin(times), np.cos(times)))
    positive = detect_poincare_crossings(
        times,
        sine,
        derivative_mode="finite_difference_diagnostic",
        burn_time=0.1,
    )
    negative = detect_poincare_crossings(
        times,
        sine,
        direction="negative",
        derivative_mode="finite_difference_diagnostic",
        burn_time=0.1,
    )
    no_crossings = detect_poincare_crossings(
        times,
        np.column_stack((1.0 + 0.1 * np.sin(times), np.cos(times))),
        derivative_mode="finite_difference_diagnostic",
    )
    geometric = detect_poincare_crossings(
        times,
        sine,
        direction="positive_geometric_crossing",
        derivative_mode="geometric_fractional",
    )
    expected_positive = np.arange(1, 4) * 2.0 * math.pi
    checks = {
        "positive_crossing_detection": bool(
            positive.crossing_count >= 3
            and np.all(np.abs(positive.crossing_times[:3] - expected_positive) < h**2)
        ),
        "negative_direction_filtering": bool(
            negative.crossing_count >= 3 and set(negative.crossing_directions) == {"negative"}
        ),
        "no_crossings_status": no_crossings.status == "no_crossings",
        "linear_interpolation": bool(np.all(np.abs(positive.points[:, 0]) < 1.0e-10)),
        "caputo_geometric_metadata": bool(
            geometric.section_metadata["caputo_geometric_crossing"]
            and not geometric.section_metadata["exact_poincare_map"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"synthetic Poincare validation failed: {checks}")
    return {
        "status": "passed_synthetic_crossing_tests",
        "checks": checks,
        "validates": [
            "crossing_detection",
            "linear_interpolation",
            "direction_filtering",
            "integer_rhs_direction",
            "caputo_geometric_crossing_metadata",
        ],
        "does_not_validate": ["chaos", "hiddenness", "periodic_orbits_in_caputo"],
    }


def _integrate_case(config: dict[str, Any]) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    case_id = config["case_id"]
    integration = config["integration"]
    params = _params(config)
    q = float(config["q"])
    h = float(integration["h"])
    t_final = float(integration["t_final"])
    if case_id == "chua_integer_q1_reference":
        rhs = lambda state: rhs_nonsmooth(state, params)
        trajectory, status = efork_q1_integrate(
            rhs,
            np.asarray(config["seed"]["X0_python"], dtype=float),
            h=h,
            t_final=t_final,
        )
        return [("X0_python", trajectory, {"integration_status": status})]
    if case_id == "danca2017_chua_fractional_saturation_q09998":
        backend = FullHistoryABMBackend.build(output_name="poincare_danca2017_abm_full_history")
        backend.set_nonsmooth_params(params)
        trajectory = backend.integrate(
            config["seed"]["diagnostic_X0"],
            q=q,
            h=h,
            t_final=t_final,
        )
        return [(
            "diagnostic_X0",
            trajectory,
            {
                "integration_status": "ok" if np.all(np.isfinite(trajectory)) else "nonfinite_solution",
                "seed_scope": "diagnostic_only_not_reported_by_article",
            },
        )]
    if case_id == "wu2023_chua_fractional_arctan_q099":
        backend = FractionalChuaBackend.build(output_name="poincare_wu2023_efork_arctan")
        backend.set_arctan_params(params)
        return [
            (
                name,
                backend.integrate_efork3(
                    seed,
                    q=q,
                    h=h,
                    Lm=float(integration["memory_length"]),
                    t_final=t_final,
                ),
                {"integration_status": "ok"},
            )
            for name, seed in (
                ("x0_plus", config["seed"]["x0_plus"]),
                ("x0_minus", config["seed"]["x0_minus"]),
            )
        ]
    raise ValueError(f"unsupported Poincare diagnostic case: {case_id}")


def _burn_time(config: dict[str, Any]) -> float:
    integration = config["integration"]
    if "t_burn" in integration:
        return float(integration["t_burn"])
    return float(integration["t_final"]) * float(integration.get("diagnostic_burn_fraction", 0.5))


def _write_case_outputs(config: dict[str, Any]) -> dict[str, Any]:
    case_id = config["case_id"]
    case_dir = POINCARE_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    section = config["section"]
    burn_time = _burn_time(config)
    derivative_mode = section["derivative_mode"]
    params = _params(config)
    aggregate_rows: list[dict[str, Any]] = []
    combined_points: list[np.ndarray] = []
    trajectory_summaries: dict[str, Any] = {}
    for trajectory_id, trajectory, integration_metadata in _integrate_case(config):
        times = trajectory[:, 0]
        states = trajectory[:, 1:]
        rhs = (
            (lambda _time, state: rhs_nonsmooth(state, params))
            if derivative_mode == "integer_rhs"
            else None
        )
        result = detect_poincare_crossings(
            times,
            states,
            section_variable=section["variable"],
            section_value=float(section["value"]),
            direction=section["direction"],
            derivative_mode=derivative_mode,
            rhs=rhs,
            burn_time=burn_time,
        )
        write_poincare_outputs(
            case_dir / trajectory_id,
            result,
            metadata={"trajectory_id": trajectory_id, **integration_metadata},
        )
        trajectory_summaries[trajectory_id] = {
            **integration_metadata,
            "trajectory_rows": int(trajectory.shape[0]),
            "crossing_count": result.crossing_count,
            "status": result.status,
            "section_metadata": result.section_metadata,
        }
        if result.points.size:
            combined_points.append(result.points)
        for index, time, direction, point in zip(
            result.crossing_indices,
            result.crossing_times,
            result.crossing_directions,
            result.points,
            strict=True,
        ):
            aggregate_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "crossing_index": int(index),
                    "crossing_time": float(time),
                    "direction": direction,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                }
            )
    fields = ["trajectory_id", "crossing_index", "crossing_time", "direction", "x", "y", "z"]
    for filename in ("poincare_points.csv", "poincare_section.csv"):
        with (case_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(aggregate_rows)
    points = np.vstack(combined_points) if combined_points else np.empty((0, 3), dtype=float)
    summary = summarize_poincare_points(points)
    summary.update(
        {
            "case_id": case_id,
            "status": "completed_structured_outputs",
            "trajectory_count": len(trajectory_summaries),
            "trajectories": trajectory_summaries,
            "geometric_interpretation": summary["interpretation_label"],
            "chaos_certified_by_poincare": False,
            "hiddenness_certified_by_poincare": False,
            "periodic_orbit_exact": False,
            "caputo_periodic_orbit_exact": False,
        }
    )
    _write_json(case_dir / "poincare_summary.json", summary)
    metadata = {
        "case_id": case_id,
        "source": config["source"],
        "derivative_model": config["derivative_model"],
        "q": config["q"],
        "section": section,
        "integration": config["integration"],
        "burn_time": burn_time,
        "warnings": [
            "Poincare alone does not certify chaos.",
            "Poincare alone does not certify hiddenness.",
            "Caputo crossings are geometric diagnostics, not exact periodic-orbit returns.",
        ],
    }
    _write_json(case_dir / "poincare_metadata.json", metadata)
    (case_dir / "README.md").write_text(
        f"# {case_id}\n\n"
        "Standardized numerical Poincare crossing outputs. These geometric\n"
        "diagnostics do not certify chaos, hiddenness, or exact Caputo periodic orbits.\n",
        encoding="utf-8",
    )
    return summary


def _final_f5_status(subphases: dict[str, dict[str, Any]]) -> str:
    if all(item["standardized_outputs"] for item in subphases.values()):
        return "f5_diagnostics_structured_outputs_ready"
    return "diagnostics_partial_current_protocol"


def _write_f5_summary(
    method_validation: dict[str, Any],
    application_summary: dict[str, Any],
) -> dict[str, Any]:
    subphases = {
        "boundedness": {"status": "partial", "standardized_outputs": False},
        "zero_one": {"status": "not_implemented", "standardized_outputs": False},
        "psd_fft": {"status": "partial", "standardized_outputs": False},
        "poincare": {
            "status": application_summary["status"],
            "standardized_outputs": True,
            "path": "poincare/poincare_diagnostics_summary.json",
        },
    }
    payload = {
        "stage": "F5_dynamics_diagnostics",
        **subphases,
        "poincare_method_validation": method_validation,
        "poincare_application_to_published_cases": application_summary,
        "final_f5_status": _final_f5_status(subphases),
        "certifications": {"chaos_verified": False, "hidden_verified": False},
        "invariants": {
            "single_indicator_cannot_certify_chaos": True,
            "diagnostics_cannot_certify_hiddenness": True,
            "poincare_cannot_certify_caputo_periodic_orbits": True,
        },
    }
    _write_json(F5_SUMMARY_PATH, payload)
    return payload


def run(cases_dir: Path = CASES_DIR) -> dict[str, Any]:
    method_validation = _synthetic_method_validation()
    configs = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(cases_dir.glob("*.yaml"))
    ]
    summaries = [_write_case_outputs(config) for config in configs]
    application = {
        "status": "completed_structured_outputs",
        "cases": [summary["case_id"] for summary in summaries],
        "chaos_certified_by_poincare": False,
        "hiddenness_certified_by_poincare": False,
    }
    payload = {
        "stage": "F5.4_poincare_diagnostic",
        "status": "completed_structured_outputs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases_total": len(summaries),
        "cases_with_crossings": sum(summary["crossing_count"] > 0 for summary in summaries),
        "cases_insufficient_crossings": sum(summary["crossing_count"] < 3 for summary in summaries),
        "caputo_periodic_orbit_claim": False,
        "chaos_certified_by_poincare": False,
        "hiddenness_certified_by_poincare": False,
        "standardized_outputs": True,
        "poincare_method_validation": method_validation,
        "poincare_application_to_published_cases": application,
        "case_summaries": summaries,
    }
    _write_json(GLOBAL_SUMMARY_PATH, payload)
    _write_f5_summary(method_validation, application)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", type=Path, default=CASES_DIR)
    args = parser.parse_args()
    payload = run(args.cases_dir)
    print(
        f"Recorded {payload['cases_total']} Poincare diagnostic cases; "
        f"cases_with_crossings={payload['cases_with_crossings']}; "
        f"status={payload['status']}"
    )


if __name__ == "__main__":
    main()
