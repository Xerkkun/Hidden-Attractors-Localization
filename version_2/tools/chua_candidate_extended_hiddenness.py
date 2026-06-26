"""Run one dense, resumable spherical hiddenness protocol for Chua candidates.

Integer and fractional candidates use the same radii, directions, stopping
rules, target classifier, output schema, and plotting inputs.  Only the
order-appropriate integrator differs: Heun at q=1 and full-history Caputo ABM
at q<1.  The result is a finite spherical-surface test, not a filled-ball or
global basin proof.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from hidden_attractors.integrations.general import integrate_general
from hidden_attractors.models.chua import (
    chua_parameters,
    equilibria_arctan,
    jacobian_arctan,
)
from hidden_attractors.systems import get_system


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_DIR = (
    ROOT
    / "outputs"
    / "arctan_hidden_candidate_search"
    / "c590_q09999_seed9_candidate_20260623"
)
EQUILIBRIUM_ORDER = {"E0": 0, "E+": 1, "E-": 2}
_WORKER_CONTEXT: dict[str, Any] = {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _float_list(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def _downsample(states: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(states, dtype=float)
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum).astype(int)
    return values[indices]


def build_directions(total: int) -> list[dict[str, Any]]:
    """Return six axes plus a deterministic spherical-Fibonacci cloud."""
    if total < 6:
        raise ValueError("directions-per-radius must be at least 6")
    rows: list[dict[str, Any]] = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            vector = np.zeros(3, dtype=float)
            vector[axis] = sign
            rows.append(
                {
                    "direction": f"axis{axis}_{'plus' if sign > 0 else 'minus'}",
                    "direction_index": len(rows),
                    "direction_family": "axis",
                    "direction_vector": vector.tolist(),
                }
            )
    fibonacci_count = total - len(rows)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for index in range(fibonacci_count):
        z = 1.0 - 2.0 * (index + 0.5) / fibonacci_count
        radial = np.sqrt(max(0.0, 1.0 - z * z))
        azimuth = golden_angle * index
        vector = np.array(
            [radial * np.cos(azimuth), radial * np.sin(azimuth), z],
            dtype=float,
        )
        rows.append(
            {
                "direction": f"fibonacci_{index:03d}",
                "direction_index": len(rows),
                "direction_family": "fibonacci",
                "direction_vector": vector.tolist(),
            }
        )
    return rows


def build_probe_tasks(
    equilibria: dict[str, Sequence[float]],
    radii: Sequence[float],
    samples_per_radius: Sequence[int],
) -> list[dict[str, Any]]:
    if len(radii) != len(samples_per_radius):
        raise ValueError("radii and samples_per_radius must have the same length")
    tasks: list[dict[str, Any]] = []
    for equilibrium in ("E0", "E+", "E-"):
        center = np.asarray(equilibria[equilibrium], dtype=float)
        for radius_index, (radius, samples) in enumerate(
            zip(radii, samples_per_radius)
        ):
            directions = build_directions(int(samples))
            for direction in directions:
                tasks.append(
                    {
                        "probe_id": (
                            f"{equilibrium}|r{radius_index:02d}|"
                            f"d{int(direction['direction_index']):03d}"
                        ),
                        "equilibrium": equilibrium,
                        "equilibrium_state": center.tolist(),
                        "radius": float(radius),
                        **direction,
                    }
                )
    return tasks


def _sampling_plan_hash(tasks: Sequence[dict[str, Any]]) -> str:
    canonical = [
        {
            "probe_id": task["probe_id"],
            "equilibrium": task["equilibrium"],
            "radius": task["radius"],
            "direction_vector": task["direction_vector"],
        }
        for task in tasks
    ]
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _build_system(parameters: dict[str, float], q: float, model: str):
    base = get_system(f"chua-{model}")
    merged = dict(base.parameters)
    merged.update(parameters)
    merged.update(
        {
            "model": model,
            "q": float(q),
            "system_id": f"chua_{'integer' if np.isclose(q, 1.0) else 'fractional'}_{model}",
        }
    )
    return replace(base, parameters=merged)


def _init_worker(
    parameters: dict[str, float],
    model: str,
    q: float,
    h: float,
    t_final: float,
    t_burn: float,
    divergence_norm: float,
    target_cloud: np.ndarray,
    contact_threshold: float,
    equilibria: Sequence[Sequence[float]],
    equilibrium_min_time: float,
    equilibrium_tol: float,
    equilibrium_derivative_tol: float,
    equilibrium_consecutive_steps: int,
) -> None:
    integrator = "heun" if np.isclose(q, 1.0) else "abm"
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(
        {
            "system": _build_system(parameters, q, model),
            "q": float(q),
            "integrator": integrator,
            "memory_mode": "not_applicable" if np.isclose(q, 1.0) else "full",
            "h": float(h),
            "t_final": float(t_final),
            "t_burn": float(t_burn),
            "divergence_norm": float(divergence_norm),
            "target_tree": cKDTree(np.asarray(target_cloud, dtype=float)),
            "symmetric_tree": cKDTree(-np.asarray(target_cloud, dtype=float)),
            "contact_threshold": float(contact_threshold),
            "equilibria": [np.asarray(state, dtype=float) for state in equilibria],
            "early_stop_config": {
                "enabled": True,
                "divergence_enabled": False,
                "equilibrium_enabled": True,
                "equilibrium_min_time": float(equilibrium_min_time),
                "equilibrium_tol": float(equilibrium_tol),
                "equilibrium_derivative_tol": float(equilibrium_derivative_tol),
                "equilibrium_consecutive_steps": int(equilibrium_consecutive_steps),
            },
        }
    )


def _run_probe(task: dict[str, Any]) -> dict[str, Any]:
    context = _WORKER_CONTEXT
    system = context["system"]
    center = np.asarray(task["equilibrium_state"], dtype=float)
    direction = np.asarray(task["direction_vector"], dtype=float)
    initial_state = center + float(task["radius"]) * direction
    row = {
        **task,
        "initial_state": initial_state.tolist(),
    }
    try:
        times, states, status = integrate_general(
            lambda _time, state: system.evaluate(state),
            initial_state,
            q=context["q"],
            h=context["h"],
            t_final=context["t_final"],
            integrator=context["integrator"],
            memory_mode=context["memory_mode"],
            memory_window_length=None,
            system=system,
            use_c_backend=True,
            divergence_norm=context["divergence_norm"],
            early_stop_config=context["early_stop_config"],
            equilibria=context["equilibria"],
        )
        finite = bool(len(states) and np.all(np.isfinite(states)))
        tail = states[times >= context["t_burn"]] if finite else np.empty((0, 3))
        tail = _downsample(tail, 6000) if len(tail) else tail
        if len(tail):
            target_nn90 = float(
                np.percentile(context["target_tree"].query(tail, k=1)[0], 90)
            )
            symmetric_nn90 = float(
                np.percentile(context["symmetric_tree"].query(tail, k=1)[0], 90)
            )
        else:
            target_nn90 = None
            symmetric_nn90 = None
        minimum_nn90 = (
            min(target_nn90, symmetric_nn90)
            if target_nn90 is not None and symmetric_nn90 is not None
            else None
        )
        contact = bool(
            status == "ok"
            and minimum_nn90 is not None
            and minimum_nn90 <= context["contact_threshold"]
        )
        if contact:
            outcome = "TARGET"
        elif status == "converged_equilibrium_early":
            outcome = "EQUILIBRIUM"
        elif status in {"diverged", "diverged_early"}:
            outcome = "DIVERGED"
        elif status == "ok":
            outcome = "OTHER"
        else:
            outcome = "NUMERICAL_FAILURE"
        row.update(
            {
                "status": status,
                "end_time": float(times[-1]) if len(times) else 0.0,
                "max_norm": (
                    float(np.max(np.linalg.norm(states, axis=1)))
                    if len(states)
                    else float("nan")
                ),
                "range": (
                    np.ptp(tail, axis=0).tolist() if len(tail) else [0.0, 0.0, 0.0]
                ),
                "target_nn90": target_nn90,
                "symmetric_nn90": symmetric_nn90,
                "minimum_nn90": minimum_nn90,
                "contact": contact,
                "outcome": outcome,
                "finite": finite,
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": f"solver_exception:{type(exc).__name__}:{exc}",
                "end_time": 0.0,
                "max_norm": float("nan"),
                "range": [0.0, 0.0, 0.0],
                "target_nn90": None,
                "symmetric_nn90": None,
                "minimum_nn90": None,
                "contact": False,
                "outcome": "NUMERICAL_FAILURE",
                "finite": False,
            }
        )
    return row


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["probe_id"])] = row
    return rows


def _append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _write_rows_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    normalized = []
    for source in rows:
        row = dict(source)
        for field in ("equilibrium_state", "direction_vector", "initial_state", "range"):
            row[field] = json.dumps(row[field], separators=(",", ":"))
        normalized.append(row)
    fields: list[str] = []
    for row in normalized:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)


def _matignon(
    parameters: dict[str, float],
    q: float,
    equilibria: dict[str, np.ndarray],
) -> dict[str, Any]:
    model = chua_parameters(model="arctan", **parameters)
    threshold = float(q) * np.pi / 2.0
    stability: dict[str, str] = {}
    margins: dict[str, float] = {}
    for name, state in equilibria.items():
        eigenvalues = np.linalg.eigvals(jacobian_arctan(state, model))
        margin = float(np.min(np.abs(np.angle(eigenvalues)) - threshold))
        margins[name] = margin
        stability[name] = "stable" if margin > 0.0 else "unstable"
    return {
        "equilibria_count": len(equilibria),
        "stability": stability,
        "margins": margins,
        "minimum_margin": min(margins.values()),
    }


def _candidate_sources(candidate_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = candidate_dir / "publication_figure_inputs.json"
    manifest = _read_json(manifest_path)
    summary_path = candidate_dir / "candidate_summary_dense.json"
    if not summary_path.exists():
        summary_path = candidate_dir / str(manifest["candidate_summary"])
    calibration_path = candidate_dir / "hiddenness_matrix_dense.json"
    if not calibration_path.exists():
        calibration_path = candidate_dir / str(manifest["hiddenness_matrix"])
    return summary_path, calibration_path, manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_dir = args.candidate_dir.resolve()
    summary_path, calibration_path, publication_manifest = _candidate_sources(
        candidate_dir
    )
    source_summary = _read_json(summary_path)
    calibration = _read_json(calibration_path)
    parameters = {
        name: float(source_summary["parameters"][name])
        for name in ("alpha", "beta", "gamma", "a1", "a2", "rho")
    }
    q = float(source_summary["q"])
    model = chua_parameters(model="arctan", **parameters)
    equilibria = equilibria_arctan(model)
    recorded_equilibria = source_summary["hiddenness_evidence"]["equilibria"]
    for name, state in equilibria.items():
        if not np.allclose(state, recorded_equilibria[name], rtol=0.0, atol=1.0e-10):
            raise ValueError(f"equilibrium mismatch for {name}")

    target_path = candidate_dir / "target.npz"
    target_data = np.load(target_path)
    target_times = np.asarray(target_data["times"], dtype=float)
    target_states = np.asarray(target_data["states"], dtype=float)
    target_burn = float(source_summary["numerical_contract"]["target_t_burn"])
    target_cloud = _downsample(target_states[target_times >= target_burn], 6000)
    if len(target_cloud) < 100:
        raise ValueError("target post-transient cloud is too short")

    radii = _float_list(args.radii)
    if args.directions_per_radius_list:
        samples_per_radius = [int(v.strip()) for v in args.directions_per_radius_list.split(",") if v.strip()]
        if len(samples_per_radius) != len(radii):
            raise ValueError(
                f"--directions-per-radius-list has {len(samples_per_radius)} values "
                f"but --radii has {len(radii)}; lengths must match"
            )
    else:
        samples_per_radius = [args.directions_per_radius] * len(radii)
    equilibrium_lists = {name: state.tolist() for name, state in equilibria.items()}
    tasks = build_probe_tasks(equilibrium_lists, radii, samples_per_radius)
    checkpoint_path = candidate_dir / args.checkpoint_name
    completed = _load_checkpoint(checkpoint_path) if args.resume else {}
    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError(
            f"{checkpoint_path} exists; use --resume or choose another checkpoint"
        )
    pending = [task for task in tasks if task["probe_id"] not in completed]
    contact_threshold = float(calibration["contact_threshold"])

    print(
        f"extended hiddenness: planned={len(tasks)} completed={len(completed)} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    if pending:
        equilibria_states = [state.tolist() for state in equilibria.values()]
        initializer_args = (
            parameters,
            "arctan",      # model
            q,
            args.h,
            args.t_final,
            args.t_burn,
            args.divergence_norm,
            target_cloud,
            contact_threshold,
            equilibria_states,              # equilibria
            float(args.t_burn * 0.5),       # equilibrium_min_time
            1.0e-6,                         # equilibrium_tol
            1.0e-4,                         # equilibrium_derivative_tol
            50,                             # equilibrium_consecutive_steps
        )
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=initializer_args,
        ) as executor:
            futures = {executor.submit(_run_probe, task): task for task in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                completed[str(row["probe_id"])] = row
                _append_checkpoint(checkpoint_path, row)
                if index % 24 == 0 or index == len(pending):
                    contacts = sum(bool(item["contact"]) for item in completed.values())
                    print(
                        f"completed {len(completed)}/{len(tasks)}; contacts={contacts}",
                        flush=True,
                    )

    rows = sorted(
        completed.values(),
        key=lambda row: (
            EQUILIBRIUM_ORDER[str(row["equilibrium"])],
            float(row["radius"]),
            int(row["direction_index"]),
        ),
    )
    full_length_tolerance = max(1.0e-9, 0.51 * float(args.h))
    numerical_failures = [
        row
        for row in rows
        if row["status"] != "ok"
        or not bool(row["finite"])
        or abs(float(row["end_time"]) - float(args.t_final)) > full_length_tolerance
    ]
    contacts = sum(bool(row["contact"]) for row in rows)
    minimum_nn90 = min((float(row["minimum_nn90"]) for row in rows if row["minimum_nn90"] is not None), default=None)
    status = (
        "completed_extended_spherical_probe_matrix"
        if len(rows) == len(tasks) and not numerical_failures
        else "incomplete_or_numerically_failed"
    )
    matrix = {
        "schema_version": "1.1",
        "candidate_id": source_summary["candidate_id"],
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters,
        "seed": source_summary["seed"],
        "q": q,
        "h": float(args.h),
        "integrator": "ABM predictor-corrector",
        "memory_mode": "full",
        "caputo_history_accumulated": True,
        "t_final": float(args.t_final),
        "t_burn": float(args.t_burn),
        "equilibria": equilibrium_lists,
        "matignon": _matignon(parameters, q, equilibria),
        "radii": radii,
        "direction_scheme": "six_axes_plus_spherical_fibonacci",
        "axis_directions_per_equilibrium_radius": 6,
        "samples_per_radius": samples_per_radius,
        "directions_per_equilibrium_radius": samples_per_radius,
        "target_reference_points": len(target_cloud),
        "target_calibration_nn90": float(calibration["target_calibration_nn90"]),
        "contact_threshold": contact_threshold,
        "contact_metric": "minimum target/symmetric 90th-percentile nearest-neighbor distance",
        "tests": len(rows),
        "planned_tests": len(tasks),
        "contacts": contacts,
        "numerical_failures": len(numerical_failures),
        "all_equilibria_tested": all(
            any(row["equilibrium"] == name for row in rows)
            for name in ("E0", "E+", "E-")
        ),
        "minimum_observed_target_or_symmetric_nn90": minimum_nn90,
        "scientific_boundary": (
            "Finite deterministic sampling of spherical surfaces only; this is "
            "not a filled-ball or global basin proof."
        ),
        "rows": rows,
    }
    matrix_path = candidate_dir / args.matrix_name
    rows_path = candidate_dir / args.rows_name
    _write_json(matrix_path, matrix)
    _write_rows_csv(rows_path, rows)

    summary = deepcopy(source_summary)
    summary["status"] = (
        "hiddenness_compatible_chaos_candidate_exploratory"
        if status.startswith("completed") and contacts == 0
        else "hiddenness_test_requires_review"
    )
    summary["hiddenness_evidence"] = {
        "tests": len(rows),
        "contacts": contacts,
        "radii": radii,
        "axis_directions_per_equilibrium_radius": 6,
        "samples_per_radius": samples_per_radius,
        "directions_per_equilibrium_radius": samples_per_radius,
        "all_equilibria_tested": matrix["all_equilibria_tested"],
        "equilibria": equilibrium_lists,
        "matignon": matrix["matignon"],
        "target_calibration_nn90": matrix["target_calibration_nn90"],
        "contact_threshold": contact_threshold,
        "minimum_observed_target_or_symmetric_nn90": minimum_nn90,
        "numerical_failures": len(numerical_failures),
        "verdict": (
            "consistent_with_hiddenness_under_finite_extended_surface_test_contract"
            if status.startswith("completed") and contacts == 0
            else "requires_review"
        ),
    }
    summary["scientific_boundary"] = matrix["scientific_boundary"]
    summary_path_out = candidate_dir / args.summary_name
    _write_json(summary_path_out, summary)
    run_config = {
        "schema_version": "1.0",
        "candidate_dir": candidate_dir.relative_to(ROOT).as_posix(),
        "matrix": matrix_path.name,
        "summary": summary_path_out.name,
        "rows": rows_path.name,
        "checkpoint": checkpoint_path.name,
        "workers": int(args.workers),
        "radii": radii,
        "samples_per_radius": samples_per_radius,
        "tests": len(tasks),
        "integrator": "abm",
        "memory_mode": "full",
        "h": float(args.h),
        "t_final": float(args.t_final),
        "t_burn": float(args.t_burn),
    }
    _write_json(candidate_dir / args.config_name, run_config)

    if args.promote:
        if status != "completed_extended_spherical_probe_matrix":
            raise RuntimeError("refusing to promote an incomplete hiddenness matrix")
        publication_manifest["candidate_summary"] = summary_path_out.name
        publication_manifest["hiddenness_matrix"] = matrix_path.name
        publication_manifest["extended_hiddenness_config"] = args.config_name
        _write_json(
            candidate_dir / "publication_figure_inputs.json",
            publication_manifest,
        )

    result = {
        "status": status,
        "tests": len(rows),
        "contacts": contacts,
        "numerical_failures": len(numerical_failures),
        "minimum_nn90": minimum_nn90,
        "matrix": matrix_path.name,
        "summary": summary_path_out.name,
        "promoted": bool(args.promote),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument(
        "--radii",
        default="1e-5,3e-5,1e-4,3e-4,1e-3,1e-2",
    )
    parser.add_argument("--directions-per-radius", type=int, default=96)
    parser.add_argument(
        "--directions-per-radius-list",
        default="",
        help="Comma-separated per-radius direction counts (overrides --directions-per-radius).",
    )
    parser.add_argument("--h", type=float, default=0.005)
    parser.add_argument("--t-final", type=float, default=180.0)
    parser.add_argument("--t-burn", type=float, default=90.0)
    parser.add_argument("--divergence-norm", type=float, default=120.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(12, max(1, os.cpu_count() or 1)),
    )
    parser.add_argument("--checkpoint-name", default="extended_hiddenness_checkpoint.jsonl")
    parser.add_argument("--matrix-name", default="hiddenness_matrix_extended.json")
    parser.add_argument("--summary-name", default="candidate_summary_extended.json")
    parser.add_argument("--rows-name", default="hiddenness_extended_rows.csv")
    parser.add_argument("--config-name", default="extended_hiddenness_run_config.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--promote", action="store_true")
    return parser


def main() -> None:
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
