#!/usr/bin/env python3
"""Shared trajectory cache for F5 dynamics diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from validation.python.run_poincare_diagnostics import CASES_DIR, _burn_time, _integrate_case  # noqa: E402


TRAJECTORY_ROOT = (
    PROJECT_ROOT / "validation" / "chaos_validation" / "dynamics_diagnostics" / "trajectories"
)
CASES = {
    path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
    for path in sorted(CASES_DIR.glob("*.yaml"))
}


@dataclass(frozen=True)
class F5Trajectory:
    """Cached post-transient trajectory and its numerical provenance."""

    case_id: str
    trajectory_id: str
    times: np.ndarray
    states: np.ndarray
    metadata: dict[str, Any]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _seed_for(config: dict[str, Any], trajectory_id: str) -> list[float]:
    seed = config["seed"]
    if trajectory_id in seed:
        return [float(value) for value in seed[trajectory_id]]
    raise KeyError(f"seed for trajectory_id={trajectory_id!r} is not configured.")


def _paths(case_id: str, trajectory_id: str) -> tuple[Path, Path, Path]:
    case_dir = TRAJECTORY_ROOT / case_id
    return (
        case_dir / f"{trajectory_id}_post_transient.npz",
        case_dir / f"{trajectory_id}_metadata.json",
        case_dir / f"{trajectory_id}_post_transient_sampled.csv",
    )


def _hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def _sample_rows(trajectory: np.ndarray, maximum: int = 5000) -> np.ndarray:
    if trajectory.shape[0] <= maximum:
        return trajectory
    indices = np.linspace(0, trajectory.shape[0] - 1, maximum, dtype=int)
    return trajectory[indices]


def _write_sampled_csv(path: Path, trajectory: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "x", "y", "z"])
        writer.writerows(trajectory)


def _save(
    config: dict[str, Any],
    trajectory_id: str,
    trajectory: np.ndarray,
    integration_metadata: dict[str, Any],
) -> F5Trajectory:
    case_id = config["case_id"]
    burn_time = _burn_time(config)
    post = np.asarray(trajectory, dtype=float)
    post = post[post[:, 0] >= burn_time]
    npz_path, metadata_path, sampled_path = _paths(case_id, trajectory_id)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, trajectory=post)
    sampled = _sample_rows(post)
    _write_sampled_csv(sampled_path, sampled)
    integration = config["integration"]
    metadata = {
        "case_id": case_id,
        "trajectory_id": trajectory_id,
        "q": float(config["q"]),
        "derivative_model": config["derivative_model"],
        "h": float(integration["h"]),
        "t_final": float(integration["t_final"]),
        "t_burn": float(burn_time),
        "integrator": integration.get("integrator", integration.get("backend")),
        "backend": integration.get("backend"),
        "memory_policy": integration.get("memory_policy", "not_applicable"),
        "memory_length": integration.get("memory_length"),
        "seed": _seed_for(config, trajectory_id),
        "seed_scope": integration_metadata.get("seed_scope", "configured_reference_seed"),
        "source_reference": config["source"],
        "trajectory_rows": int(trajectory.shape[0]),
        "post_transient_rows": int(post.shape[0]),
        "sampled_rows": int(sampled.shape[0]),
        "nonfinite_count": int(np.size(post) - np.count_nonzero(np.isfinite(post))),
        "post_transient_sha256": _hash_array(post),
        "post_transient_cache": npz_path.relative_to(PROJECT_ROOT).as_posix(),
        "post_transient_sampled_csv": sampled_path.relative_to(PROJECT_ROOT).as_posix(),
        **integration_metadata,
    }
    _write_json(metadata_path, metadata)
    return F5Trajectory(case_id, trajectory_id, post[:, 0], post[:, 1:], metadata)


def _load(case_id: str, trajectory_id: str) -> F5Trajectory | None:
    npz_path, metadata_path, _ = _paths(case_id, trajectory_id)
    if not npz_path.exists() or not metadata_path.exists():
        return None
    with np.load(npz_path) as payload:
        trajectory = np.asarray(payload["trajectory"], dtype=float)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("post_transient_sha256") != _hash_array(trajectory):
        raise RuntimeError(f"trajectory cache hash mismatch: {npz_path}")
    return F5Trajectory(case_id, trajectory_id, trajectory[:, 0], trajectory[:, 1:], metadata)


def configured_trajectory_ids(case_id: str) -> list[str]:
    """Return deterministic trajectory identifiers configured for one F5 case."""

    if case_id not in CASES:
        raise KeyError(f"unsupported F5 case: {case_id}")
    if case_id == "chua_integer_q1_reference":
        return ["X0_python"]
    if case_id == "danca2017_chua_fractional_saturation_q09998":
        return ["diagnostic_X0"]
    if case_id == "wu2023_chua_fractional_arctan_q099":
        return ["x0_plus", "x0_minus"]
    raise KeyError(f"unsupported F5 case: {case_id}")


def _generate_case(case_id: str) -> list[F5Trajectory]:
    config = CASES[case_id]
    return [
        _save(config, trajectory_id, trajectory, integration_metadata)
        for trajectory_id, trajectory, integration_metadata in _integrate_case(config)
    ]


def load_or_generate_f5_trajectory(
    case_id: str,
    trajectory_id: str | None = None,
) -> F5Trajectory | list[F5Trajectory]:
    """Load cached post-transient data or integrate once with the Poincare contract."""

    expected = configured_trajectory_ids(case_id)
    if trajectory_id is not None and trajectory_id not in expected:
        raise KeyError(f"unsupported trajectory_id={trajectory_id!r} for case_id={case_id!r}")
    requested = expected if trajectory_id is None else [trajectory_id]
    loaded = [_load(case_id, item) for item in requested]
    if any(item is None for item in loaded):
        _generate_case(case_id)
        loaded = [_load(case_id, item) for item in requested]
    results = [item for item in loaded if item is not None]
    if len(results) != len(requested):
        raise RuntimeError(f"failed to create trajectory cache for {case_id}")
    return results if trajectory_id is None else results[0]


__all__ = [
    "CASES",
    "F5Trajectory",
    "TRAJECTORY_ROOT",
    "configured_trajectory_ids",
    "load_or_generate_f5_trajectory",
]
