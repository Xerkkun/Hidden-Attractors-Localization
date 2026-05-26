"""Shared infrastructure for the isolated non-smooth Chua memory matrix.

The experiment scripts in this directory orchestrate maintained
``hidden_attractors`` APIs.  They deliberately do not alter the validated
workflow or implement an unreviewed fractional integrator.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterable, Sequence


# Each Python worker owns an independent numerical task.  Native libraries and
# numerical BLAS/OpenMP runtimes must therefore stay sequential inside it.
VERSION2_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_CACHE = VERSION2_ROOT / ".runtime_cache" / "chua_nonsmooth_memory_matrix"
_RUNTIME_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_RUNTIME_CACHE / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(_RUNTIME_CACHE / "xdg")
for _thread_env in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_env] = "1"

if str(VERSION2_ROOT) not in sys.path:
    sys.path.insert(0, str(VERSION2_ROOT))

import numpy as np

from hidden_attractors.io import json_safe, read_json, write_csv, write_json
from hidden_attractors.seed_generation import HarmonicSeed


THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MPLCONFIGDIR": str(_RUNTIME_CACHE / "matplotlib"),
    "XDG_CACHE_HOME": str(_RUNTIME_CACHE / "xdg"),
}
CLASS_LABELS = (
    "target_candidate_plus",
    "target_candidate_minus",
    "equilibrium_E0",
    "equilibrium_Eplus",
    "equilibrium_Eminus",
    "infinity",
    "bounded_other",
    "periodic_or_quasiperiodic",
    "numerical_failure",
)


def configure_worker_environment() -> None:
    """Enforce one native thread per independent Python worker."""

    os.environ.update(THREAD_ENV)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read task or result rows while preserving manifest string values."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def matrix_root_from_tasks(tasks_path: str | Path) -> Path:
    """Resolve the output root from a task CSV, not from a foreign absolute path."""

    task_file = Path(tasks_path).expanduser().resolve()
    if task_file.parent.name != "tasks":
        raise ValueError("tasks CSV must be inside <matrix_root>/tasks/.")
    return task_file.parent.parent


def load_matrix(tasks_path: str | Path) -> tuple[Path, dict[str, Any], list[dict[str, str]]]:
    """Return ``(root, manifest, tasks)`` for one generated task table."""

    root = matrix_root_from_tasks(tasks_path)
    return root, read_json(root / "experiment_matrix.json"), read_csv_rows(tasks_path)


def git_commit() -> str:
    """Return the repository commit used for result provenance."""

    try:
        result = subprocess.run(
            ["git", "-C", str(VERSION2_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def optional_float(value: Any) -> float | None:
    """Convert a manifest cell to a finite float, treating blanks as null."""

    if value in {"", None, "None", "null"}:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def metadata(
    manifest: dict[str, Any],
    row: dict[str, Any],
    *,
    stage: str,
    q: float | None,
    integrator: str,
    memory_policy: str,
    workers: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build provenance required for every independently generated result."""

    contract = dict(manifest["contract"])
    result = {
        "stage": stage,
        "git_commit": git_commit(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "numerical_contract": contract,
        "q": q,
        "h": contract.get("h"),
        "t_final": contract.get("t_final"),
        "t_burn": contract.get("t_burn"),
        "Lm": optional_float(row.get("memory_length", contract.get("memory_length"))),
        "integrator": integrator,
        "memory_policy": memory_policy,
        "exp_id": row.get("exp_id", "shared"),
        "cache_key": row.get("cache_key", ""),
        "parallel_contract": {"python_workers": max(1, int(workers)), **THREAD_ENV},
    }
    if extra:
        result.update(extra)
    return result


def shared_status_path(root: Path, cache_name: str) -> Path:
    """Return the non-colliding status path of a shared cache task."""

    return root / "shared" / "status" / str(cache_name) / "status.json"


def is_ok_status(path: Path, outputs: Sequence[Path] | None = None) -> bool:
    """Return whether a prior task completed and all required outputs remain."""

    try:
        if not path.exists() or read_json(path).get("status") != "ok":
            return False
        return outputs is None or all(output.exists() for output in outputs)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def write_status(path: Path, *, status: str, meta: dict[str, Any], outputs: Sequence[str], reason: str = "") -> None:
    """Write a task status envelope used for idempotent resume behavior."""

    write_json(
        path,
        {
            "status": status,
            "reason": reason,
            "metadata": meta,
            "outputs": list(outputs),
        },
    )


def assert_unique_cache_keys(rows: Sequence[dict[str, str]]) -> None:
    """Refuse a parallel launch whose tasks can collide on one cache key."""

    keys = [str(row.get("cache_key", "")) for row in rows]
    duplicates = sorted({key for key in keys if key and keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"task table contains repeated cache_key values: {duplicates}")


def run_process_pool(
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    jobs: Sequence[dict[str, Any]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    """Execute independent tasks in processes while keeping serial mode simple."""

    configure_worker_environment()
    count = max(1, int(workers))
    if count == 1:
        return [worker(job) for job in jobs]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=count, initializer=configure_worker_environment) as executor:
        return list(executor.map(worker, jobs))


def write_trajectory(path: str | Path, trajectory: np.ndarray, *, max_rows: int | None = None) -> None:
    """Store a ``t,x,y,z`` trajectory, optionally evenly subsampled."""

    values = np.asarray(trajectory, dtype=float)
    if max_rows is not None and values.shape[0] > int(max_rows):
        index = np.linspace(0, values.shape[0] - 1, int(max_rows)).astype(int)
        values = values[index]
    rows = [{"t": row[0], "x": row[1], "y": row[2], "z": row[3]} for row in values]
    write_csv(path, rows, ["t", "x", "y", "z"])


def load_trajectory(path: str | Path) -> np.ndarray:
    """Load one trajectory written by :func:`write_trajectory`."""

    rows = read_csv_rows(path)
    return np.asarray([[float(row[key]) for key in ("t", "x", "y", "z")] for row in rows], dtype=float)


def seed_payload(seed: HarmonicSeed, *, family: str, transfer_power: str, q: float, meta: dict[str, Any]) -> dict[str, Any]:
    """Serialize a classical describing-function seed without Machado fields."""

    return {
        "seed_family": family,
        "transfer_power": transfer_power,
        "df_family": "classic",
        "periodic_interpretation": "Weyl/Liouville-Weyl first-harmonic asymptotic seed for subsequent Caputo validation",
        "q_transfer": float(q),
        "omega": float(seed.omega),
        "gain": float(seed.gain),
        "amplitude": float(seed.amplitude),
        "branch_index": int(seed.branch_index),
        "method": "classic",
        "seed": np.asarray(seed.seed, dtype=float),
        "eigenvector": [[float(value.real), float(value.imag)] for value in np.asarray(seed.eigenvector, dtype=complex)],
        "matched_eigenvalue": complex(seed.matched_eigenvalue),
        "metadata": meta,
    }


def harmonic_seed_from_payload(data: dict[str, Any]) -> HarmonicSeed:
    """Reconstruct the package seed dataclass from a stored JSON artifact."""

    vector = np.asarray([complex(*value) for value in data["eigenvector"]], dtype=complex)
    eigenvalue = complex(*data["matched_eigenvalue"])
    return HarmonicSeed(
        seed=np.asarray(data["seed"], dtype=float),
        eigenvector=vector,
        matched_eigenvalue=eigenvalue,
        omega=float(data["omega"]),
        gain=float(data["gain"]),
        amplitude=float(data["amplitude"]),
        branch_index=int(data["branch_index"]),
        method="classic",
        mu=None,
    )


def json_result(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON result using the package serializer for NumPy values."""

    write_json(path, json_safe(payload))


def experiment_spec(manifest: dict[str, Any], exp_id: str) -> dict[str, Any]:
    """Find one experiment specification by its stable experiment identifier."""

    for item in manifest["experiments"]:
        if item["exp_id"] == exp_id:
            return dict(item)
    raise KeyError(f"unknown exp_id {exp_id}")


def full_history_horizon(total_time: float, h: float) -> float:
    """Choose a finite backend storage horizon that cannot truncate ``[0,T]``."""

    return float(total_time) + float(h)


def status_counts(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    """Count categorical outcomes for JSON reports and figures."""

    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(field, ""))
        counts[label] = counts.get(label, 0) + 1
    return counts
