#!/usr/bin/env python3
"""Generate reusable algebra, classical-DF seeds, Nyquist plots, and test balls."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from _common import (
    VERSION2_ROOT,
    json_result,
    load_matrix,
    metadata,
    run_process_pool,
    seed_payload,
    shared_status_path,
    is_ok_status,
    assert_unique_cache_keys,
    write_status,
)

from hidden_attractors.io import write_csv
from hidden_attractors.models import chua_nonsmooth_parameters, equilibria_nonsmooth, jacobian_nonsmooth, rhs_nonsmooth
from hidden_attractors.plotting import plot_lure_nyquist_describing_function
from hidden_attractors.seed_generation import find_lure_harmonic_seed
from hidden_attractors.systems import get_system
from hidden_attractors.workflows.integer_lure import integer_lure_seed
from hidden_attractors.workflows.protocol import sample_uniform_ball


def _validate_default_contract(manifest: dict[str, Any]) -> None:
    """Ensure package built-ins represent the generated non-smooth contract."""

    contract = manifest["contract"]
    params = chua_nonsmooth_parameters()
    for field in ("alpha", "beta", "gamma", "m0", "m1"):
        if not np.isclose(float(contract[field]), float(getattr(params, field))):
            raise ValueError(f"built-in chua-nonsmooth API does not match contract override for {field}.")


def _algebra(root: Path, manifest: dict[str, Any], row: dict[str, str], workers: int) -> list[str]:
    """Calculate equilibria, regional Jacobians, Matignon margins, and Lur'e split."""

    q = float(manifest["contract"]["q_target"])
    params = chua_nonsmooth_parameters()
    system = get_system("chua-nonsmooth")
    equilibria = equilibria_nonsmooth(params)
    matrices = {name: jacobian_nonsmooth(state, params) for name, state in equilibria.items()}
    algebra_meta = metadata(manifest, row, stage="shared_algebra", q=q, integrator="not_applicable", memory_policy="not_applicable", workers=workers)
    q_boundary = q * np.pi / 2.0
    eigen_records: dict[str, Any] = {}
    for name, matrix in matrices.items():
        eig = np.linalg.eigvals(matrix)
        margins = np.abs(np.angle(eig)) - q_boundary
        eigen_records[name] = {
            "eigenvalues": eig,
            "matignon_boundary_rad": q_boundary,
            "margins_rad": margins,
            "locally_asymptotically_stable": bool(np.all(margins > 0.0)),
        }
    outputs = [
        "shared/algebra/equilibria.json",
        "shared/algebra/jacobians.json",
        "shared/algebra/eigenvalues_matignon.json",
        "shared/algebra/lure_split.json",
    ]
    json_result(root / outputs[0], {"equilibria": equilibria, "residual_norms": {key: float(np.linalg.norm(rhs_nonsmooth(value, params))) for key, value in equilibria.items()}, "metadata": algebra_meta})
    json_result(root / outputs[1], {"jacobians": matrices, "metadata": algebra_meta})
    json_result(root / outputs[2], {"criterion": "Matignon: abs(arg(lambda)) > q*pi/2", "q": q, "records": eigen_records, "metadata": algebra_meta})
    lure = system.lure
    if lure is None:
        raise ValueError("chua-nonsmooth does not expose its required Lur'e split.")
    json_result(
        root / outputs[3],
        {
            "equation": "D^q x = P x + b psi(r^T x)",
            "P": lure.matrix,
            "b": lure.input_vector,
            "r": lure.output_vector,
            "nonlinearity": "psi(sigma)=(m0-m1)*clip(sigma,-1,1)",
            "metadata": algebra_meta,
        },
    )
    return outputs


def _seed(root: Path, manifest: dict[str, Any], row: dict[str, str], workers: int, *, fractional: bool) -> list[str]:
    """Build a centered classical DF seed through the maintained Lur'e API."""

    system = get_system("chua-nonsmooth")
    if system.lure is None:
        raise ValueError("the Chua experiment requires a Lur'e representation.")
    q = float(manifest["contract"]["q_target"]) if fractional else 1.0
    family = "fractional" if fractional else "integer_like"
    transfer = "s^q" if fractional else "s"
    seed = (
        find_lure_harmonic_seed(q=q, system=system.lure, method="classic", branch_index=0)
        if fractional
        else integer_lure_seed(system, method="classic", branch_index=0)
    )
    meta = metadata(manifest, row, stage="shared_seed", q=q, integrator="describing_function", memory_policy="Weyl_seed_not_time_integration", workers=workers)
    output_json = f"shared/seeds/{family}_seed.json"
    output_png = f"figures/shared/nyquist_{family}.png"
    json_result(root / output_json, seed_payload(seed, family=family, transfer_power=transfer, q=q, meta=meta))
    plot_lure_nyquist_describing_function(
        system.lure,
        seed,
        root / output_png,
        q=q,
        method="classic",
        title=f"Classical DF closure: {transfer}",
    )
    return [output_json, output_png]


def _clouds(root: Path, manifest: dict[str, Any], row: dict[str, str], workers: int) -> list[str]:
    """Create identical reproducible equilibrium-centered balls for all tests."""

    contract = manifest["contract"]
    q = float(contract["q_target"])
    eqs = equilibria_nonsmooth(chua_nonsmooth_parameters())
    rng = np.random.default_rng(20260525)
    meta = metadata(manifest, row, stage="shared_equilibrium_clouds", q=q, integrator="not_applicable", memory_policy="initial_condition_sampling", workers=workers)
    name_map = {"E0": "E0", "E+": "Eplus", "E-": "Eminus"}
    outputs: list[str] = []
    for eq_id, center in eqs.items():
        result_rows: list[dict[str, Any]] = []
        for radius in contract["radii"]:
            points = sample_uniform_ball(center, float(radius), int(contract["samples_per_radius"]), rng)
            for sample_id, point in enumerate(points):
                result_rows.append(
                    {
                        "equilibrium": eq_id,
                        "radius": float(radius),
                        "sample_id": sample_id,
                        "x0": float(point[0]),
                        "y0": float(point[1]),
                        "z0": float(point[2]),
                        "distance_from_equilibrium": float(np.linalg.norm(point - center)),
                        "cache_key": row["cache_key"],
                    }
                )
        path = f"shared/equilibrium_clouds/{name_map[eq_id]}.csv"
        write_csv(root / path, result_rows)
        outputs.append(path)
    json_result(root / "shared/equilibrium_clouds/metadata.json", {"metadata": meta, "files": outputs})
    return outputs + ["shared/equilibrium_clouds/metadata.json"]


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    """Execute one cache task, with idempotent completion through status files."""

    root = Path(job["root"])
    manifest = job["manifest"]
    row = job["row"]
    workers = int(job["workers"])
    status_path = shared_status_path(root, row["cache_name"])
    required = [root / path for path in row["outputs"].split(";")]
    if not bool(job["force"]) and is_ok_status(status_path, required):
        return {"task_id": row["task_id"], "status": "skipped_ok"}
    try:
        _validate_default_contract(manifest)
        if row["cache_name"] == "algebra":
            outputs = _algebra(root, manifest, row, workers)
        elif row["cache_name"] == "seed_integer_like":
            outputs = _seed(root, manifest, row, workers, fractional=False)
        elif row["cache_name"] == "seed_fractional":
            outputs = _seed(root, manifest, row, workers, fractional=True)
        elif row["cache_name"] == "equilibrium_clouds":
            outputs = _clouds(root, manifest, row, workers)
        else:
            raise ValueError(f"unknown cache task {row['cache_name']}")
        meta = metadata(manifest, row, stage="shared_cache", q=float(manifest["contract"]["q_target"]), integrator="mixed_shared_stage", memory_policy="not_applicable", workers=workers)
        write_status(status_path, status="ok", meta=meta, outputs=outputs)
        return {"task_id": row["task_id"], "status": "ok"}
    except Exception as exc:
        meta = metadata(manifest, row, stage="shared_cache", q=float(manifest["contract"]["q_target"]), integrator="mixed_shared_stage", memory_policy="not_applicable", workers=workers)
        write_status(status_path, status="failed", meta=meta, outputs=[], reason=str(exc))
        return {"task_id": row["task_id"], "status": "failed", "reason": str(exc)}


def main() -> None:
    """Run selected shared-cache tasks from the generated matrix CSV."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=str(VERSION2_ROOT / "outputs/chua_nonsmooth_fractional_memory_matrix/tasks/shared_cache_tasks.csv"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root, manifest, rows = load_matrix(args.tasks)
    assert_unique_cache_keys(rows)
    jobs = [{"root": str(root), "manifest": manifest, "row": row, "workers": args.workers, "force": args.force} for row in rows]
    results = run_process_pool(run_one, jobs, workers=args.workers)
    for result in results:
        print(f"{result['task_id']}: {result['status']}")


if __name__ == "__main__":
    main()
