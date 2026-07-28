#!/usr/bin/env python3
"""Reejecución completa del caso no suave usado en el artículo 07.

El programa corrige la mezcla previa entre una trayectoria abreviada y el
contrato nominal. Reconstruye la raíz de función descriptiva sesgada, ejecuta
la continuación afín de Caputo con memoria completa, genera una trayectoria
de referencia de 300 unidades, realiza las 675 sondas esféricas declaradas y,
cuando se solicita, recorre un plan volumétrico de hasta 26,400 sondas. El
recorrido termina después de completar el primer radio con contactos.

La salida contiene únicamente datos teórico-numéricos del experimento:
parámetros, condiciones iniciales, calendario de continuación, puntos
sondeados, decisiones y figuras. No registra información de Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve()
VERSION2 = HERE.parents[1]
ROOT = VERSION2.parent
if str(VERSION2) not in sys.path:
    sys.path.insert(0, str(VERSION2))

from hidden_attractors.diagnostics.periodicity import (  # noqa: E402
    classify_post_transient_periodicity,
)
from hidden_attractors.integrations.fractional_c import (  # noqa: E402
    fractional_integrate,
)
from hidden_attractors.models.chua import chua_parameters  # noqa: E402
from hidden_attractors.systems import get_system  # noqa: E402
from hidden_attractors.verification.equilibria import solve_equilibria  # noqa: E402
from hidden_attractors.verification.hiddenness import (  # noqa: E402
    evaluate_target_match,
    generate_neighborhood_points,
)
from hidden_attractors.workflows.biased_chua import (  # noqa: E402
    biased_saturation_df,
    build_biased_seed,
    find_biased_branches,
    run_affine_continuation,
    sample_ball,
)


CONFIG_PATH = VERSION2 / "configs" / "examples" / "chua_nonsmooth_biased_df_search.yaml"
DEFAULT_OUTPUT = ROOT / "outputs" / "paper07_nonsmooth_corrected"

_WORKER_SYSTEM: Any = None
_WORKER_EQS: list[np.ndarray] = []
_WORKER_REF: np.ndarray | None = None
_WORKER_Q = 0.0
_WORKER_H = 0.0
_WORKER_T_FINAL = 0.0
_WORKER_T_BURN = 0.0
_WORKER_EQ_TOL = 0.0
_WORKER_MATCH_METRIC = ""
_WORKER_MATCH_TOL = 0.0
_WORKER_MATCH_PERCENTILE = 90.0

_EXT_SYSTEM: Any = None
_EXT_EQS: list[np.ndarray] = []
_EXT_REF_TREE: cKDTree | None = None
_EXT_Q = 0.0
_EXT_H = 0.0
_EXT_T_FINAL = 0.0
_EXT_T_BURN = 0.0
_EXT_EQ_TOL = 0.0
_EXT_MATCH_TOL = 0.0
_EXT_MATCH_PERCENTILE = 90.0
_EXT_HARD_DIVERGENCE_NORM = 120.0
_EXT_EARLY_STOP: dict[str, Any] = {}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, np.integer):
        return value.item()
    return value


def _sha256_payload(payload: Any) -> str:
    canonical = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _init_probe_worker(
    system_parameters: dict[str, float],
    equilibria: list[np.ndarray],
    reference_tail: np.ndarray,
    q: float,
    h: float,
    t_final: float,
    t_burn: float,
    equilibrium_tol: float,
    match_metric: str,
    match_tol: float,
    match_percentile: float,
) -> None:
    global _WORKER_SYSTEM, _WORKER_EQS, _WORKER_REF
    global _WORKER_Q, _WORKER_H, _WORKER_T_FINAL, _WORKER_T_BURN
    global _WORKER_EQ_TOL, _WORKER_MATCH_METRIC, _WORKER_MATCH_TOL
    global _WORKER_MATCH_PERCENTILE

    _WORKER_SYSTEM = get_system("chua-nonsmooth")
    _WORKER_SYSTEM.parameters.update(system_parameters)
    _WORKER_EQS = [np.asarray(e, dtype=float) for e in equilibria]
    _WORKER_REF = np.asarray(reference_tail, dtype=float)
    _WORKER_Q = float(q)
    _WORKER_H = float(h)
    _WORKER_T_FINAL = float(t_final)
    _WORKER_T_BURN = float(t_burn)
    _WORKER_EQ_TOL = float(equilibrium_tol)
    _WORKER_MATCH_METRIC = str(match_metric)
    _WORKER_MATCH_TOL = float(match_tol)
    _WORKER_MATCH_PERCENTILE = float(match_percentile)


def _run_one_probe(payload: tuple[int, np.ndarray]) -> dict[str, Any]:
    sample_index, x0 = payload
    x0 = np.asarray(x0, dtype=float)
    t_start = time.perf_counter()

    try:
        _, states, status, _ = fractional_integrate(
            rhs=lambda t, x: _WORKER_SYSTEM.rhs(x, _WORKER_SYSTEM.parameters),
            x0=x0,
            q=_WORKER_Q,
            h=_WORKER_H,
            t_final=_WORKER_T_FINAL,
            method="abm",
            memory_mode="full",
            system=_WORKER_SYSTEM,
            use_c_backend=True,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "sample_index": sample_index,
            "destination": "numerical_failure",
            "status": f"exception:{exc}",
            "elapsed_s": time.perf_counter() - t_start,
        }

    if status in ("diverged", "diverged_early", "nonfinite_solution") or len(states) == 0:
        return {
            "sample_index": sample_index,
            "destination": "divergence",
            "status": status,
            "elapsed_s": time.perf_counter() - t_start,
        }

    n_burn = int(np.ceil(_WORKER_T_BURN / _WORKER_H))
    tail = states[n_burn:] if len(states) > n_burn else states
    final = states[-1]
    eq_distances = [float(np.linalg.norm(final - eq)) for eq in _WORKER_EQS]

    if min(eq_distances) <= _WORKER_EQ_TOL:
        destination = "stable_equilibrium"
    elif evaluate_target_match(
        tail,
        _WORKER_REF,
        metric=_WORKER_MATCH_METRIC,
        tolerance=_WORKER_MATCH_TOL,
        nn_percentile=_WORKER_MATCH_PERCENTILE,
    ):
        destination = "target_attractor"
    else:
        destination = "other_attractor"

    return {
        "sample_index": sample_index,
        "destination": destination,
        "status": "ok",
        "final_x": float(final[0]),
        "final_y": float(final[1]),
        "final_z": float(final[2]),
        "min_final_equilibrium_distance": min(eq_distances),
        "tail_points": int(len(tail)),
        "elapsed_s": time.perf_counter() - t_start,
    }


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _select_positive_biased_root(cfg: dict[str, Any]) -> tuple[Any, dict[str, float]]:
    sys_cfg = cfg["system"]
    params = chua_parameters(
        model="nonsmooth",
        alpha=float(sys_cfg["parameters"]["alpha"]),
        beta=float(sys_cfg["parameters"]["beta"]),
        gamma=float(sys_cfg["parameters"]["gamma"]),
        m0=float(sys_cfg["parameters"]["m0"]),
        m1=float(sys_cfg["parameters"]["m1"]),
    )
    roots = find_biased_branches(
        params,
        float(sys_cfg["q"]),
        cfg["step2_biased_df_search"],
    )
    positive = [r for r in roots if float(r["c"]) > 0.05]
    if not positive:
        raise RuntimeError("No se encontró la raíz sesgada positiva esperada.")
    root = min(positive, key=lambda r: float(r["residual_norm"]))
    return params, root


def _build_reference(
    cfg: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    sys_cfg = cfg["system"]
    int_cfg = cfg["integrator"]
    search_cfg = cfg["step2_biased_df_search"]
    q = float(sys_cfg["q"])
    h = float(int_cfg["h"])

    params, root = _select_positive_biased_root(cfg)
    gain = float(params.m0 - params.m1)
    psi0, n1 = biased_saturation_df(
        float(root["A"]),
        float(root["c"]),
        gain,
        int(search_cfg["n_theta"]),
    )
    seed_data = build_biased_seed(
        params,
        q,
        float(root["A"]),
        float(root["c"]),
        float(root["omega"]),
        psi0,
        n1,
    )
    seed = np.asarray(seed_data["seed"], dtype=float)

    eta_values = np.arange(
        0.0,
        1.0 + 0.5 * float(search_cfg["eta_step"]),
        float(search_cfg["eta_step"]),
    ).tolist()
    continuation = run_affine_continuation(
        params=params,
        q=q,
        h=h,
        seed_x0=seed,
        A=float(root["A"]),
        c=float(root["c"]),
        psi0=psi0,
        N1=n1,
        lambda_values=eta_values,
        t_transient=float(search_cfg["t_transient"]),
        t_keep=float(search_cfg["t_keep"]),
        div_threshold=float(search_cfg["div_threshold"]),
    )
    if len(continuation) != len(eta_values) or any(
        step["status"] != "ok" for step in continuation
    ):
        raise RuntimeError("La continuación no alcanzó eta=1 bajo el contrato nominal.")

    continuation_rows = []
    continuation_trajectory_dir = output_dir / "continuation_stage_trajectories"
    continuation_trajectory_dir.mkdir(parents=True, exist_ok=True)
    for idx, step in enumerate(continuation):
        trajectory = np.asarray(step["trajectory"], dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != 4:
            raise RuntimeError(
                f"Trayectoria inválida en la etapa {idx}: {trajectory.shape}."
            )
        trajectory_path = (
            continuation_trajectory_dir
            / f"continuation_eta_{idx:03d}.csv"
        )
        pd.DataFrame(
            trajectory,
            columns=["t", "x", "y", "z"],
        ).to_csv(trajectory_path, index=False)
        continuation_rows.append(
            {
                "stage": idx,
                "eta": float(step["lambda_value"]),
                "status": step["status"],
                "trajectory_file": trajectory_path.relative_to(output_dir).as_posix(),
                "trajectory_rows": int(trajectory.shape[0]),
                "x_in": float(step["x_in"][0]),
                "y_in": float(step["x_in"][1]),
                "z_in": float(step["x_in"][2]),
                "x_out": float(step["x_out"][0]),
                "y_out": float(step["x_out"][1]),
                "z_out": float(step["x_out"][2]),
                "x_out_norm": float(step["x_out_norm"]),
            }
        )
    pd.DataFrame(continuation_rows).to_csv(
        output_dir / "continuation_stages.csv",
        index=False,
    )

    system = get_system("chua-nonsmooth")
    system_parameters = {
        "alpha": float(sys_cfg["parameters"]["alpha"]),
        "beta": float(sys_cfg["parameters"]["beta"]),
        "gamma": float(sys_cfg["parameters"]["gamma"]),
        "m0": float(sys_cfg["parameters"]["m0"]),
        "m1": float(sys_cfg["parameters"]["m1"]),
    }
    system.parameters.update(system_parameters)
    x_final = np.asarray(continuation[-1]["x_out"], dtype=float)
    sim_t, sim_x, sim_status, _ = fractional_integrate(
        rhs=lambda t, x: system.rhs(x, system.parameters),
        x0=x_final,
        q=q,
        h=h,
        t_final=float(search_cfg["t_sim_final"]),
        method="abm",
        memory_mode="full",
        system=system,
        use_c_backend=True,
    )
    if sim_status != "ok":
        raise RuntimeError(f"La trayectoria de referencia terminó con {sim_status}.")

    trajectory = np.column_stack((sim_t, sim_x))
    pd.DataFrame(trajectory, columns=["t", "x", "y", "z"]).to_csv(
        output_dir / "reference_trajectory_full.csv",
        index=False,
    )
    post = trajectory[trajectory[:, 0] >= float(search_cfg["t_sim_transient"])]
    pd.DataFrame(post, columns=["t", "x", "y", "z"]).to_csv(
        output_dir / "reference_trajectory_posttransient.csv",
        index=False,
    )
    periodicity = classify_post_transient_periodicity(post, h=h)

    candidate = {
        "q": q,
        "alpha": system_parameters["alpha"],
        "beta": system_parameters["beta"],
        "gamma": system_parameters["gamma"],
        "m0": system_parameters["m0"],
        "m1": system_parameters["m1"],
        "A": float(root["A"]),
        "c": float(root["c"]),
        "omega": float(root["omega"]),
        "residual_norm": float(root["residual_norm"]),
        "psi0": float(psi0),
        "N1": float(n1),
        "seed": seed.tolist(),
        "continuation_endpoint": x_final.tolist(),
        "continuation_eta_values": eta_values,
        "continuation_t_transient_per_stage": float(search_cfg["t_transient"]),
        "continuation_t_keep_per_stage": float(search_cfg["t_keep"]),
        "reference_t_final": float(search_cfg["t_sim_final"]),
        "reference_t_burn": float(search_cfg["t_sim_transient"]),
        "reference_status": sim_status,
        "posttransient_classification": _jsonable(periodicity),
    }
    with (output_dir / "candidate_and_reference.json").open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(candidate), stream, ensure_ascii=False, indent=2)
    return candidate, post


def _load_reference(output_dir: Path) -> tuple[dict[str, Any], np.ndarray]:
    with (output_dir / "candidate_and_reference.json").open(encoding="utf-8") as stream:
        candidate = json.load(stream)
    post = pd.read_csv(output_dir / "reference_trajectory_posttransient.csv")
    return candidate, post[["t", "x", "y", "z"]].to_numpy(dtype=float)


def _probe(
    cfg: dict[str, Any],
    candidate: dict[str, Any],
    reference_post: np.ndarray,
    output_dir: Path,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sys_cfg = cfg["system"]
    int_cfg = cfg["integrator"]
    probe_cfg = cfg["step3_hiddenness"]
    q = float(sys_cfg["q"])
    h = float(int_cfg["h"])
    system_parameters = {
        "alpha": float(sys_cfg["parameters"]["alpha"]),
        "beta": float(sys_cfg["parameters"]["beta"]),
        "gamma": float(sys_cfg["parameters"]["gamma"]),
        "m0": float(sys_cfg["parameters"]["m0"]),
        "m1": float(sys_cfg["parameters"]["m1"]),
    }
    system = get_system("chua-nonsmooth")
    system.parameters.update(system_parameters)
    equilibria = solve_equilibria(system)
    equilibrium_items = [(name, np.asarray(point, dtype=float)) for name, point in equilibria.items()]
    equilibrium_points = [point for _, point in equilibrium_items]

    reference_tail = reference_post[
        reference_post[:, 0] >= float(probe_cfg["t_burn_probe"])
    ][:, 1:4]
    radii = [float(v) for v in probe_cfg["radii"]]
    samples = [int(v) for v in probe_cfg["samples_per_radius"]]
    base_seed = int(cfg["experiment"]["random_seed"])

    tasks_by_group: list[tuple[str, np.ndarray, float, int, np.ndarray]] = []
    global_index = 0
    for eq_name, eq_point in equilibrium_items:
        for radius_index, (radius, n_samples) in enumerate(zip(radii, samples)):
            points = generate_neighborhood_points(
                eq_point=eq_point,
                radius=radius,
                num_samples=n_samples,
                mode=str(probe_cfg["sampling_mode"]),
                seed=base_seed + radius_index,
            )
            tasks_by_group.append((eq_name, eq_point, radius, global_index, points))
            global_index += len(points)

    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_init_probe_worker,
        initargs=(
            system_parameters,
            equilibrium_points,
            reference_tail,
            q,
            h,
            float(probe_cfg["t_final_probe"]),
            float(probe_cfg["t_burn_probe"]),
            float(probe_cfg["equilibrium_tol"]),
            str(probe_cfg["match_metric"]),
            float(probe_cfg["match_tol"]),
            float(probe_cfg["match_percentile"]),
        ),
    ) as pool:
        completed = 0
        total = sum(len(group[-1]) for group in tasks_by_group)
        for eq_name, eq_point, radius, start_index, points in tasks_by_group:
            payloads = [(start_index + idx, point) for idx, point in enumerate(points)]
            results = pool.map(_run_one_probe, payloads)
            for local_index, (point, result) in enumerate(zip(points, results)):
                direction = (point - eq_point) / radius
                rows.append(
                    {
                        "sample_index": start_index + local_index,
                        "equilibrium": eq_name,
                        "radius": radius,
                        "radius_index": radii.index(radius),
                        "sampling_seed": base_seed + radii.index(radius),
                        "x0": float(point[0]),
                        "y0": float(point[1]),
                        "z0": float(point[2]),
                        "direction_x": float(direction[0]),
                        "direction_y": float(direction[1]),
                        "direction_z": float(direction[2]),
                        **result,
                    }
                )
            completed += len(results)
            counts = pd.Series([r["destination"] for r in results]).value_counts().to_dict()
            print(
                f"{eq_name:>3s} r={radius:.1e}: {len(results):3d} sondas, "
                f"TARGET={counts.get('target_attractor', 0)}, "
                f"EQ={counts.get('stable_equilibrium', 0)}, "
                f"OTHER={counts.get('other_attractor', 0)}, "
                f"DIV={counts.get('divergence', 0)}, "
                f"FAIL={counts.get('numerical_failure', 0)} "
                f"({completed}/{total})",
                flush=True,
            )

    runs = pd.DataFrame(rows).sort_values("sample_index").reset_index(drop=True)
    runs.to_csv(output_dir / "probe_runs.csv", index=False)

    categories = [
        "target_attractor",
        "stable_equilibrium",
        "other_attractor",
        "divergence",
        "numerical_failure",
    ]
    summary_rows: list[dict[str, Any]] = []
    for eq_name, _ in equilibrium_items:
        for radius in radii:
            subset = runs[
                (runs["equilibrium"] == eq_name)
                & np.isclose(runs["radius"].astype(float), radius)
            ]
            row: dict[str, Any] = {
                "equilibrium": eq_name,
                "radius": radius,
                "samples": int(len(subset)),
            }
            for category in categories:
                row[category] = int((subset["destination"] == category).sum())
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "probe_summary.csv", index=False)

    target_hits = int((runs["destination"] == "target_attractor").sum())
    failures = int((runs["destination"] == "numerical_failure").sum())
    result = {
        "candidate": candidate,
        "equilibria": {name: point.tolist() for name, point in equilibrium_items},
        "probe_contract": {
            "operator": "Caputo",
            "integrator": "ABM-PECE",
            "memory": "full",
            "q": q,
            "h": h,
            "t_final": float(probe_cfg["t_final_probe"]),
            "t_burn": float(probe_cfg["t_burn_probe"]),
            "radii": radii,
            "samples_per_radius": samples,
            "samples_per_equilibrium": int(sum(samples)),
            "samples_total": int(len(runs)),
            "sampling_mode": str(probe_cfg["sampling_mode"]),
            "random_generator": "NumPy default_rng (PCG64)",
            "base_seed": base_seed,
            "seed_schedule": "base_seed + radius_index; repeated for each equilibrium",
            "equilibrium_tol": float(probe_cfg["equilibrium_tol"]),
            "target_match_metric": str(probe_cfg["match_metric"]),
            "target_match_tol": float(probe_cfg["match_tol"]),
            "target_match_percentile": float(probe_cfg["match_percentile"]),
            "hard_divergence_norm": 120.0,
            "early_divergence_rule": {
                "norm": 80.0,
                "consecutive_steps": 5,
                "growth_factor": 1.25,
            },
            "python_fallback": False,
            "reference_tail_points": int(len(reference_tail)),
        },
        "target_hits_total": target_hits,
        "numerical_failures_total": failures,
        "compatible_with_hiddenness_under_tested_radii": target_hits == 0 and failures == 0,
        "elapsed_probe_seconds": time.perf_counter() - t_start,
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(result), stream, ensure_ascii=False, indent=2)
    return runs, summary, result


def _target_robustness(
    cfg: dict[str, Any],
    candidate: dict[str, Any],
    reference_post: np.ndarray,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    """Comprueba que perturbaciones locales de la semilla reproducen la referencia."""
    sys_cfg = cfg["system"]
    probe_cfg = cfg["step3_hiddenness"]
    q = float(sys_cfg["q"])
    h = float(cfg["integrator"]["h"])
    system_parameters = {
        "alpha": float(sys_cfg["parameters"]["alpha"]),
        "beta": float(sys_cfg["parameters"]["beta"]),
        "gamma": float(sys_cfg["parameters"]["gamma"]),
        "m0": float(sys_cfg["parameters"]["m0"]),
        "m1": float(sys_cfg["parameters"]["m1"]),
    }
    system = get_system("chua-nonsmooth")
    system.parameters.update(system_parameters)
    equilibria = solve_equilibria(system)
    equilibrium_points = [np.asarray(point, dtype=float) for point in equilibria.values()]
    reference_tail = reference_post[
        reference_post[:, 0] >= float(probe_cfg["t_burn_probe"])
    ][:, 1:4]

    center = np.asarray(candidate["continuation_endpoint"], dtype=float)
    radii = [1.0e-6, 1.0e-4, 1.0e-2]
    samples_per_radius = 12
    base_seed = 142
    tasks: list[tuple[float, int, np.ndarray]] = []
    sample_index = 0
    for radius_index, radius in enumerate(radii):
        points = generate_neighborhood_points(
            eq_point=center,
            radius=radius,
            num_samples=samples_per_radius,
            mode="sphere_random",
            seed=base_seed + radius_index,
        )
        for point in points:
            tasks.append((radius, sample_index, point))
            sample_index += 1

    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_init_probe_worker,
        initargs=(
            system_parameters,
            equilibrium_points,
            reference_tail,
            q,
            h,
            float(probe_cfg["t_final_probe"]),
            float(probe_cfg["t_burn_probe"]),
            float(probe_cfg["equilibrium_tol"]),
            str(probe_cfg["match_metric"]),
            float(probe_cfg["match_tol"]),
            float(probe_cfg["match_percentile"]),
        ),
    ) as pool:
        results = pool.map(
            _run_one_probe,
            [(sample_id, point) for _, sample_id, point in tasks],
        )

    rows = []
    for (radius, sample_id, point), probe_result in zip(tasks, results):
        rows.append(
            {
                "sample_index": sample_id,
                "radius": radius,
                "sampling_seed": base_seed + radii.index(radius),
                "x0": float(point[0]),
                "y0": float(point[1]),
                "z0": float(point[2]),
                **probe_result,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "target_reproduction_runs.csv", index=False)
    target_matches = int((frame["destination"] == "target_attractor").sum())
    failures = int(
        frame["destination"].isin(["numerical_failure", "divergence"]).sum()
    )
    return {
        "center": center.tolist(),
        "radii": radii,
        "samples_per_radius": samples_per_radius,
        "samples_total": int(len(frame)),
        "base_seed": base_seed,
        "target_matches": target_matches,
        "failures_or_divergences": failures,
        "all_perturbations_reproduce_target": (
            target_matches == len(frame) and failures == 0
        ),
    }


def _fixed_nn_sample(points: np.ndarray, max_points: int = 2000) -> np.ndarray:
    """Reproduce the deterministic subsampling used by ``evaluate_target_match``."""
    values = np.asarray(points, dtype=float)
    if len(values) <= max_points:
        return values
    indices = np.random.default_rng(0).choice(
        len(values),
        max_points,
        replace=False,
    )
    return values[indices]


def _init_extended_probe_worker(
    system_parameters: dict[str, float],
    equilibria: list[np.ndarray],
    reference_sample: np.ndarray,
    q: float,
    h: float,
    t_final: float,
    t_burn: float,
    equilibrium_tol: float,
    match_tol: float,
    match_percentile: float,
    hard_divergence_norm: float,
    early_stop: dict[str, Any],
) -> None:
    global _EXT_SYSTEM, _EXT_EQS, _EXT_REF_TREE
    global _EXT_Q, _EXT_H, _EXT_T_FINAL, _EXT_T_BURN
    global _EXT_EQ_TOL, _EXT_MATCH_TOL, _EXT_MATCH_PERCENTILE
    global _EXT_HARD_DIVERGENCE_NORM, _EXT_EARLY_STOP

    _EXT_SYSTEM = get_system("chua-nonsmooth")
    _EXT_SYSTEM.parameters.update(system_parameters)
    _EXT_EQS = [np.asarray(point, dtype=float) for point in equilibria]
    _EXT_REF_TREE = cKDTree(np.asarray(reference_sample, dtype=float))
    _EXT_Q = float(q)
    _EXT_H = float(h)
    _EXT_T_FINAL = float(t_final)
    _EXT_T_BURN = float(t_burn)
    _EXT_EQ_TOL = float(equilibrium_tol)
    _EXT_MATCH_TOL = float(match_tol)
    _EXT_MATCH_PERCENTILE = float(match_percentile)
    _EXT_HARD_DIVERGENCE_NORM = float(hard_divergence_norm)
    _EXT_EARLY_STOP = dict(early_stop)


def _run_one_extended_probe(task: dict[str, Any]) -> dict[str, Any]:
    """Integrate and classify one extended probe without storing its trajectory."""
    identity = {
        "probe_id": int(task["probe_id"]),
        "equilibrium": str(task["equilibrium"]),
        "equilibrium_index": int(task["equilibrium_index"]),
        "radius": float(task["radius"]),
        "radius_index": int(task["radius_index"]),
        "sample_index": int(task["sample_index"]),
        "sampling_seed": int(task["sampling_seed"]),
        "x0": float(task["x0"]),
        "y0": float(task["y0"]),
        "z0": float(task["z0"]),
        "radial_fraction": float(task["radial_fraction"]),
    }
    x0 = np.array(
        [identity["x0"], identity["y0"], identity["z0"]],
        dtype=float,
    )
    started = time.perf_counter()

    try:
        times, states, status, info = fractional_integrate(
            rhs=lambda t, x: _EXT_SYSTEM.rhs(x, _EXT_SYSTEM.parameters),
            x0=x0,
            q=_EXT_Q,
            h=_EXT_H,
            t_final=_EXT_T_FINAL,
            method="abm",
            memory_mode="full",
            system=_EXT_SYSTEM,
            use_c_backend=True,
            divergence_norm=_EXT_HARD_DIVERGENCE_NORM,
            allow_python_fallback=False,
            early_stop_config=_EXT_EARLY_STOP,
            equilibria=_EXT_EQS,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            **identity,
            "destination": "numerical_failure",
            "status": f"exception:{type(exc).__name__}:{exc}",
            "end_time": 0.0,
            "steps": 0,
            "max_norm": float("nan"),
            "final_x": float("nan"),
            "final_y": float("nan"),
            "final_z": float("nan"),
            "min_final_equilibrium_distance": float("nan"),
            "tail_points": 0,
            "nn_percentile_score": float("nan"),
            "used_c_backend": False,
            "elapsed_s": time.perf_counter() - started,
        }

    if len(states) == 0:
        return {
            **identity,
            "destination": "numerical_failure",
            "status": f"{status}:empty_solution",
            "end_time": 0.0,
            "steps": 0,
            "max_norm": float("nan"),
            "final_x": float("nan"),
            "final_y": float("nan"),
            "final_z": float("nan"),
            "min_final_equilibrium_distance": float("nan"),
            "tail_points": 0,
            "nn_percentile_score": float("nan"),
            "used_c_backend": bool(info.get("used_c_backend", False)),
            "elapsed_s": time.perf_counter() - started,
        }

    final = np.asarray(states[-1], dtype=float)
    eq_distances = [float(np.linalg.norm(final - eq)) for eq in _EXT_EQS]
    common = {
        **identity,
        "status": str(status),
        "end_time": float(times[-1]) if len(times) else 0.0,
        "steps": int(len(states)),
        "max_norm": float(np.linalg.norm(states, axis=1).max()),
        "final_x": float(final[0]),
        "final_y": float(final[1]),
        "final_z": float(final[2]),
        "min_final_equilibrium_distance": min(eq_distances),
        "used_c_backend": bool(info.get("used_c_backend", False)),
        "elapsed_s": time.perf_counter() - started,
    }

    if status in ("diverged", "diverged_early", "nonfinite_solution"):
        return {
            **common,
            "destination": "divergence",
            "tail_points": 0,
            "nn_percentile_score": float("nan"),
        }
    if status == "converged_equilibrium_early":
        return {
            **common,
            "destination": "stable_equilibrium",
            "tail_points": 0,
            "nn_percentile_score": float("nan"),
        }
    if status != "ok":
        return {
            **common,
            "destination": "numerical_failure",
            "tail_points": 0,
            "nn_percentile_score": float("nan"),
        }
    if min(eq_distances) <= _EXT_EQ_TOL:
        return {
            **common,
            "destination": "stable_equilibrium",
            "tail_points": 0,
            "nn_percentile_score": float("nan"),
        }

    n_burn = int(np.ceil(_EXT_T_BURN / _EXT_H))
    tail = states[n_burn:] if len(states) > n_burn else states[int(len(states) / 2) :]
    trajectory_sample = _fixed_nn_sample(tail)
    distances, _ = _EXT_REF_TREE.query(trajectory_sample, k=1, workers=1)
    score = float(np.percentile(distances, _EXT_MATCH_PERCENTILE))
    destination = (
        "target_attractor" if score <= _EXT_MATCH_TOL else "other_attractor"
    )
    return {
        **common,
        "destination": destination,
        "tail_points": int(len(tail)),
        "nn_percentile_score": score,
    }


def _build_extended_probe_plan(
    cfg: dict[str, Any],
    equilibrium_items: list[tuple[str, np.ndarray]],
    max_per_radius: int,
) -> pd.DataFrame:
    ext_cfg = cfg["step4_extended_hiddenness"]
    radius_plan = [
        (float(radius), int(samples))
        for radius, samples in ext_cfg["radius_plan"]
    ]
    base_seed = int(cfg["experiment"]["random_seed"])
    rows: list[dict[str, Any]] = []
    probe_id = 0

    for equilibrium_index, (eq_name, eq_point) in enumerate(equilibrium_items):
        for radius_index, (radius, planned_samples) in enumerate(radius_plan):
            n_samples = (
                min(planned_samples, max_per_radius)
                if max_per_radius > 0
                else planned_samples
            )
            sampling_seed = base_seed + equilibrium_index * 100 + radius_index * 10
            points = sample_ball(
                np.asarray(eq_point, dtype=float),
                radius,
                n_samples,
                sampling_seed,
            )
            for sample_index, point in enumerate(points):
                displacement = np.asarray(point, dtype=float) - eq_point
                radial_fraction = float(np.linalg.norm(displacement) / radius)
                rows.append(
                    {
                        "probe_id": probe_id,
                        "equilibrium": eq_name,
                        "equilibrium_index": equilibrium_index,
                        "radius": radius,
                        "radius_index": radius_index,
                        "sample_index": sample_index,
                        "sampling_seed": sampling_seed,
                        "x0": float(point[0]),
                        "y0": float(point[1]),
                        "z0": float(point[2]),
                        "radial_fraction": radial_fraction,
                    }
                )
                probe_id += 1
    return pd.DataFrame(rows)


def _load_extended_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    raw_text = path.read_text(encoding="utf-8")
    raw_lines = raw_text.splitlines()
    missing_terminal_newline = bool(raw_text) and not raw_text.endswith(
        ("\n", "\r")
    )
    nonempty_indices = [
        index for index, line in enumerate(raw_lines) if line.strip()
    ]
    last_nonempty = nonempty_indices[-1] if nonempty_indices else -1
    valid_rows: list[dict[str, Any]] = []
    repair_trailing_fragment = False

    def _reject_nonfinite(token: str) -> None:
        raise ValueError(f"constante JSON no finita: {token}")

    for index, line in enumerate(raw_lines):
        if not line.strip():
            continue
        line_number = index + 1
        try:
            row = json.loads(line, parse_constant=_reject_nonfinite)
        except (json.JSONDecodeError, ValueError) as exc:
            if index == last_nonempty:
                repair_trailing_fragment = True
                print(
                    f"Checkpoint: se repara el fragmento final incompleto "
                    f"de la línea {line_number}.",
                    flush=True,
                )
                continue
            raise RuntimeError(
                f"Checkpoint corrupto antes de su última línea "
                f"(línea {line_number})."
            ) from exc
        probe_id = int(row["probe_id"])
        if probe_id in completed:
            raise RuntimeError(
                f"Checkpoint con probe_id duplicado: {probe_id}."
            )
        completed[probe_id] = row
        valid_rows.append(row)

    if repair_trailing_fragment or missing_terminal_newline:
        temporary = path.with_suffix(path.suffix + ".repair.tmp")
        serialized = "".join(
            json.dumps(
                _jsonable(row),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in valid_rows
        )
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(path)
    return completed


def _validate_extended_checkpoint_identity(
    completed: dict[int, dict[str, Any]],
    plan: pd.DataFrame,
) -> None:
    plan_rows = {
        int(row["probe_id"]): row
        for row in plan.to_dict("records")
    }
    unexpected = sorted(set(completed) - set(plan_rows))
    if unexpected:
        raise RuntimeError(
            "El checkpoint contiene probe_id fuera del plan vigente: "
            f"{unexpected[:5]}."
        )

    exact_fields = (
        "equilibrium",
        "equilibrium_index",
        "radius_index",
        "sample_index",
        "sampling_seed",
    )
    float_fields = (
        "radius",
        "x0",
        "y0",
        "z0",
        "radial_fraction",
    )
    for probe_id, saved in completed.items():
        expected = plan_rows[probe_id]
        for field in exact_fields:
            if str(saved.get(field)) != str(expected[field]):
                raise RuntimeError(
                    f"Checkpoint incompatible en probe_id={probe_id}, "
                    f"campo {field}."
                )
        for field in float_fields:
            try:
                matches = np.isclose(
                    float(saved[field]),
                    float(expected[field]),
                    rtol=0.0,
                    atol=1.0e-14,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Checkpoint incompleto en probe_id={probe_id}, "
                    f"campo {field}."
                ) from exc
            if not bool(matches):
                raise RuntimeError(
                    f"Checkpoint incompatible en probe_id={probe_id}, "
                    f"campo {field}."
                )


def _extended_summary(
    runs: pd.DataFrame,
    equilibrium_items: list[tuple[str, np.ndarray]],
    radius_plan: list[tuple[float, int]],
) -> pd.DataFrame:
    categories = [
        "target_attractor",
        "stable_equilibrium",
        "other_attractor",
        "divergence",
        "numerical_failure",
    ]
    rows: list[dict[str, Any]] = []
    for eq_name, _ in equilibrium_items:
        for radius, _ in radius_plan:
            subset = runs[
                (runs["equilibrium"] == eq_name)
                & np.isclose(runs["radius"].astype(float), radius)
            ]
            row: dict[str, Any] = {
                "equilibrium": eq_name,
                "radius": radius,
                "samples": int(len(subset)),
            }
            for category in categories:
                row[category] = int((subset["destination"] == category).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_extended_probe_summary(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    language: str,
) -> None:
    eq_order = list(dict.fromkeys(summary["equilibrium"].tolist()))
    eq_labels = {"E0": r"$E_0$", "E+": r"$E_+$", "E-": r"$E_-$"}
    radii = sorted(summary["radius"].unique().tolist())
    target = np.zeros((len(eq_order), len(radii)), dtype=float)
    labels = np.empty_like(target, dtype=object)
    for row_index, eq_name in enumerate(eq_order):
        for column_index, radius in enumerate(radii):
            row = summary[
                (summary["equilibrium"] == eq_name)
                & np.isclose(summary["radius"].astype(float), radius)
            ].iloc[0]
            target[row_index, column_index] = (
                float(row["target_attractor"]) / float(row["samples"])
                if int(row["samples"]) > 0
                else 0.0
            )
            labels[row_index, column_index] = (
                f"{int(row['target_attractor'])}/{int(row['samples'])}"
            )

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(10.8, 7.2),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.9, 1.1]},
    )
    observed_max = float(target.max()) if target.size else 0.0
    color_max = max(0.02, min(1.0, 1.05 * observed_max))
    image = ax0.imshow(
        target,
        vmin=0.0,
        vmax=color_max,
        cmap="YlOrRd",
        aspect="auto",
    )
    for row_index in range(len(eq_order)):
        for column_index in range(len(radii)):
            ax0.text(
                column_index,
                row_index,
                labels[row_index, column_index],
                ha="center",
                va="center",
                fontsize=8.5,
                color=(
                    "white"
                    if target[row_index, column_index] > 0.55 * color_max
                    else "black"
                ),
            )
    ax0.axvline(6.5, color="#1f2937", linestyle="--", linewidth=1.0)
    ax0.set_xticks(
        range(len(radii)),
        [f"{radius:.0e}" for radius in radii],
        rotation=40,
    )
    ax0.set_yticks(
        range(len(eq_order)),
        [eq_labels.get(eq_name, eq_name) for eq_name in eq_order],
    )
    ax0.set_xlabel("Radio $r$" if language == "es" else "Radius $r$")
    ax0.set_ylabel("Equilibrio" if language == "es" else "Equilibrium")
    colorbar = fig.colorbar(image, ax=ax0, fraction=0.025, pad=0.02)
    colorbar.set_label(
        "Fracción de contactos"
        if language == "es"
        else "Target-contact fraction"
    )

    categories = [
        (
            "stable_equilibrium",
            "#4c78a8",
            "Equilibrio" if language == "es" else "Equilibrium",
        ),
        (
            "other_attractor",
            "#f2cf5b",
            "Otro" if language == "es" else "Other",
        ),
        (
            "target_attractor",
            "#d62728",
            "Objetivo" if language == "es" else "Target",
        ),
        (
            "divergence",
            "#7f7f7f",
            "Divergencia" if language == "es" else "Divergence",
        ),
        (
            "numerical_failure",
            "#000000",
            "Fallo" if language == "es" else "Failure",
        ),
    ]
    grouped = runs.groupby(["radius", "destination"]).size().unstack(fill_value=0)
    x_positions = np.arange(len(radii))
    totals = grouped.sum(axis=1).reindex(radii, fill_value=0).to_numpy(dtype=float)
    bottom = np.zeros(len(radii), dtype=float)
    for key, color, label in categories:
        values = (
            grouped[key].reindex(radii, fill_value=0).to_numpy(dtype=float)
            if key in grouped.columns
            else np.zeros(len(radii), dtype=float)
        )
        fractions = np.divide(
            values,
            totals,
            out=np.zeros_like(values),
            where=totals > 0,
        )
        ax1.bar(x_positions, fractions, bottom=bottom, color=color, label=label)
        bottom += fractions
    ax1.axvline(6.5, color="#1f2937", linestyle="--", linewidth=1.0)
    ax1.set_xticks(
        x_positions,
        [f"{radius:.0e}" for radius in radii],
        rotation=40,
    )
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("Radio $r$" if language == "es" else "Radius $r$")
    ax1.set_ylabel(
        "Fracción de destinos" if language == "es" else "Destination fraction"
    )
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=5,
        fontsize=8,
    )

    suffix = "es" if language == "es" else "en"
    for extension in ("pdf", "png"):
        fig.savefig(
            output_dir / f"nonsmooth_extended_probe_summary_{suffix}.{extension}",
            dpi=260,
        )
    plt.close(fig)


def _run_extended_probe(
    cfg: dict[str, Any],
    candidate: dict[str, Any],
    reference_post: np.ndarray,
    output_dir: Path,
    workers: int,
    resume: bool,
    max_per_radius: int,
) -> dict[str, Any]:
    ext_cfg = cfg["step4_extended_hiddenness"]
    sys_cfg = cfg["system"]
    system_parameters = {
        "alpha": float(sys_cfg["parameters"]["alpha"]),
        "beta": float(sys_cfg["parameters"]["beta"]),
        "gamma": float(sys_cfg["parameters"]["gamma"]),
        "m0": float(sys_cfg["parameters"]["m0"]),
        "m1": float(sys_cfg["parameters"]["m1"]),
    }
    system = get_system("chua-nonsmooth")
    system.parameters.update(system_parameters)
    equilibria = solve_equilibria(system)
    equilibrium_items = [
        (name, np.asarray(point, dtype=float))
        for name, point in equilibria.items()
    ]
    equilibrium_points = [point for _, point in equilibrium_items]
    radius_plan = [
        (float(radius), int(samples))
        for radius, samples in ext_cfg["radius_plan"]
    ]
    early_stop = {
        "enabled": True,
        "equilibrium_enabled": True,
        "divergence_enabled": True,
        **dict(ext_cfg["early_stopping"]),
    }
    hard_divergence_norm = float(early_stop.pop("hard_divergence_norm"))
    stop_after_first_contact = bool(
        ext_cfg.get("stop_after_first_contact_radius", False)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "extended_probe_plan.csv"
    plan = _build_extended_probe_plan(cfg, equilibrium_items, max_per_radius)
    plan_csv = plan.to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    plan_hash = hashlib.sha256(plan_csv.encode("utf-8")).hexdigest()

    reference_tail = reference_post[
        reference_post[:, 0] >= float(ext_cfg["t_burn_probe"])
    ][:, 1:4]
    reference_sample = _fixed_nn_sample(reference_tail)
    reference_sample_path = output_dir / "target_cloud_nn_sample.csv"
    reference_sample_csv = pd.DataFrame(
        reference_sample,
        columns=["x", "y", "z"],
    ).to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    reference_hash = hashlib.sha256(
        reference_sample_csv.encode("utf-8")
    ).hexdigest()

    contract = {
        "contract_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "system": {
            "system_id": "chua-nonsmooth",
            "parameters": system_parameters,
            "equilibria": {
                name: point.tolist() for name, point in equilibrium_items
            },
        },
        "candidate": {
            key: candidate[key]
            for key in (
                "q",
                "A",
                "c",
                "omega",
                "residual_norm",
                "seed",
                "continuation_endpoint",
            )
        },
        "integrator": {
            "operator": "Caputo",
            "method": "ABM-PECE",
            "q": float(sys_cfg["q"]),
            "h": float(cfg["integrator"]["h"]),
            "memory_mode": "full",
            "t_final": float(ext_cfg["t_final_probe"]),
            "t_burn": float(ext_cfg["t_burn_probe"]),
            "native_backend_required": True,
            "hard_divergence_norm": hard_divergence_norm,
            "early_stopping": early_stop,
        },
        "sampling": {
            "mode": str(ext_cfg["sampling_mode"]),
            "generator": "sample_ball; NumPy default_rng (PCG64)",
            "base_seed": int(cfg["experiment"]["random_seed"]),
            "seed_schedule": (
                "base_seed + equilibrium_index*100 + radius_index*10"
            ),
            "radius_plan_nominal": radius_plan,
            "max_per_radius_override": int(max_per_radius),
            "samples_total": int(len(plan)),
            "plan_file": plan_path.name,
            "plan_sha256": plan_hash,
        },
        "classification": {
            "equilibrium_tol": float(ext_cfg["equilibrium_tol"]),
            "target_match_metric": str(ext_cfg["match_metric"]),
            "target_match_tol": float(ext_cfg["match_tol"]),
            "target_match_percentile": float(ext_cfg["match_percentile"]),
            "target_cloud_source": (
                "../reference_trajectory_posttransient.csv"
            ),
            "target_cloud_points_before_subsampling": int(len(reference_tail)),
            "target_cloud_subsampling": (
                "default_rng(0).choice(max_points=2000, replace=False)"
            ),
            "probe_tail_subsampling": (
                "default_rng(0).choice(max_points=2000, replace=False)"
            ),
            "nearest_neighbor_implementation": "SciPy cKDTree exact query",
            "target_cloud_sample_file": reference_sample_path.name,
            "target_cloud_sample_sha256": reference_hash,
        },
        "termination": {
            "rule": (
                (
                    "Process radii in ascending order across all equilibria; "
                    "after the first radius with one or more target contacts, "
                    "finish every planned probe at that radius and omit all "
                    "larger radii from the declared experiment."
                )
                if stop_after_first_contact
                else (
                    "Process every planned radius in ascending order across "
                    "all equilibria."
                )
            ),
            "stop_after_first_contact_radius": stop_after_first_contact,
            "radius_group_scope": "all equilibria",
        },
    }
    contract_hash_payload = {
        key: value
        for key, value in contract.items()
        if key != "created_utc"
    }
    contract["contract_sha256"] = _sha256_payload(contract_hash_payload)
    contract_path = output_dir / "extended_numerical_contract.json"
    checkpoint_path = output_dir / "extended_checkpoint.jsonl"
    if resume and checkpoint_path.exists() and not contract_path.exists():
        raise RuntimeError(
            "Existe un checkpoint sin contrato numérico; no se puede reanudar."
        )
    if resume and contract_path.exists():
        previous_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        previous_hash = previous_contract.get("contract_sha256")
        previous_hash_payload = {
            key: value
            for key, value in previous_contract.items()
            if key not in ("created_utc", "contract_sha256")
        }
        if (
            not previous_hash
            or _sha256_payload(previous_hash_payload) != previous_hash
        ):
            raise RuntimeError(
                "El contrato extendido guardado carece de una huella integral "
                "válida; no se puede reanudar."
            )
        if previous_hash != contract["contract_sha256"]:
            raise RuntimeError(
                "El contrato extendido guardado no coincide íntegramente con "
                "el contrato actual; no se puede reanudar."
            )
        contract["created_utc"] = previous_contract.get(
            "created_utc",
            contract["created_utc"],
        )
    plan_path.write_bytes(plan_csv.encode("utf-8"))
    reference_sample_path.write_bytes(reference_sample_csv.encode("utf-8"))
    if _sha256_file(plan_path) != plan_hash:
        raise RuntimeError("La huella escrita del plan extendido no coincide.")
    if _sha256_file(reference_sample_path) != reference_hash:
        raise RuntimeError("La huella escrita de la nube objetivo no coincide.")
    _write_json_atomic(contract_path, contract)

    if checkpoint_path.exists() and not resume:
        raise RuntimeError(
            f"Ya existe {checkpoint_path.name}; use --resume-extended "
            "o elija otro directorio."
        )
    completed = _load_extended_checkpoint(checkpoint_path) if resume else {}
    _validate_extended_checkpoint_identity(completed, plan)
    tasks = plan.to_dict("records")
    total = len(tasks)
    started = time.perf_counter()
    print(
        f"Sondeo extendido: {len(completed)}/{total} filas recuperadas; "
        f"{workers} procesos; paro tras completar el primer radio con contactos.",
        flush=True,
    )

    context = mp.get_context("spawn")
    progress_step = max(25, total // 200)
    first_contact_radius_index: int | None = None
    declared_radius_indices: list[int] = []
    completed_at_start = len(completed)
    with checkpoint_path.open("a", encoding="utf-8", buffering=1) as checkpoint:
        with context.Pool(
            processes=workers,
            initializer=_init_extended_probe_worker,
            initargs=(
                system_parameters,
                equilibrium_points,
                reference_sample,
                float(sys_cfg["q"]),
                float(cfg["integrator"]["h"]),
                float(ext_cfg["t_final_probe"]),
                float(ext_cfg["t_burn_probe"]),
                float(ext_cfg["equilibrium_tol"]),
                float(ext_cfg["match_tol"]),
                float(ext_cfg["match_percentile"]),
                hard_divergence_norm,
                early_stop,
            ),
        ) as pool:
            for radius_index, (radius, _) in enumerate(radius_plan):
                radius_tasks = [
                    task
                    for task in tasks
                    if int(task["radius_index"]) == radius_index
                ]
                pending_at_radius = [
                    task
                    for task in radius_tasks
                    if int(task["probe_id"]) not in completed
                ]
                for result in pool.imap_unordered(
                    _run_one_extended_probe,
                    pending_at_radius,
                    chunksize=1,
                ):
                    probe_id = int(result["probe_id"])
                    completed[probe_id] = result
                    checkpoint.write(
                        json.dumps(
                            _jsonable(result),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    n_completed = len(completed)
                    if (
                        (n_completed - completed_at_start) % progress_step == 0
                        or all(
                            int(task["probe_id"]) in completed
                            for task in radius_tasks
                        )
                    ):
                        elapsed = time.perf_counter() - started
                        new_completed = n_completed - completed_at_start
                        rate = new_completed / elapsed if elapsed > 0 else 0.0
                        eta = (
                            (total - n_completed) / rate if rate > 0 else float("inf")
                        )
                        counts = pd.Series(
                            [row["destination"] for row in completed.values()]
                        ).value_counts()
                        print(
                            f"EXT {n_completed}/{total} "
                            f"TARGET={int(counts.get('target_attractor', 0))} "
                            f"EQ={int(counts.get('stable_equilibrium', 0))} "
                            f"OTHER={int(counts.get('other_attractor', 0))} "
                            f"DIV={int(counts.get('divergence', 0))} "
                            f"FAIL={int(counts.get('numerical_failure', 0))} "
                            f"rate={rate:.2f}/s ETA={eta/60.0:.1f} min",
                            flush=True,
                        )

                missing_at_radius = [
                    int(task["probe_id"])
                    for task in radius_tasks
                    if int(task["probe_id"]) not in completed
                ]
                if missing_at_radius:
                    raise RuntimeError(
                        f"El radio r={radius:g} quedó incompleto: "
                        f"{len(missing_at_radius)} sondas pendientes."
                    )
                declared_radius_indices.append(radius_index)
                contacts_at_radius = sum(
                    completed[int(task["probe_id"])]["destination"]
                    == "target_attractor"
                    for task in radius_tasks
                )
                print(
                    f"Radio r={radius:g} completo: "
                    f"{contacts_at_radius}/{len(radius_tasks)} contactos.",
                    flush=True,
                )
                if (
                    contacts_at_radius > 0
                    and first_contact_radius_index is None
                ):
                    first_contact_radius_index = radius_index
                if contacts_at_radius > 0 and stop_after_first_contact:
                    print(
                        f"Paro contractual activado en r={radius:g}; "
                        "no se incorporan radios mayores.",
                        flush=True,
                    )
                    break

    declared_probe_ids = {
        int(task["probe_id"])
        for task in tasks
        if int(task["radius_index"]) in declared_radius_indices
    }
    if any(probe_id not in completed for probe_id in declared_probe_ids):
        raise RuntimeError(
            "El sondeo extendido declarado contiene sondas pendientes."
        )

    runs = pd.DataFrame(
        [completed[probe_id] for probe_id in sorted(declared_probe_ids)]
    )
    runs.to_csv(
        output_dir / "extended_probe_runs.csv",
        index=False,
        float_format="%.17g",
    )
    excluded_probe_ids = sorted(set(completed) - declared_probe_ids)
    if excluded_probe_ids:
        pd.DataFrame(
            [completed[probe_id] for probe_id in excluded_probe_ids]
        ).to_csv(
            output_dir / "post_cutoff_completed_runs_excluded.csv",
            index=False,
            float_format="%.17g",
        )
    declared_radius_plan = [
        radius_plan[index] for index in declared_radius_indices
    ]
    summary = _extended_summary(runs, equilibrium_items, declared_radius_plan)
    summary.to_csv(output_dir / "extended_probe_summary.csv", index=False)

    local = runs[runs["radius"].astype(float) <= 1.0e-2 + 1.0e-15]
    macro = runs[runs["radius"].astype(float) > 1.0e-2 + 1.0e-15]
    local_target_hits = int(
        (local["destination"] == "target_attractor").sum()
    )
    local_failures = int(
        (local["destination"] == "numerical_failure").sum()
    )
    macro_target_hits = int(
        (macro["destination"] == "target_attractor").sum()
    )
    macro_failures = int(
        (macro["destination"] == "numerical_failure").sum()
    )
    used_c_backend_all = bool(runs["used_c_backend"].astype(bool).all())
    first_contact_radius = (
        radius_plan[first_contact_radius_index][0]
        if first_contact_radius_index is not None
        else None
    )
    first_contact_hits = (
        int(
            (
                np.isclose(
                    runs["radius"].astype(float),
                    float(first_contact_radius),
                )
                & (runs["destination"] == "target_attractor")
            ).sum()
        )
        if first_contact_radius is not None
        else 0
    )
    result = {
        "status": (
            "stopped_after_first_contact_radius"
            if first_contact_radius is not None and stop_after_first_contact
            else (
                "complete_without_target_contacts"
                if first_contact_radius is None
                else "complete_all_radii"
            )
        ),
        "samples_total": int(len(runs)),
        "planned_samples_total": int(total),
        "checkpoint_rows_total": int(len(completed)),
        "post_cutoff_completed_rows_excluded": int(len(excluded_probe_ids)),
        "termination": {
            "rule": contract["termination"]["rule"],
            "first_contact_radius": first_contact_radius,
            "first_contact_hits": first_contact_hits,
            "first_contact_radius_samples": (
                int(
                    np.isclose(
                        runs["radius"].astype(float),
                        float(first_contact_radius),
                    ).sum()
                )
                if first_contact_radius is not None
                else 0
            ),
            "last_declared_radius": float(runs["radius"].max()),
        },
        "samples_per_equilibrium": {
            name: int((runs["equilibrium"] == name).sum())
            for name, _ in equilibrium_items
        },
        "destination_counts": {
            key: int((runs["destination"] == key).sum())
            for key in (
                "target_attractor",
                "stable_equilibrium",
                "other_attractor",
                "divergence",
                "numerical_failure",
            )
        },
        "local_protocol": {
            "radius_max": 1.0e-2,
            "radii": [
                radius
                for radius, _ in declared_radius_plan
                if radius <= 1.0e-2
            ],
            "samples_total": int(len(local)),
            "target_hits": local_target_hits,
            "numerical_failures": local_failures,
            "compatible_with_hiddenness_under_sampled_local_balls": (
                local_target_hits == 0 and local_failures == 0
            ),
        },
        "macro_basin_exploration": {
            "radius_min_exclusive": 1.0e-2,
            "radii": [
                radius
                for radius, _ in declared_radius_plan
                if radius > 1.0e-2
            ],
            "samples_total": int(len(macro)),
            "target_hits": macro_target_hits,
            "numerical_failures": macro_failures,
        },
        "all_runs_used_native_backend": used_c_backend_all,
        "plan_sha256": plan_hash,
        "target_cloud_sample_sha256": reference_hash,
        "elapsed_new_probe_seconds": time.perf_counter() - started,
    }
    _write_json_atomic(output_dir / "extended_result.json", result)
    _plot_extended_probe_summary(runs, summary, output_dir, "es")
    _plot_extended_probe_summary(runs, summary, output_dir, "en")
    return result


def _plot_reference(reference_post: np.ndarray, output_dir: Path) -> None:
    t, x, y, z = reference_post.T
    stride = max(1, len(t) // 12000)

    fig = plt.figure(figsize=(10.2, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax3 = fig.add_subplot(grid[:, 0], projection="3d")
    ax3.plot(x[::stride], y[::stride], z[::stride], lw=0.55, color="#234a91")
    ax3.set_xlabel("$x$")
    ax3.set_ylabel("$y$")
    ax3.set_zlabel("$z$")
    ax3.view_init(elev=24, azim=-56)

    axt = fig.add_subplot(grid[0, 1])
    axt.plot(t[::stride], x[::stride], lw=0.55, color="#8b1a1a")
    axt.set_xlabel("$t$")
    axt.set_ylabel("$x(t)$")
    axt.grid(alpha=0.25)

    axp = fig.add_subplot(grid[1, 1])
    axp.plot(x[::stride], z[::stride], lw=0.55, color="#166534")
    axp.set_xlabel("$x$")
    axp.set_ylabel("$z$")
    axp.grid(alpha=0.25)

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"nonsmooth_corrected_reference.{ext}", dpi=260)
    plt.close(fig)


def _plot_probe_summary(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    language: str,
) -> None:
    eq_order = list(dict.fromkeys(summary["equilibrium"].tolist()))
    eq_labels = {
        "E0": r"$E_0$",
        "E+": r"$E_+$",
        "E-": r"$E_-$",
    }
    radii = sorted(summary["radius"].unique().tolist())
    target = np.zeros((len(eq_order), len(radii)), dtype=float)
    labels = np.empty_like(target, dtype=object)
    for i, eq in enumerate(eq_order):
        for j, radius in enumerate(radii):
            row = summary[
                (summary["equilibrium"] == eq)
                & np.isclose(summary["radius"].astype(float), radius)
            ].iloc[0]
            target[i, j] = row["target_attractor"] / row["samples"]
            labels[i, j] = f"{int(row['target_attractor'])}/{int(row['samples'])}"

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    image = ax0.imshow(target, vmin=0.0, vmax=1.0, cmap="YlOrRd", aspect="auto")
    for i in range(len(eq_order)):
        for j in range(len(radii)):
            ax0.text(j, i, labels[i, j], ha="center", va="center", fontsize=8.5)
    ax0.set_xticks(range(len(radii)), [f"{r:.0e}" for r in radii], rotation=35)
    ax0.set_yticks(
        range(len(eq_order)),
        [eq_labels.get(eq, eq) for eq in eq_order],
    )
    ax0.set_xlabel("Radio $r$" if language == "es" else "Radius $r$")
    ax0.set_ylabel("Equilibrio" if language == "es" else "Equilibrium")
    colorbar = fig.colorbar(image, ax=ax0, fraction=0.047, pad=0.03)
    colorbar.set_label(
        "Fracción de contactos"
        if language == "es"
        else "Target-contact fraction"
    )

    categories = [
        ("stable_equilibrium", "#4c78a8", "Equilibrio" if language == "es" else "Equilibrium"),
        ("other_attractor", "#f2cf5b", "Otro" if language == "es" else "Other"),
        ("target_attractor", "#d62728", "Objetivo" if language == "es" else "Target"),
        ("divergence", "#7f7f7f", "Divergencia" if language == "es" else "Divergence"),
        ("numerical_failure", "#000000", "Fallo" if language == "es" else "Failure"),
    ]
    grouped = (
        runs.groupby(["equilibrium", "destination"]).size().unstack(fill_value=0)
    )
    bottom = np.zeros(len(eq_order), dtype=float)
    totals = grouped.sum(axis=1).reindex(eq_order).to_numpy(dtype=float)
    for key, color, label in categories:
        values = (
            grouped[key].reindex(eq_order).to_numpy(dtype=float)
            if key in grouped.columns
            else np.zeros(len(eq_order), dtype=float)
        )
        fractions = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
        ax1.bar(eq_order, fractions, bottom=bottom, color=color, label=label)
        bottom += fractions
    ax1.set_xticks(
        range(len(eq_order)),
        [eq_labels.get(eq, eq) for eq in eq_order],
    )
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel(
        "Fracción de destinos" if language == "es" else "Destination fraction"
    )
    ax1.set_xlabel("Equilibrio" if language == "es" else "Equilibrium")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, fontsize=8)

    suffix = "es" if language == "es" else "en"
    for ext in ("pdf", "png"):
        fig.savefig(
            output_dir / f"nonsmooth_corrected_probe_summary_{suffix}.{ext}",
            dpi=260,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 4))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-reference",
        action="store_true",
        help="Reutiliza una referencia completa ya generada en el mismo directorio.",
    )
    parser.add_argument(
        "--reuse-probes",
        action="store_true",
        help="Reutiliza las 675 sondas ya guardadas y ejecuta sólo controles y figuras.",
    )
    parser.add_argument(
        "--run-extended",
        action="store_true",
        help=(
            "Recorre el plan volumétrico de hasta 26,400 sondas y termina "
            "tras completar el primer radio con contactos."
        ),
    )
    parser.add_argument(
        "--resume-extended",
        action="store_true",
        help="Reanuda el sondeo extendido a partir de su checkpoint JSONL.",
    )
    parser.add_argument(
        "--extended-dirname",
        default="extended",
        help="Subdirectorio de salida del sondeo extendido.",
    )
    parser.add_argument(
        "--extended-max-per-radius",
        type=int,
        default=0,
        help=(
            "Limita muestras por grupo para una prueba técnica; 0 conserva "
            "íntegro el plan científico."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_config()

    contract = {
        "system": cfg["system"],
        "integrator": {
            "method": cfg["integrator"]["method"],
            "h": cfg["integrator"]["h"],
            "memory_mode": cfg["integrator"]["memory_mode"],
        },
        "parameter_grid": cfg["parameter_grid"],
        "biased_search": cfg["step2_biased_df_search"],
        "hiddenness_probe": cfg["step3_hiddenness"],
        "random_seed": cfg["experiment"]["random_seed"],
    }
    with (output_dir / "numerical_contract.json").open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(contract), stream, ensure_ascii=False, indent=2)

    if args.reuse_reference:
        candidate, reference_post = _load_reference(output_dir)
    else:
        print("Reconstruyendo raíz, continuación y trayectoria de referencia...", flush=True)
        candidate, reference_post = _build_reference(cfg, output_dir)
        print("Trayectoria de referencia completa.", flush=True)

    if args.reuse_probes:
        runs = pd.read_csv(output_dir / "probe_runs.csv")
        summary = pd.read_csv(output_dir / "probe_summary.csv")
        with (output_dir / "result.json").open(encoding="utf-8") as stream:
            result = json.load(stream)
    else:
        runs, summary, result = _probe(
            cfg,
            candidate,
            reference_post,
            output_dir,
            workers=max(1, int(args.workers)),
        )
    result["target_robustness"] = _target_robustness(
        cfg,
        candidate,
        reference_post,
        output_dir,
        workers=max(1, int(args.workers)),
    )
    with (output_dir / "result.json").open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(result), stream, ensure_ascii=False, indent=2)
    _plot_reference(reference_post, output_dir)
    _plot_probe_summary(runs, summary, output_dir, "es")
    _plot_probe_summary(runs, summary, output_dir, "en")

    if args.run_extended:
        extended_result = _run_extended_probe(
            cfg,
            candidate,
            reference_post,
            output_dir / args.extended_dirname,
            workers=max(1, int(args.workers)),
            resume=bool(args.resume_extended),
            max_per_radius=max(0, int(args.extended_max_per_radius)),
        )
        result["extended_probe"] = extended_result
        with (output_dir / "result.json").open("w", encoding="utf-8") as stream:
            json.dump(_jsonable(result), stream, ensure_ascii=False, indent=2)

    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
