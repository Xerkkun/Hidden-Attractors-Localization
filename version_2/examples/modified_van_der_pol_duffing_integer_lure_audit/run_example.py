#!/usr/bin/env python3
"""Run the reproducible integer MAVPD Lur'e localization audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.integrate import solve_ivp

EXAMPLE_DIR = Path(__file__).resolve().parent
VERSION2 = EXAMPLE_DIR.parents[1]
ROOT = VERSION2.parent
for _path in (VERSION2, ROOT):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from hidden_attractors.analysis.trajectory import (
    classify_trajectory_against_equilibria,
    cloud_median_distance,
    sample_rows,
)
from hidden_attractors.analysis.zero_one import zero_one_test
from hidden_attractors.plotting import (
    plot_integer_hiddenness_controls,
    plot_integer_lure_continuation,
    plot_lure_nyquist_describing_function,
    plot_lure_transfer_components,
    plot_phase_projections,
    plot_phase_space,
    plot_time_series,
    plot_trajectory_spectra,
)
from hidden_attractors.plotting.export import intercept_and_export_path
from hidden_attractors.seed_generation import (
    find_integer_lure_omega_gain_candidates_direct,
    lure_transfer_function,
)
from hidden_attractors.systems.modified_van_der_pol_duffing import mavpd_2023_system
from hidden_attractors.workflows import ContinuationPlan
from hidden_attractors.workflows.integer_lure import (
    IntegerHiddennessProbe,
    continue_integer_lure_seed,
    final_integer_lure_attractor,
    integer_lure_seed,
    integrate_integer_lure,
)


CONFIG_PATH = EXAMPLE_DIR / "reproducibility.yaml"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_trajectory(path: Path, trajectory: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, trajectory, delimiter=",", header="t,y1,y2,y3", comments="")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(VERSION2.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_config(*, config_path: Path = CONFIG_PATH, quick: bool = False) -> dict[str, Any]:
    """Load the frozen YAML and apply only its declared quick overrides."""

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if float(cfg["system"]["q"]) != 1.0:
        raise ValueError("this audit is restricted to q=1")
    if quick:
        for section, overrides in cfg.get("quick", {}).items():
            cfg[section].update(overrides)
    return cfg


def output_dir(cfg: dict[str, Any]) -> Path:
    return VERSION2 / Path(cfg["outputs"]["output_dir"])


def figure_dir(cfg: dict[str, Any]) -> Path:
    return VERSION2 / Path(cfg["outputs"]["figures_dir"])


def default_probe_path(cfg: dict[str, Any]) -> Path:
    return VERSION2 / Path(cfg["hiddenness"]["probe_input"])


def build_system(cfg: dict[str, Any], regime: str):
    """Construct one frozen parameter regime from the registered equations."""

    values = dict(cfg["system"]["shared_parameters"])
    values["xi"] = float(cfg["system"][regime]["xi"])
    return mavpd_2023_system(values)


def _tail_cloud(trajectory: np.ndarray, *, t_start: float, max_points: int) -> np.ndarray:
    data = np.asarray(trajectory, dtype=float)
    tail = data[data[:, 0] >= float(t_start), 1:]
    if tail.size == 0:
        tail = data[:, 1:]
    return sample_rows(tail, int(max_points))


def _cloud_scale(cloud: np.ndarray) -> float:
    return max(float(np.linalg.norm(np.ptp(np.asarray(cloud), axis=0))), 1.0e-12)


def _cloud_distance_norm(first: np.ndarray, second: np.ndarray) -> float:
    distance = cloud_median_distance(first, second)
    scale = max(_cloud_scale(first), _cloud_scale(second))
    return float(distance / scale)


def _trajectory_signature(trajectory: np.ndarray) -> dict[str, Any]:
    states = np.asarray(trajectory, dtype=float)[:, 1:]
    span = np.ptp(states, axis=0)
    return {
        "mean": np.mean(states, axis=0),
        "std": np.std(states, axis=0),
        "span": span,
        "span_norm": float(np.linalg.norm(span)),
        "max_norm": float(np.max(np.linalg.norm(states, axis=1))),
    }


def _poincare_section(trajectory: np.ndarray) -> np.ndarray:
    """Return linearly interpolated negative-to-positive y2=0 crossings."""

    data = np.asarray(trajectory, dtype=float)
    y2 = data[:, 2]
    indices = np.where((y2[:-1] < 0.0) & (y2[1:] >= 0.0))[0]
    rows: list[np.ndarray] = []
    for index in indices:
        denominator = y2[index + 1] - y2[index]
        if denominator == 0.0:
            continue
        fraction = -y2[index] / denominator
        rows.append(data[index] + fraction * (data[index + 1] - data[index]))
    return np.asarray(rows, dtype=float)


def _poincare_periodicity(
    trajectory: np.ndarray, *, section: np.ndarray | None = None
) -> dict[str, Any]:
    """Test period-one through period-four recurrence on the y2=0 section."""

    crossings = _poincare_section(trajectory) if section is None else np.asarray(section)
    if len(crossings) < 12:
        return {
            "n_crossings": int(len(crossings)),
            "periodic_candidate": False,
            "reason": "too_few_section_crossings",
        }
    # Normalize recurrence by the full attractor extent.  Normalizing by the
    # tiny residual spread of an already-collapsed Poincare point would turn
    # interpolation noise into an artificial O(1) drift.
    scale = max(
        float(np.linalg.norm(np.ptp(np.asarray(trajectory)[:, 1:], axis=0))),
        1.0e-12,
    )
    candidates: list[dict[str, Any]] = []
    for return_period in range(1, 5):
        state_errors = np.linalg.norm(
            crossings[return_period:, 1:] - crossings[:-return_period, 1:], axis=1
        )
        time_periods = crossings[return_period:, 0] - crossings[:-return_period, 0]
        tail_count = min(80, len(state_errors))
        candidates.append(
            {
                "return_period": return_period,
                "period_mean": float(np.mean(time_periods[-tail_count:])),
                "period_std": float(np.std(time_periods[-tail_count:])),
                "recurrence_median_norm": float(
                    np.median(state_errors[-tail_count:]) / scale
                ),
                "recurrence_max_norm": float(np.max(state_errors[-tail_count:]) / scale),
            }
        )
    qualifying = [
        row
        for row in candidates
        if row["recurrence_median_norm"] <= 1.0e-3
        and row["recurrence_max_norm"] <= 5.0e-3
    ]
    best = (
        min(qualifying, key=lambda row: row["return_period"])
        if qualifying
        else min(candidates, key=lambda row: row["recurrence_median_norm"])
    )
    return {
        "section": "y2=0_negative_to_positive",
        "section_interpolation": (
            "linear_sample_interpolation" if section is None else "DOP853_event_dense_interpolation"
        ),
        "n_crossings": int(len(crossings)),
        "attractor_scale": scale,
        "section_state_ranges": np.ptp(crossings[:, 1:], axis=0),
        "tested_return_periods": candidates,
        "selected": best,
        "periodic_candidate": bool(qualifying),
        "interpretation": "finite_recurrence_diagnostic_not_an_analytic_periodicity_proof",
    }


def _mavpd_rhs_and_jacobian(system):
    p = {key: float(value) for key, value in system.parameters.items()}

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        y1, y2, y3 = state
        return np.array(
            [
                p["delta"] * (p["gamma"] * y1 + y2 - y1**3),
                y1 - p["xi"] * y2 - y3,
                p["rho"] * y2,
            ],
            dtype=float,
        )

    def jacobian(state: np.ndarray) -> np.ndarray:
        y1 = float(state[0])
        return np.array(
            [
                [p["delta"] * (p["gamma"] - 3.0 * y1 * y1), p["delta"], 0.0],
                [1.0, -p["xi"], -1.0],
                [0.0, p["rho"], 0.0],
            ],
            dtype=float,
        )

    return rhs, jacobian


def _uniform_eval_times(duration: float, step: float) -> np.ndarray:
    count = int(np.floor(float(duration) / float(step)))
    times = float(step) * np.arange(count + 1, dtype=float)
    if times[-1] < float(duration) - 1.0e-12:
        times = np.append(times, float(duration))
    return times


def _final_dop853(
    system,
    x0: Sequence[float],
    *,
    t_burn: float,
    t_keep: float,
    sample_step: float,
    max_step: float,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    """Independently refine one candidate with SciPy DOP853."""

    rhs, _ = _mavpd_rhs_and_jacobian(system)
    burn = solve_ivp(
        rhs,
        (0.0, float(t_burn)),
        np.asarray(x0, dtype=float),
        method="DOP853",
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )
    if not burn.success:
        raise RuntimeError(f"DOP853 burn failed: {burn.message}")
    evaluation_times = _uniform_eval_times(float(t_keep), float(sample_step))
    def positive_y2_event(_time: float, state: np.ndarray) -> float:
        return float(state[1])

    positive_y2_event.direction = 1.0  # type: ignore[attr-defined]
    positive_y2_event.terminal = False  # type: ignore[attr-defined]
    kept = solve_ivp(
        rhs,
        (0.0, float(t_keep)),
        burn.y[:, -1],
        method="DOP853",
        t_eval=evaluation_times,
        events=positive_y2_event,
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
    )
    if not kept.success:
        raise RuntimeError(f"DOP853 keep failed: {kept.message}")
    trajectory = np.column_stack((kept.t, kept.y.T))
    event_times = kept.t_events[0]
    event_states = kept.y_events[0]
    section = np.column_stack((event_times, event_states))
    return burn.y[:, -1].copy(), trajectory, {
        "method": "DOP853",
        "rtol": float(rtol),
        "atol": float(atol),
        "max_step": float(max_step),
        "sample_step": float(sample_step),
        "burn_nfev": int(burn.nfev),
        "keep_nfev": int(kept.nfev),
        "section_method": "DOP853_positive_y2_event_dense_interpolation",
        "section_crossings": int(len(section)),
    }, section


def _dop853_variational_qr(system, x0: Sequence[float], cfg: dict[str, Any]) -> dict[str, Any]:
    """Estimate integer Lyapunov exponents with independent DOP853+QR."""

    rhs, jacobian = _mavpd_rhs_and_jacobian(system)
    burn_time = float(cfg["lyapunov_t_burn"])
    accumulation_time = float(cfg["lyapunov_t_accumulate"])
    interval = float(cfg["lyapunov_qr_interval"])
    state = np.asarray(x0, dtype=float).copy()
    basis = np.eye(3, dtype=float)
    log_sums = np.zeros(3, dtype=float)
    elapsed = 0.0
    history: list[dict[str, Any]] = []
    total_time = burn_time + accumulation_time
    n_segments = int(np.ceil(total_time / interval))

    def augmented_rhs(time_value: float, augmented: np.ndarray) -> np.ndarray:
        current = augmented[:3]
        tangent = augmented[3:].reshape(3, 3)
        return np.concatenate((rhs(time_value, current), (jacobian(current) @ tangent).ravel()))

    for segment in range(n_segments):
        start = segment * interval
        stop = min((segment + 1) * interval, total_time)
        if stop <= start:
            break
        augmented0 = np.concatenate((state, basis.ravel()))
        solved = solve_ivp(
            augmented_rhs,
            (start, stop),
            augmented0,
            method="DOP853",
            rtol=float(cfg["rtol"]),
            atol=float(cfg["atol"]),
            max_step=float(cfg["max_step"]),
        )
        if not solved.success:
            raise RuntimeError(f"DOP853 variational segment failed: {solved.message}")
        state = solved.y[:3, -1]
        tangent = solved.y[3:, -1].reshape(3, 3)
        basis, upper = np.linalg.qr(tangent)
        diagonal = np.diag(upper)
        signs = np.where(diagonal < 0.0, -1.0, 1.0)
        basis = basis * signs
        diagonal = np.abs(diagonal)
        if stop > burn_time:
            effective = stop - max(start, burn_time)
            if effective < stop - start - 1.0e-12:
                # The configured burn is an exact multiple of the QR interval
                # in maintained runs. Refuse an ambiguous partial segment.
                raise RuntimeError("lyapunov burn must align with the QR interval")
            log_sums += np.log(np.maximum(diagonal, np.finfo(float).tiny))
            elapsed += effective
            if len(history) == 0 or elapsed - history[-1]["time"] >= 5.0 - 1.0e-12:
                history.append({"time": elapsed, "exponents": log_sums / elapsed})
    exponents = log_sums / elapsed
    return {
        "method": "independent_DOP853_variational_QR",
        "t_burn": burn_time,
        "t_accumulate": elapsed,
        "qr_interval": interval,
        "exponents": exponents,
        "sum_exponents": float(np.sum(exponents)),
        "history": history,
        "interpretation": "one_exponent_tending_to_zero_and_two_negative_supports_a_stable_limit_cycle",
    }


def validate_system_declaration(system) -> dict[str, Any]:
    """Check equations, exact Lur'e equality, Jacobian, and equilibria."""

    if system.lure is None:
        raise ValueError("MAVPD must provide an explicit scalar Lur'e declaration")
    samples = (
        np.zeros(3),
        np.array([0.37, -0.21, 0.58]),
        np.array([-1.2, 0.4, -0.8]),
    )
    lure_residuals = [
        float(np.linalg.norm(system.evaluate(state) - system.lure.evaluate(state)))
        for state in samples
    ]
    point = samples[1]
    step = 1.0e-7
    finite_difference = np.column_stack(
        [
            (
                system.evaluate(point + step * direction)
                - system.evaluate(point - step * direction)
            )
            / (2.0 * step)
            for direction in np.eye(3)
        ]
    )
    jacobian_residual = float(
        np.linalg.norm(system.jacobian_matrix(point) - finite_difference)
    )
    equilibria = []
    for label, state in system.equilibrium_points().items():
        jacobian = system.jacobian_matrix(state)
        equilibria.append(
            {
                "label": label,
                "state": state,
                "rhs_residual": float(np.linalg.norm(system.evaluate(state))),
                "jacobian": jacobian,
                "eigenvalues": np.linalg.eigvals(jacobian),
                "locally_asymptotically_stable": bool(
                    np.max(np.real(np.linalg.eigvals(jacobian))) < 0.0
                ),
            }
        )
    if max(lure_residuals) > 1.0e-12:
        raise RuntimeError("the exact Lur'e split does not reproduce the MAVPD equations")
    if jacobian_residual > 2.0e-7:
        raise RuntimeError("the MAVPD analytic Jacobian failed its finite-difference check")
    if max(row["rhs_residual"] for row in equilibria) > 1.0e-12:
        raise RuntimeError("a declared MAVPD equilibrium does not satisfy the equations")
    return {
        "system": system.name,
        "q": 1.0,
        "parameters": dict(system.parameters),
        "reference": dict(system.reference),
        "lure": {
            "matrix": system.lure.matrix,
            "input_vector": system.lure.input_vector,
            "output_vector": system.lure.output_vector,
            "nonlinearity": "psi(sigma)=sigma^3",
            "describing_function": "N(A)=3*A^2/4",
            "transfer_convention": "c^T (P - s I)^(-1) b",
            "form": "exact_scalar",
        },
        "max_lure_field_residual": max(lure_residuals),
        "lure_field_residuals": lure_residuals,
        "jacobian_finite_difference_residual": jacobian_residual,
        "equilibria": equilibria,
        "status": "passed",
    }


def run_contract(cfg: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    primary = context.setdefault("primary_system", build_system(cfg, "primary"))
    negative = context.setdefault(
        "negative_system", build_system(cfg, "negative_control")
    )
    result = {
        "primary_xi_3p1": validate_system_declaration(primary),
        "negative_xi_3p5": validate_system_declaration(negative),
        "contract_decision": "both_parameter_regimes_pass_exact_algebraic_contract",
    }
    _write_json(output_dir(cfg) / "00_system_contract.json", result)
    context["contract"] = result
    return result


def derive_direct_seed_records(
    system, cfg: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Derive all direct branches and phases without consulting Table 1."""

    direct_cfg = cfg["direct_route"]
    if direct_cfg["route"] != "direct_integer_transfer" or direct_cfg["fallback_route"] is not None:
        raise ValueError("the maintained MAVPD audit requires the direct route without scan fallback")
    all_pairs = find_integer_lure_omega_gain_candidates_direct(
        system.lure,
        wmin=float(direct_cfg["omega_min"]),
        wmax=float(direct_cfg["omega_max"]),
        compatible_only=False,
    )
    compatible_pairs = find_integer_lure_omega_gain_candidates_direct(
        system.lure,
        wmin=float(direct_cfg["omega_min"]),
        wmax=float(direct_cfg["omega_max"]),
        compatible_only=True,
    )
    p = system.parameters
    polynomial = np.array(
        [
            1.0,
            float(p["delta"]) - 2.0 * float(p["rho"]) + float(p["xi"]) ** 2,
            float(p["rho"]) * (float(p["rho"]) - float(p["delta"])),
        ]
    )
    z_roots = sorted(
        float(root.real)
        for root in np.roots(polynomial)
        if abs(float(root.imag)) <= 1.0e-10 and float(root.real) > 0.0
    )
    omega_from_polynomial = [float(np.sqrt(root)) for root in z_roots]
    if len(all_pairs) != len(z_roots) or compatible_pairs != all_pairs:
        raise RuntimeError("the direct polynomial branches and compatible DF branches disagree")
    if not np.allclose(
        [pair[0] for pair in all_pairs], omega_from_polynomial, rtol=0.0, atol=2.0e-10
    ):
        raise RuntimeError("the direct transfer roots do not match the analytic z polynomial")

    seed_entries: list[dict[str, Any]] = []
    seed_payloads: list[dict[str, Any]] = []
    for branch in [int(value) for value in direct_cfg["branch_order"]]:
        for phase in [float(value) for value in direct_cfg["seed_phases"]]:
            seed = integer_lure_seed(
                system,
                branch_index=branch,
                method=str(direct_cfg["method"]),
                theta=phase,
                wmin=float(direct_cfg["omega_min"]),
                wmax=float(direct_cfg["omega_max"]),
            )
            route_id = f"branch_{branch}_phase_{phase / np.pi:.1f}pi"
            transfer = lure_transfer_function(seed.omega, 1.0, system.lure)
            payload = {
                "route_id": route_id,
                "execution_order": len(seed_entries),
                "branch_index": branch,
                "phase": phase,
                "omega": seed.omega,
                "gain": seed.gain,
                "amplitude": seed.amplitude,
                "seed": seed.seed,
                "search_route": seed.search_route,
                "transfer_value": transfer,
                "nyquist_closure_residual": abs(1.0 + seed.gain * transfer),
                "describing_function_residual": abs(
                    float(system.lure.describing_function(seed.amplitude)) - seed.gain
                ),
                "published_table_used": False,
            }
            seed_entries.append({**payload, "seed_object": seed})
            seed_payloads.append(payload)
    if not seed_entries or seed_entries[0]["branch_index"] != 0:
        raise RuntimeError("branch 0 must be executed first")
    result = {
        "route": "direct_integer_transfer_no_grid",
        "frequency_grid_used": False,
        "fallback_used": False,
        "analytic_z_polynomial_coefficients": polynomial,
        "analytic_positive_z_roots": z_roots,
        "analytic_positive_omega_roots": omega_from_polynomial,
        "omega_gain_pairs_all": all_pairs,
        "omega_gain_pairs_compatible": compatible_pairs,
        "seed_execution_order": seed_payloads,
        "table_1_role": "post_derivation_only_not_read_by_this_function",
    }
    return result, seed_entries


def _cluster_candidate_records(
    records: list[dict[str, Any]], *, threshold: float, max_points: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    assignment: dict[str, int] = {}
    for record in records:
        cloud = _tail_cloud(record["trajectory"], t_start=0.0, max_points=max_points)
        assigned = None
        assigned_distance = float("nan")
        for index, cluster in enumerate(clusters):
            distance = _cloud_distance_norm(cloud, cluster["cloud"])
            if distance <= threshold:
                assigned = index
                assigned_distance = distance
                break
        if assigned is None:
            assigned = len(clusters)
            assigned_distance = 0.0
            clusters.append(
                {
                    "cloud": cloud,
                    "representative": record,
                    "members": [],
                }
            )
        clusters[assigned]["members"].append(
            {"route_id": record["route_id"], "distance_to_representative": assigned_distance}
        )
        assignment[record["route_id"]] = assigned

    selected_index = max(
        range(len(clusters)),
        key=lambda index: _trajectory_signature(
            clusters[index]["representative"]["trajectory"]
        )["span_norm"],
    )
    semantics: dict[int, str] = {selected_index: "outer_recurrent_candidate"}
    remaining = [index for index in range(len(clusters)) if index != selected_index]
    for index in remaining:
        mean_y1 = float(
            _trajectory_signature(clusters[index]["representative"]["trajectory"])[
                "mean"
            ][0]
        )
        semantics[index] = "inner_positive" if mean_y1 >= 0.0 else "inner_negative"

    payload_clusters = []
    runtime_clusters: dict[str, Any] = {}
    for index, cluster in enumerate(clusters):
        semantic = semantics[index]
        representative = cluster["representative"]
        signature = _trajectory_signature(representative["trajectory"])
        payload_clusters.append(
            {
                "cluster_id": index,
                "semantic_label": semantic,
                "representative_route": representative["route_id"],
                "members": cluster["members"],
                "signature": signature,
            }
        )
        runtime_clusters[semantic] = {
            "cluster_id": index,
            "cloud": cluster["cloud"],
            "representative": representative,
            "signature": signature,
        }
    for route_id, index in list(assignment.items()):
        assignment[route_id] = semantics[index]
    return (
        {
            "method": "greedy_symmetric_median_cloud_distance",
            "threshold": threshold,
            "n_clusters": len(clusters),
            "selection_rule": "largest_blind_state_span_norm",
            "selected_cluster": "outer_recurrent_candidate",
            "clusters": payload_clusters,
            "route_assignment": assignment,
        },
        runtime_clusters,
    )


def run_primary_route(cfg: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    system = context.setdefault("primary_system", build_system(cfg, "primary"))
    if "contract" not in context:
        run_contract(cfg, context)
    direct, seed_entries = derive_direct_seed_records(system, cfg)
    _write_json(output_dir(cfg) / "01_primary_direct_branches.json", direct)

    continuation_cfg = cfg["continuation"]
    continuation_phases = [
        float(value) for value in cfg["direct_route"]["continuation_phases"]
    ]
    plan = ContinuationPlan(
        tuple(float(value) for value in continuation_cfg["lambda_values"]),
        {
            "public_parameter": "lambda",
            "source": "A_plus_kbc",
            "target": "exact_cubic_MAVPD",
        },
    )
    final_cfg = cfg["final_simulation"]
    candidates: list[dict[str, Any]] = []
    trace_payload: list[dict[str, Any]] = []
    trace_csv_rows: list[list[Any]] = []
    continuation_objects: dict[str, Any] = {}
    for entry in seed_entries:
        phase = float(entry["phase"])
        use_continuation = any(abs(phase - value) <= 1.0e-12 for value in continuation_phases)
        if use_continuation:
            steps = continue_integer_lure_seed(
                system,
                entry["seed_object"],
                plan=plan,
                t_transient=float(continuation_cfg["t_transient"]),
                t_keep=float(continuation_cfg["t_keep"]),
                h=float(continuation_cfg["h"]),
                div_threshold=float(continuation_cfg["div_threshold"]),
            )
            if len(steps) != len(plan.lambda_values) or steps[-1].status != "ok":
                raise RuntimeError(f"continuation failed for {entry['route_id']}")
            continuation_objects[entry["route_id"]] = steps
            x_start = steps[-1].x_out
            step_rows = []
            for step_index, step in enumerate(steps):
                relative = (
                    Path("continuation_steps")
                    / f"{entry['route_id']}_lambda_{step_index:02d}.csv"
                )
                _write_trajectory(
                    output_dir(cfg) / relative,
                    sample_rows(step.trajectory, 1001),
                )
                step_row = {
                    "step": step_index,
                    "lambda": step.lambda_value,
                    "x_in": step.x_in,
                    "x_out": step.x_out,
                    "status": step.status,
                    "trajectory": relative.as_posix(),
                }
                step_rows.append(step_row)
                trace_csv_rows.append(
                    [
                        entry["route_id"],
                        step_index,
                        step.lambda_value,
                        step.status,
                        *step.x_in.tolist(),
                        *step.x_out.tolist(),
                    ]
                )
        else:
            steps = []
            step_rows = []
            x_start = entry["seed_object"].seed
        target_seed, trajectory, status = final_integer_lure_attractor(
            system,
            x_start,
            t_burn=float(final_cfg["t_burn"]),
            t_keep=float(final_cfg["t_keep"]),
            h=float(final_cfg["h"]),
            div_threshold=float(final_cfg["div_threshold"]),
        )
        if status != "ok":
            raise RuntimeError(f"final integration failed for {entry['route_id']}: {status}")
        relative = Path("primary_candidates") / f"{entry['route_id']}.csv"
        _write_trajectory(output_dir(cfg) / relative, trajectory)
        candidate = {
            "route_id": entry["route_id"],
            "branch_index": entry["branch_index"],
            "phase": entry["phase"],
            "continuation_used": use_continuation,
            "target_seed": target_seed,
            "trajectory": trajectory,
            "trajectory_path": relative.as_posix(),
            "signature": _trajectory_signature(trajectory),
        }
        candidates.append(candidate)
        trace_payload.append(
            {
                "route_id": entry["route_id"],
                "branch_index": entry["branch_index"],
                "phase": entry["phase"],
                "continuation_used": use_continuation,
                "steps": step_rows,
                "final_trajectory": relative.as_posix(),
                "final_signature": candidate["signature"],
            }
        )

    _write_json(
        output_dir(cfg) / "02_primary_continuation_and_integration.json",
        {
            "plan": {
                "lambda_values": plan.lambda_values,
                "h": continuation_cfg["h"],
                "t_transient": continuation_cfg["t_transient"],
                "t_keep": continuation_cfg["t_keep"],
            },
            "routes": trace_payload,
        },
    )
    trace_csv = output_dir(cfg) / "02_primary_continuation_trace.csv"
    trace_csv.parent.mkdir(parents=True, exist_ok=True)
    with trace_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "route_id",
                "step",
                "lambda",
                "status",
                "y1_in",
                "y2_in",
                "y3_in",
                "y1_out",
                "y2_out",
                "y3_out",
            ]
        )
        writer.writerows(trace_csv_rows)

    cluster_payload, runtime_clusters = _cluster_candidate_records(
        candidates,
        threshold=float(cfg["direct_route"]["cluster_threshold"]),
        max_points=int(cfg["direct_route"]["cluster_cloud_points"]),
    )
    if cluster_payload["n_clusters"] != 3:
        raise RuntimeError("the blind primary route did not recover the expected three clusters")
    selected = runtime_clusters["outer_recurrent_candidate"]["representative"]
    exploratory_trajectory = selected["trajectory"]
    adaptive_cfg = cfg["adaptive_refinement"]
    refined_seed, refined_trajectory, adaptive_solver, section = _final_dop853(
        system,
        selected["target_seed"],
        t_burn=float(adaptive_cfg["t_burn"]),
        t_keep=float(adaptive_cfg["t_keep"]),
        sample_step=float(adaptive_cfg["sample_step"]),
        max_step=float(adaptive_cfg["max_step"]),
        rtol=float(adaptive_cfg["rtol"]),
        atol=float(adaptive_cfg["atol"]),
    )
    periodicity = _poincare_periodicity(refined_trajectory, section=section)
    if not periodicity["periodic_candidate"]:
        raise RuntimeError("strict DOP853 refinement did not recover the stable outer cycle")
    section_path = output_dir(cfg) / "04_primary_poincare_section.csv"
    np.savetxt(
        section_path,
        section,
        delimiter=",",
        header="t,y1,y2,y3",
        comments="",
    )
    window_diagnostics = []
    for window in [float(value) for value in adaptive_cfg["convergence_windows"]]:
        subset = refined_trajectory[refined_trajectory[:, 0] <= window]
        subset_section = section[section[:, 0] <= window]
        window_diagnostics.append(
            {
                "window": window,
                "poincare": _poincare_periodicity(subset, section=subset_section),
            }
        )
    stride = int(adaptive_cfg["zero_one_stride"])
    zero_one = zero_one_test(
        refined_trajectory[::stride, 1],
        n_c=int(adaptive_cfg["zero_one_c_values"]),
        random_seed=20230801,
        max_samples=int(adaptive_cfg["zero_one_max_samples"]),
    )
    lyapunov = _dop853_variational_qr(system, refined_seed, adaptive_cfg)
    selected["exploratory_efork_trajectory"] = exploratory_trajectory
    selected["trajectory"] = refined_trajectory
    selected["target_seed"] = refined_seed
    selected["signature"] = _trajectory_signature(refined_trajectory)
    runtime_clusters["outer_recurrent_candidate"]["cloud"] = _tail_cloud(
        refined_trajectory,
        t_start=0.0,
        max_points=int(cfg["direct_route"]["cluster_cloud_points"]),
    )
    runtime_clusters["outer_recurrent_candidate"]["representative"] = selected
    runtime_clusters["outer_recurrent_candidate"]["signature"] = selected["signature"]
    _write_json(output_dir(cfg) / "03_primary_candidate_clusters.json", cluster_payload)
    cluster_csv = output_dir(cfg) / "03_primary_candidate_clusters.csv"
    with cluster_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "route_id",
                "cluster",
                "branch_index",
                "phase",
                "continuation_used",
                "span_norm",
                "mean_y1",
            ]
        )
        for candidate in candidates:
            writer.writerow(
                [
                    candidate["route_id"],
                    cluster_payload["route_assignment"][candidate["route_id"]],
                    candidate["branch_index"],
                    candidate["phase"],
                    candidate["continuation_used"],
                    candidate["signature"]["span_norm"],
                    candidate["signature"]["mean"][0],
                ]
            )
    _write_trajectory(
        output_dir(cfg) / "04_primary_outer_candidate_exploratory_efork.csv",
        exploratory_trajectory,
    )
    _write_trajectory(
        output_dir(cfg) / "04_primary_outer_candidate.csv",
        sample_rows(selected["trajectory"], 60001),
    )
    result = {
        "candidate_label": "outer_recurrent_candidate",
        "candidate_route": selected["route_id"],
        "selection_used_table_1": False,
        "selection_rule": cluster_payload["selection_rule"],
        "signature": selected["signature"],
        "exploratory_integrator": "hidden_attractors_fo_EFORK_q1",
        "adaptive_refinement": adaptive_solver,
        "poincare_convergence_windows": window_diagnostics,
        "periodicity": periodicity,
        "zero_one": zero_one,
        "lyapunov": lyapunov,
        "local_dynamic_class": "stable_periodic_candidate",
        "source_dynamic_label": "quasiperiodic_hidden_attractor",
        "source_reproduction_discrepancy": (
            "strict_local_DOP853_sections_and_variational_exponents_support_a_stable_"
            "periodic_cycle_for_the_declared_equations"
        ),
        "scientific_status": "positive_periodic_candidate_pending_finite_hiddenness_controls",
    }
    _write_json(output_dir(cfg) / "04_primary_candidate_summary.json", result)
    context.update(
        {
            "primary_direct": direct,
            "primary_seed_entries": seed_entries,
            "primary_candidates": candidates,
            "primary_clusters": cluster_payload,
            "primary_runtime_clusters": runtime_clusters,
            "primary_selected": selected,
            "primary_continuation_objects": continuation_objects,
            "primary_candidate_summary": result,
        }
    )
    return result


def _load_probe_contract(
    path: Path, *, cfg: dict[str, Any], system
) -> list[dict[str, Any]]:
    expected_hash = str(cfg["hiddenness"]["expected_sha256"]).lower()
    actual_hash = _sha256(path)
    if actual_hash.lower() != expected_hash:
        raise RuntimeError(
            f"shared probe CSV checksum mismatch: expected {expected_hash}, got {actual_hash}"
        )
    equilibrium_map = {
        "P1": ("E0", system.equilibrium_points()["E0"]),
        "P2": ("E+", system.equilibrium_points()["E+"]),
        "P3": ("E-", system.equilibrium_points()["E-"]),
    }
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            source_label = str(raw["equilibrium"])
            if source_label not in equilibrium_map:
                raise ValueError(f"unknown probe equilibrium label {source_label}")
            equilibrium, equilibrium_state = equilibrium_map[source_label]
            x0 = np.array([raw["y1"], raw["y2"], raw["y3"]], dtype=float)
            radius = float(raw["radius"])
            distance = float(np.linalg.norm(x0 - equilibrium_state))
            if abs(distance - radius) > 2.0e-12:
                raise RuntimeError(f"probe {raw['sample_id']} is not on its declared radius")
            rows.append(
                {
                    "sample_id": int(raw["sample_id"]),
                    "source_equilibrium": source_label,
                    "equilibrium": equilibrium,
                    "radius": radius,
                    "direction_id": int(raw["direction_id"]),
                    "x0": x0,
                    "distance_from_equilibrium": distance,
                }
            )
    if len(rows) != int(cfg["hiddenness"]["expected_rows"]):
        raise RuntimeError("shared probe CSV does not contain the expected 108 rows")
    return rows


def _classify_probe_against_clusters(
    trajectory: np.ndarray,
    runtime_clusters: dict[str, Any],
    *,
    t_start: float,
    max_points: int,
) -> tuple[str, dict[str, float]]:
    cloud = _tail_cloud(trajectory, t_start=t_start, max_points=max_points)
    distances = {
        label: _cloud_distance_norm(cloud, cluster["cloud"])
        for label, cluster in runtime_clusters.items()
    }
    assigned = min(distances, key=distances.get)
    return assigned, distances


def run_primary_hiddenness(
    cfg: dict[str, Any], context: dict[str, Any], probe_path: Path
) -> dict[str, Any]:
    if "primary_selected" not in context:
        run_primary_route(cfg, context)
    system = context["primary_system"]
    rows = _load_probe_contract(probe_path, cfg=cfg, system=system)
    destination = output_dir(cfg) / "05_hiddenness_initial_conditions.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(probe_path, destination)
    hidden_cfg = cfg["hiddenness"]
    equilibria = system.equilibrium_points()
    runtime_clusters = context["primary_runtime_clusters"]
    probe_objects: list[IntegerHiddennessProbe] = []
    probe_payloads: list[dict[str, Any]] = []
    counts = {
        "inner_positive": 0,
        "inner_negative": 0,
        "outer_recurrent_candidate": 0,
        "unmatched": 0,
    }
    target_hits = 0
    for row in rows:
        trajectory, status = integrate_integer_lure(
            system,
            row["x0"],
            t_final=float(hidden_cfg["t_final"]),
            h=float(hidden_cfg["h"]),
            div_threshold=float(hidden_cfg["div_threshold"]),
        )
        metrics = classify_trajectory_against_equilibria(
            trajectory,
            equilibria,
            divergence_norm=float(hidden_cfg["div_threshold"]),
            equilibrium_tol=float(hidden_cfg["equilibrium_tol"]),
            t_start=float(hidden_cfg["t_burn"]),
        )
        assigned, distances = _classify_probe_against_clusters(
            trajectory,
            runtime_clusters,
            t_start=float(hidden_cfg["t_burn"]),
            max_points=int(hidden_cfg["max_cloud_points"]),
        )
        matched = bool(
            distances[assigned] <= float(hidden_cfg["cluster_assignment_tol"])
        )
        cluster = assigned if matched else "unmatched"
        counts[cluster] += 1
        target_hit = bool(
            status == "ok"
            and metrics["final_class"] == "bounded_nontrivial"
            and distances["outer_recurrent_candidate"]
            <= float(hidden_cfg["target_cloud_tol"])
        )
        target_hits += int(target_hit)
        probe = IntegerHiddennessProbe(
            equilibrium=row["equilibrium"],
            radius=row["radius"],
            sample_id=row["sample_id"],
            x0=row["x0"],
            status=status,
            final_class=str(metrics["final_class"]),
            target_hit=target_hit,
            cloud_distance=float("nan"),
            cloud_distance_norm=distances["outer_recurrent_candidate"],
            trajectory=trajectory,
            metrics=metrics,
            sampling_mode="shared_explicit_direction_csv",
            distance_from_equilibrium=row["distance_from_equilibrium"],
        )
        probe_objects.append(probe)
        probe_payloads.append(
            {
                **row,
                "status": status,
                "final_class": metrics["final_class"],
                "assigned_cluster": cluster,
                "matched_cluster": matched,
                "cluster_distances_norm": distances,
                "target_hit": target_hit,
            }
        )
    if target_hits != 0:
        raise RuntimeError("the shared primary probe contract contacted the outer candidate")
    summary = {
        "probe_contract": {
            "path": _display_path(probe_path),
            "sha256": _sha256(probe_path),
            "rows": len(rows),
            "construction": hidden_cfg["sampling_contract"],
            "h": float(hidden_cfg["h"]),
            "t_final": float(hidden_cfg["t_final"]),
            "t_burn": float(hidden_cfg["t_burn"]),
            "t_keep": float(hidden_cfg["t_final"]) - float(hidden_cfg["t_burn"]),
            "equilibrium_tol": float(hidden_cfg["equilibrium_tol"]),
            "target_cloud_tol": float(hidden_cfg["target_cloud_tol"]),
            "max_cloud_points": int(hidden_cfg["max_cloud_points"]),
        },
        "cluster_counts": counts,
        "target_hits": target_hits,
        "hidden_candidate_allowed": True,
        "interpretation": "finite_shared_equilibrium_neighborhood_screen_not_global_proof",
        "scientific_status": "outer_periodic_candidate_supported_under_finite_108_probe_contract",
        "probes": probe_payloads,
    }
    _write_json(output_dir(cfg) / "05_primary_hiddenness_summary.json", summary)
    probe_csv = output_dir(cfg) / "05_primary_hiddenness_classification.csv"
    with probe_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "equilibrium",
                "radius",
                "direction_id",
                "y1",
                "y2",
                "y3",
                "status",
                "final_class",
                "assigned_cluster",
                "target_distance_norm",
                "target_hit",
            ]
        )
        for row in probe_payloads:
            writer.writerow(
                [
                    row["sample_id"],
                    row["equilibrium"],
                    row["radius"],
                    row["direction_id"],
                    *row["x0"].tolist(),
                    row["status"],
                    row["final_class"],
                    row["assigned_cluster"],
                    row["cluster_distances_norm"]["outer_recurrent_candidate"],
                    row["target_hit"],
                ]
            )
    context.update(
        {
            "primary_probe_rows": rows,
            "primary_probes": probe_objects,
            "primary_hiddenness": summary,
        }
    )
    return summary


def _cluster_negative_candidates(
    records: list[dict[str, Any]], *, threshold: float, max_points: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, runtime = _cluster_candidate_records(
        records, threshold=threshold, max_points=max_points
    )
    outer = runtime.pop("outer_recurrent_candidate")
    outer_mean = float(outer["signature"]["mean"][0])
    replacement = "cycle_positive" if outer_mean >= 0.0 else "cycle_negative"
    runtime[replacement] = outer
    for cluster in payload["clusters"]:
        if cluster["semantic_label"] == "outer_recurrent_candidate":
            cluster["semantic_label"] = replacement
        elif cluster["semantic_label"] == "inner_positive":
            cluster["semantic_label"] = "cycle_positive"
        elif cluster["semantic_label"] == "inner_negative":
            cluster["semantic_label"] = "cycle_negative"
    for route_id, label in payload["route_assignment"].items():
        if label == "outer_recurrent_candidate":
            payload["route_assignment"][route_id] = replacement
        elif label == "inner_positive":
            payload["route_assignment"][route_id] = "cycle_positive"
        elif label == "inner_negative":
            payload["route_assignment"][route_id] = "cycle_negative"
    renamed: dict[str, Any] = {}
    for label, data in runtime.items():
        if label == "inner_positive":
            renamed["cycle_positive"] = data
        elif label == "inner_negative":
            renamed["cycle_negative"] = data
        else:
            renamed[label] = data
    payload["selected_cluster"] = "first_branch_zero_cycle"
    payload["selection_rule"] = "mandatory_branch_zero_phase_zero_first"
    return payload, renamed


def run_negative_audit(
    cfg: dict[str, Any], context: dict[str, Any], probe_path: Path
) -> dict[str, Any]:
    system = context.setdefault(
        "negative_system", build_system(cfg, "negative_control")
    )
    if "contract" not in context:
        run_contract(cfg, context)
    direct, seed_entries = derive_direct_seed_records(system, cfg)
    negative_cfg = cfg["negative_audit"]
    candidates: list[dict[str, Any]] = []
    for entry in seed_entries:
        target_seed, trajectory, status = final_integer_lure_attractor(
            system,
            entry["seed_object"].seed,
            t_burn=float(negative_cfg["direct_t_burn"]),
            t_keep=float(negative_cfg["direct_t_keep"]),
            h=float(negative_cfg["h"]),
            div_threshold=float(negative_cfg["div_threshold"]),
        )
        if status != "ok":
            raise RuntimeError(f"negative direct route failed for {entry['route_id']}")
        relative = Path("negative_candidates") / f"{entry['route_id']}.csv"
        _write_trajectory(output_dir(cfg) / relative, trajectory)
        candidates.append(
            {
                "route_id": entry["route_id"],
                "branch_index": entry["branch_index"],
                "phase": entry["phase"],
                "target_seed": target_seed,
                "trajectory": trajectory,
                "trajectory_path": relative.as_posix(),
                "signature": _trajectory_signature(trajectory),
            }
        )
    cluster_payload, runtime_clusters = _cluster_negative_candidates(
        candidates,
        threshold=float(cfg["direct_route"]["cluster_threshold"]),
        max_points=int(cfg["direct_route"]["cluster_cloud_points"]),
    )
    if len(runtime_clusters) != 2:
        raise RuntimeError("xi=3.5 direct routes did not collapse to two symmetric cycles")
    target = candidates[0]
    target_label = cluster_payload["route_assignment"][target["route_id"]]

    probe_rows = _load_probe_contract(probe_path, cfg=cfg, system=system)
    probe_payloads = []
    target_probe_trajectory = None
    first_target_contact = None
    target_hits_by_cluster = {label: 0 for label in runtime_clusters}
    final_class_counts: dict[str, int] = {}
    equilibria = system.equilibrium_points()
    for row_index, row in enumerate(probe_rows):
        _, trajectory, status = final_integer_lure_attractor(
            system,
            row["x0"],
            t_burn=float(negative_cfg["probe_t_burn"]),
            t_keep=float(negative_cfg["probe_t_keep"]),
            h=float(negative_cfg["h"]),
            div_threshold=float(negative_cfg["div_threshold"]),
        )
        metrics = classify_trajectory_against_equilibria(
            trajectory,
            equilibria,
            divergence_norm=float(negative_cfg["div_threshold"]),
            equilibrium_tol=float(cfg["hiddenness"]["equilibrium_tol"]),
            t_start=0.0,
        )
        assigned, distances = _classify_probe_against_clusters(
            trajectory,
            runtime_clusters,
            t_start=0.0,
            max_points=int(cfg["hiddenness"]["max_cloud_points"]),
        )
        final_class = str(metrics["final_class"])
        final_class_counts[final_class] = final_class_counts.get(final_class, 0) + 1
        bounded = bool(status == "ok" and final_class == "bounded_nontrivial")
        hits = {
            label: bool(
                bounded
                and distance <= float(cfg["hiddenness"]["target_cloud_tol"])
            )
            for label, distance in distances.items()
        }
        for label, hit in hits.items():
            target_hits_by_cluster[label] += int(hit)
        matched = bool(
            distances[assigned] <= float(cfg["hiddenness"]["cluster_assignment_tol"])
        )
        assigned_cluster = assigned if matched else "unmatched"
        target_hit = hits[target_label]
        if target_hit and target_probe_trajectory is None:
            target_probe_trajectory = trajectory
        if target_hit and first_target_contact is None:
            first_target_contact = {
                "probe_row_index": row_index,
                "sample_id": row["sample_id"],
                "equilibrium": row["equilibrium"],
                "radius": row["radius"],
                "direction_id": row["direction_id"],
                "target_cluster": target_label,
                "target_distance_norm": distances[target_label],
            }
        probe_payloads.append(
            {
                **row,
                "status": status,
                "final_class": final_class,
                "assigned_cluster": assigned_cluster,
                "matched_cluster": matched,
                "cluster_distances_norm": distances,
                "cluster_hits": hits,
                "target_distance_norm": distances[target_label],
                "target_hit": target_hit,
            }
        )
    target_hits = target_hits_by_cluster[target_label]
    if target_hits == 0:
        raise RuntimeError("the xi=3.5 shared 108-probe audit found no target contact")

    fallback_cfg = cfg["fallback_negative"]
    fallback_records: list[dict[str, Any]] = []
    for branch in [int(value) for value in fallback_cfg["branches"]]:
        for phase in [float(value) for value in fallback_cfg["phases"]]:
            seed = integer_lure_seed(
                system,
                branch_index=branch,
                theta=phase,
                wmin=float(cfg["direct_route"]["omega_min"]),
                wmax=float(cfg["direct_route"]["omega_max"]),
            )
            route_id = f"fallback_branch_{branch}_phase_{phase / np.pi:.1f}pi"
            _, trajectory, status = final_integer_lure_attractor(
                system,
                seed.seed,
                t_burn=float(fallback_cfg["t_burn"]),
                t_keep=float(fallback_cfg["t_keep"]),
                h=float(fallback_cfg["h"]),
                div_threshold=float(fallback_cfg["div_threshold"]),
            )
            fallback_records.append(
                {
                    "route_id": route_id,
                    "source": "direct_branch_phase",
                    "branch": branch,
                    "phase": phase,
                    "x0": seed.seed,
                    "status": status,
                    "trajectory": trajectory,
                }
            )
    for index, x0 in enumerate(fallback_cfg["initial_conditions"]):
        route_id = f"fallback_initial_condition_{index}"
        _, trajectory, status = final_integer_lure_attractor(
            system,
            x0,
            t_burn=float(fallback_cfg["t_burn"]),
            t_keep=float(fallback_cfg["t_keep"]),
            h=float(fallback_cfg["h"]),
            div_threshold=float(fallback_cfg["div_threshold"]),
        )
        fallback_records.append(
            {
                "route_id": route_id,
                "source": "deterministic_initial_condition",
                "branch": None,
                "phase": None,
                "x0": np.asarray(x0, dtype=float),
                "status": status,
                "trajectory": trajectory,
            }
        )
    fallback_payloads = []
    for record in fallback_records:
        cloud = _tail_cloud(
            record["trajectory"],
            t_start=0.0,
            max_points=int(cfg["direct_route"]["cluster_cloud_points"]),
        )
        distances = {
            label: _cloud_distance_norm(cloud, cluster["cloud"])
            for label, cluster in runtime_clusters.items()
        }
        assigned = min(distances, key=distances.get)
        fallback_payloads.append(
            {
                key: value
                for key, value in record.items()
                if key != "trajectory"
            }
            | {
                "assigned_cluster": assigned,
                "cluster_distances_norm": distances,
                "signature": _trajectory_signature(record["trajectory"]),
            }
        )
    result = {
        "parameters": dict(system.parameters),
        "direct_route": direct,
        "direct_clusters": cluster_payload,
        "blind_target_route": target["route_id"],
        "blind_target_cluster": target_label,
        "finite_probe_contract": {
            "path": _display_path(probe_path),
            "sha256": _sha256(probe_path),
            "rows_used": len(probe_rows),
            "selection": "all_rows_from_shared_3_equilibria_x_3_radii_x_12_direction_contract",
            "h": float(negative_cfg["h"]),
            "t_final": float(negative_cfg["probe_t_burn"])
            + float(negative_cfg["probe_t_keep"]),
            "t_burn": float(negative_cfg["probe_t_burn"]),
            "t_keep": float(negative_cfg["probe_t_keep"]),
            "equilibrium_tol": float(cfg["hiddenness"]["equilibrium_tol"]),
            "target_cloud_tol": float(cfg["hiddenness"]["target_cloud_tol"]),
            "cluster_assignment_tol": float(
                cfg["hiddenness"]["cluster_assignment_tol"]
            ),
            "target_hits": target_hits,
            "target_hits_by_cluster": target_hits_by_cluster,
            "final_class_counts": final_class_counts,
            "first_target_contact": first_target_contact,
            "probes": probe_payloads,
        },
        "fallback": {
            "trigger": fallback_cfg["trigger"],
            "executed_after_target_contact": True,
            "first_target_contact_recorded_before_execution": True,
            "frequency_grid_used": False,
            "records": fallback_payloads,
        },
        "scientific_decision": "hiddenness_not_reproduced_under_exact_equations_and_finite_controls",
        "interpretation": "shared_108_probe_contacts_block_the_hidden_label_for_both_cycles_under_the_tested_contract",
    }
    _write_json(output_dir(cfg) / "06_negative_xi3p5_audit.json", result)
    negative_csv = output_dir(cfg) / "06_negative_probe_contacts.csv"
    with negative_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "equilibrium",
                "radius",
                "direction_id",
                "y1",
                "y2",
                "y3",
                "status",
                "final_class",
                "assigned_cluster",
                "matched_cluster",
                "cycle_positive_distance_norm",
                "cycle_negative_distance_norm",
                "target_distance_norm",
                "target_hit",
                "cycle_positive_hit",
                "cycle_negative_hit",
            ]
        )
        for row in probe_payloads:
            writer.writerow(
                [
                    row["sample_id"],
                    row["equilibrium"],
                    row["radius"],
                    row["direction_id"],
                    *row["x0"].tolist(),
                    row["status"],
                    row["final_class"],
                    row["assigned_cluster"],
                    row["matched_cluster"],
                    row["cluster_distances_norm"]["cycle_positive"],
                    row["cluster_distances_norm"]["cycle_negative"],
                    row["target_distance_norm"],
                    row["target_hit"],
                    row["cluster_hits"]["cycle_positive"],
                    row["cluster_hits"]["cycle_negative"],
                ]
            )
    fallback_csv = output_dir(cfg) / "07_negative_fallback_scan.csv"
    with fallback_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "route_id",
                "source",
                "branch",
                "phase",
                "y1_0",
                "y2_0",
                "y3_0",
                "status",
                "assigned_cluster",
                "span_norm",
            ]
        )
        for row in fallback_payloads:
            writer.writerow(
                [
                    row["route_id"],
                    row["source"],
                    row["branch"],
                    row["phase"],
                    *np.asarray(row["x0"]).tolist(),
                    row["status"],
                    row["assigned_cluster"],
                    row["signature"]["span_norm"],
                ]
            )
    context.update(
        {
            "negative_direct": direct,
            "negative_seed_entries": seed_entries,
            "negative_candidates": candidates,
            "negative_runtime_clusters": runtime_clusters,
            "negative_target": target,
            "negative_target_probe_trajectory": target_probe_trajectory,
            "negative_audit": result,
            "negative_fallback_records": fallback_records,
        }
    )
    return result


def _posthoc_case(
    system,
    reference: dict[str, Any],
    target_trajectory: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    posthoc_cfg = cfg["posthoc_table_1"]
    _, trajectory, status = final_integer_lure_attractor(
        system,
        reference["reported_seed"],
        t_burn=float(posthoc_cfg["t_burn"]),
        t_keep=float(posthoc_cfg["t_keep"]),
        h=float(posthoc_cfg["h"]),
        div_threshold=float(posthoc_cfg["div_threshold"]),
    )
    max_points = int(cfg["hiddenness"]["max_cloud_points"])
    reference_cloud = _tail_cloud(trajectory, t_start=0.0, max_points=max_points)
    target_cloud = _tail_cloud(target_trajectory, t_start=0.0, max_points=max_points)
    cloud_distance = _cloud_distance_norm(reference_cloud, target_cloud)
    omega = float(reference["reported_omega"])
    gain = float(reference["reported_gain"])
    transfer = lure_transfer_function(omega, 1.0, system.lure)
    direct_pairs = find_integer_lure_omega_gain_candidates_direct(
        system.lure,
        wmin=float(cfg["direct_route"]["omega_min"]),
        wmax=float(cfg["direct_route"]["omega_max"]),
        compatible_only=True,
    )
    closest = min(direct_pairs, key=lambda pair: abs(pair[0] - omega))
    result = {
        "role": posthoc_cfg["role"],
        "used_as_search_input": posthoc_cfg["used_as_search_input"],
        "reported_omega": omega,
        "reported_gain": gain,
        "reported_seed": reference["reported_seed"],
        "posthoc_integrator": "hidden_attractors_fo_EFORK_q1",
        "integration_status": status,
        "target_cloud_distance_norm": cloud_distance,
        "same_numerical_cluster_as_blind_target": bool(
            status == "ok"
            and cloud_distance <= float(cfg["hiddenness"]["target_cloud_tol"])
        ),
        "printed_transfer_value": transfer,
        "printed_nyquist_closure_residual": abs(1.0 + gain * transfer),
        "printed_cubic_df_residual_using_seed_y1_as_amplitude": abs(
            float(system.lure.describing_function(abs(float(reference["reported_seed"][0]))))
            - gain
        ),
        "nearest_exact_direct_branch": closest,
        "printed_vs_nearest_direct_difference": {
            "omega": abs(omega - closest[0]),
            "gain": abs(gain - closest[1]),
        },
        "interpretation": "posthoc_dynamic_seed_match_does_not_remove_the_printed_Lure_value_discrepancy",
    }
    return result, trajectory


def run_posthoc_table_audit(
    cfg: dict[str, Any], context: dict[str, Any], probe_path: Path
) -> dict[str, Any]:
    if "primary_hiddenness" not in context:
        run_primary_hiddenness(cfg, context, probe_path)
    if "negative_audit" not in context:
        run_negative_audit(cfg, context, probe_path)
    references = cfg["posthoc_table_1"]
    primary, primary_trajectory = _posthoc_case(
        context["primary_system"],
        references["primary_xi_3p1"],
        context["primary_selected"]["trajectory"],
        cfg,
    )
    negative, negative_trajectory = _posthoc_case(
        context["negative_system"],
        references["negative_xi_3p5"],
        context["negative_target"]["trajectory"],
        cfg,
    )
    if not primary["same_numerical_cluster_as_blind_target"]:
        raise RuntimeError("the xi=3.1 Table 1 seed did not reach the blind outer cluster")
    if not negative["same_numerical_cluster_as_blind_target"]:
        raise RuntimeError("the xi=3.5 Table 1 seed did not reach the blind branch-0 cycle")
    _write_trajectory(output_dir(cfg) / "08_posthoc_table1_xi3p1.csv", primary_trajectory)
    _write_trajectory(output_dir(cfg) / "08_posthoc_table1_xi3p5.csv", negative_trajectory)
    result = {
        "operational_search_read_table_1": False,
        "primary_xi_3p1": primary,
        "negative_xi_3p5": negative,
        "overall_decision": (
            "published_seeds_match_posthoc_dynamic_clusters_but_printed_frequency_gain_values_"
            "do_not_match_the_exact_direct_Lure_branches"
        ),
    }
    _write_json(output_dir(cfg) / "08_posthoc_table1_audit.json", result)
    context.update(
        {
            "posthoc": result,
            "posthoc_primary_trajectory": primary_trajectory,
            "posthoc_negative_trajectory": negative_trajectory,
        }
    )
    return result


def _plot_candidate_clusters(context: dict[str, Any], path: Path) -> None:
    colors = {
        "inner_positive": "#2563eb",
        "inner_negative": "#dc2626",
        "outer_recurrent_candidate": "#15803d",
    }
    fig = plt.figure(figsize=(8.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    shown: set[str] = set()
    assignment = context["primary_clusters"]["route_assignment"]
    for record in context["primary_candidates"]:
        label = assignment[record["route_id"]]
        data = sample_rows(record["trajectory"], 1800)
        legend = label if label not in shown else None
        shown.add(label)
        ax.plot(
            data[:, 1],
            data[:, 2],
            data[:, 3],
            lw=0.65,
            alpha=0.78,
            color=colors[label],
            label=legend,
        )
    ax.set_xlabel("y1")
    ax.set_ylabel("y2")
    ax.set_zlabel("y3")
    ax.set_title("Blind D0 clusters, xi=3.1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    intercept_and_export_path(fig, path, "mavpd_primary_clusters")
    plt.close(fig)


def _plot_negative_comparison(context: dict[str, Any], path: Path) -> None:
    fig = plt.figure(figsize=(8.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    rows = [
        (context["negative_target"]["trajectory"], "blind branch-0 cycle", "#2563eb"),
        (
            context["negative_target_probe_trajectory"],
            "shared 108-probe TARGET contact",
            "#dc2626",
        ),
        (context["posthoc_negative_trajectory"], "Table 1 seed, posthoc", "#15803d"),
    ]
    for trajectory, label, color in rows:
        data = sample_rows(trajectory, 1800)
        ax.plot(data[:, 1], data[:, 2], data[:, 3], lw=0.7, alpha=0.8, label=label, color=color)
    ax.set_xlabel("y1")
    ax.set_ylabel("y2")
    ax.set_zlabel("y3")
    ax.set_title("xi=3.5 negative hiddenness audit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    intercept_and_export_path(fig, path, "mavpd_negative_audit")
    plt.close(fig)


def run_figures(
    cfg: dict[str, Any], context: dict[str, Any], probe_path: Path
) -> dict[str, Any]:
    if "posthoc" not in context:
        run_posthoc_table_audit(cfg, context, probe_path)
    figures = figure_dir(cfg)
    figures.mkdir(parents=True, exist_ok=True)
    selected = context["primary_selected"]
    seed_entry = next(
        entry
        for entry in context["primary_seed_entries"]
        if entry["route_id"] == selected["route_id"]
    )
    seed = seed_entry["seed_object"]
    system = context["primary_system"]
    paths: list[Path] = []

    def record(path: Path) -> Path:
        paths.append(path)
        return path

    plot_lure_nyquist_describing_function(
        system.lure,
        seed,
        record(figures / "primary_nyquist_df.png"),
        q=1.0,
        wmin=0.1,
        wmax=25.0,
        amin=0.02,
        amax=1.2,
        title="MAVPD xi=3.1 direct D0 closure",
    )
    plot_lure_transfer_components(
        system.lure,
        seed,
        record(figures / "primary_transfer_components.png"),
        q=1.0,
        wmin=0.1,
        wmax=25.0,
        nscan=1200,
        title="Visualization of exact direct roots (grid not used for search)",
    )
    continuation = context["primary_continuation_objects"].get(selected["route_id"])
    if continuation:
        plot_integer_lure_continuation(
            continuation,
            record(figures / "primary_lambda_continuation.png"),
            title="MAVPD branch 1 continuation to xi=3.1 target",
        )
    plot_phase_space(
        selected["trajectory"],
        record(figures / "primary_outer_candidate.png"),
        title="MAVPD xi=3.1 blind outer candidate (DOP853-qualified cycle)",
    )
    plot_phase_projections(
        selected["trajectory"],
        record(figures / "primary_outer_projections.png"),
        title="MAVPD xi=3.1 outer candidate",
    )
    plot_time_series(
        selected["trajectory"],
        record(figures / "primary_outer_timeseries.png"),
        title="MAVPD xi=3.1 outer candidate",
    )
    plot_integer_hiddenness_controls(
        selected["trajectory"],
        context["primary_probes"],
        record(figures / "primary_shared_hiddenness_controls.png"),
        title="Shared 108-probe finite hiddenness screen",
        max_probe_points=35,
    )
    spectrum_paths = plot_trajectory_spectra(
        selected["trajectory"],
        figures,
        method="fft",
        prefix="primary_outer",
    )
    paths.extend(Path(path) for path in spectrum_paths)
    cluster_path = record(figures / "primary_blind_clusters.png")
    _plot_candidate_clusters(context, cluster_path)
    negative_path = record(figures / "negative_xi3p5_comparison.png")
    _plot_negative_comparison(context, negative_path)
    result = {
        "figures": [_display_path(path) for path in paths],
        "note": "transfer component grid is visualization only; search used exact polynomial roots",
    }
    _write_json(output_dir(cfg) / "09_figure_manifest.json", result)
    context["figures"] = result
    return result


def run_selected(
    steps: Sequence[str],
    *,
    quick: bool = False,
    config_path: Path = CONFIG_PATH,
    output_override: Path | None = None,
    probe_input: Path | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path=config_path, quick=quick)
    if output_override is not None:
        destination = output_override.resolve()
        cfg["outputs"]["output_dir"] = str(destination)
        cfg["outputs"]["figures_dir"] = str(destination / "figures")
    probe_path = (probe_input or default_probe_path(cfg)).resolve()
    context: dict[str, Any] = {"quick": quick, "probe_path": probe_path}
    operations = {
        "contract": lambda: run_contract(cfg, context),
        "primary": lambda: run_primary_route(cfg, context),
        "hiddenness": lambda: run_primary_hiddenness(cfg, context, probe_path),
        "negative": lambda: run_negative_audit(cfg, context, probe_path),
        "posthoc": lambda: run_posthoc_table_audit(cfg, context, probe_path),
        "figures": lambda: run_figures(cfg, context, probe_path),
    }
    phase_timings: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for step in steps:
        started = time.perf_counter()
        operations[step]()
        phase_timings.append({"phase": step, "seconds": time.perf_counter() - started})
    total_seconds = time.perf_counter() - total_started
    timing_csv = output_dir(cfg) / "phase_timings.csv"
    timing_csv.parent.mkdir(parents=True, exist_ok=True)
    with timing_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "seconds"])
        for row in phase_timings:
            writer.writerow([row["phase"], f"{row['seconds']:.12f}"])
        writer.writerow(["total", f"{total_seconds:.12f}"])
    manifest = {
        "case_id": cfg["case_id"],
        "steps": list(steps),
        "quick": quick,
        "q": cfg["system"]["q"],
        "output_dir": _display_path(output_dir(cfg)),
        "figures_dir": _display_path(figure_dir(cfg)),
        "probe_input": _display_path(probe_path),
        "probe_sha256": _sha256(probe_path),
        "frequency_grid_used_for_search": False,
        "table_1_used_as_search_input": False,
        "phase_timings": phase_timings,
        "total_seconds": total_seconds,
        "python": sys.version,
        "platform": platform.platform(),
    }
    _write_json(output_dir(cfg) / "run_manifest.json", manifest)
    print(f"total_seconds={total_seconds:.9f}")
    print(f"output_dir={output_dir(cfg)}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["contract", "primary", "hiddenness", "negative", "posthoc", "figures"],
        default=["contract", "primary", "hiddenness", "negative", "posthoc", "figures"],
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--probe-input",
        type=Path,
        help="Explicit shared 108-condition CSV; defaults to the copy stored with this example.",
    )
    args = parser.parse_args()
    run_selected(
        args.steps,
        quick=args.quick,
        config_path=args.config,
        output_override=args.output_dir,
        probe_input=args.probe_input,
    )


if __name__ == "__main__":
    main()
