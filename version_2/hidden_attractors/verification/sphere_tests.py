"""Reproducible finite-horizon surface/ball neighborhood sweeps."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from .hiddenness import generate_neighborhood_points, run_neighborhood_probe
from .terminal_policy import PROBE_POLICY
from ..plotting.plot_sphere_tests import plot_sphere_test_results


def _json_safe(value):
    """Keep nonfinite samples in NPZ; encode their JSON diagnostics as null."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def run_single_sphere_probe(payload: Tuple) -> Tuple[int, Dict[str, Any]]:
    """Integrate once and retain actual times and classifier diagnostics."""
    values = list(payload)
    (idx, system, x0, transfer_mode, integrator, t_final, t_burn, h, ref_tail,
     stable_eqs, eq_tol, div_norm, metric, tol, dynamics_mode, memory_mode,
     memory_window, early_stop, equilibria) = values[:19]
    q = values[19] if len(values) > 19 else None
    policy = values[20] if len(values) > 20 else {}
    x0 = np.asarray(x0, dtype=float)
    try:
        res = run_neighborhood_probe(
            system=system, x0=x0, transfer_mode=transfer_mode, integrator=integrator,
            t_final=t_final, t_burn=t_burn, h=h, ref_tail=ref_tail,
            stable_equilibria=stable_eqs, equilibrium_tol=eq_tol,
            divergence_norm=div_norm, target_match_metric=metric,
            target_match_tol=tol, dynamics_mode=dynamics_mode, memory_mode=memory_mode,
            memory_window_length=memory_window, early_stop_config=early_stop,
            equilibria_dict=equilibria, q_dynamics_effective=q, **policy)
        trajectory = res["trajectory"]
        final = trajectory[-1] if len(trajectory) else np.full_like(x0, np.nan)
        finite_final = bool(np.isfinite(final).all())
        eq_distance = (float(min(np.linalg.norm(final-eq) for eq in stable_eqs))
                       if stable_eqs and finite_final else None)
        # This is expressly a centroid diagnostic, not the configured contact metric.
        times = np.asarray(res.get("times", []))
        valid_tail = trajectory[times >= t_burn] if len(times) == len(trajectory) else np.empty((0, len(x0)))
        centroid_distance = (float(np.linalg.norm(np.mean(valid_tail, axis=0)-np.mean(ref_tail, axis=0)))
                             if len(valid_tail) and len(ref_tail) and np.isfinite(valid_tail).all()
                             and np.isfinite(ref_tail).all() else None)
        return idx, {**res, "x0": x0.tolist(), "final_state": final.tolist(),
                     "distance_to_target": centroid_distance,
                     "distance_to_equilibrium": eq_distance}
    except Exception as exc:
        return idx, {"x0": x0.tolist(), "destination": "numerical_failure",
                     "final_state": [None]*len(x0), "trajectory": np.empty((0, len(x0))),
                     "times": np.empty(0), "status": f"exception:{type(exc).__name__}:{exc}",
                     "classification_policy": policy.get("classification_policy", PROBE_POLICY),
                     "diagnostics": {"reason": str(exc), "tail_samples": 0},
                     "distance_to_target": None, "distance_to_equilibrium": None}


def run_sphere_probe_sweep(system: Any, config: Dict[str, Any],
                           equilibria: Dict[str, np.ndarray], stable_eqs: List[np.ndarray],
                           ref_tail: np.ndarray, output_dir: str, workers: int = 1,
                           q_dynamics_effective: Optional[float] = None, *,
                           output_claim=None) -> Dict[str, Any]:
    """Save complete traces, initials/directions and the finite classification contract.

    Surface samples are explicitly distinct from volume-uniform balls. A finite
    absence of reference contacts is never a mathematical hiddenness proof.
    """
    from .._run_output import claim_output_directory, require_output_claim, complete_output_claim
    st = dict(config.get("sphere_tests", {}))
    if not st:
        raise ValueError("sphere_tests must explicitly declare radii, sampling and horizon.")
    radii = list(st.get("radii", []))
    if not radii or any(not np.isfinite(r) or r <= 0 for r in radii):
        raise ValueError("sphere_tests.radii must contain positive finite radii.")
    mode = st.get("directions_mode", "sphere_random")
    if mode not in ("sphere_random", "ball_random", "coordinate_axes", "hybrid"):
        raise ValueError(f"Unknown sampling mode: {mode}")
    policy = {"classification_policy": st.get("classification_policy", PROBE_POLICY),
              "terminal_window": st.get("terminal_window"),
              "equilibrium_span_tol": st.get("equilibrium_span_tol"),
              "target_match_nn_percentile": config.get("hiddenness", {}).get("target_match_nn_percentile", 90.)}
    hiddenness_config = config.get("hiddenness", {})
    contact_metric = hiddenness_config.get("target_match_metric", config["target_match_metric"])
    contact_tol = hiddenness_config.get("target_match_tol", config["target_match_tol"])
    if policy["classification_policy"] not in (PROBE_POLICY, "legacy_v1"):
        raise ValueError("Unknown neighborhood classification policy.")
    selection = st.get("equilibrium_selection", "all")
    names = list(equilibria) if selection == "all" else ([selection] if isinstance(selection, str) else list(selection))
    if any(name not in equilibria for name in names) or not names:
        raise ValueError("Select at least one known equilibrium for neighborhood tests.")
    t_final, t_burn, h = (float(st.get(key, default)) for key, default in
                         (("t_final", 80.), ("t_burn", 20.), ("h", .01)))
    if not (0 <= t_burn < t_final and h > 0 and all(np.isfinite(v) for v in (t_final,t_burn,h))):
        raise ValueError("Require finite 0 <= t_burn < t_final and h > 0.")
    counts = []
    for index, radius in enumerate(radii):
        count = (st["samples_per_radius"][index] if st.get("samples_per_radius") is not None
                 else st.get("samples_initial", 20)*st.get("samples_growth_factor", 2.)**index)
        if isinstance(count, bool) or not np.isfinite(count) or int(count) != count or count <= 0:
            raise ValueError("Neighborhood sample counts must be positive integers.")
        counts.append(int(count))
    own_claim = output_claim is None
    claim = claim_output_directory(output_dir, run_kind="neighborhood_probe_sweep") if own_claim else output_claim
    output = require_output_claim(output_dir, claim)
    traces = output/"sphere_test_trajectories"
    traces.mkdir(exist_ok=False)
    all_runs, summaries, csv_rows = [], [], []
    labels = {"equilibrium_terminal": "EQ", "reference_contact": "TARGET",
              "other_finite": "OTHER", "threshold_crossed": "DIV",
              "numerical_failure": "FAIL", "inconclusive": "INCONCLUSIVE",
              "stable_equilibrium": "EQ", "target_attractor": "TARGET",
              "other_attractor": "OTHER", "divergence": "DIV"}
    for eq_index, eq_name in enumerate(names):
        center = np.asarray(equilibria[eq_name], dtype=float)
        for radius_index, (radius, requested) in enumerate(zip(radii, counts)):
            seed = int(st.get("random_seed", 42))+radius_index
            # Same direction stream at different equilibria is intentional and recorded.
            points = generate_neighborhood_points(center, radius, requested, mode, seed)
            payloads = [(index, system, point, config["transfer_mode"], config["integrator"],
                         t_final, t_burn, h, ref_tail, stable_eqs,
                         config["equilibrium_tol"], config["divergence_norm"],
                         contact_metric, contact_tol,
                         config["dynamics_mode"], config["memory_mode"], config.get("memory_window_length"),
                         config.get("early_stop"), equilibria, q_dynamics_effective, policy)
                        for index, point in enumerate(points)]
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    runs = [result for _, result in pool.map(run_single_sphere_probe, payloads)]
            else:
                runs = [run_single_sphere_probe(payload)[1] for payload in payloads]
            stats = {key: 0 for key in ("EQ","TARGET","OTHER","DIV","FAIL","INCONCLUSIVE")}
            for index, (point, run) in enumerate(zip(points, runs)):
                displacement = point-center
                distance = float(np.linalg.norm(displacement))
                direction = displacement/distance if distance else np.zeros_like(center)
                fraction = distance/radius
                if (mode != "ball_random" and abs(fraction-1.) > 1e-8) or fraction > 1.+1e-8:
                    raise ValueError("Sampling geometry failed its recorded radius check.")
                stem = f"eq{eq_index:03d}_radius{radius_index:03d}_sample{index:06d}"
                trace = traces/(stem+".npz")
                np.savez_compressed(trace, times=run["times"], states=run["trajectory"], x0=point,
                                    equilibrium=center, direction=direction, radius=radius,
                                    radial_fraction=fraction)
                digest = hashlib.sha256(trace.read_bytes()).hexdigest()
                record = {**run, "equilibrium": eq_name, "radius": float(radius),
                          "sample_index": index, "direction": direction.tolist(),
                          "direction_norm": float(np.linalg.norm(direction)),
                          "radial_fraction": fraction, "sampling_domain": "ball" if mode == "ball_random" else "sphere_surface",
                          "trajectory_available": bool(len(run["trajectory"])),
                          "trajectory_file": trace.relative_to(output).as_posix(),
                          "trajectory_sha256": digest}
                all_runs.append(record)
                stats[labels.get(run["destination"], "INCONCLUSIVE")] += 1
                csv_rows.append({"system_id": config["system_id"], "equilibrium": eq_name,
                                 "radius": radius, "sample_index": index,
                                 "x0": json.dumps(point.tolist()), "direction": json.dumps(direction.tolist()),
                                 "radial_fraction": fraction, "sampling_domain": record["sampling_domain"],
                                 "destination": run["destination"], "status": run["status"],
                                 "classification_policy": run["classification_policy"],
                                 "final_state": json.dumps(_json_safe(run["final_state"]), allow_nan=False),
                                 "diagnostics": json.dumps(_json_safe(run["diagnostics"]), allow_nan=False),
                                 "trajectory_file": record["trajectory_file"], "trajectory_sha256": digest})
            summaries.append({"system_id": config["system_id"], "equilibrium": eq_name,
                              "radius": float(radius), "samples": len(points),
                              "requested_samples": requested, "classification_policy": policy["classification_policy"], **stats})
            print(f"[{config['system_id']}] Neighborhood {eq_name} r={radius:g}, completed={len(runs)}, {stats}")
            if config.get("plot_sphere_tests", True):
                plot_sphere_test_results(eq_name, center, radius, runs, str(output),
                                        st.get("trajectory_plot_fraction", .25), st.get("max_trajectories_to_plot", 60))
    metadata = {"schema_version": "2.0", "classification_policy": policy["classification_policy"],
                "meaning": "finite_horizon_observations_not_asymptotic_attractors",
                "equilibria_selected": names, "radii": radii, "requested_samples_per_radius": counts,
                "directions_mode": mode, "sampling_domain": "ball" if mode == "ball_random" else "sphere_surface",
                "sampling_measure": "uniform_volume" if mode == "ball_random" else ("uniform_surface" if mode == "sphere_random" else "deterministic_axes_or_mixed_surface"),
                "random_seed": st.get("random_seed", 42), "rng": "numpy.default_rng/PCG64",
                "stream_policy": "base_seed_plus_radius_index_shared_across_equilibria",
                "t_final": t_final, "t_burn": t_burn, "h": h, "terminal_window": policy["terminal_window"],
                "minimum_terminal_samples": 3, "equilibrium_distance_tol": config["equilibrium_tol"],
                "equilibrium_span_tol": policy["equilibrium_span_tol"] if policy["equilibrium_span_tol"] is not None else 2*config["equilibrium_tol"],
                "divergence_norm": config["divergence_norm"], "reference_contact_metric": contact_metric,
                "reference_contact_tol": contact_tol,
                "reference_contact_config_source": "hiddenness section with global fallback",
                "target_match_nn_percentile": policy["target_match_nn_percentile"],
                "integrator": config["integrator"], "memory_mode": config["memory_mode"],
                "memory_window_length": config.get("memory_window_length"),
                "q_dynamics_effective": q_dynamics_effective,
                "early_stop_effective": "disabled_heuristics_explicit_norm_threshold_retained" if policy["classification_policy"] == PROBE_POLICY else "legacy_config",
                "equilibria": {key: np.asarray(value).tolist() for key, value in equilibria.items()},
                "reference_tail_sha256": hashlib.sha256(np.asarray(ref_tail,dtype='<f8').tobytes()).hexdigest()}
    np.savez_compressed(output/"sphere_tests_reference.npz", states=ref_tail)
    with (output/"sphere_tests_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]) if csv_rows else [])
        writer.writeheader(); writer.writerows(csv_rows)
    _print_and_save_hiddenness_tables(summaries, str(output))
    (output/"sphere_tests_summary.json").write_text(json.dumps(summaries, indent=2),encoding="utf-8")
    (output/"sphere_tests_metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2, allow_nan=False),encoding="utf-8")
    with (output/"sphere_tests_runs.jsonl").open("w",encoding="utf-8") as handle:
        for run in all_runs:
            handle.write(json.dumps(_json_safe({key:value for key,value in run.items() if key not in ("trajectory","times")}), allow_nan=False)+"\n")
    if own_claim:
        complete_output_claim(claim)
    return {"probe_runs": all_runs, "summary_records": summaries, "metadata": metadata}


def _print_and_save_hiddenness_tables(summary_records, output_dir):
    """Describe observed contacts; never turn zero finite contacts into a proof."""
    columns = ["system_id","equilibrium","radius","samples","EQ","TARGET","OTHER","DIV","FAIL","INCONCLUSIVE","criterio_local"]
    for record in summary_records:
        if record.get("TARGET",0):
            criterion = "REFERENCE_CONTACT_OBSERVED"
        elif record.get("FAIL",0) or record.get("INCONCLUSIVE",0) or not record.get("samples",0):
            criterion = "INCONCLUSIVE"
        else:
            criterion = "NO_REFERENCE_CONTACT_OBSERVED"
        record["criterio_local"] = criterion
    output = Path(output_dir)
    with (output/"hiddenness_summary.csv").open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore")
        writer.writeheader();writer.writerows(summary_records)
    (output/"hiddenness_summary.json").write_text(json.dumps(summary_records,indent=2),encoding="utf-8")
    text = "# Finite neighborhood observations\n\nTARGET means finite reference contact; DIV means an observed norm threshold.\nNo finite absence of contacts proves hiddenness.\n\n"
    text += "| "+" | ".join(columns)+" |\n|"+"|".join("---" for _ in columns)+"|\n"
    for record in summary_records:
        text += "| "+" | ".join(str(record.get(key,0)) for key in columns)+" |\n"
    (output/"hiddenness_summary.md").write_text(text,encoding="utf-8")
