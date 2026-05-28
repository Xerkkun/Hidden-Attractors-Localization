import os
import csv
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple, Optional
from .hiddenness import generate_neighborhood_points, run_neighborhood_probe
from ..plotting.plot_sphere_tests import plot_sphere_test_results

def run_single_sphere_probe(payload: Tuple) -> Tuple[int, Dict[str, Any]]:
    """Worker function to simulate a single sphere initial condition."""
    payload_list = list(payload)
    idx = payload_list[0]
    system = payload_list[1]
    x0 = payload_list[2]
    transfer_mode = payload_list[3]
    integrator = payload_list[4]
    t_final = payload_list[5]
    t_burn = payload_list[6]
    h = payload_list[7]
    ref_tail = payload_list[8]
    stable_eqs = payload_list[9]
    eq_tol = payload_list[10]
    div_norm = payload_list[11]
    metric = payload_list[12]
    tol = payload_list[13]
    dynamics_mode = payload_list[14]
    memory_mode = payload_list[15]
    memory_window_length = payload_list[16]
    early_stop_config = payload_list[17]
    equilibria_dict = payload_list[18]
    q_dynamics_effective = payload_list[19] if len(payload_list) > 19 else None
    
    try:
        res = run_neighborhood_probe(
            system=system,
            x0=x0,
            transfer_mode=transfer_mode,
            integrator=integrator,
            t_final=t_final,
            t_burn=t_burn,
            h=h,
            ref_tail=ref_tail,
            stable_equilibria=stable_eqs,
            equilibrium_tol=eq_tol,
            divergence_norm=div_norm,
            target_match_metric=metric,
            target_match_tol=tol,
            dynamics_mode=dynamics_mode,
            memory_mode=memory_mode,
            memory_window_length=memory_window_length,
            early_stop_config=early_stop_config,
            equilibria_dict=equilibria_dict,
            q_dynamics_effective=q_dynamics_effective
        )
        dest = res["destination"]
        
        if dest in ("equilibrium_stable", "stable_equilibrium"):
            dest = "stable_equilibrium"
            
        final_state = res["trajectory"][-1] if len(res["trajectory"]) > 0 else x0
        
        dist_eq = 0.0
        if stable_eqs:
            dist_eq = float(min(np.linalg.norm(final_state - eq) for eq in stable_eqs))
            
        dist_target = 9999.0
        if len(ref_tail) > 0:
            tail_for_dist = res["trajectory"][-100:] if len(res["trajectory"]) > 100 else res["trajectory"]
            dist_target = float(np.linalg.norm(np.mean(tail_for_dist, axis=0) - np.mean(ref_tail, axis=0)))
            
        return idx, {
            "x0": x0.tolist(),
            "destination": dest,
            "final_state": final_state.tolist(),
            "trajectory": res["trajectory"],
            "status": res["status"],
            "distance_to_target": dist_target,
            "distance_to_equilibrium": dist_eq
        }
    except Exception as e:
        return idx, {
            "x0": x0.tolist(),
            "destination": "numerical_failure",
            "final_state": x0.tolist(),
            "trajectory": np.empty((0, 3)),
            "status": f"exception: {str(e)}",
            "distance_to_target": 9999.0,
            "distance_to_equilibrium": 9999.0
        }

def run_sphere_probe_sweep(
    system: Any,
    config: Dict[str, Any],
    equilibria: Dict[str, np.ndarray],
    stable_eqs: List[np.ndarray],
    ref_tail: np.ndarray,
    output_dir: str,
    workers: int = 1,
    q_dynamics_effective: Optional[float] = None
) -> Dict[str, Any]:
    """
    Executes the complete equilibrium neighborhood sphere probe sweep with early stopping.
    Saves CSV and JSON results, produces transparent 3D sphere plots,
    and returns a summary dictionary.
    """
    if q_dynamics_effective is None:
        import warnings
        warnings.warn("q_dynamics_effective is omitted, falling back to legacy dynamics_mode logic", UserWarning)
    st_config = config.get("sphere_tests", {})
    if not st_config:
        st_config = {
            "enabled": True,
            "equilibrium_selection": "all",
            "radii": [1e-5, 1e-4, 1e-3, 1e-2],
            "samples_initial": 20,
            "samples_growth_factor": 2.0,
            "directions_mode": "sphere_random",
            "random_seed": 42,
            "t_final": 80.0,
            "t_burn": 20.0,
            "h": 0.01,
            "trajectory_plot_fraction": 0.25,
            "max_trajectories_to_plot": 60,
            "samples_per_radius": None,
            "early_stop_enabled": True
        }
        
    system_id = config["system_id"]
    
    eq_selection = st_config.get("equilibrium_selection", "all")
    selected_eqs = {}
    if eq_selection == "all":
        selected_eqs = equilibria.copy()
    elif isinstance(eq_selection, list):
        for name in eq_selection:
            if name in equilibria:
                selected_eqs[name] = equilibria[name]
    elif isinstance(eq_selection, str):
        if eq_selection in equilibria:
            selected_eqs[eq_selection] = equilibria[eq_selection]
            
    if not selected_eqs:
        print(f"[{system_id}] WARNING: No equilibria selected for sphere tests.")
        return {"probe_runs": [], "summary_records": []}
        
    radii_list = st_config.get("radii", [1e-5, 1e-4, 1e-3, 1e-2])
    
    all_runs = []
    summary_records = []
    detailed_csv_rows = []
    
    for eq_name, eq_pt in selected_eqs.items():
        for r_idx, radius in enumerate(radii_list):
            if st_config.get("samples_per_radius") is not None:
                n_samples = st_config["samples_per_radius"][r_idx]
            else:
                n_samples = int(st_config["samples_initial"] * (st_config["samples_growth_factor"] ** r_idx))
                
            pts = generate_neighborhood_points(
                eq_point=eq_pt,
                radius=radius,
                num_samples=n_samples,
                mode=st_config.get("directions_mode", "sphere_random"),
                seed=st_config.get("random_seed", 42) + r_idx
            )
            
            payloads = []
            for idx, pt in enumerate(pts):
                payloads.append((
                    idx, system, pt, config["transfer_mode"], config["integrator"],
                    st_config.get("t_final", 80.0), st_config.get("t_burn", 20.0), st_config.get("h", 0.01),
                    ref_tail, stable_eqs, config["equilibrium_tol"], config["divergence_norm"],
                    config["target_match_metric"], config["target_match_tol"],
                    config["dynamics_mode"], config["memory_mode"], config["memory_window_length"],
                    config.get("early_stop"), equilibria, q_dynamics_effective
                ))
                
            completed = 0
            radius_runs = [None] * n_samples
            
            def log_progress(count):
                pct = (count / n_samples) * 100.0
                print(f"[{system_id}] Esferas {eq_name} radio={radius:.0e}: {count}/{n_samples}, {pct:.1f}%")
                
            log_progress(0)
            
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(run_single_sphere_probe, pay) for pay in payloads]
                    for fut in as_completed(futures):
                        idx, res_dict = fut.result()
                        radius_runs[idx] = res_dict
                        completed += 1
                        if completed % max(1, n_samples // 10) == 0 or completed == n_samples:
                            log_progress(completed)
            else:
                for pay in payloads:
                    idx, res_dict = run_single_sphere_probe(pay)
                    radius_runs[idx] = res_dict
                    completed += 1
                    if completed % max(1, n_samples // 10) == 0 or completed == n_samples:
                        log_progress(completed)
                        
            stats = {
                "target_attractor": 0,
                "stable_equilibrium": 0,
                "divergence": 0,
                "other_attractor": 0,
                "numerical_failure": 0,
                "unclassified": 0
            }
            
            for run_res in radius_runs:
                dest = run_res["destination"]
                if dest in stats:
                    stats[dest] += 1
                else:
                    stats["unclassified"] += 1
                    
            print(f"[{system_id}] Esferas {eq_name} radio={radius:.0e} terminado:")
            print(f"    target_attractor={stats['target_attractor']}")
            print(f"    stable_equilibrium={stats['stable_equilibrium']}")
            print(f"    divergence={stats['divergence']}")
            print(f"    other_attractor={stats['other_attractor']}")
            print(f"    numerical_failure={stats['numerical_failure']}")
            
            for idx, run_res in enumerate(radius_runs):
                all_runs.append({
                    "equilibrium": eq_name,
                    "radius": radius,
                    "x0": run_res["x0"],
                    "destination": run_res["destination"],
                    "trajectory": run_res["trajectory"]
                })
                detailed_csv_rows.append([
                    system_id, eq_name, radius, idx,
                    run_res["x0"][0], run_res["x0"][1], run_res["x0"][2],
                    run_res["destination"],
                    run_res["final_state"][0], run_res["final_state"][1], run_res["final_state"][2],
                    run_res["status"],
                    run_res["distance_to_target"],
                    run_res["distance_to_equilibrium"],
                    st_config.get("t_final", 80.0), st_config.get("t_burn", 20.0),
                    config["integrator"], config["memory_mode"]
                ])
                
            summary_records.append({
                "system_id": system_id,
                "equilibrium": eq_name,
                "radius": radius,
                "samples": n_samples,
                "EQ": stats["stable_equilibrium"],
                "TARGET": stats["target_attractor"],
                "OTHER": stats["other_attractor"],
                "DIV": stats.get("divergence", 0),
                "FAIL": stats.get("numerical_failure", 0)
            })
            
            if config.get("plot_sphere_tests", True):
                plot_sphere_test_results(
                    eq_name=eq_name,
                    eq_pt=eq_pt,
                    radius=radius,
                    probe_runs=radius_runs,
                    output_dir=output_dir,
                    trajectory_plot_fraction=st_config.get("trajectory_plot_fraction", 0.25),
                    max_trajectories_to_plot=st_config.get("max_trajectories_to_plot", 60)
                )

    csv_path = os.path.join(output_dir, "sphere_tests_results.csv")
    headers = [
        "system_id", "equilibrium", "radius", "sample_index", "x0_x", "x0_y", "x0_z",
        "destination", "final_x", "final_y", "final_z", "status",
        "distance_to_target", "distance_to_equilibrium", "t_final", "t_burn",
        "integrator", "memory_mode"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(detailed_csv_rows)
        
    summary_json_path = os.path.join(output_dir, "sphere_tests_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=4)
        
    _print_and_save_hiddenness_tables(summary_records, output_dir)
    
    return {"probe_runs": all_runs, "summary_records": summary_records}

def _print_and_save_hiddenness_tables(summary_records: List[Dict[str, Any]], output_dir: str) -> None:
    """Renders the Markdown summary table to console, CSV, and summary files."""
    headers = ["system_id", "equilibrium", "radius", "samples", "EQ", "TARGET", "OTHER", "DIV", "FAIL"]
    
    print("\n" + "="*95)
    print(" RESUMEN DE PRUEBAS DE OCULTEDAD (VECINDARIOS DE EQUILIBRIOS) ")
    print("="*95)
    print(f"| {' | '.join(f'{h:<11}' if h != 'system_id' else f'{h:<22}' for h in headers)} |")
    print(f"|{'-'*93}|")
    for r in summary_records:
        rad_str = f"{r['radius']:.0e}"
        row = [
            f"{r['system_id']:<22}",
            f"{r['equilibrium']:<11}",
            f"{rad_str:<11}",
            f"{r['samples']:<11}",
            f"{r['EQ']:<11}",
            f"{r['TARGET']:<11}",
            f"{r['OTHER']:<11}",
            f"{r['DIV']:<11}",
            f"{r['FAIL']:<11}"
        ]
        print(f"| {' | '.join(row)} |")
    print("="*95 + "\n")

    md_path = os.path.join(output_dir, "hiddenness_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Neighborhood Probe Hiddenness Verification Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join("---" for _ in headers) + "|\n")
        for r in summary_records:
            f.write(f"| {r['system_id']} | {r['equilibrium']} | {r['radius']:.0e} | {r['samples']} | {r['EQ']} | {r['TARGET']} | {r['OTHER']} | {r['DIV']} | {r['FAIL']} |\n")
            
    csv_summary_path = os.path.join(output_dir, "hiddenness_summary.csv")
    with open(csv_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in summary_records:
            writer.writerow([r[h] for h in headers])
            
    json_summary_path = os.path.join(output_dir, "hiddenness_summary.json")
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=4)
