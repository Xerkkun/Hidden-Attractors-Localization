import os
import json
import csv
import yaml
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from ..systems.registry import get_system_by_id
from ..lure.decomposition import validate_lure_decomposition
from ..lure.transfer import W_eval
from ..lure.nyquist import find_harmonic_candidates
from ..lure.seeds import build_lure_seed
from ..continuation.continuation_integer import run_integer_continuation
from ..continuation.continuation_fractional import run_fractional_continuation
from ..integrators.abm import caputo_abm_integrate
from ..integrators.efork import efork_integrate
from ..verification.equilibria import solve_equilibria
from ..verification.stability import classify_equilibrium_stability
from ..verification.hiddenness import run_neighborhood_probe, generate_neighborhood_points
from ..verification.basins import generate_basin_slice
from ..verification.classifiers import classify_hiddenness_verdict

# Import plotting routines dynamically if enabled
from ..plotting.plot_transfer import plot_nyquist_transfer
from ..plotting.plot_df import plot_describing_function
from ..plotting.plot_continuation import plot_continuation_eta
from ..plotting.plot_trajectories import plot_attractor_trajectories, plot_neighborhood_control_spheres
from ..plotting.plot_basins import plot_basin_slices

from .configs import DEFAULT_CONFIG

def run_centered_lure_df_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the full 7-phase centered Lur'e describing function workflow."""
    # Inject defaults for any missing keys
    for k, v in DEFAULT_CONFIG.items():
        config.setdefault(k, v)
        
    system_id = config["system_id"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Fase 1/7: cargando configuración... 0%
    # -------------------------------------------------------------------------
    print(f"[{system_id}] Fase 1/7: cargando configuración... 0%")
    
    # Instantiate the system
    sys_kwargs = {}
    if config["q"] is not None:
        sys_kwargs["q"] = config["q"]
    system = get_system_by_id(system_id, **sys_kwargs)
    q = system.q
    
    # Save the effective configuration used
    effective_config_path = os.path.join(output_dir, "effective_config.yaml")
    effective_config = config.copy()
    effective_config["q"] = q # Store actual q
    with open(effective_config_path, "w", encoding="utf-8") as f:
        yaml.dump(effective_config, f, default_flow_style=False)
        
    # Validate the Lur'e split
    if not validate_lure_decomposition(system):
        print(f"[{system_id}] WARNING: Lur'e decomposition vector field mismatch.")
        
    # -------------------------------------------------------------------------
    # Fase 2/7: calculando equilibrios... 15%
    # -------------------------------------------------------------------------
    print(f"[{system_id}] Fase 2/7: calculando equilibrios... 15%")
    equilibria = solve_equilibria(system)
    
    eq_stability = {}
    stable_eqs = []
    unstable_eqs = []
    
    for eq_name, eq_pt in equilibria.items():
        stability_res = classify_equilibrium_stability(system, eq_pt)
        eq_stability[eq_name] = stability_res
        if stability_res["stable"]:
            stable_eqs.append(eq_pt)
        else:
            unstable_eqs.append(eq_pt)
            
    # -------------------------------------------------------------------------
    # Fase 3/7: construyendo transferencia... 30%
    # -------------------------------------------------------------------------
    print(f"[{system_id}] Fase 3/7: construyendo transferencia... 30%")
    # Prepare frequency grid to log Nyquist behavior
    omega_grid = np.linspace(config["omega_min"], config["omega_max"], config["grid_size_omega"])
    w_vals = []
    for w in omega_grid:
        try:
            val = W_eval(w, q, config["transfer_mode"], system.P, system.b, system.r)
            w_vals.append(val)
        except Exception:
            w_vals.append(complex(np.nan, np.nan))
    w_vals = np.array(w_vals)
    
    # -------------------------------------------------------------------------
    # Fase 4/7: buscando semillas DF... 45%
    # -------------------------------------------------------------------------
    print(f"[{system_id}] Fase 4/7: buscando semillas DF... 45%")
    candidates = find_harmonic_candidates(
        system=system,
        transfer_mode=config["transfer_mode"],
        seed_strategy=config["seed_strategy"],
        df_residual_tol=config["df_residual_tol"],
        omega_min=config["omega_min"],
        omega_max=config["omega_max"],
        amplitude_min=config["amplitude_min"],
        amplitude_max=config["amplitude_max"],
        grid_size_omega=config["grid_size_omega"],
        grid_size_amplitude=config["grid_size_amplitude"],
        root_refinement=config["root_refinement"]
    )
    
    n_candidates = len(candidates)
    
    if n_candidates == 0:
        print(f"[{system_id}] No DF seed candidates found.")
        summary = _build_summary_dict(config, system, equilibria, unstable_eqs, candidates, None, None, None, None, [], "df_seed_not_found")
        _save_summary(summary, output_dir)
        return summary
        
    # Select the first candidate as the primary seed
    # and generate the positive and negative initial seeds
    A0, omega0, k = candidates[0]
    seed_pos, seed_neg = build_lure_seed(
        system, A0, omega0, k,
        seed_sign_convention=config["seed_sign_convention"],
        q=q,
        transfer_mode=config["transfer_mode"],
        theta=config.get("seed_theta", 0.0),
        seed_construction=config.get("seed_construction", "modal"),
    )
    
    # -------------------------------------------------------------------------
    # Fase 5/7: continuación eta... 60%
    # -------------------------------------------------------------------------
    print(f"[{system_id}] Fase 5/7: continuación eta... 60%")
    # Setup eta/lambda list: 5 points in [0, 1] for fast, robust tracking
    lambda_grid = np.linspace(0.0, 1.0, 5)
    
    # Run continuation from the positive seed
    if config["continuation_mode"] == "integer":
        cont_steps = run_integer_continuation(
            system=system,
            seed_x0=seed_pos,
            k_gain=k,
            lambda_values=lambda_grid,
            t_transient=30.0,
            t_keep=30.0,
            h=config["h"],
            div_threshold=config["divergence_norm"],
            integrator=config["integrator"]
        )
    else: # fractional
        cont_steps = run_fractional_continuation(
            system=system,
            seed_x0=seed_pos,
            k_gain=k,
            lambda_values=lambda_grid,
            t_transient=30.0,
            t_keep=30.0,
            h=config["h"],
            memory_mode=config["memory_mode"],
            memory_window_length=config["memory_window_length"],
            div_threshold=config["divergence_norm"],
            integrator=config["integrator"],
            use_c_backend=use_c_backend_check(config)
        )
        
    cont_success = bool(len(cont_steps) == len(lambda_grid) and cont_steps[-1]["status"] == "ok")
    
    # -------------------------------------------------------------------------
    # Fase 6/7: simulación final... 75%
    # -------------------------------------------------------------------------
    print(f"[{system_id}] Fase 6/7: simulación final... 75%")
    final_traj = None
    final_status = "continuation_failed"
    
    if cont_success:
        # Final point of continuation as initial state
        x_final_seed = cont_steps[-1]["x_out"].copy()
        
        # Long final integration at eta = 1.0
        if config["continuation_mode"] == "integer":
            if config["integrator"] == "abm":
                t_fin, x_fin, final_status = caputo_abm_integrate(
                    system.evaluate_rhs, x_final_seed, q=1.0, h=config["h"], t_final=config["t_final"], divergence_norm=config["divergence_norm"], system=system
                )
            else: # efork
                t_fin, x_fin, final_status = efork_integrate(
                    system, x_final_seed, q=1.0, h=config["h"], t_final=config["t_final"], memory_mode="full"
                )
        else: # fractional
            if config["integrator"] == "abm":
                t_fin, x_fin, final_status = caputo_abm_integrate(
                    system.evaluate_rhs, x_final_seed, q=q, h=config["h"], t_final=config["t_final"], divergence_norm=config["divergence_norm"],
                    memory_mode=config["memory_mode"], memory_window_length=config["memory_window_length"], system=system
                )
            else: # efork
                t_fin, x_fin, final_status = efork_integrate(
                    system, x_final_seed, q=q, h=config["h"], t_final=config["t_final"], memory_mode="full"
                )
                
        if final_status == "ok":
            final_traj = np.column_stack((t_fin, x_fin))
            # Save trajectory CSV
            traj_csv_path = os.path.join(output_dir, "final_attractor.csv")
            with open(traj_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["t", "x", "y", "z"])
                writer.writerows(final_traj.tolist())
                
    # Evaluate final tail states
    ref_tail = np.empty((0, 3))
    if final_traj is not None:
        n_burn = int(np.ceil(config["t_burn"] / config["h"]))
        if len(final_traj) > n_burn:
            ref_tail = final_traj[n_burn:, 1:]
            
    # -------------------------------------------------------------------------
    # Fase 7/7: pruebas de ocultedad... 90%
    # -------------------------------------------------------------------------
    print(f"[{system_id}] Fase 7/7: pruebas de ocultedad... 90%")
    
    probe_results = []
    target_hits = 0
    numerical_fails = 0
    seed_reached_attractor = bool(final_traj is not None and final_status == "ok" and len(ref_tail) > 10)
    
    # Neighborhood / stability settings
    radii_list = [1e-5, 1e-4, 1e-3, 1e-2] # Default explicit values
    
    if config["run_hiddenness_tests"] and seed_reached_attractor:
        # Generate radii list
        # We tested equilibria neighborhood sampling
        for eq_name, eq_pt in equilibria.items():
            stability_res = eq_stability[eq_name]
            if stability_res["stable"]:
                # We skip neighborhood sampling of stable equilibria for hiddenness since they are not source points
                continue
                
            for r_idx, radius in enumerate(radii_list):
                pts = generate_neighborhood_points(
                    eq_pt, radius, num_samples=config["samples_per_radius"], mode=config["directions_mode"], seed=config["random_seed"]
                )
                for pt in pts:
                    probe_res = run_neighborhood_probe(
                        system=system,
                        x0=pt,
                        transfer_mode=config["transfer_mode"],
                        integrator=config["integrator"],
                        t_final=config["t_final"],
                        t_burn=config["t_burn"],
                        h=config["h"],
                        ref_tail=ref_tail,
                        stable_equilibria=stable_eqs,
                        equilibrium_tol=config["equilibrium_tol"],
                        divergence_norm=config["divergence_norm"],
                        target_match_metric=config["target_match_metric"],
                        target_match_tol=config["target_match_tol"]
                    )
                    
                    probe_results.append({
                        "equilibrium": eq_name,
                        "radius": radius,
                        "x0": pt.tolist(),
                        "destination": probe_res["destination"],
                        "trajectory": probe_res["trajectory"]
                    })
                    
                    if probe_res["destination"] == "target_attractor":
                        target_hits += 1
                    elif probe_res["destination"] == "numerical_failure":
                        numerical_fails += 1
                        
        # Save neighborhood probe raw CSV
        probe_csv_path = os.path.join(output_dir, "hiddenness_probes.csv")
        with open(probe_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["equilibrium", "radius", "x0_x", "x0_y", "x0_z", "destination"])
            for r in probe_results:
                writer.writerow([r["equilibrium"], r["radius"], r["x0"][0], r["x0"][1], r["x0"][2], r["destination"]])
                
    # Classify overall hiddenness verdict
    if config["run_hiddenness_tests"]:
        verdict = classify_hiddenness_verdict(
            target_hits_from_equilibria=target_hits,
            equilibria_count=len(equilibria),
            unstable_equilibria_count=len(unstable_eqs),
            seed_reached_attractor=seed_reached_attractor,
            numerical_failures=numerical_fails
        )
    else:
        verdict = "df_seed_found" if seed_reached_attractor else "df_seed_not_found"
        
    # Evaluate basin slices if enabled
    basin_data = None
    if config["run_basin_slices"] and seed_reached_attractor:
        # Compute first plane (e.g. xy)
        u_grid, v_grid, basin_mat = generate_basin_slice(
            plane="xy",
            system=system,
            transfer_mode=config["transfer_mode"],
            integrator=config["integrator"],
            ref_tail=ref_tail,
            stable_eqs=stable_eqs,
            fixed_values={"z": config["basin_fixed_z"], "y": config["basin_fixed_y"], "x": config["basin_fixed_x"]},
            extent=config["basin_extent"],
            grid_n=config["basin_grid_n"],
            center=equilibria["E0"].tolist(),
            workers=config["workers"],
            eq_tol=config["equilibrium_tol"],
            div_norm=config["divergence_norm"],
            metric=config["target_match_metric"],
            tol=config["target_match_tol"]
        )
        basin_data = {"xy": (u_grid, v_grid, basin_mat)}
        
        # Save basin metadata
        basin_meta = {
            "extent": config["basin_extent"],
            "grid_n": config["basin_grid_n"],
            "center": equilibria["E0"].tolist(),
            "planes": ["xy"]
        }
        with open(os.path.join(output_dir, "basin_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(basin_meta, f, indent=4)
            
    # -------------------------------------------------------------------------
    # terminado... 100%
    # -------------------------------------------------------------------------
    print(f"[{system_id}] terminado... 100%")
    
    # 1. Trigger plots if enabled
    if config["plot_enabled"]:
        # Save figures to final_pdf_figs / output_dir
        plot_nyquist_transfer(omega_grid, w_vals, candidates, config, output_dir)
        plot_describing_function(system, candidates, config, output_dir)
        plot_continuation_eta(cont_steps, config, output_dir)
        if final_traj is not None:
            plot_attractor_trajectories(final_traj, equilibria, config, output_dir)
            if len(probe_results) > 0:
                plot_neighborhood_control_spheres(final_traj, probe_results, equilibria, config, output_dir)
        if basin_data is not None:
            plot_basin_slices(basin_data, config, output_dir)
            
    # 2. Build and save summary
    summary = _build_summary_dict(
        config=config,
        system=system,
        equilibria=equilibria,
        unstable_eqs=unstable_eqs,
        candidates=candidates,
        A0=A0 if n_candidates > 0 else None,
        omega0=omega0 if n_candidates > 0 else None,
        k=k if n_candidates > 0 else None,
        cont_success=cont_success,
        probe_results=probe_results,
        verdict=verdict,
        final_traj=final_traj
    )
    _save_summary(summary, output_dir)
    
    # 3. Print Markdown summary table in the terminal
    _print_terminal_table(summary)
    
    return summary

def use_c_backend_check(config: Dict[str, Any]) -> bool:
    """Helper to check if C backend should be used."""
    # EFORK is always C for fractional; ABM uses C by default.
    return True

def _build_summary_dict(
    config: Dict[str, Any],
    system: Any,
    equilibria: Dict[str, np.ndarray],
    unstable_eqs: List[np.ndarray],
    candidates: List[Tuple[float, float, float]],
    A0: Optional[float],
    omega0: Optional[float],
    k: Optional[float],
    cont_success: Optional[bool],
    probe_results: List[Dict[str, Any]],
    verdict: str,
    final_traj: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    n_candidates = len(candidates)
    target_hits = sum(1 for r in probe_results if r["destination"] == "target_attractor")
    numerical_fails = sum(1 for r in probe_results if r["destination"] == "numerical_failure")
    
    # Determine final state bounded classification
    final_class = "numerical_failure"
    if final_traj is not None:
        final_norm = np.linalg.norm(final_traj[-1, 1:])
        if final_norm > config["divergence_norm"]:
            final_class = "simulation_unbounded"
        else:
            final_class = "simulation_bounded"
            
    notes = "El balance armónico (DF) es una heurística tipo Weyl; la ocultedad fue probada bajo radios, muestras, tiempo e integrador especificados."
    
    return {
        "system_id": config["system_id"],
        "transfer_mode": config["transfer_mode"],
        "continuation_mode": config["continuation_mode"],
        "integrator": config["integrator"],
        "memory_mode": config["memory_mode"],
        "q": system.q,
        "seed_strategy": config["seed_strategy"],
        "n_df_candidates": n_candidates,
        "selected_seed": "pos" if n_candidates > 0 else "none",
        "omega0": float(omega0) if omega0 is not None else float("nan"),
        "amplitude_a0": float(A0) if A0 is not None else float("nan"),
        "k": float(k) if k is not None else float("nan"),
        "continuation_success": cont_success,
        "final_class": final_class if final_traj is not None else "continuation_failed",
        "equilibria_count": len(equilibria),
        "hiddenness_tests_enabled": config["run_hiddenness_tests"],
        "radii_tested": [1e-5, 1e-4, 1e-3, 1e-2] if config["run_hiddenness_tests"] else [],
        "samples_per_radius": config["samples_per_radius"] if config["run_hiddenness_tests"] else 0,
        "equilibrium_contacts": target_hits,
        "target_hits_from_equilibria": target_hits,
        "target_hits_from_seed": 1 if final_traj is not None and final_class == "simulation_bounded" else 0,
        "basin_slices_enabled": config["run_basin_slices"],
        "status": verdict,
        "notes": notes
    }

def _save_summary(summary: Dict[str, Any], output_dir: str) -> None:
    # Save JSON summary
    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    # Save CSV summary
    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

def _print_terminal_table(summary: Dict[str, Any]) -> None:
    print("\n" + "="*80)
    print(" RESUMEN FINAL DEL WORKFLOW DE LOCALIZACIÓN ")
    print("="*80)
    print(f"| {'Variable':<30} | {'Valor':<43} |")
    print(f"|{'-'*32}|{'-'*45}|")
    for k, v in summary.items():
        if k == "notes":
            continue
        val_str = str(v)
        if len(val_str) > 43:
            val_str = val_str[:40] + "..."
        print(f"| {k:<30} | {val_str:<43} |")
    print("="*80)
    print(f"Nota: {summary['notes']}\n")
