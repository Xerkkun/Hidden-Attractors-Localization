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
from ..lure.seeds import build_lure_seed, build_modal_lure_seed
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
from ..plotting.plot_continuation import (
    plot_continuation_eta,
    plot_continuation_tracking,
)
from ..plotting.plot_trajectories import plot_attractor_trajectories, plot_neighborhood_control_spheres, plot_flexible_attractor_and_projections, plot_timeseries_data
from ..verification.sphere_tests import run_sphere_probe_sweep

from .configs import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_eta_grid(cont_cfg: dict) -> np.ndarray:
    """Build the eta (lambda) grid for numerical continuation.

    Modes
    -----
    ``"adaptive"``    : Dense log-like preset that concentrates points near 0
                        and spreads them toward 1 — ideal for most systems.
    ``"logarithmic"`` : ``n_eta`` points spaced logarithmically from
                        ``eta_min`` (or 0) to ``eta_max``.
    ``"linear"``      : ``n_eta`` points spaced linearly.
    Manual override   : if ``eta_values`` is not None, it is used directly.
    """
    # Manual list takes priority
    if cont_cfg.get("eta_values") is not None:
        return np.asarray(cont_cfg["eta_values"], dtype=float)

    mode       = cont_cfg.get("eta_grid_mode", "adaptive")
    eta_min    = float(cont_cfg.get("eta_min", 1e-3))
    eta_max    = float(cont_cfg.get("eta_max", 1.0))
    n_eta      = int(cont_cfg.get("n_eta", 21))
    start_zero = bool(cont_cfg.get("start_at_zero", False))

    if mode == "adaptive":
        # Default adaptive preset — matches the plan specification
        base = np.array([1e-3, 3e-3, 1e-2, 3e-2, 0.07, 0.12,
                         0.2,  0.35, 0.5,  0.7,  0.85, 1.0], dtype=float)
        grid = base[(base >= eta_min) & (base <= eta_max)]
        if len(grid) == 0:
            grid = np.linspace(eta_min, eta_max, max(n_eta, 5))
    elif mode == "logarithmic":
        start = 0.0 if start_zero else eta_min
        if start <= 0.0:
            grid = np.geomspace(eta_min, eta_max, n_eta)
        else:
            grid = np.geomspace(start, eta_max, n_eta)
    else:  # linear
        start = 0.0 if start_zero else eta_min
        grid = np.linspace(start, eta_max, n_eta)

    if start_zero and grid[0] != 0.0:
        grid = np.concatenate(([0.0], grid))

    return grid


def _save_continuation_trace(cont_steps: list, output_dir: str) -> None:
    """Save per-step continuation metadata to CSV + JSON and per-step trajectory CSVs."""
    if not cont_steps:
        return

    # ── trace CSV ─────────────────────────────────────────────────────────
    trace_csv = os.path.join(output_dir, "continuation_trace.csv")
    fieldnames = [
        "step_idx", "lambda_value", "status",
        "x_in_norm", "x_out_norm", "max_norm",
        "n_steps", "t_end",
        "used_c_backend", "rhs_source", "early_stop_reason",
    ]
    rows = []
    for idx, s in enumerate(cont_steps):
        rows.append({
            "step_idx":          idx,
            "lambda_value":      s.get("lambda_value", float("nan")),
            "status":            s.get("status", ""),
            "x_in_norm":         s.get("x_in_norm",  float(np.linalg.norm(s.get("x_in",  [0])))),
            "x_out_norm":        s.get("x_out_norm", float(np.linalg.norm(s.get("x_out", [0])))),
            "max_norm":          s.get("max_norm",   float("nan")),
            "n_steps":           s.get("n_steps",    0),
            "t_end":             s.get("t_end",      float("nan")),
            "used_c_backend":    s.get("used_c_backend", False),
            "rhs_source":        s.get("rhs_source",  ""),
            "early_stop_reason": s.get("early_stop_reason", ""),
        })

    with open(trace_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # ── trace JSON ────────────────────────────────────────────────────────
    trace_json = os.path.join(output_dir, "continuation_trace.json")
    json_rows = []
    for r in rows:
        jr = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
              for k, v in r.items()}
        json_rows.append(jr)
    with open(trace_json, "w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2)

    # ── per-step trajectory CSVs ──────────────────────────────────────────
    traj_dir = os.path.join(output_dir, "continuation_steps")
    os.makedirs(traj_dir, exist_ok=True)
    for idx, s in enumerate(cont_steps):
        traj = s.get("trajectory")
        if traj is None or len(traj) == 0:
            continue
        fname = os.path.join(traj_dir, f"continuation_eta_{idx:03d}.csv")
        # traj shape: (N, 1+dim) → columns t, x, y, z, ...
        dim = traj.shape[1] - 1
        header = ["t"] + [f"x{i}" for i in range(dim)]
        with open(fname, "w", newline="", encoding="utf-8") as f:
            w2 = csv.writer(f)
            w2.writerow(header)
            w2.writerows(traj.tolist())


def run_centered_lure_df_workflow(config: dict) -> dict:
    """Execute the full 7-phase centered Lur'e describing function workflow with early stopping and visual updates."""
    # Inject defaults for any missing keys
    for k, v in DEFAULT_CONFIG.items():
        config.setdefault(k, v)
        
    system_id = config["system_id"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    run_id = config.get("run_id", "no_run_id")
    
    # -------------------------------------------------------------------------
    # Fase 1/7: cargando configuración... 0%
    # -------------------------------------------------------------------------
    print(f"[{run_id}][{system_id}] Fase 1/7: cargando configuración... 0%")
    
    # Instantiate the system with describing_function_mode forwarded if ChuaArctanSystem
    sys_kwargs = {}
    if config["q"] is not None:
        sys_kwargs["q"] = config["q"]
    if "chua_fractional_arctan" in system_id:
        sys_kwargs["describing_function_mode"] = config["describing_function_mode"]
        
    system = get_system_by_id(system_id, **sys_kwargs)
    q = system.q
    
    # Save the effective configuration used (both YAML and JSON)
    effective_config_path = os.path.join(output_dir, "effective_config.yaml")
    effective_config = config.copy()
    effective_config["q"] = q # Store actual q
    with open(effective_config_path, "w", encoding="utf-8") as f:
        yaml.dump(effective_config, f, default_flow_style=False)
        
    effective_config_json_path = os.path.join(output_dir, "effective_config.json")
    with open(effective_config_json_path, "w", encoding="utf-8") as f:
        json.dump(effective_config, f, indent=4)
        
    # Validate the Lur'e split
    if not validate_lure_decomposition(system):
        print(f"[{run_id}][{system_id}] WARNING: Lur'e decomposition vector field mismatch.")
        
    # Extract separate simulation times
    fs_cfg = config.get("final_simulation", {})
    t_final = fs_cfg.get("t_final", config.get("t_final", 500.0))
    t_burn = fs_cfg.get("t_burn", config.get("t_burn", 120.0))
    
    # -------------------------------------------------------------------------
    # Fase 2/7: calculando equilibrios... 15%
    # -------------------------------------------------------------------------
    print(f"[{run_id}][{system_id}] Fase 2/7: calculando equilibrios... 15%")
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
    print(f"[{run_id}][{system_id}] Fase 3/7: construyendo transferencia... 30%")
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
    print(f"[{run_id}][{system_id}] Fase 4/7: buscando semillas DF... 45%")
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
        root_refinement=config["root_refinement"],
        q=q,
        describing_function_mode=config["describing_function_mode"]
    )
    
    n_candidates = len(candidates)
    
    if n_candidates == 0:
        print(f"[{run_id}][{system_id}] No DF seed candidates found.")
        summary = _build_summary_dict(
            config, system, equilibria, unstable_eqs, candidates, None, None, None, None, [], "df_seed_not_found",
            final_traj=None, matched_ev=None, target_lam=None, modal_res=None, norm_res=None
        )
        _save_summary(summary, output_dir)
        return summary
        
    # Plot candidate seed trajectories if plot_enabled is true
    if config["plot_enabled"] and n_candidates > 0:
        max_seeds_to_plot = config.get("max_seed_candidates_to_plot", 3)
        for idx in range(min(n_candidates, max_seeds_to_plot)):
            c_A0, c_omega0, c_k = candidates[idx]
            try:
                # Reconstruct seed
                c_seed_pos, _ = build_lure_seed(
                    system, c_A0, c_omega0, c_k,
                    seed_sign_convention=config["seed_sign_convention"],
                    q=q,
                    transfer_mode=config["transfer_mode"],
                    theta=config.get("seed_theta", 0.0),
                    seed_construction=config.get("seed_construction", "modal"),
                )
                
                # Decide dynamics q based on dynamics_mode
                dyn_mode = config["dynamics_mode"]
                if dyn_mode == "integer":
                    c_active_q = 1.0
                elif dyn_mode == "fractional":
                    c_active_q = q
                elif dyn_mode == "system":
                    c_active_q = 1.0 if q == 1.0 else q
                else:
                    c_active_q = q
                
                # Integrate candidate with early stop
                if c_active_q == 1.0:
                    if config["integrator"] == "abm":
                        c_t_fin, c_x_fin, c_status = caputo_abm_integrate(
                            system.evaluate_rhs, c_seed_pos, q=1.0, h=config["h"], t_final=t_final, divergence_norm=config["divergence_norm"], system=system,
                            early_stop_config=config.get("early_stop"), equilibria=list(equilibria.values())
                        )
                    else:  # efork
                        c_t_fin, c_x_fin, c_status = efork_integrate(
                            system, c_seed_pos, q=1.0, h=config["h"], t_final=t_final, memory_mode="full",
                            divergence_norm=config["divergence_norm"],
                            early_stop_config=config.get("early_stop"), equilibria=list(equilibria.values())
                        )
                else:
                    if config["integrator"] == "abm":
                        c_t_fin, c_x_fin, c_status = caputo_abm_integrate(
                            system.evaluate_rhs, c_seed_pos, q=c_active_q, h=config["h"], t_final=t_final, divergence_norm=config["divergence_norm"],
                            memory_mode=config["memory_mode"], memory_window_length=config["memory_window_length"], system=system,
                            early_stop_config=config.get("early_stop"), equilibria=list(equilibria.values())
                        )
                    else:  # efork
                        c_t_fin, c_x_fin, c_status = efork_integrate(
                            system, c_seed_pos, q=c_active_q, h=config["h"], t_final=t_final,
                            memory_mode=config["memory_mode"], memory_window_length=config["memory_window_length"],
                            divergence_norm=config["divergence_norm"],
                            early_stop_config=config.get("early_stop"), equilibria=list(equilibria.values())
                        )
                
                if c_status in ("ok", "diverged_early", "converged_equilibrium_early"):
                    c_traj = np.column_stack((c_t_fin, c_x_fin))
                    plot_flexible_attractor_and_projections(
                        trajectory=c_traj,
                        equilibria=equilibria,
                        config=config,
                        output_dir=output_dir,
                        file_prefix=f"seed_candidate_{idx:02d}"
                    )
                    # Render and save candidate time series and CSV
                    if config.get("plot_timeseries", True):
                        plot_timeseries_data(
                            trajectory=c_traj,
                            config=config,
                            output_dir=output_dir,
                            file_prefix=f"seed_candidate_{idx:02d}"
                        )
            except Exception as e:
                print(f"[{run_id}][{system_id}] WARNING: Candidate seed {idx} plotting simulation failed: {e}")
 
    # Select the candidate based on branch_index
    branch_idx = config.get("branch_index", 0)
    if branch_idx >= len(candidates):
        print(f"[{run_id}][{system_id}] branch_index {branch_idx} out of range (found {len(candidates)} candidates). Selecting index 0.")
        branch_idx = 0
        
    A0, omega0, k = candidates[branch_idx]
    
    # Reconstruct the seed
    seed_pos, seed_neg = build_lure_seed(
        system, A0, omega0, k,
        seed_sign_convention=config["seed_sign_convention"],
        q=q,
        transfer_mode=config["transfer_mode"],
        theta=config.get("seed_theta", 0.0),
        seed_construction=config.get("seed_construction", "modal"),
    )
    
    # Compute residuals for modal metrics
    if config["seed_construction"] == "modal":
        _, v_norm, matched_ev, target_lam = build_modal_lure_seed(
            system, A0, omega0, k, q=q, transfer_mode=config["transfer_mode"], theta=config.get("seed_theta", 0.0)
        )
        norm_res = float(np.abs(system.r.astype(complex) @ v_norm - 1.0))
        P0 = system.P.astype(complex) + k * np.outer(system.b.astype(complex), system.r.astype(complex))
        modal_res = float(np.linalg.norm(P0 @ v_norm - matched_ev * v_norm))
    else:
        matched_ev = complex(0.0, omega0)
        target_lam = complex(0.0, omega0)
        modal_res = 0.0
        norm_res = 0.0
    
    # -------------------------------------------------------------------------
    # Fase 5/7: continuación eta... 60%
    # -------------------------------------------------------------------------
    print(f"[{run_id}][{system_id}] Fase 5/7: continuación eta... 60%")
    
    cont_cfg = config.get("continuation", {})
    
    # ── A2: build the eta grid ─────────────────────────────────────────────
    lambda_grid = build_eta_grid(cont_cfg)
    
    # ── Period-based transient / keep times ───────────────────────────────
    if cont_cfg.get("use_period_based_times", True) and omega0 is not None and omega0 > 0:
        T0 = 2.0 * np.pi / omega0
        t_transient_cont = cont_cfg.get("t_transient") or float(cont_cfg.get("periods_transient", 20)) * T0
        t_keep_cont      = cont_cfg.get("t_keep")      or float(cont_cfg.get("periods_keep",      10)) * T0
    else:
        t_transient_cont = float(cont_cfg.get("t_transient") or 30.0)
        t_keep_cont      = float(cont_cfg.get("t_keep")      or 30.0)
    
    # ── B2-B3: build fractional harmonic prehistory ────────────────────────
    pre_hist_t = None
    pre_hist_x = None
    is_fractional_cont = config["continuation_mode"] == "fractional"
    
    if is_fractional_cont and cont_cfg.get("build_fractional_harmonic_history", True):
        try:
            _, v_norm_pre, _, _ = build_modal_lure_seed(
                system, A0, omega0, k,
                q=q,
                transfer_mode=config["transfer_mode"],
                theta=config.get("seed_theta", 0.0)
            )
            T0_pre = 2.0 * np.pi / omega0
            n_hist_periods = int(cont_cfg.get("harmonic_history_periods", 10))
            h_val = config["h"]
            
            # Memory window length in time
            if config["memory_mode"] == "window" and config["memory_window_length"] is not None:
                Lm_time = float(config["memory_window_length"]) * h_val
            else:
                Lm_time = n_hist_periods * T0_pre
            
            # Sample prehistory times t_j in [-Lm_time, 0]
            n_pre = int(np.ceil(Lm_time / h_val))
            pre_hist_t = np.linspace(-Lm_time, 0.0, n_pre + 1)
            # x(t_j) = A0 * Re(v_norm * exp(i*omega0*t_j))
            pre_hist_x = np.array([
                A0 * np.real(v_norm_pre * np.exp(1j * omega0 * tj))
                for tj in pre_hist_t
            ])
            print(f"[{run_id}][{system_id}] Prehistoria armónica: {len(pre_hist_t)} puntos, Lm={Lm_time:.2f}s")
        except Exception as exc_pre:
            print(f"[{run_id}][{system_id}] WARNING: No se pudo construir prehistoria armónica: {exc_pre}")
            pre_hist_t = None
            pre_hist_x = None
    
    # ── Continuation early-stop config ────────────────────────────────────
    cont_early_stop = config.get("early_stop", {}).copy()
    if not cont_cfg.get("early_stop_enabled", True):
        cont_early_stop["enabled"] = False
    
    # Run continuation from the positive seed
    if config["continuation_mode"] == "integer":
        cont_steps = run_integer_continuation(
            system=system,
            seed_x0=seed_pos,
            k_gain=k,
            lambda_values=lambda_grid,
            h=config["h"],
            t_transient=t_transient_cont,
            t_keep=t_keep_cont,
            div_threshold=config["divergence_norm"],
            integrator=config["integrator"],
            early_stop_config=cont_early_stop,
            equilibria=list(equilibria.values()),
        )
    else:  # fractional
        cont_steps = run_fractional_continuation(
            system=system,
            seed_x0=seed_pos,
            k_gain=k,
            lambda_values=lambda_grid,
            h=config["h"],
            t_transient=t_transient_cont,
            t_keep=t_keep_cont,
            memory_mode=config["memory_mode"],
            memory_window_length=config["memory_window_length"],
            div_threshold=config["divergence_norm"],
            integrator=config["integrator"],
            use_c_backend=use_c_backend_check(config),
            history_times=pre_hist_t,
            history_states=pre_hist_x,
            early_stop_config=cont_early_stop,
            equilibria=list(equilibria.values()),
            require_c_backend=cont_cfg.get("require_c_backend", True),
            allow_python_fallback=cont_cfg.get("allow_python_fallback", False),
        )
    
    # ── Step logging ──────────────────────────────────────────────────────
    _save_continuation_trace(cont_steps, output_dir)
    
    # ── Evaluate success ──────────────────────────────────────────────────
    # Success = all steps completed with status "ok"
    successful_steps = [s for s in cont_steps if s["status"] == "ok"]
    last_status = cont_steps[-1]["status"] if cont_steps else "no_steps"
    cont_success = bool(
        len(cont_steps) == len(lambda_grid)
        and last_status == "ok"
    )
    
    # -------------------------------------------------------------------------
    # Fase 6/7: simulación final... 75%
    # -------------------------------------------------------------------------
    print(f"[{run_id}][{system_id}] Fase 6/7: simulación final... 75%")
    final_traj = None
    final_status = "continuation_failed"
    
    if cont_success:
        x_final_seed = cont_steps[-1]["x_out"].copy()
        
        # Decide dynamics q based on dynamics_mode
        dyn_mode = config["dynamics_mode"]
        if dyn_mode == "integer":
            active_q = 1.0
        elif dyn_mode == "fractional":
            active_q = q
        elif dyn_mode == "system":
            active_q = 1.0 if q == 1.0 else q
        else:
            raise ValueError(f"Unknown dynamics_mode: {dyn_mode}")
            
        # Long final integration at eta = 1.0 with early stop support
        if active_q == 1.0:
            if config["integrator"] == "abm":
                t_fin, x_fin, final_status = caputo_abm_integrate(
                    system.evaluate_rhs, x_final_seed, q=1.0, h=config["h"], t_final=t_final, divergence_norm=config["divergence_norm"], system=system,
                    early_stop_config=config.get("early_stop"), equilibria=list(equilibria.values())
                )
            else: # efork
                t_fin, x_fin, final_status = efork_integrate(
                    system, x_final_seed, q=1.0, h=config["h"], t_final=t_final, memory_mode="full",
                    divergence_norm=config["divergence_norm"],
                    early_stop_config=config.get("early_stop"), equilibria=list(equilibria.values())
                )
        else: # fractional
            if config["integrator"] == "abm":
                t_fin, x_fin, final_status = caputo_abm_integrate(
                    system.evaluate_rhs, x_final_seed, q=active_q, h=config["h"], t_final=t_final, divergence_norm=config["divergence_norm"],
                    memory_mode=config["memory_mode"], memory_window_length=config["memory_window_length"], system=system,
                    early_stop_config=config.get("early_stop"), equilibria=list(equilibria.values())
                )
            else: # efork
                t_fin, x_fin, final_status = efork_integrate(
                    system, x_final_seed, q=active_q, h=config["h"], t_final=t_final,
                    memory_mode=config["memory_mode"], memory_window_length=config["memory_window_length"],
                    divergence_norm=config["divergence_norm"],
                    early_stop_config=config.get("early_stop"), equilibria=list(equilibria.values())
                )
                
        if final_status in ("ok", "diverged_early", "converged_equilibrium_early"):
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
        n_burn = int(np.ceil(t_burn / config["h"]))
        if len(final_traj) > n_burn:
            ref_tail = final_traj[n_burn:, 1:]
            
    # -------------------------------------------------------------------------
    # Fase 7/7: pruebas de ocultedad y cuencas... 90%
    # -------------------------------------------------------------------------
    print(f"[{run_id}][{system_id}] Fase 7/7: pruebas de ocultedad y cuencas... 90%")
    
    probe_results = []
    target_hits = 0
    numerical_fails = 0
    seed_reached_attractor = bool(final_traj is not None and final_status in ("ok", "diverged_early", "converged_equilibrium_early") and len(ref_tail) > 10)
    
    # Run sphere verification tests
    if (config["run_sphere_tests"] or config["run_hiddenness_tests"]):
        if seed_reached_attractor:
            sphere_results = run_sphere_probe_sweep(
                system=system,
                config=config,
                equilibria=equilibria,
                stable_eqs=stable_eqs,
                ref_tail=ref_tail,
                output_dir=output_dir,
                workers=config["workers"]
            )
            probe_results = sphere_results["probe_runs"]
            target_hits = sum(1 for r in probe_results if r["destination"] == "target_attractor")
            numerical_fails = sum(1 for r in probe_results if r["destination"] == "numerical_failure")
            
            verdict = classify_hiddenness_verdict(
                target_hits_from_equilibria=target_hits,
                equilibria_count=len(equilibria),
                unstable_equilibria_count=len(unstable_eqs),
                seed_reached_attractor=seed_reached_attractor,
                numerical_failures=numerical_fails
            )
        else:
            print(f"[{run_id}][{system_id}] WARNING: Seed did not reach the target attractor. Skipping sphere tests.")
            verdict = "df_seed_not_found"
    else:
        verdict = "df_seed_found" if seed_reached_attractor else "df_seed_not_found"
        
    # Evaluate basin slices if enabled
    basin_data_accum = []
    if config["run_basin_slices"] and seed_reached_attractor:
        basin_cfg = config.get("basin", {})
        planes = basin_cfg.get("planes", ["xy", "xz", "yz"])
        around_eq = basin_cfg.get("around_equilibria", True)
        eq_sel = basin_cfg.get("equilibrium_selection", "all")
        
        # Determine equilibria selection for basin scans
        selected_eq_basin = {}
        if eq_sel == "all":
            selected_eq_basin = equilibria.copy()
        elif isinstance(eq_sel, list):
            for name in eq_sel:
                if name in equilibria:
                    selected_eq_basin[name] = equilibria[name]
        elif isinstance(eq_sel, str):
            if eq_sel in equilibria:
                selected_eq_basin[eq_sel] = equilibria[eq_sel]
                
        # Perform scans with early stopping
        if around_eq:
            for eq_name, eq_pt in selected_eq_basin.items():
                for plane in planes:
                    u, v, mat = generate_basin_slice(
                        plane=plane,
                        system=system,
                        transfer_mode=config["transfer_mode"],
                        integrator=config["integrator"],
                        ref_tail=ref_tail,
                        stable_eqs=stable_eqs,
                        fixed_values={"z": float(eq_pt[2]), "y": float(eq_pt[1]), "x": float(eq_pt[0])},
                        grid_n=basin_cfg.get("grid_n", 150),
                        center=eq_pt.tolist(),
                        t_final=basin_cfg.get("t_final", 80.0),
                        t_burn=basin_cfg.get("t_burn", 20.0),
                        h=basin_cfg.get("h", 0.01),
                        workers=config["workers"],
                        eq_tol=config["equilibrium_tol"],
                        div_norm=config["divergence_norm"],
                        metric=config["target_match_metric"],
                        tol=config["target_match_tol"],
                        dynamics_mode=config["dynamics_mode"],
                        memory_mode=config["memory_mode"],
                        memory_window_length=config["memory_window_length"],
                        around_equilibria=True,
                        local_radius=basin_cfg.get("local_radius", 2.0),
                        eq_name=eq_name,
                        system_id=system_id,
                        early_stop_config=config.get("early_stop"),
                        equilibria_dict=equilibria
                    )
                    # Plot slice
                    if config["plot_enabled"]:
                        from ..plotting.plot_basins import plot_basin_slice_file
                        plot_basin_slice_file(plane, u, v, mat, eq_name, config, output_dir)
                    # Accumulate for CSV export
                    for i, u_val in enumerate(u):
                        for j, v_val in enumerate(v):
                            basin_data_accum.append([plane, eq_name, float(u_val), float(v_val), int(mat[i, j])])
        else:
            center_pt = equilibria.get("E0", np.zeros(3)).tolist()
            for plane in planes:
                u, v, mat = generate_basin_slice(
                    plane=plane,
                    system=system,
                    transfer_mode=config["transfer_mode"],
                    integrator=config["integrator"],
                    ref_tail=ref_tail,
                    stable_eqs=stable_eqs,
                    fixed_values={"z": basin_cfg.get("fixed_z", 0.0), "y": basin_cfg.get("fixed_y", 0.0), "x": basin_cfg.get("fixed_x", 0.0)},
                    grid_n=basin_cfg.get("grid_n", 150),
                    center=center_pt,
                    t_final=basin_cfg.get("t_final", 80.0),
                    t_burn=basin_cfg.get("t_burn", 20.0),
                    h=basin_cfg.get("h", 0.01),
                    workers=config["workers"],
                    eq_tol=config["equilibrium_tol"],
                    div_norm=config["divergence_norm"],
                    metric=config["target_match_metric"],
                    tol=config["target_match_tol"],
                    dynamics_mode=config["dynamics_mode"],
                    memory_mode=config["memory_mode"],
                    memory_window_length=config["memory_window_length"],
                    x_interval=basin_cfg.get("x_interval"),
                    y_interval=basin_cfg.get("y_interval"),
                    z_interval=basin_cfg.get("z_interval"),
                    around_equilibria=False,
                    eq_name="global",
                    system_id=system_id,
                    early_stop_config=config.get("early_stop"),
                    equilibria_dict=equilibria
                )
                # Plot slice
                if config["plot_enabled"]:
                    from ..plotting.plot_basins import plot_basin_slice_file
                    plot_basin_slice_file(plane, u, v, mat, "global", config, output_dir)
                # Accumulate for CSV export
                for i, u_val in enumerate(u):
                    for j, v_val in enumerate(v):
                        basin_data_accum.append([plane, "global", float(u_val), float(v_val), int(mat[i, j])])
                        
        # Save basin results database CSV
        if len(basin_data_accum) > 0:
            basin_csv_path = os.path.join(output_dir, "basin_results.csv")
            with open(basin_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["plane", "equilibrium", "u_val", "v_val", "classification_code"])
                writer.writerows(basin_data_accum)
                
        # Save basin metadata JSON
        basin_meta = {
            "system_id": system_id,
            "grid_n": basin_cfg.get("grid_n", 150),
            "planes": planes,
            "around_equilibria": around_eq,
            "local_radius": basin_cfg.get("local_radius", 2.0),
            "x_interval": basin_cfg.get("x_interval"),
            "y_interval": basin_cfg.get("y_interval"),
            "z_interval": basin_cfg.get("z_interval")
        }
        with open(os.path.join(output_dir, "basin_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(basin_meta, f, indent=4)
            
    # -------------------------------------------------------------------------
    # terminado... 100%
    # -------------------------------------------------------------------------
    print(f"[{run_id}][{system_id}] terminado... 100%")
    
    # 1. Trigger plots if enabled
    if config["plot_enabled"]:
        plot_nyquist_transfer(omega_grid, w_vals, candidates, config, output_dir)
        plot_describing_function(system, candidates, config, output_dir)
        plot_continuation_eta(cont_steps, config, output_dir)
        if len(cont_steps) >= 1:
            try:
                plot_continuation_tracking(cont_steps, config, output_dir)
            except Exception as exc_track:
                print(f"[{run_id}][{system_id}] WARNING: tracking plots failed: {exc_track}")
        if final_traj is not None:
            # Saves final_attractor_3d.png and separate xy, xz, yz projection files
            plot_flexible_attractor_and_projections(final_traj, equilibria, config, output_dir, "final_attractor")
            # Save final time series and CSV
            if config.get("plot_timeseries", True):
                plot_timeseries_data(final_traj, config, output_dir, "final")
                
            if len(probe_results) > 0:
                plot_neighborhood_control_spheres(final_traj, probe_results, equilibria, config, output_dir)
                
        # Save Matignon fractional stability plot
        if config.get("plot_matignon", True):
            from ..plotting.plot_matignon import plot_matignon_equilibria
            plot_matignon_equilibria(system, equilibria, config, output_dir)
            
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
        final_traj=final_traj,
        matched_ev=matched_ev,
        target_lam=target_lam,
        modal_res=modal_res,
        norm_res=norm_res
    )
    _save_summary(summary, output_dir)
    
    # 3. Print Markdown summary table in the terminal
    _print_terminal_table(summary)
    
    # 4. Print final output folder path
    print("Resultados guardados en:")
    print(f"    {output_dir}/\n")
    
    return summary

def use_c_backend_check(config: Dict[str, Any]) -> bool:
    """Helper to check if C backend should be used."""
    return config.get("use_c_backend", True)

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
    final_traj: Optional[np.ndarray],
    matched_ev: Optional[complex],
    target_lam: Optional[complex],
    modal_res: Optional[float],
    norm_res: Optional[float]
) -> Dict[str, Any]:
    n_candidates = len(candidates)
    target_hits = sum(1 for r in probe_results if r["destination"] == "target_attractor")
    
    final_class = "numerical_failure"
    if final_traj is not None:
        final_norm = np.linalg.norm(final_traj[-1, 1:])
        if final_norm > config["divergence_norm"]:
            final_class = "simulation_unbounded"
        else:
            final_class = "simulation_bounded"
            
    notes = "El balance armónico (DF) es una heurística tipo Weyl; la ocultedad fue probada bajo radios, muestras, tiempo e integrador especificados."
    
    # Format complex values to string safely
    def _c_str(val: Optional[complex]) -> str:
        if val is None:
            return ""
        return f"{val.real:+.12f}{val.imag:+.12f}j"
        
    return {
        "system_id": config["system_id"],
        "transfer_mode": config["transfer_mode"],
        "seed_strategy": config["seed_strategy"],
        "seed_construction": config["seed_construction"],
        "dynamics_mode": config["dynamics_mode"],
        "continuation_mode": config["continuation_mode"],
        "integrator": config["integrator"],
        "memory_mode": config["memory_mode"],
        "branch_index": config.get("branch_index", 0),
        "n_df_candidates": n_candidates,
        "selected_seed": "pos" if n_candidates > 0 else "none",
        "omega0": float(omega0) if omega0 is not None else float("nan"),
        "amplitude_a0": float(A0) if A0 is not None else float("nan"),
        "k": float(k) if k is not None else float("nan"),
        "matched_eigenvalue": _c_str(matched_ev),
        "lambda0": _c_str(target_lam),
        "modal_residual": float(modal_res) if modal_res is not None else float("nan"),
        "normalisation_residual": float(norm_res) if norm_res is not None else float("nan"),
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
        csv_writer = csv.writer(f)
        csv_writer.writerow(summary.keys())
        csv_writer.writerow(summary.values())

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
