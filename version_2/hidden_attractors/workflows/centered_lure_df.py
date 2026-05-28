import os
import json
import csv
import yaml
import numpy as np
import dataclasses
from typing import Any, Dict, List, Tuple, Optional
from ..systems import get_system
from ..lure.decomposition import validate_lure_decomposition
from ..lure.transfer import W_eval, W_precompute_spectral, W_eval_from_cache
from ..lure.nyquist import find_harmonic_candidates
from ..lure.seeds import build_lure_seed, build_modal_lure_seed
from ..continuation.continuation_integer import run_integer_continuation
from ..continuation.continuation_fractional import run_fractional_continuation
from ..integrations.selector import integrate
from ..contracts import validate_contracts
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
from ..plotting.plot_trajectories import (
    plot_attractor_trajectories,
    plot_neighborhood_control_spheres,
    plot_flexible_attractor_and_projections,
    plot_timeseries_data,
)
from ..verification.sphere_tests import run_sphere_probe_sweep

DEFAULT_CONFIG = {
    "system_id": "chua_fractional_saturation",
    "q": None,
    "transfer_mode": "fractional",
    "continuation_mode": "fractional",
    "dynamics_mode": "system",
    "integrator": "efork3",
    "memory_mode": "full",
    "memory_window_length": 4000,
    "run_hiddenness_tests": False,
    "run_basin_slices": False,
    "run_sphere_tests": False,
    "run_robustness": False,
    "workers": 1,
    "seed_mode": "fractional",
    "memory_policy": "full_caputo",
    "memory_window_steps": 4000,
    "memory_window_time": None,
    "transfer_convention": "standard",
    "harmonic_condition": "1_minus_WN",
    "q_seed": None,
    "q_dynamics": None,
    "plot_enabled": True,
    "save_figures": True,
    "plot_attractors": True,
    "plot_seed_trajectories": True,
    "plot_transfer": True,
    "plot_describing_function": True,
    "plot_residual_map": True,
    "plot_continuation": True,
    "plot_sphere_tests": True,
    "plot_timeseries": True,
    "plot_matignon": True,
    "max_seed_candidates_to_plot": 3,
    "plot_each_phase": False,
    "output_dir": None,
    "seed_strategy": "k_phi",
    "seed_sign_convention": "kuznetsov",
    "seed_construction": "modal",
    "seed_theta": 0.0,
    "hiddenness_equilibria_filter": "all",
    "describing_function_mode": "auto",
    "branch_index": 0,
    "omega_min": 0.01,
    "omega_max": 20.0,
    "amplitude_min": 0.01,
    "amplitude_max": 20.0,
    "grid_size_omega": 200,
    "grid_size_amplitude": 200,
    "root_refinement": True,
    "df_residual_tol": 1e-2,
    "samples_per_radius": 15,
    "radial_growth_factor": 1.0,
    "directions_mode": "sphere_random",
    "random_seed": 42,
    "h": 0.01,
    "divergence_norm": 120.0,
    "equilibrium_tol": 0.5,
    "target_match_metric": "nn_percentile",
    "target_match_tol": 0.5,
    "attractor_plots": {
        "enabled": True,
        "include_equilibria": False,
        "use_tail_after_burn": True,
        "max_seed_candidates_to_plot": 3,
        "line_width": 0.7,
        "point_size": 0.0
    },
    "early_stop": {
        "enabled": True,
        "divergence_enabled": True,
        "divergence_norm": 80.0,
        "divergence_derivative_norm": None,
        "divergence_consecutive_steps": 5,
        "divergence_growth_factor": 1.25,
        "equilibrium_enabled": True,
        "equilibrium_tol": 1e-3,
        "equilibrium_derivative_tol": 1e-4,
        "equilibrium_consecutive_steps": 200,
        "equilibrium_min_time": 5.0
    },
    "final_simulation": {
        "t_final": 500.0,
        "t_burn": 120.0
    },
    "sphere_tests": {
        "enabled": False,
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
    },
    "continuation": {
        "eta_grid_mode": "adaptive",
        "eta_values": None,
        "eta_min": 1.0e-3,
        "eta_max": 1.0,
        "n_eta": 21,
        "start_at_zero": False,
        "t_transient": None,
        "t_keep": None,
        "periods_transient": 20,
        "periods_keep": 10,
        "use_period_based_times": True,
        "early_stop_enabled": True,
        "require_c_backend": True,
        "allow_python_fallback": False,
        "build_fractional_harmonic_history": True,
        "harmonic_history_periods": 10
    },
    "basin": {
        "enabled": False,
        "planes": ["xy", "xz", "yz"],
        "grid_n": 150,
        "x_interval": [-10.0, 10.0],
        "y_interval": [-10.0, 10.0],
        "z_interval": [-10.0, 10.0],
        "fixed_x": 0.0,
        "fixed_y": 0.0,
        "fixed_z": 0.0,
        "around_equilibria": True,
        "equilibrium_selection": "all",
        "local_radius": 2.0,
        "t_final": 80.0,
        "t_burn": 20.0,
        "h": 0.01,
        "early_stop_enabled": True
    }
}

def build_eta_grid(cont_cfg: dict) -> np.ndarray:
    if cont_cfg.get("eta_values") is not None:
        return np.asarray(cont_cfg["eta_values"], dtype=float)

    mode       = cont_cfg.get("eta_grid_mode", "adaptive")
    eta_min    = float(cont_cfg.get("eta_min", 1e-3))
    eta_max    = float(cont_cfg.get("eta_max", 1.0))
    n_eta      = int(cont_cfg.get("n_eta", 21))
    start_zero = bool(cont_cfg.get("start_at_zero", False))

    if mode == "adaptive":
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
    else:
        start = 0.0 if start_zero else eta_min
        grid = np.linspace(start, eta_max, n_eta)

    if start_zero and grid[0] != 0.0:
        grid = np.concatenate(([0.0], grid))

    return grid

def _save_continuation_trace(cont_steps: list, output_dir: str) -> None:
    if not cont_steps:
        return

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

    trace_json = os.path.join(output_dir, "continuation_trace.json")
    json_rows = []
    for r in rows:
        jr = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
              for k, v in r.items()}
        json_rows.append(jr)
    with open(trace_json, "w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2)

    traj_dir = os.path.join(output_dir, "continuation_steps")
    os.makedirs(traj_dir, exist_ok=True)
    for idx, s in enumerate(cont_steps):
        traj = s.get("trajectory")
        if traj is None or len(traj) == 0:
            continue
        fname = os.path.join(traj_dir, f"continuation_eta_{idx:03d}.csv")
        dim = traj.shape[1] - 1
        header = ["t"] + [f"x{i}" for i in range(dim)]
        with open(fname, "w", newline="", encoding="utf-8") as f:
            w2 = csv.writer(f)
            w2.writerow(header)
            w2.writerows(traj.tolist())

def _get_lure_matrix(system: Any) -> Any:
    return system.lure.matrix if system.lure is not None else getattr(system, "P", None)

def _get_lure_input_vector(system: Any) -> Any:
    return system.lure.input_vector if system.lure is not None else getattr(system, "b", None)

def _get_lure_output_vector(system: Any) -> Any:
    return system.lure.output_vector if system.lure is not None else getattr(system, "r", None)

def _get_lure_nonlinearity(system: Any) -> Any:
    return system.lure.nonlinearity if system.lure is not None else getattr(system, "psi", None)

def _get_describing_function(system: Any) -> Any:
    return system.lure.describing_function if system.lure is not None else getattr(system, "describing_function", None)

def _evaluate_rhs(system: Any, x: Any) -> Any:
    return system.evaluate(x)

def _effective_q(config: dict, system: Any) -> float:
    q = config.get("q", system.parameters.get("q", 1.0))
    if q is None:
        q = 1.0
    return float(q)

def _apply_compatibility_adapter(system: Any, q: float, merged_params: dict) -> None:
    """Transitory compatibility adapter attaching legacy properties to the system.

    This function dynamically injects attributes (`q`, parameter fields, `P`,
    `b`, `r`, `psi`, `describing_function`, `evaluate_rhs`) into the system
    instance to maintain backwards-compatibility with package modules that
    perform numerical or algebraic checks under legacy names.

    Why this is required (Active dependencies):
    -------------------------------------------
    - `hidden_attractors/lure/seeds.py` (Lure seed generation, expects `system.P`, `system.b`, `system.r`)
    - `hidden_attractors/lure/decomposition.py` (Lure decomposition, checks `system.evaluate_rhs`)
    - `hidden_attractors/lure/describing_function.py` (Fourier integration, uses `system.psi` and `system.describing_function`)
    - `hidden_attractors/integrations/numba_kernels.py` & `hidden_attractors/integrations/efork.py` (Numerical solvers, expect `system.P`, `system.b`, `system.r`, `system.psi` for optimized calculations)
    - `hidden_attractors/continuation/continuation_integer.py` & `hidden_attractors/continuation/continuation_fractional.py` (Homotopy tracking, uses `system.P`, `system.b`, `system.r`, `system.psi`)
    - `hidden_attractors/verification/jacobian.py` (Piecewise Jacobian calculations, expect `system.P`)
    - `hidden_attractors/verification/hiddenness.py` (Verification logic, calls `system.evaluate_rhs`)
    - `hidden_attractors/plotting/*` (Visualization routines, e.g., plot_df.py, plot_transfer.py, expect `system.P`, `system.b`, `system.r`, `system.describing_function`)

    TODO (Pending Refactoring):
    --------------------------
    - Refactor `hidden_attractors/integrations/numba_kernels.py` and `hidden_attractors/integrations/efork.py` to read parameters directly from `system.lure.matrix`, `system.lure.input_vector`, etc., instead of relying on top-level `system.P`.
    - Adapt `hidden_attractors/lure/seeds.py` and `hidden_attractors/continuation/` to consume the standard `system.lure` sub-object attributes.
    - Update `hidden_attractors/verification/hiddenness.py` and `hidden_attractors/verification/jacobian.py` to consume the unified `system.evaluate(x)` and `system.lure.matrix` respectively.
    """
    object.__setattr__(system, "q", q)
    for k, v in merged_params.items():
        try:
            object.__setattr__(system, k, v)
        except AttributeError:
            pass
            
    if system.lure is not None:
        object.__setattr__(system, "P", system.lure.matrix)
        object.__setattr__(system, "b", system.lure.input_vector)
        object.__setattr__(system, "r", system.lure.output_vector)
        object.__setattr__(system, "describing_function", system.lure.describing_function)
        object.__setattr__(system, "psi", system.lure.nonlinearity)
    object.__setattr__(system, "evaluate_rhs", lambda x: system.evaluate(x))

def run_workflow_integration(system, x0, q_val, h, t_final, config, equilibria):
    integrator = config["integrator"]
    memory_mode = config["memory_mode"]
    memory_window_length = config.get("memory_window_steps") or config.get("memory_window_length")
    divergence_norm = config["divergence_norm"]
    early_stop = config.get("early_stop")
    
    return integrate(
        rhs=lambda x: system.evaluate(x),
        x0=x0,
        q=q_val,
        h=h,
        t_final=t_final,
        integrator=integrator,
        memory_mode=memory_mode,
        memory_window_length=memory_window_length,
        divergence_norm=divergence_norm,
        system=system,
        use_c_backend=config.get("use_c_backend", True),
        allow_python_fallback=config.get("allow_python_fallback", True),
        early_stop_config=early_stop,
        equilibria=equilibria
    )

def use_c_backend_check(config: Dict[str, Any]) -> bool:
    return config.get("use_c_backend", True)

def run_centered_lure_df_workflow(config: dict) -> dict:
    """Execute the full 7-phase centered Lur'e describing function workflow with early stopping."""
    for k, v in DEFAULT_CONFIG.items():
        config.setdefault(k, v)
        
    system_id = config["system_id"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    run_id = config.get("run_id", "no_run_id")
    
    print(f"[{run_id}][{system_id}] Fase 1/7: cargando configuración... 0%")
    
    sys_kwargs = {}
    if "chua_fractional_arctan" in system_id or "chua_arctan" in system_id:
        # Filter parameter overrides to pass only chua parameters
        for pk in ("alpha", "beta", "gamma", "m0", "m1", "a1", "a2", "rho"):
            if pk in config:
                sys_kwargs[pk] = config[pk]
    else:
        for pk in ("alpha", "beta", "gamma", "m0", "m1"):
            if pk in config:
                sys_kwargs[pk] = config[pk]
                
    # Use version_2 systems module
    name_map = {
        "chua_piecewise": "chua-nonsmooth",
        "chua_integer_saturation": "chua-nonsmooth",
        "chua_fractional_saturation": "chua-nonsmooth",
        "chua_integer_arctan": "chua-arctan",
        "chua_fractional_arctan": "chua-arctan",
        "chua_arctan_wu2023": "fractional-chua-arctan-wu2023",
    }
    normalized_sys_id = name_map.get(system_id, system_id)
    system = get_system(normalized_sys_id)
    
    # Merge overrides and build adapter attributes
    merged_params = dict(system.parameters)
    merged_params.update(sys_kwargs)
    system = dataclasses.replace(system, parameters=merged_params)
    
    q = _effective_q(config, system)
    _apply_compatibility_adapter(system, q, merged_params)
    
    if config.get("q_seed") is not None:
        q_seed = config["q_seed"]
    elif config.get("seed_mode") == "integer":
        q_seed = 1.0
    else:
        q_seed = _effective_q(config, system)

    if config.get("q_dynamics") is not None:
        q_dynamics = config["q_dynamics"]
    elif config.get("dynamics_mode") == "integer":
        q_dynamics = 1.0
    elif config.get("dynamics_mode") == "fractional":
        q_dynamics = _effective_q(config, system)
    else:
        q_dynamics = _effective_q(config, system)

    if config.get("continuation_mode") == "integer":
        q_continuation = 1.0
    else:
        q_continuation = q_dynamics if q_dynamics < 1.0 else _effective_q(config, system)

    if config.get("seed_mode") == "integer":
        seed_transfer_mode = "integer"
    else:
        seed_transfer_mode = "fractional"

    validation_config = config.copy()
    validation_config["q_seed"] = q_seed
    validation_config["q_dynamics"] = q_dynamics
    validation_config["q_continuation"] = q_continuation
    validation_config["seed_transfer_mode"] = seed_transfer_mode
    validation_config["q"] = q
    validate_contracts(validation_config, resolved=True)
    
    effective_config_path = os.path.join(output_dir, "effective_config.yaml")
    effective_config = config.copy()
    effective_config["q"] = q
    effective_config["seed_mode"] = config["seed_mode"]
    effective_config["q_seed_effective"] = q_seed
    effective_config["q_dynamics_effective"] = q_dynamics
    effective_config["q_continuation_effective"] = q_continuation
    effective_config["seed_transfer_mode"] = seed_transfer_mode
    effective_config["transfer_convention"] = config["transfer_convention"]
    effective_config["harmonic_condition"] = config["harmonic_condition"]
    effective_config["memory_policy"] = config["memory_policy"]
    effective_config["history_policy"] = "full_caputo" if config["memory_policy"] == "full_caputo" else ("finite_window" if config["memory_policy"] == "finite_window" else "none")
    
    with open(effective_config_path, "w", encoding="utf-8") as f:
        yaml.dump(effective_config, f, default_flow_style=False)
        
    effective_config_json_path = os.path.join(output_dir, "effective_config.json")
    with open(effective_config_json_path, "w", encoding="utf-8") as f:
        json.dump(effective_config, f, indent=4)
        
    if not validate_lure_decomposition(system):
        print(f"[{run_id}][{system_id}] WARNING: Lur'e decomposition vector field mismatch.")
        
    fs_cfg = config.get("final_simulation", {})
    t_final = fs_cfg.get("t_final", config.get("t_final", 500.0))
    t_burn = fs_cfg.get("t_burn", config.get("t_burn", 120.0))
    
    print(f"[{run_id}][{system_id}] Fase 2/7: calculando equilibrios... 15%")
    equilibria = solve_equilibria(system)
    
    eq_stability = {}
    stable_eqs = []
    unstable_eqs = []
    marginal_eqs = []
    
    for eq_name, eq_pt in equilibria.items():
        stability_res = classify_equilibrium_stability(system, eq_pt)
        eq_stability[eq_name] = stability_res
        if stability_res["stability_class"] == "stable":
            stable_eqs.append(eq_pt)
        elif stability_res["stability_class"] == "marginal_or_inconclusive":
            marginal_eqs.append(eq_pt)
            unstable_eqs.append(eq_pt)
        else:
            unstable_eqs.append(eq_pt)
            
    print(f"[{run_id}][{system_id}] Fase 3/7: construyendo transferencia... 30%")
    omega_grid = np.linspace(config["omega_min"], config["omega_max"], config["grid_size_omega"])

    _W_cache = W_precompute_spectral(
        _get_lure_matrix(system), _get_lure_input_vector(system), _get_lure_output_vector(system),
        transfer_convention=config["transfer_convention"],
    )

    try:
        w_vals = W_eval_from_cache(omega_grid, q_seed, seed_transfer_mode, _W_cache)
    except Exception:
        w_vals = np.full(len(omega_grid), complex(np.nan, np.nan))
        for _i, _w in enumerate(omega_grid):
            try:
                w_vals[_i] = W_eval(
                    _w, q_seed, seed_transfer_mode,
                    _get_lure_matrix(system), _get_lure_input_vector(system), _get_lure_output_vector(system),
                    transfer_convention=config["transfer_convention"],
                )
            except Exception:
                pass

    print(f"[{run_id}][{system_id}] Fase 4/7: buscando semillas DF... 45%")

    _pass_precomputed = config["seed_strategy"] == "nyquist_df"
    candidates = find_harmonic_candidates(
        system=system,
        transfer_mode=seed_transfer_mode,
        seed_strategy=config["seed_strategy"],
        df_residual_tol=config["df_residual_tol"],
        omega_min=config["omega_min"],
        omega_max=config["omega_max"],
        amplitude_min=config["amplitude_min"],
        amplitude_max=config["amplitude_max"],
        grid_size_omega=config["grid_size_omega"],
        grid_size_amplitude=config["grid_size_amplitude"],
        root_refinement=config["root_refinement"],
        q=q_seed,
        describing_function_mode=config["describing_function_mode"],
        transfer_convention=config["transfer_convention"],
        harmonic_condition=config["harmonic_condition"],
        precomputed_W_vals=w_vals if _pass_precomputed else None,
        precomputed_omega_grid=omega_grid if _pass_precomputed else None,
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
        
    if config["plot_enabled"] and n_candidates > 0:
        max_seeds_to_plot = config.get("max_seed_candidates_to_plot", 3)
        for idx in range(min(n_candidates, max_seeds_to_plot)):
            c_A0, c_omega0, c_k = candidates[idx]
            try:
                c_seed_pos, _ = build_lure_seed(
                    system, c_A0, c_omega0, c_k,
                    seed_sign_convention=config["seed_sign_convention"],
                    q=q_seed,
                    transfer_mode=seed_transfer_mode,
                    theta=config.get("seed_theta", 0.0),
                    seed_construction=config.get("seed_construction", "modal"),
                )
                
                c_active_q = q_dynamics
                
                c_t_fin, c_x_fin, c_status = run_workflow_integration(
                    system=system,
                    x0=c_seed_pos,
                    q_val=c_active_q,
                    h=config["h"],
                    t_final=t_final,
                    config=config,
                    equilibria=list(equilibria.values())
                )
                
                if c_status in ("ok", "diverged_early", "converged_equilibrium_early"):
                    c_traj = np.column_stack((c_t_fin, c_x_fin))
                    
                    if c_status == "ok":
                        target_dir = output_dir
                        file_prefix = f"seed_candidate_{idx:02d}"
                    elif c_status == "diverged_early":
                        target_dir = os.path.join(output_dir, "diagnostics", "diverged_seeds")
                        file_prefix = f"seed_diverged_{idx:02d}"
                    else:
                        target_dir = os.path.join(output_dir, "diagnostics", "equilibrium_converged_seeds")
                        file_prefix = f"seed_converged_{idx:02d}"
                        
                    os.makedirs(target_dir, exist_ok=True)
                    
                    plot_flexible_attractor_and_projections(
                        trajectory=c_traj,
                        equilibria=equilibria,
                        config=config,
                        output_dir=target_dir,
                        file_prefix=file_prefix
                    )
                    if config.get("plot_timeseries", True):
                        plot_timeseries_data(
                            trajectory=c_traj,
                            config=config,
                            output_dir=target_dir,
                            file_prefix=file_prefix
                        )
            except Exception as e:
                print(f"[{run_id}][{system_id}] WARNING: Candidate seed {idx} plotting simulation failed: {e}")
 
    branch_idx = config.get("branch_index", 0)
    if branch_idx >= len(candidates):
        print(f"[{run_id}][{system_id}] branch_index {branch_idx} out of range. Selecting index 0.")
        branch_idx = 0
        
    A0, omega0, k = candidates[branch_idx]
    
    seed_pos, seed_neg = build_lure_seed(
        system, A0, omega0, k,
        seed_sign_convention=config["seed_sign_convention"],
        q=q_seed,
        transfer_mode=seed_transfer_mode,
        theta=config.get("seed_theta", 0.0),
        seed_construction=config.get("seed_construction", "modal"),
    )
    
    if config["seed_construction"] == "modal":
        _, v_norm, matched_ev, target_lam = build_modal_lure_seed(
            system, A0, omega0, k, q=q_seed, transfer_mode=seed_transfer_mode, theta=config.get("seed_theta", 0.0)
        )
        r_vec = _get_lure_output_vector(system)
        P_mat = _get_lure_matrix(system)
        b_vec = _get_lure_input_vector(system)
        norm_res = float(np.abs(r_vec.astype(complex) @ v_norm - 1.0))
        P0 = P_mat.astype(complex) + k * np.outer(b_vec.astype(complex), r_vec.astype(complex))
        modal_res = float(np.linalg.norm(P0 @ v_norm - matched_ev * v_norm))
    else:
        matched_ev = complex(0.0, omega0)
        target_lam = complex(0.0, omega0)
        modal_res = 0.0
        norm_res = 0.0
    
    print(f"[{run_id}][{system_id}] Fase 5/7: continuación eta... 60%")
    
    cont_cfg = config.get("continuation", {})
    lambda_grid = build_eta_grid(cont_cfg)
    
    if cont_cfg.get("use_period_based_times", True) and omega0 is not None and omega0 > 0:
        T0 = 2.0 * np.pi / omega0
        t_transient_cont = cont_cfg.get("t_transient") or float(cont_cfg.get("periods_transient", 20)) * T0
        t_keep_cont      = cont_cfg.get("t_keep")      or float(cont_cfg.get("periods_keep",      10)) * T0
    else:
        t_transient_cont = float(cont_cfg.get("t_transient") or 30.0)
        t_keep_cont      = float(cont_cfg.get("t_keep")      or 30.0)
    
    pre_hist_t = None
    pre_hist_x = None
    is_fractional_cont = config["continuation_mode"] == "fractional"
    
    if is_fractional_cont and cont_cfg.get("build_fractional_harmonic_history", True):
        try:
            _, v_norm_pre, _, _ = build_modal_lure_seed(
                system, A0, omega0, k,
                q=q_seed,
                transfer_mode=seed_transfer_mode,
                theta=config.get("seed_theta", 0.0)
            )
            T0_pre = 2.0 * np.pi / omega0
            n_hist_periods = int(cont_cfg.get("harmonic_history_periods", 10))
            h_val = config["h"]
            
            if config["memory_mode"] == "window" and config["memory_window_length"] is not None:
                Lm_time = float(config["memory_window_length"]) * h_val
            else:
                Lm_time = n_hist_periods * T0_pre
            
            n_pre = int(np.ceil(Lm_time / h_val))
            pre_hist_t = np.linspace(-Lm_time, 0.0, n_pre + 1)
            pre_hist_x = np.array([
                A0 * np.real(v_norm_pre * np.exp(1j * omega0 * tj))
                for tj in pre_hist_t
            ])
            print(f"[{run_id}][{system_id}] Prehistoria armónica: {len(pre_hist_t)} puntos, Lm={Lm_time:.2f}s")
        except Exception as exc_pre:
            print(f"[{run_id}][{system_id}] WARNING: No se pudo construir prehistoria armónica: {exc_pre}")
            pre_hist_t = None
            pre_hist_x = None
    
    cont_early_stop = config.get("early_stop", {}).copy()
    if not cont_cfg.get("early_stop_enabled", True):
        cont_early_stop["enabled"] = False
    
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
    else:
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
            require_c_backend=cont_cfg.get("require_c_backend", False),
            allow_python_fallback=cont_cfg.get("allow_python_fallback", True),
            q=q_continuation,
        )
    
    _save_continuation_trace(cont_steps, output_dir)
    
    successful_steps = [s for s in cont_steps if s["status"] == "ok"]
    last_status = cont_steps[-1]["status"] if cont_steps else "no_steps"
    cont_success = bool(
        len(cont_steps) == len(lambda_grid)
        and last_status == "ok"
    )
    
    print(f"[{run_id}][{system_id}] Fase 6/7: simulación final... 75%")
    final_traj = None
    final_status = "continuation_failed"
    
    if cont_success:
        x_final_seed = cont_steps[-1]["x_out"].copy()
        active_q = q_dynamics
            
        t_fin, x_fin, final_status = run_workflow_integration(
            system=system,
            x0=x_final_seed,
            q_val=active_q,
            h=config["h"],
            t_final=t_final,
            config=config,
            equilibria=list(equilibria.values())
        )
                
        if final_status == "ok":
            final_traj = np.column_stack((t_fin, x_fin))
            traj_csv_path = os.path.join(output_dir, "final_attractor.csv")
            with open(traj_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["t", "x", "y", "z"])
                writer.writerows(final_traj.tolist())
                
    ref_tail = np.empty((0, 3))
    if final_status == "ok" and final_traj is not None:
        n_burn = int(np.ceil(t_burn / config["h"]))
        if len(final_traj) > n_burn:
            ref_tail = final_traj[n_burn:, 1:]
            
    print(f"[{run_id}][{system_id}] Fase 7/7: pruebas de ocultedad y cuencas... 90%")
    
    probe_results = []
    target_hits = 0
    numerical_fails = 0
    
    if final_status == "converged_equilibrium_early":
        final_status = "converged_to_equilibrium"
        
    seed_reached_attractor = bool(final_traj is not None and final_status == "ok" and len(ref_tail) > 10)
    
    if (config["run_sphere_tests"] or config["run_hiddenness_tests"]):
        if seed_reached_attractor:
            sphere_results = run_sphere_probe_sweep(
                system=system,
                config=config,
                equilibria=equilibria,
                stable_eqs=stable_eqs,
                ref_tail=ref_tail,
                output_dir=output_dir,
                workers=config["workers"],
                q_dynamics_effective=q_dynamics
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
        
    basin_data_accum = []
    if config["run_basin_slices"] and seed_reached_attractor:
        basin_cfg = config.get("basin", {})
        planes = basin_cfg.get("planes", ["xy", "xz", "yz"])
        around_eq = basin_cfg.get("around_equilibria", True)
        eq_sel = basin_cfg.get("equilibrium_selection", "all")
        
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
                        equilibria_dict=equilibria,
                        q_dynamics_effective=q_dynamics
                    )
                    if config["plot_enabled"]:
                        from ..plotting.plot_basins import plot_basin_slice_file
                        plot_basin_slice_file(plane, u, v, mat, eq_name, config, output_dir)
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
                    equilibria_dict=equilibria,
                    q_dynamics_effective=q_dynamics
                )
                if config["plot_enabled"]:
                    from ..plotting.plot_basins import plot_basin_slice_file
                    plot_basin_slice_file(plane, u, v, mat, "global", config, output_dir)
                for i, u_val in enumerate(u):
                    for j, v_val in enumerate(v):
                        basin_data_accum.append([plane, "global", float(u_val), float(v_val), int(mat[i, j])])
                        
        if len(basin_data_accum) > 0:
            basin_csv_path = os.path.join(output_dir, "basin_results.csv")
            with open(basin_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["plane", "equilibrium", "u_val", "v_val", "classification_code"])
                writer.writerows(basin_data_accum)
                
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
            
    print(f"[{run_id}][{system_id}] terminado... 100%")
    
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
            plot_flexible_attractor_and_projections(final_traj, equilibria, config, output_dir, "final_attractor")
            if config.get("plot_timeseries", True):
                plot_timeseries_data(final_traj, config, output_dir, "final")
                
            if len(probe_results) > 0:
                plot_neighborhood_control_spheres(final_traj, probe_results, equilibria, config, output_dir)
                
        if config.get("plot_matignon", True):
            from ..plotting.plot_matignon import plot_matignon_equilibria
            plot_matignon_equilibria(system, equilibria, config, output_dir)
            
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
        norm_res=norm_res,
        marginal_eqs=marginal_eqs
    )
    _save_summary(summary, output_dir)
    
    if config.get("plot_enabled", True):
        try:
            from ..plotting.generate_publication_figures import generate_all_publication_figures
            generate_all_publication_figures(output_dir, config)
        except Exception as plot_exc:
            print(f"[{run_id}][{system_id}] WARNING: Publication figures generation failed: {plot_exc}")
            import traceback
            traceback.print_exc()
            
    _print_terminal_table(summary)
    
    print("Resultados guardados en:")
    print(f"    {output_dir}/\n")
    
    return summary

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
    norm_res: Optional[float],
    marginal_eqs: Optional[List[np.ndarray]] = None
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
    
    def _c_str(val: Optional[complex]) -> str:
        if val is None:
            return ""
        return f"{val.real:+.12f}{val.imag:+.12f}j"
        
    if config.get("q_seed") is not None:
        q_seed = config["q_seed"]
    elif config.get("seed_mode") == "integer":
        q_seed = 1.0
    else:
        q_seed = _effective_q(config, system)

    if config.get("q_dynamics") is not None:
        q_dynamics = config["q_dynamics"]
    elif config.get("dynamics_mode") == "integer":
        q_dynamics = 1.0
    elif config.get("dynamics_mode") == "fractional":
        q_dynamics = _effective_q(config, system)
    else:
        q_dynamics = _effective_q(config, system)

    if config.get("continuation_mode") == "integer":
        q_continuation = 1.0
    else:
        q_continuation = q_dynamics if q_dynamics < 1.0 else _effective_q(config, system)

    if config.get("seed_mode") == "integer":
        seed_transfer_mode = "integer"
    else:
        seed_transfer_mode = "fractional"
        
    history_policy = "full_caputo" if config.get("memory_policy") == "full_caputo" else ("finite_window" if config.get("memory_policy") == "finite_window" else "none")

    return {
        "system_id": config["system_id"],
        "transfer_mode": config["transfer_mode"],
        "seed_strategy": config["seed_strategy"],
        "seed_construction": config["seed_construction"],
        "dynamics_mode": config["dynamics_mode"],
        "continuation_mode": config["continuation_mode"],
        "integrator": config["integrator"],
        "memory_mode": config["memory_mode"],
        "seed_mode": config.get("seed_mode"),
        "q_seed_effective": q_seed,
        "q_continuation_effective": q_continuation,
        "q_dynamics_effective": q_dynamics,
        "seed_transfer_mode": seed_transfer_mode,
        "transfer_convention": config.get("transfer_convention"),
        "harmonic_condition": config.get("harmonic_condition"),
        "memory_policy": config.get("memory_policy"),
        "history_policy": history_policy,
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
        "marginal_equilibria_count": len(marginal_eqs) if marginal_eqs is not None else 0,
        "marginal_eqs": [eq.tolist() for eq in marginal_eqs] if marginal_eqs is not None else [],
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
    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
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
