import yaml
import os
import time
from typing import Any, Dict
from src.contracts import validate_contracts

DEFAULT_CONFIG = {
    "system_id": "chua_fractional_saturation",
    "q": None,  # Will fallback to system default
    "transfer_mode": "fractional",
    "continuation_mode": "fractional",
    "dynamics_mode": "system",
    "integrator": "efork",
    "memory_mode": "full",
    "memory_window_length": 4000,
    "run_hiddenness_tests": False,
    "run_basin_slices": False,
    "run_sphere_tests": False,
    "run_robustness": False,
    "workers": 1,
    
    # Explicit Contracts
    "seed_mode": "fractional",
    "memory_policy": "full_caputo",
    "memory_window_steps": 4000,
    "memory_window_time": None,
    "transfer_convention": "standard",
    "harmonic_condition": "1_minus_WN",
    "q_seed": None,
    "q_dynamics": None,

    
    # Plotting configuration
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
    
    # Grid search and solver tolerances
    "omega_min": 0.01,
    "omega_max": 20.0,
    "amplitude_min": 0.01,
    "amplitude_max": 20.0,
    "grid_size_omega": 200,
    "grid_size_amplitude": 200,
    "root_refinement": True,
    "df_residual_tol": 1e-2,
    
    # Neighborhood / stability settings
    "samples_per_radius": 15,
    "radial_growth_factor": 1.0,
    "directions_mode": "sphere_random",
    "random_seed": 42,
    
    "h": 0.01,
    "divergence_norm": 120.0,
    "equilibrium_tol": 0.5,
    "target_match_metric": "centroid_distance",
    "target_match_tol": 0.5,
    
    # New Nested Configurations
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
    
    # Sphere Probe configuration
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
    
    # Continuation configuration
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

    # Basin Probe configuration
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

def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file, validate and fill defaults."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        
    if config_data is None:
        config_data = {}
        
    # Merge defaults recursively for nested dictionary settings
    config = {}
    for k, v in DEFAULT_CONFIG.items():
        if isinstance(v, dict):
            config[k] = v.copy()
        else:
            config[k] = v
            
    for k, v in config_data.items():
        if isinstance(v, dict) and k in config and isinstance(config[k], dict):
            config[k].update(v)
        else:
            config[k] = v
            
    # Backwards compatibility check with warnings
    if "t_final" in config_data:
        print("WARNING: Using top-level 't_final' config is deprecated. Please configure 'final_simulation', 'sphere_tests', and 'basin' separately.")
        config["final_simulation"]["t_final"] = float(config_data["t_final"])
    if "t_burn" in config_data:
        print("WARNING: Using top-level 't_burn' config is deprecated. Please configure 'final_simulation', 'sphere_tests', and 'basin' separately.")
        config["final_simulation"]["t_burn"] = float(config_data["t_burn"])
    
    # Normalize and map integrator name
    if config.get("integrator") == "efork":
        config["integrator"] = "efork3"

    # Map memory policy <-> memory mode
    if config.get("memory_policy") == "full_caputo":
        config["memory_mode"] = "full"
    elif config.get("memory_policy") == "finite_window":
        config["memory_mode"] = "window"

    if config.get("memory_mode") == "full":
        config["memory_policy"] = "full_caputo"
    elif config.get("memory_mode") == "window":
        config["memory_policy"] = "finite_window"

    if config.get("memory_window_steps") is not None:
        config["memory_window_length"] = int(config["memory_window_steps"])
    elif config.get("memory_window_length") is not None:
        config["memory_window_steps"] = int(config["memory_window_length"])

    if config.get("memory_window_time") is not None and config.get("h") is not None:
        config["memory_window_steps"] = int(np.round(float(config["memory_window_time"]) / float(config["h"])))
        config["memory_window_length"] = config["memory_window_steps"]

    # Run central contracts validation
    validate_contracts(config)

    # Validate critical keys
    if config["transfer_mode"] not in {"integer", "fractional"}:
        raise ValueError(f"Invalid transfer_mode: {config['transfer_mode']}")
    if config["continuation_mode"] not in {"integer", "fractional"}:
        raise ValueError(f"Invalid continuation_mode: {config['continuation_mode']}")
    if config["dynamics_mode"] not in {"integer", "fractional", "system"}:
        raise ValueError(f"Invalid dynamics_mode: {config['dynamics_mode']}")
    if config["integrator"] not in {"abm", "efork3", "efork_q1", "heun"}:
        raise ValueError(f"Invalid integrator: {config['integrator']}")
    if config["memory_mode"] not in {"full", "window", "none"}:
        raise ValueError(f"Invalid memory_mode: {config['memory_mode']}")
    if config["seed_strategy"] not in {"k_phi", "imw_gain", "nyquist_df"}:
        raise ValueError(f"Invalid seed_strategy: {config['seed_strategy']}")
    if config["seed_sign_convention"] not in {"kuznetsov", "wu"}:
        raise ValueError(f"Invalid seed_sign_convention: {config['seed_sign_convention']}")
    if config["seed_construction"] not in {"modal", "closed_form_integer"}:
        raise ValueError(f"Invalid seed_construction: {config['seed_construction']}")
    if config["hiddenness_equilibria_filter"] not in {"all", "unstable_only"}:
        raise ValueError(f"Invalid hiddenness_equilibria_filter: {config['hiddenness_equilibria_filter']}")
    if config["describing_function_mode"] not in {"auto", "closed_form", "piecewise_closed_form", "quadrature", "segmented_quadrature"}:
        raise ValueError(f"Invalid describing_function_mode: {config['describing_function_mode']}")
        
    # Window checks
    if config["memory_mode"] == "window" and (config["memory_window_length"] is None or config["memory_window_length"] <= 0):
        raise ValueError("memory_window_length must be a positive integer when memory_mode is 'window'.")
        
    # Cast numeric configs to correct types
    for float_key in ["omega_min", "omega_max", "amplitude_min", "amplitude_max", "df_residual_tol", "h", "divergence_norm", "equilibrium_tol", "target_match_tol", "q", "seed_theta"]:
        if float_key in config and config[float_key] is not None:
            config[float_key] = float(config[float_key])
            
    for int_key in ["grid_size_omega", "grid_size_amplitude", "workers", "samples_per_radius", "memory_window_length", "branch_index", "max_seed_candidates_to_plot"]:
        if int_key in config and config[int_key] is not None:
            config[int_key] = int(config[int_key])
            
    # Handle early_stop casts
    es = config["early_stop"]
    for float_key in ["divergence_norm", "divergence_growth_factor", "equilibrium_tol", "equilibrium_derivative_tol", "equilibrium_min_time"]:
        if float_key in es and es[float_key] is not None:
            es[float_key] = float(es[float_key])
    for int_key in ["divergence_consecutive_steps", "equilibrium_consecutive_steps"]:
        if int_key in es and es[int_key] is not None:
            es[int_key] = int(es[int_key])
            
    # Handle final_simulation casts
    fs = config["final_simulation"]
    for float_key in ["t_final", "t_burn"]:
        if float_key in fs and fs[float_key] is not None:
            fs[float_key] = float(fs[float_key])
            
    # Handle sphere_tests casts
    st = config["sphere_tests"]
    for float_key in ["t_final", "t_burn", "h", "trajectory_plot_fraction", "samples_growth_factor"]:
        if float_key in st and st[float_key] is not None:
            st[float_key] = float(st[float_key])
    for int_key in ["samples_initial", "random_seed", "max_trajectories_to_plot"]:
        if int_key in st and st[int_key] is not None:
            st[int_key] = int(st[int_key])
    if "radii" in st and st["radii"] is not None:
        st["radii"] = [float(r) for r in st["radii"]]
    if "samples_per_radius" in st and st["samples_per_radius"] is not None:
        st["samples_per_radius"] = [int(s) for s in st["samples_per_radius"]]
        
    # Handle continuation casts
    cont = config["continuation"]
    for float_key in ["eta_min", "eta_max"]:
        if float_key in cont and cont[float_key] is not None:
            cont[float_key] = float(cont[float_key])
    for int_key in ["n_eta", "periods_transient", "periods_keep", "harmonic_history_periods"]:
        if int_key in cont and cont[int_key] is not None:
            cont[int_key] = int(cont[int_key])
    for bool_key in ["start_at_zero", "early_stop_enabled", "require_c_backend",
                     "allow_python_fallback", "build_fractional_harmonic_history",
                     "use_period_based_times"]:
        if bool_key in cont and cont[bool_key] is not None:
            cont[bool_key] = bool(cont[bool_key])
    if "t_transient" in cont and cont["t_transient"] is not None:
        cont["t_transient"] = float(cont["t_transient"])
    if "t_keep" in cont and cont["t_keep"] is not None:
        cont["t_keep"] = float(cont["t_keep"])
    if "eta_values" in cont and cont["eta_values"] is not None:
        cont["eta_values"] = [float(v) for v in cont["eta_values"]]
    if cont.get("eta_grid_mode") not in {"linear", "logarithmic", "adaptive", None}:
        raise ValueError(f"Invalid continuation.eta_grid_mode: {cont['eta_grid_mode']}")

    # Handle basin casts
    b = config["basin"]
    for float_key in ["fixed_x", "fixed_y", "fixed_z", "local_radius", "t_final", "t_burn", "h"]:
        if float_key in b and b[float_key] is not None:
            b[float_key] = float(b[float_key])
    for int_key in ["grid_n"]:
        if int_key in b and b[int_key] is not None:
            b[int_key] = int(b[int_key])
    for list_key in ["x_interval", "y_interval", "z_interval"]:
        if list_key in b and b[list_key] is not None:
            b[list_key] = [float(val) for val in b[list_key]]
            
    # Set default output_dir if not specified
    if config["output_dir"] is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        config["output_dir"] = os.path.join("outputs", config["system_id"], timestamp)
        
    return config
