import yaml
import os
import time
from typing import Any, Dict

DEFAULT_CONFIG = {
    "system_id": "chua_fractional_saturation",
    "q": None,  # Will fallback to system default
    "transfer_mode": "fractional",
    "continuation_mode": "fractional",
    "integrator": "efork",
    "memory_mode": "full",
    "memory_window_length": 4000,
    "run_hiddenness_tests": True,
    "run_basin_slices": False,
    "run_robustness": False,
    "workers": 1,
    "plot_enabled": True,
    "plot_each_phase": False,
    "save_figures": True,
    "output_dir": None,
    "seed_strategy": "nyquist_df",
    "seed_sign_convention": "kuznetsov",
    
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
    
    # Classify settings
    "t_final": 500.0,
    "t_burn": 120.0,
    "h": 0.01,
    "divergence_norm": 120.0,
    "equilibrium_tol": 0.5,
    "target_match_metric": "centroid_distance",
    "target_match_tol": 0.5,
    
    # Basin Settings
    "basin_planes": ["xy", "xz", "yz"],
    "basin_grid_n": 40,
    "basin_extent": 8.0,
    "basin_fixed_z": 0.0,
    "basin_fixed_y": 0.0,
    "basin_fixed_x": 0.0,
    "basin_around_equilibria": True
}

def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file, validate and fill defaults."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
        
    if config_data is None:
        config_data = {}
        
    # Merge defaults
    config = DEFAULT_CONFIG.copy()
    config.update(config_data)
    
    # Validate critical keys
    if config["transfer_mode"] not in {"integer", "fractional"}:
        raise ValueError(f"Invalid transfer_mode: {config['transfer_mode']}")
    if config["continuation_mode"] not in {"integer", "fractional"}:
        raise ValueError(f"Invalid continuation_mode: {config['continuation_mode']}")
    if config["integrator"] not in {"abm", "efork"}:
        raise ValueError(f"Invalid integrator: {config['integrator']}")
    if config["memory_mode"] not in {"full", "window"}:
        raise ValueError(f"Invalid memory_mode: {config['memory_mode']}")
    if config["seed_strategy"] not in {"nyquist_df", "k_phi"}:
        raise ValueError(f"Invalid seed_strategy: {config['seed_strategy']}")
    if config["seed_sign_convention"] not in {"kuznetsov", "wu"}:
        raise ValueError(f"Invalid seed_sign_convention: {config['seed_sign_convention']}")
        
    # Window checks
    if config["memory_mode"] == "window" and (config["memory_window_length"] is None or config["memory_window_length"] <= 0):
        raise ValueError("memory_window_length must be a positive integer when memory_mode is 'window'.")
        
    # Cast numeric configs to correct types to handle PyYAML notation parsing
    for float_key in ["omega_min", "omega_max", "amplitude_min", "amplitude_max", "df_residual_tol", "t_final", "t_burn", "h", "divergence_norm", "equilibrium_tol", "target_match_tol", "basin_extent", "radial_growth_factor", "q"]:
        if float_key in config and config[float_key] is not None:
            config[float_key] = float(config[float_key])
            
    for int_key in ["grid_size_omega", "grid_size_amplitude", "workers", "samples_per_radius", "basin_grid_n", "memory_window_length"]:
        if int_key in config and config[int_key] is not None:
            config[int_key] = int(config[int_key])
            
    # Set default output_dir if not specified
    if config["output_dir"] is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        config["output_dir"] = os.path.join("outputs", config["system_id"], timestamp)
        
    return config
