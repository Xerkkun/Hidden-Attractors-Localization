"""YAML configuration loader and normalizer for hidden-attractor workflows.

Stability: experimental

This module is the single source of truth for loading, normalizing, validating
and saving experiment configurations.  It supports two YAML schemas:

Hierarchical schema (current / recommended)
    Uses nested sections: ``experiment``, ``system``, ``modes``, ``integrator``,
    ``stages``, ``simulation``, ``plots``, ``basin``, ``bifurcation``, etc.

Flat schema (legacy, deprecated)
    Top-level keys like ``system_id``, ``q``, ``integrator``, ``t_final``, …
    Triggers deprecation warnings when detected.

Usage
-----
::

    from hidden_attractors.workflows.config_loader import load_config, save_effective_config

    cfg = load_config("configs/examples/chua_fractional_centered_lure_df.yaml")
    # cfg is a fully-normalized flat dict ready for workflow functions.

    save_effective_config(cfg, output_dir="outputs/run_001")
"""

from __future__ import annotations

import os
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# ---------------------------------------------------------------------------
# Default configuration (hierarchical → normalized flat)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    # ── System ───────────────────────────────────────────────────────────────
    "system_id": "chua_fractional_saturation",
    "q": None,  # fallback to system default

    # ── Modes ────────────────────────────────────────────────────────────────
    "transfer_mode": "fractional",
    "seed_mode": "fractional",
    "continuation_mode": "fractional",
    "dynamics_mode": "system",

    # ── Integrator ───────────────────────────────────────────────────────────
    "integrator": "efork3",
    "h": 0.001,
    "memory_mode": "full",
    "memory_policy": "full_caputo",
    "memory_window_steps": 400,
    "memory_window_length": 400,
    "memory_window_time": None,
    "use_c_backend": True,
    "allow_python_fallback": True,

    # ── Stages ───────────────────────────────────────────────────────────────
    "run_seed_search": True,
    "run_continuation": True,
    "run_final_simulation": True,
    "run_hiddenness_tests": False,
    "run_sphere_tests": False,
    "run_basin_slices": False,
    "run_bifurcation": False,
    "run_attractor_only": False,

    # ── Seed search ──────────────────────────────────────────────────────────
    "seed_strategy": "k_phi",
    "seed_sign_convention": "kuznetsov",
    "seed_construction": "modal",
    "seed_theta": 0.0,
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
    "hiddenness_equilibria_filter": "all",
    "transfer_convention": "standard",
    "harmonic_condition": "1_minus_WN",
    "q_seed": None,
    "q_dynamics": None,

    # ── Classical route feature flags ────────────────────────────────────────────
    # machado_enabled: Machado generalised DF with mu parameter.
    # Default False: not in the published classical route.
    "machado_enabled": False,
    # biased_enabled: biased seeds with sigma0 != 0.
    # Default False: only centred seeds in the classical published route.
    "biased_enabled": False,
    # seed_filter: post-search quality filter on harmonic residual / rho_H.
    # Default disabled: all candidates from the base search are accepted.
    "seed_filter": {
        "enabled": False,
        "harmonic_residual_keep": 0.05,
        "rho_H_keep": 0.3,
    },

    # ── Workers / reproducibility ─────────────────────────────────────────────
    "workers": 1,
    "random_seed": 42,

    # ── Divergence ───────────────────────────────────────────────────────────
    "divergence_norm": 120.0,
    "equilibrium_tol": 0.5,
    "target_match_metric": "nn_percentile",
    "target_match_tol": 0.5,

    # ── Output ───────────────────────────────────────────────────────────────
    "output_dir": None,
    "run_id": "auto",

    # ── Plotting ─────────────────────────────────────────────────────────────
    "plot_enabled": True,
    "save_figures": True,
    "plot_attractors": True,
    "plot_timeseries": True,
    "plot_transfer": True,
    "plot_describing_function": True,
    "plot_residual_map": True,
    "plot_continuation": True,
    "plot_sphere_tests": True,
    "plot_matignon": True,
    "plot_basin": True,
    "plot_bifurcation": True,
    "max_seed_candidates_to_plot": 3,

    # ── Nested sections (preserved as dicts) ─────────────────────────────────
    "early_stop": {
        "enabled": True,
        "divergence_enabled": True,
        "divergence_norm": 80.0,
        "divergence_consecutive_steps": 5,
        "divergence_growth_factor": 1.25,
        "equilibrium_enabled": True,
        "equilibrium_tol": 1e-3,
        "equilibrium_derivative_tol": 1e-4,
        "equilibrium_consecutive_steps": 200,
        "equilibrium_min_time": 5.0,
    },

    "final_simulation": {
        "t_final": 500.0,
        "t_burn": 120.0,
        "initial_condition": None,
        "divergence_norm": 120.0,
    },

    "continuation": {
        # lambda_values: explicit list overrides the adaptive eta grid.
        # When provided, these values are used exactly (no adaptive fallback).
        # Official contract values: [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
        "lambda_values": None,
        "eta_grid_mode": "adaptive",
        "eta_values": None,
        "eta_min": 1.0e-3,
        "eta_max": 1.0,
        "n_eta": 21,
        "start_at_zero": False,
        "use_period_based_times": True,
        "periods_transient": 20,
        "periods_keep": 10,
        "build_fractional_harmonic_history": True,
        "harmonic_history_periods": 10,
        "early_stop_enabled": True,
        "require_c_backend": True,
        "allow_python_fallback": False,
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
        "early_stop_enabled": True,
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
        "early_stop_enabled": True,
    },

    "bifurcation": {
        "enabled": False,
        "parameter": "beta",
        "values": {"min": 8.0, "max": 16.0, "n": 300},
        "continuation_between_values": True,
        "initial_condition": [0.1, 0.0, 0.0],
        "discard_time": 200.0,
        "sample_time": 200.0,
        "h": 0.01,
        "coordinate": "x",
        "sampling": {
            "method": "local_maxima",
            "max_points_per_parameter": 200,
        },
        "save_csv": True,
        "save_plot": True,
        "workers": 1,
    },

    "attractor_plots": {
        "include_equilibria": False,
        "use_tail_after_burn": True,
        "max_seed_candidates_to_plot": 3,
        "line_width": 0.7,
        "point_size": 0.0,
    },

    # ── Robustness (second priority, disabled by default) ───────────────────
    "robustness": {
        "enabled": False,
    },
}

# ---------------------------------------------------------------------------
# Hierarchical → flat mapping
# ---------------------------------------------------------------------------

def _flatten_hierarchical(raw: Dict[str, Any]) -> Dict[str, Any]:  # noqa: C901
    """Convert new hierarchical YAML schema to internal flat dict."""
    flat: Dict[str, Any] = {}

    # experiment section
    exp = raw.get("experiment", {})
    if exp:
        flat["output_dir"] = exp.get("output_dir", None)
        flat["run_id"] = exp.get("run_id", "auto")
        flat["random_seed"] = exp.get("random_seed", 42)
        flat["_experiment_name"] = exp.get("name", "")
        flat["_description"] = exp.get("description", "")

    # system section
    sys_sec = raw.get("system", {})
    if sys_sec:
        flat["system_id"] = sys_sec.get("system_id", _DEFAULTS["system_id"])
        flat["q"] = sys_sec.get("q", None)
        params = sys_sec.get("parameters", {})
        flat.update(params)  # alpha, beta, gamma, m0/m, m1/n, etc.

    # modes section
    modes = raw.get("modes", {})
    if modes:
        flat["transfer_mode"] = modes.get("transfer_mode", _DEFAULTS["transfer_mode"])
        flat["seed_mode"] = modes.get("seed_mode", _DEFAULTS["seed_mode"])
        flat["continuation_mode"] = modes.get("continuation_mode", _DEFAULTS["continuation_mode"])
        flat["dynamics_mode"] = modes.get("dynamics_mode", _DEFAULTS["dynamics_mode"])

    # integrator section
    integ = raw.get("integrator", {})
    if isinstance(integ, dict):
        flat["integrator"] = integ.get("name", _DEFAULTS["integrator"])
        flat["h"] = integ.get("h", _DEFAULTS["h"])
        if "memory_mode" in integ:
            flat["memory_mode"] = integ["memory_mode"]
        if "memory_policy" in integ:
            flat["memory_policy"] = integ["memory_policy"]
        if "memory_window_steps" in integ:
            flat["memory_window_steps"] = integ["memory_window_steps"]
        elif "memory_window_length" in integ:
            flat["memory_window_steps"] = integ["memory_window_length"]
        if "memory_window_time" in integ:
            flat["memory_window_time"] = integ["memory_window_time"]
        flat["use_c_backend"] = integ.get("use_c_backend", _DEFAULTS["use_c_backend"])
        flat["allow_python_fallback"] = integ.get("allow_python_fallback", _DEFAULTS["allow_python_fallback"])
    elif isinstance(integ, str):
        flat["integrator"] = integ  # flat legacy key inside hierarchical doc

    # stages section
    stages = raw.get("stages", {})
    if stages:
        flat["run_seed_search"] = stages.get("seed_search", _DEFAULTS["run_seed_search"])
        flat["run_continuation"] = stages.get("continuation", _DEFAULTS["run_continuation"])
        flat["run_final_simulation"] = stages.get("final_simulation", _DEFAULTS["run_final_simulation"])
        flat["run_hiddenness_tests"] = stages.get("hiddenness_tests", _DEFAULTS["run_hiddenness_tests"])
        flat["run_sphere_tests"] = stages.get("sphere_tests", _DEFAULTS["run_sphere_tests"])
        flat["run_basin_slices"] = stages.get("basin_slices", _DEFAULTS["run_basin_slices"])
        flat["run_bifurcation"] = stages.get("bifurcation", _DEFAULTS["run_bifurcation"])
        flat["run_attractor_only"] = stages.get("attractor_only", _DEFAULTS["run_attractor_only"])

    # seed_search section
    ss = raw.get("seed_search", {})
    if ss:
        for k, dk in [
            ("strategy", "seed_strategy"),
            ("construction", "seed_construction"),
            ("branch_index", "branch_index"),
            ("omega_min", "omega_min"),
            ("omega_max", "omega_max"),
            ("amplitude_min", "amplitude_min"),
            ("amplitude_max", "amplitude_max"),
            ("grid_size_omega", "grid_size_omega"),
            ("grid_size_amplitude", "grid_size_amplitude"),
            ("df_residual_tol", "df_residual_tol"),
            ("root_refinement", "root_refinement"),
            ("describing_function_mode", "describing_function_mode"),
            ("seed_sign_convention", "seed_sign_convention"),
            ("seed_theta", "seed_theta"),
            # Classical route feature flags
            ("machado_enabled", "machado_enabled"),
            ("biased_enabled", "biased_enabled"),
        ]:
            if k in ss:
                flat[dk] = ss[k]
        # seed_filter is a nested dict — pass through directly
        if "seed_filter" in ss:
            flat["seed_filter"] = ss["seed_filter"]

    # simulation section
    sim = raw.get("simulation", {})
    if sim:
        flat.setdefault("final_simulation", {})
        flat["final_simulation"]["t_final"] = sim.get("t_final", _DEFAULTS["final_simulation"]["t_final"])
        flat["final_simulation"]["t_burn"] = sim.get("t_burn", _DEFAULTS["final_simulation"]["t_burn"])
        flat["final_simulation"]["initial_condition"] = sim.get("initial_condition", None)
        flat["final_simulation"]["divergence_norm"] = sim.get("divergence_norm",
                                                               _DEFAULTS["final_simulation"]["divergence_norm"])

    # plots section
    plots = raw.get("plots", {})
    if plots:
        flat["plot_enabled"] = plots.get("enabled", _DEFAULTS["plot_enabled"])
        flat["save_figures"] = plots.get("save_figures", _DEFAULTS["save_figures"])
        flat["plot_attractors"] = plots.get("attractor", _DEFAULTS["plot_attractors"])
        flat["plot_timeseries"] = plots.get("timeseries", _DEFAULTS["plot_timeseries"])
        flat["plot_transfer"] = plots.get("transfer", _DEFAULTS["plot_transfer"])
        flat["plot_describing_function"] = plots.get("describing_function", _DEFAULTS["plot_describing_function"])
        flat["plot_residual_map"] = plots.get("residual_map", _DEFAULTS["plot_residual_map"])
        flat["plot_continuation"] = plots.get("continuation", _DEFAULTS["plot_continuation"])
        flat["plot_sphere_tests"] = plots.get("sphere_tests", _DEFAULTS["plot_sphere_tests"])
        flat["plot_basin"] = plots.get("basin", _DEFAULTS["plot_basin"])
        flat["plot_bifurcation"] = plots.get("bifurcation", _DEFAULTS["plot_bifurcation"])
        flat["plot_matignon"] = plots.get("matignon", _DEFAULTS["plot_matignon"])
        flat["max_seed_candidates_to_plot"] = plots.get("max_seed_candidates_to_plot",
                                                          _DEFAULTS["max_seed_candidates_to_plot"])

    # nested sections passed through directly
    for section in ("continuation", "sphere_tests", "basin", "bifurcation", "early_stop", "attractor_plots", "robustness"):
        if section in raw:
            flat[section] = raw[section]

    # If continuation is present in raw and has lambda_values, inject into flat continuation
    cont_raw = raw.get("continuation", {})
    if isinstance(cont_raw, dict) and "lambda_values" in cont_raw:
        flat.setdefault("continuation", {})
        flat["continuation"]["lambda_values"] = cont_raw["lambda_values"]

    return flat


def _is_hierarchical(raw: Dict[str, Any]) -> bool:
    """Return True if the YAML looks like the new hierarchical schema."""
    hierarchical_keys = {"experiment", "system", "modes", "stages", "simulation", "plots"}
    return bool(hierarchical_keys.intersection(raw.keys()))


def _detect_and_warn_legacy(raw: Dict[str, Any]) -> None:
    """Emit deprecation warnings for flat/legacy YAML keys."""
    legacy_top_level = {
        "system_id", "integrator", "q", "h", "t_final", "t_burn",
        "workflow_mode", "dynamics_order",
    }
    found = legacy_top_level.intersection(raw.keys())
    if found:
        warnings.warn(
            f"Detected legacy flat YAML keys: {sorted(found)}. "
            "Please migrate to the hierarchical schema. "
            "See version_2/configs/examples/ for reference YAMLs. "
            "Legacy support will be removed in a future version.",
            DeprecationWarning,
            stacklevel=4,
        )


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into a copy of *base*."""
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _apply_defaults(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing keys from _DEFAULTS, merging nested dicts."""
    result = deepcopy(_DEFAULTS)
    for k, v in flat.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Normalization / casting
# ---------------------------------------------------------------------------

def _normalize(cfg: Dict[str, Any]) -> Dict[str, Any]:  # noqa: C901
    """Apply normalization: aliases, type casting, derived keys."""

    # Integrator aliases
    if cfg.get("integrator") == "efork":
        cfg["integrator"] = "efork3"

    # memory_policy <-> memory_mode
    mp = cfg.get("memory_policy")
    mm = cfg.get("memory_mode")
    if mp is not None and mm is not None:
        if mp == "full_caputo" and mm != "full":
            raise ValueError(f"Incompatible settings: memory_policy='{mp}' and memory_mode='{mm}'")
        if mp == "finite_window" and mm != "window":
            raise ValueError(f"Incompatible settings: memory_policy='{mp}' and memory_mode='{mm}'")
        if mp == "none" and mm != "none":
            raise ValueError(f"Incompatible settings: memory_policy='{mp}' and memory_mode='{mm}'")
            
    if mp == "full_caputo":
        cfg["memory_mode"] = "full"
    elif mp == "finite_window":
        cfg["memory_mode"] = "window"
    elif mp == "none":
        cfg["memory_mode"] = "none"
        
    if mm == "full":
        cfg["memory_policy"] = "full_caputo"
    elif mm == "window":
        cfg["memory_policy"] = "finite_window"
    elif mm == "none":
        cfg["memory_policy"] = "none"

    # memory_window_steps / memory_window_length / memory_window_time
    import numpy as np
    if cfg.get("memory_window_time") is not None and cfg.get("h") is not None:
        steps = int(round(float(cfg["memory_window_time"]) / float(cfg["h"])))
        cfg["memory_window_steps"] = steps
        cfg["memory_window_length"] = steps
    elif cfg.get("memory_window_steps") is not None:
        cfg["memory_window_length"] = int(cfg["memory_window_steps"])
    elif cfg.get("memory_window_length") is not None:
        cfg["memory_window_steps"] = int(cfg["memory_window_length"])

    # Legacy t_final / t_burn at top level → final_simulation section
    if "t_final" in cfg and not isinstance(cfg.get("final_simulation"), dict):
        cfg.setdefault("final_simulation", {})
    if "t_final" in cfg:
        cfg["final_simulation"]["t_final"] = float(cfg.pop("t_final"))
    if "t_burn" in cfg:
        cfg["final_simulation"]["t_burn"] = float(cfg.pop("t_burn"))

    # Legacy workflow_mode → stages
    wm = cfg.pop("workflow_mode", None)
    if wm == "simulate_attractor_only":
        cfg["run_attractor_only"] = True
        cfg["run_seed_search"] = False
        cfg["run_continuation"] = False
        cfg["run_final_simulation"] = False

    # Cast numeric scalars
    for key in ("q", "h", "divergence_norm", "equilibrium_tol", "omega_min",
                "omega_max", "amplitude_min", "amplitude_max", "df_residual_tol",
                "target_match_tol", "seed_theta"):
        if key in cfg and cfg[key] is not None:
            cfg[key] = float(cfg[key])

    for key in ("grid_size_omega", "grid_size_amplitude", "workers", "branch_index",
                "memory_window_length", "memory_window_steps", "max_seed_candidates_to_plot",
                "random_seed"):
        if key in cfg and cfg[key] is not None:
            cfg[key] = int(cfg[key])

    # Nested section casts
    _cast_nested_floats(cfg.get("early_stop", {}),
                        ["divergence_norm", "divergence_growth_factor", "equilibrium_tol",
                         "equilibrium_derivative_tol", "equilibrium_min_time"])
    _cast_nested_ints(cfg.get("early_stop", {}),
                      ["divergence_consecutive_steps", "equilibrium_consecutive_steps"])

    fs = cfg.get("final_simulation", {})
    _cast_nested_floats(fs, ["t_final", "t_burn", "divergence_norm"])

    st = cfg.get("sphere_tests", {})
    _cast_nested_floats(st, ["t_final", "t_burn", "h", "samples_growth_factor"])
    _cast_nested_ints(st, ["samples_initial", "random_seed"])
    if "radii" in st and st["radii"] is not None:
        st["radii"] = [float(r) for r in st["radii"]]

    cont = cfg.get("continuation", {})
    _cast_nested_floats(cont, ["eta_min", "eta_max"])
    _cast_nested_ints(cont, ["n_eta", "periods_transient", "periods_keep", "harmonic_history_periods"])

    basin = cfg.get("basin", {})
    _cast_nested_floats(basin, ["fixed_x", "fixed_y", "fixed_z", "local_radius", "t_final", "t_burn", "h"])
    _cast_nested_ints(basin, ["grid_n"])
    for lk in ("x_interval", "y_interval", "z_interval"):
        if lk in basin and basin[lk] is not None:
            basin[lk] = [float(v) for v in basin[lk]]

    bif = cfg.get("bifurcation", {})
    _cast_nested_floats(bif, ["discard_time", "sample_time", "h"])
    if "values" in bif and isinstance(bif["values"], dict):
        _cast_nested_floats(bif["values"], ["min", "max"])
        _cast_nested_ints(bif["values"], ["n"])

    return cfg


def _cast_nested_floats(d: Dict, keys: list) -> None:
    for k in keys:
        if k in d and d[k] is not None:
            d[k] = float(d[k])


def _cast_nested_ints(d: Dict, keys: list) -> None:
    for k in keys:
        if k in d and d[k] is not None:
            d[k] = int(d[k])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(cfg: Dict[str, Any]) -> None:
    """Raise ValueError / UserWarning for invalid combinations."""
    from hidden_attractors.integrations.selector import validate_integrator_compatibility

    integrator = cfg.get("integrator", "efork3")
    q = cfg.get("q")

    # Validate integrator × q compatibility (only if q is specified)
    if q is not None:
        validate_integrator_compatibility(integrator, float(q))

    # Transfer / continuation / dynamics modes
    valid = {"integer", "fractional"}
    for key in ("transfer_mode", "seed_mode", "continuation_mode"):
        val = cfg.get(key)
        if val is not None and val not in valid:
            raise ValueError(f"Invalid {key}: '{val}'. Must be one of {valid}.")

    dm = cfg.get("dynamics_mode")
    if dm is not None and dm not in {"integer", "fractional", "system"}:
        raise ValueError(f"Invalid dynamics_mode: '{dm}'.")

    # Route-separation validation
    if cfg.get("machado_enabled") and cfg.get("transfer_mode") != "fractional":
        raise ValueError("Generalised Machado Describing Function is only supported when transfer_mode is 'fractional'.")

    if cfg.get("transfer_mode") == "integer":
        if cfg.get("seed_mode") == "fractional" or cfg.get("continuation_mode") == "fractional":
            raise ValueError("Invalid mode mixture: transfer_mode is 'integer' but seed_mode or continuation_mode is 'fractional'. For published integer reproduction, all must be 'integer'.")
        # Warning when trying to claim fractional outputs with transfer_mode: integer
        if q is not None and q < 1.0:
            import warnings
            warnings.warn(
                "You are using transfer_mode='integer' on a fractional system (q < 1). "
                "Ensure that this is only used for published integer reproduction, "
                "and NEVER mix these results with fractional claims.",
                UserWarning,
                stacklevel=2
            )

    mm = cfg.get("memory_mode")
    if mm is not None and mm not in {"full", "window", "none"}:
        raise ValueError(f"Invalid memory_mode: '{mm}'.")

    if mm == "window":
        wl = cfg.get("memory_window_length") or cfg.get("memory_window_steps")
        if not wl or int(wl) <= 0:
            raise ValueError(
                "memory_window_length must be a positive integer when memory_mode='window'."
            )

    seed_strat = cfg.get("seed_strategy")
    if seed_strat is not None and seed_strat not in {"k_phi", "imw_gain", "nyquist_df"}:
        raise ValueError(f"Invalid seed_strategy: '{seed_strat}'.")

    # System parameters validation
    system_id = cfg.get("system_id", "chua_fractional_saturation")
    is_arctan = "arctan" in system_id or "wu2023" in system_id
    
    invalid_for_nonsmooth = {"m", "n", "a1", "a2", "rho"}
    invalid_for_arctan = {"m", "n", "m0", "m1"}
    
    if "m" in cfg or "n" in cfg:
        raise ValueError("Legacy parameter keys 'm' and 'n' are no longer supported. Please use 'a1', 'a2', 'rho' for arctan model, or 'm0', 'm1' for nonsmooth model.")
        
    if is_arctan:
        for k in invalid_for_arctan:
            if k in cfg:
                raise ValueError(f"Parameter '{k}' is invalid for arctan system '{system_id}'. Allowed parameters: alpha, beta, gamma, a1, a2, rho.")
    else:
        for k in invalid_for_nonsmooth:
            if k in cfg:
                raise ValueError(f"Parameter '{k}' is invalid for nonsmooth system '{system_id}'. Allowed parameters: alpha, beta, gamma, m0, m1.")


# ---------------------------------------------------------------------------
# Output directory resolution
# ---------------------------------------------------------------------------

def _resolve_output_dir(cfg: Dict[str, Any]) -> str:
    """Return (and set in cfg) the resolved output directory."""
    od = cfg.get("output_dir")
    if od:
        return str(od)
    run_id = cfg.get("run_id", "auto")
    if run_id == "auto":
        run_id = time.strftime("%Y%m%d_%H%M%S")
    system_id = cfg.get("system_id", "experiment")
    resolved = os.path.join("outputs", system_id, run_id)
    cfg["output_dir"] = resolved
    return resolved


def _normalize_memory_config(flat: Dict[str, Any]) -> None:
    """Normalize and infer memory_mode / memory_policy before defaults are applied."""
    mp = flat.get("memory_policy")
    mm = flat.get("memory_mode")
    
    # Rules:
    # - Si el usuario define solo memory_policy: full_caputo, inferir memory_mode = full.
    # - Si el usuario define solo memory_policy: finite_window, inferir memory_mode = window.
    # - Si el usuario define solo memory_mode: full, inferir memory_policy = full_caputo.
    # - Si el usuario define solo memory_mode: window, inferir memory_policy = finite_window.
    if mp is not None and mm is None:
        if mp == "full_caputo":
            flat["memory_mode"] = "full"
        elif mp == "finite_window":
            flat["memory_mode"] = "window"
        elif mp == "none":
            flat["memory_mode"] = "none"
        else:
            raise ValueError(f"Unknown memory_policy: '{mp}'")
    elif mm is not None and mp is None:
        if mm == "full":
            flat["memory_policy"] = "full_caputo"
        elif mm == "window":
            flat["memory_policy"] = "finite_window"
        elif mm == "none":
            flat["memory_policy"] = "none"
        else:
            raise ValueError(f"Unknown memory_mode: '{mm}'")
    elif mm is not None and mp is not None:
        compat = {
            ("full", "full_caputo"),
            ("window", "finite_window"),
            ("none", "none")
        }
        if (mm, mp) not in compat:
            raise ValueError(f"Incompatible memory settings: memory_mode='{mm}' and memory_policy='{mp}'")
            
    # Check if window parameters are required and handle time/steps logic
    resolved_mm = flat.get("memory_mode")
    if resolved_mm == "window":
        mw_steps = flat.get("memory_window_steps")
        mw_len = flat.get("memory_window_length")
        mw_time = flat.get("memory_window_time")
        if mw_steps is None and mw_len is None and mw_time is None:
            raise ValueError("memory_window_length, memory_window_steps or memory_window_time must be specified when memory_mode='window'.")
            
        h = flat.get("h") or _DEFAULTS["h"]
        if mw_time is not None:
            steps = int(round(float(mw_time) / float(h)))
            flat["memory_window_steps"] = steps
            flat["memory_window_length"] = steps
        elif mw_steps is not None:
            flat["memory_window_length"] = int(mw_steps)
        elif mw_len is not None:
            flat["memory_window_steps"] = int(mw_len)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str | Path, allow_legacy: bool = True) -> Dict[str, Any]:
    """Load, normalize and validate a YAML config file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.
    allow_legacy : bool
        If False, raise an error for legacy flat-schema YAMLs.

    Returns
    -------
    dict
        Fully normalized, type-cast configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If required keys are missing or values are invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if _is_hierarchical(raw):
        flat = _flatten_hierarchical(raw)
        # Pass through any top-level keys not covered by hierarchical mapping
        for k, v in raw.items():
            if k not in ("experiment", "system", "modes", "integrator", "stages",
                         "seed_search", "continuation", "simulation", "plots",
                         "basin", "bifurcation", "sphere_tests", "early_stop",
                         "attractor_plots"):
                flat.setdefault(k, v)
    else:
        _detect_and_warn_legacy(raw)
        if not allow_legacy:
            raise ValueError(
                f"Config file '{path}' uses the legacy flat schema. "
                "Migrate to the hierarchical schema or pass allow_legacy=True."
            )
        flat = dict(raw)

    _normalize_memory_config(flat)
    cfg = _apply_defaults(flat)
    cfg = _normalize(cfg)
    _validate(cfg)
    _resolve_output_dir(cfg)

    return cfg


def save_effective_config(cfg: Dict[str, Any], output_dir: Optional[str] = None) -> Path:
    """Serialize the effective config to ``effective_config.yaml`` in *output_dir*.

    Parameters
    ----------
    cfg : dict
        Normalized config as returned by ``load_config``.
    output_dir : str, optional
        Override directory.  Defaults to ``cfg['output_dir']``.

    Returns
    -------
    Path
        Path to the written file.
    """
    directory = Path(output_dir or cfg.get("output_dir", "outputs"))
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "effective_config.yaml"

    # Convert non-serializable items (numpy arrays, etc.)
    import json

    def _clean(obj: Any) -> Any:
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        return obj

    with open(out, "w", encoding="utf-8") as fh:
        yaml.dump(_clean(dict(cfg)), fh, default_flow_style=False, allow_unicode=True, sort_keys=True)

    return out


def _set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict from a dotted key string."""
    parts = dotted_key.split(".")
    current = cfg
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


_INTEGRATOR_SUBKEY_MAPPING = {
    "name": "integrator",
    "h": "h",
    "memory_mode": "memory_mode",
    "memory_policy": "memory_policy",
    "memory_window_steps": "memory_window_steps",
    "memory_window_length": "memory_window_steps",
    "memory_window_time": "memory_window_time",
    "use_c_backend": "use_c_backend",
    "allow_python_fallback": "allow_python_fallback",
}


def apply_cli_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply CLI override values to a loaded config, then re-validate.

    Parameters
    ----------
    cfg : dict
        Config as returned by ``load_config``.
    overrides : dict
        Key-value pairs from CLI arguments (None values are ignored).

    Returns
    -------
    dict
        Updated config.
    """
    mapped_overrides = {}
    for k, v in overrides.items():
        if v is None:
            continue
        if k.startswith("integrator."):
            subkey = k.split(".", 1)[1]
            if subkey in _INTEGRATOR_SUBKEY_MAPPING:
                mapped_overrides[_INTEGRATOR_SUBKEY_MAPPING[subkey]] = v
                continue
        elif k.startswith("simulation."):
            subkey = k.split(".", 1)[1]
            mapped_overrides[f"final_simulation.{subkey}"] = v
            continue
        mapped_overrides[k] = v

    # Clear matching memory parameter if overridden to allow proper inference
    if "memory_mode" in mapped_overrides:
        cfg.pop("memory_policy", None)
    if "memory_policy" in mapped_overrides:
        cfg.pop("memory_mode", None)

    for k, v in mapped_overrides.items():
        if "." in k:
            _set_nested(cfg, k, v)
        else:
            cfg[k] = v

    _normalize_memory_config(cfg)
    cfg = _normalize(cfg)
    _validate(cfg)
    return cfg
