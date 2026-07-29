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

    cfg = load_config("my_workflow.yaml")
    # Scientific and numerical values in this file must be explicit.

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
    "system_id": None,
    "q": None,

    # ── Modes ────────────────────────────────────────────────────────────────
    "transfer_mode": None,
    "seed_mode": None,
    "continuation_mode": None,
    "dynamics_mode": None,

    # ── Explicit multi-order sections ────────────────────────────────────────
    "seed": {
        "df_order": None,
        "transfer_mode": None,
        "q_seed": None,
        "family": None,
    },
    "dynamics": {
        "dynamics_order": None,
        "q_dynamics": None,
    },

    # ── Integrator ───────────────────────────────────────────────────────────
    "integrator": None,
    "h": None,
    "memory_mode": None,
    "memory_policy": None,
    "memory_window_steps": None,
    "memory_window_length": None,
    "memory_window_time": None,
    "use_c_backend": False,
    "allow_python_fallback": False,

    # ── Stages ───────────────────────────────────────────────────────────────
    "run_seed_search": False,
    "run_continuation": False,
    "run_final_simulation": False,
    "run_hiddenness_tests": False,
    "run_sphere_tests": False,
    "run_basin_slices": False,
    "run_bifurcation": False,
    "run_attractor_only": False,

    # ── Seed search ──────────────────────────────────────────────────────────
    "seed_strategy": None,
    "seed_sign_convention": None,
    "seed_construction": None,
    "seed_theta": None,
    "describing_function_mode": None,
    "branch_index": None,
    "omega_min": None,
    "omega_max": None,
    "amplitude_min": None,
    "amplitude_max": None,
    "grid_size_omega": None,
    "grid_size_amplitude": None,
    "root_refinement": None,
    "df_residual_tol": None,
    "hiddenness_equilibria_filter": None,
    "transfer_convention": None,
    "harmonic_condition": None,
    "q_seed": None,
    "q_dynamics": None,

    # ── Classical route feature flags ────────────────────────────────────────────
    "machado_enabled": False,
    "biased_enabled": False,
    "seed_filter": {
        "enabled": False,
    },

    # ── Workers / reproducibility ─────────────────────────────────────────────
    "workers": 1,
    "random_seed": None,
    "random_seed_policy": None,

    # ── Divergence ───────────────────────────────────────────────────────────
    "divergence_norm": None,
    "equilibrium_tol": None,
    "target_match_metric": None,
    "target_match_tol": None,

    # ── Output ───────────────────────────────────────────────────────────────
    "output_dir": None,
    "run_id": "auto",

    # ── Plotting ─────────────────────────────────────────────────────────────
    "plot_enabled": False,
    "save_figures": False,
    "plot_attractors": False,
    "plot_timeseries": False,
    "plot_transfer": False,
    "plot_describing_function": False,
    "plot_residual_map": False,
    "plot_continuation": False,
    "plot_sphere_tests": False,
    "plot_matignon": False,
    "plot_basin": False,
    "plot_bifurcation": False,
    "max_seed_candidates_to_plot": None,

    # ── Nested sections (preserved as dicts) ─────────────────────────────────
    "early_stop": {
        "enabled": False,
    },

    "final_simulation": {},

    "continuation": {
        "continuation_order": None,
        "q_continuation": None,
        "lambda_values": None,
        "eta_grid_mode": None,
        "eta_values": None,
        "eta_min": None,
        "eta_max": None,
        "n_eta": None,
        "start_at_zero": None,
        "use_period_based_times": None,
        "periods_transient": None,
        "periods_keep": None,
        "t_transient": None,
        "t_keep": None,
        "build_fractional_harmonic_history": None,
        "harmonic_history_periods": None,
        "early_stop_enabled": False,
        "require_c_backend": False,
        "allow_python_fallback": False,
    },

    "sphere_tests": {
        "enabled": False,
    },

    "basin": {
        "enabled": False,
    },

    "bifurcation": {
        "enabled": False,
    },

    "attractor_plots": {
        "enabled": False,
        "include_equilibria": False,
        "use_tail_after_burn": True,
    },

    # Optional robustness diagnostics
    "robustness": {
        "enabled": False,
    },

    # ── Hiddenness contract parameters ───────────────────────────────────────
    "hiddenness": {},

    # ── Validation / Bibliography ──────────────────────────────────────────
    "validation": {
        "strict_bibliography": False,
        "claims_manifest": None,
        "fail_on_missing_references": False,
        "fail_on_unregistered_references": False,
    },

    # ── Figures (unified plotting) ───────────────────────────────────────────
    "figures": {
        "enabled": False,
        "output_root": "outputs/figures",
        "export_formats": [],
        "export_targets": [],
        "write_manifest": False,
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
        for source_key, target_key in (
            ("output_dir", "output_dir"),
            ("run_id", "run_id"),
            ("random_seed", "random_seed"),
            ("name", "_experiment_name"),
            ("description", "_description"),
        ):
            if source_key in exp:
                flat[target_key] = exp[source_key]

    # system section
    sys_sec = raw.get("system", {})
    if sys_sec:
        if "system_id" in sys_sec:
            flat["system_id"] = sys_sec["system_id"]
        if "q" in sys_sec:
            flat["q"] = sys_sec["q"]
        params = sys_sec.get("parameters", {})
        flat.update(params)  # alpha, beta, gamma, m0/m, m1/n, etc.

    # modes section
    modes = raw.get("modes", {})
    if modes:
        for key in (
            "transfer_mode",
            "seed_mode",
            "continuation_mode",
            "dynamics_mode",
        ):
            if key in modes:
                flat[key] = modes[key]

    # integrator section
    integ = raw.get("integrator", {})
    if isinstance(integ, dict):
        if "name" in integ:
            flat["integrator"] = integ["name"]
        if "h" in integ:
            flat["h"] = integ["h"]
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
        if "use_c_backend" in integ:
            flat["use_c_backend"] = integ["use_c_backend"]
        if "allow_python_fallback" in integ:
            flat["allow_python_fallback"] = integ["allow_python_fallback"]
    elif isinstance(integ, str):
        flat["integrator"] = integ  # flat legacy key inside hierarchical doc

    # stages section
    stages = raw.get("stages", {})
    if stages:
        stage_mapping = {
            "seed_search": "run_seed_search",
            "continuation": "run_continuation",
            "final_simulation": "run_final_simulation",
            "hiddenness_tests": "run_hiddenness_tests",
            "sphere_tests": "run_sphere_tests",
            "basin_slices": "run_basin_slices",
            "bifurcation": "run_bifurcation",
            "attractor_only": "run_attractor_only",
        }
        for source_key, target_key in stage_mapping.items():
            if source_key in stages:
                flat[target_key] = stages[source_key]

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
        for key in ("t_final", "t_burn", "initial_condition", "divergence_norm"):
            if key in sim:
                flat["final_simulation"][key] = sim[key]
        if "divergence_norm" in sim:
            flat["divergence_norm"] = sim["divergence_norm"]

    # plots section
    plots = raw.get("plots", {})
    if plots:
        plot_mapping = {
            "enabled": "plot_enabled",
            "save_figures": "save_figures",
            "attractor": "plot_attractors",
            "timeseries": "plot_timeseries",
            "transfer": "plot_transfer",
            "describing_function": "plot_describing_function",
            "residual_map": "plot_residual_map",
            "continuation": "plot_continuation",
            "sphere_tests": "plot_sphere_tests",
            "basin": "plot_basin",
            "bifurcation": "plot_bifurcation",
            "matignon": "plot_matignon",
            "max_seed_candidates_to_plot": "max_seed_candidates_to_plot",
        }
        for source_key, target_key in plot_mapping.items():
            if source_key in plots:
                flat[target_key] = plots[source_key]

    for section in ("seed", "dynamics", "continuation", "sphere_tests", "basin", "bifurcation", "early_stop", "attractor_plots", "robustness", "hiddenness", "validation", "figures"):
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
            "Use the packaged workflow_contract.yaml as the schema reference. "
            "This compatibility input is deprecated; use the normalized schema.",
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

    # ── Multi-Order Contract Normalization and Legacy Compatibility ──
    # Ensure nested sub-dicts exist
    if "seed" not in cfg or cfg["seed"] is None:
        cfg["seed"] = {}
    elif not isinstance(cfg["seed"], dict):
        cfg["seed"] = {"df_order": cfg["seed"]}

    if "continuation" not in cfg or cfg["continuation"] is None:
        cfg["continuation"] = {}

    if "dynamics" not in cfg or cfg["dynamics"] is None:
        cfg["dynamics"] = {}

    # Extract/resolve system order q
    q = cfg.get("q")

    # Map seed_mode to seed.df_order
    if cfg["seed"].get("df_order") is None:
        sm = cfg.get("seed_mode")
        if sm == "integer":
            cfg["seed"]["df_order"] = "integer"
        elif sm == "fractional":
            cfg["seed"]["df_order"] = "fractional"
        else:
            tm = cfg.get("transfer_mode")
            if tm in ("published_integer_laplace", "integer"):
                cfg["seed"]["df_order"] = "integer"
            elif tm in ("fractional_spectral", "fractional"):
                cfg["seed"]["df_order"] = "fractional"

    # Map transfer_mode
    if cfg["seed"].get("transfer_mode") is None:
        tm = cfg.get("transfer_mode")
        if tm is not None:
            if tm == "integer":
                cfg["seed"]["transfer_mode"] = "published_integer_laplace"
            elif tm == "fractional":
                cfg["seed"]["transfer_mode"] = "fractional_spectral"
            else:
                cfg["seed"]["transfer_mode"] = tm

    # Map q_seed
    if cfg["seed"].get("q_seed") is None:
        qs = cfg.get("q_seed")
        if qs is not None:
            cfg["seed"]["q_seed"] = float(qs)
        elif cfg["seed"].get("df_order") == "integer":
            cfg["seed"]["q_seed"] = 1.0
        elif cfg["seed"].get("df_order") == "fractional":
            cfg["seed"]["q_seed"] = q

    # Map continuation_mode to continuation.continuation_order
    if cfg["continuation"].get("continuation_order") is None:
        cm = cfg.get("continuation_mode")
        if cm == "integer":
            cfg["continuation"]["continuation_order"] = "integer"
        elif cm == "fractional":
            cfg["continuation"]["continuation_order"] = "fractional"

    # Map dynamics_mode to dynamics.dynamics_order
    if cfg["dynamics"].get("dynamics_order") is None:
        dm = cfg.get("dynamics_mode")
        if dm == "integer":
            cfg["dynamics"]["dynamics_order"] = "integer"
        elif dm == "fractional":
            cfg["dynamics"]["dynamics_order"] = "fractional"
        elif dm == "system" and q is not None:
            cfg["dynamics"]["dynamics_order"] = (
                "fractional" if float(q) < 1.0 else "integer"
            )

    # Map q_dynamics
    if cfg["dynamics"].get("q_dynamics") is None:
        qd = cfg.get("q_dynamics")
        if qd is not None:
            cfg["dynamics"]["q_dynamics"] = float(qd)
        elif cfg["dynamics"].get("dynamics_order") == "integer":
            cfg["dynamics"]["q_dynamics"] = 1.0
        elif cfg["dynamics"].get("dynamics_order") == "fractional":
            cfg["dynamics"]["q_dynamics"] = q

    # Map q_continuation
    if cfg["continuation"].get("q_continuation") is None:
        qc = cfg.get("q_continuation")
        if qc is not None:
            cfg["continuation"]["q_continuation"] = float(qc)
        else:
            if cfg["continuation"].get("continuation_order") == "integer":
                cfg["continuation"]["q_continuation"] = 1.0
            elif cfg["continuation"].get("continuation_order") == "fractional":
                qd_val = cfg["dynamics"].get("q_dynamics")
                cfg["continuation"]["q_continuation"] = qd_val if (qd_val is not None and qd_val < 1.0) else q

    # Sync back to top level for legacy support
    cfg["q_seed"] = cfg["seed"].get("q_seed")
    cfg["q_dynamics"] = cfg["dynamics"].get("q_dynamics")
    cfg["q_continuation"] = cfg["continuation"].get("q_continuation")
    if cfg["seed"].get("df_order") is not None:
        cfg["seed_mode"] = cfg["seed"]["df_order"]
    if cfg["continuation"].get("continuation_order") is not None:
        cfg["continuation_mode"] = cfg["continuation"]["continuation_order"]
    if cfg.get("dynamics_mode") is None:
        cfg["dynamics_mode"] = cfg["dynamics"].get("dynamics_order")

    # Apply type casting to nested sections
    if cfg["seed"].get("q_seed") is not None:
        cfg["seed"]["q_seed"] = float(cfg["seed"]["q_seed"])
    if cfg["continuation"].get("q_continuation") is not None:
        cfg["continuation"]["q_continuation"] = float(cfg["continuation"]["q_continuation"])
    if cfg["dynamics"].get("q_dynamics") is not None:
        cfg["dynamics"]["q_dynamics"] = float(cfg["dynamics"]["q_dynamics"])

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

    hid = cfg.get("hiddenness", {})
    _cast_nested_floats(hid, ["target_match_tol", "target_match_nn_percentile"])
    _cast_nested_ints(hid, ["min_ref_tail_points", "min_probe_tail_points"])
    if "required_radii" in hid and hid["required_radii"] is not None:
        hid["required_radii"] = [float(r) for r in hid["required_radii"]]

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

def _get_nested_value(cfg: Dict[str, Any], dotted_path: str) -> Any:
    value: Any = cfg
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _require_config_values(
    cfg: Dict[str, Any],
    paths: tuple[str, ...],
    *,
    context: str,
) -> None:
    missing = []
    for path in paths:
        value = _get_nested_value(cfg, path)
        if value is None or value == "" or (
            isinstance(value, (list, tuple, dict)) and not value
        ):
            missing.append(path)
    if missing:
        raise ValueError(
            f"{context} requires explicit configuration values for: "
            + ", ".join(missing)
        )


def _validate_explicit_stage_contracts(cfg: Dict[str, Any]) -> None:
    """Reject enabled calculations that rely on package-wide scientific values."""
    stage_keys = (
        "run_seed_search",
        "run_continuation",
        "run_final_simulation",
        "run_hiddenness_tests",
        "run_sphere_tests",
        "run_basin_slices",
        "run_bifurcation",
        "run_attractor_only",
    )
    nested_stage_keys = ("sphere_tests", "basin", "bifurcation")
    active = any(bool(cfg.get(key)) for key in stage_keys) or any(
        bool((cfg.get(key) or {}).get("enabled")) for key in nested_stage_keys
    )
    if not active:
        return

    _require_config_values(
        cfg,
        ("system_id", "q"),
        context="An enabled scientific calculation",
    )

    integration_requested = any(
        bool(cfg.get(key))
        for key in (
            "run_continuation",
            "run_final_simulation",
            "run_hiddenness_tests",
            "run_sphere_tests",
            "run_basin_slices",
            "run_bifurcation",
            "run_attractor_only",
        )
    )
    if integration_requested:
        _require_config_values(
            cfg,
            ("integrator", "h", "dynamics_mode"),
            context="An enabled integration stage",
        )
        if float(cfg["q"]) < 1.0:
            _require_config_values(
                cfg,
                ("memory_mode", "memory_policy"),
                context="A fractional integration stage",
            )

    if cfg.get("run_seed_search"):
        _require_config_values(
            cfg,
            (
                "transfer_mode",
                "seed_mode",
                "seed_strategy",
                "seed_construction",
                "describing_function_mode",
                "omega_min",
                "omega_max",
                "amplitude_min",
                "amplitude_max",
                "grid_size_omega",
                "grid_size_amplitude",
                "root_refinement",
                "df_residual_tol",
            ),
            context="Seed search",
        )

    if cfg.get("run_continuation"):
        _require_config_values(
            cfg,
            ("continuation_mode",),
            context="Continuation",
        )
        continuation = cfg.get("continuation") or {}
        explicit_values = continuation.get("lambda_values") or continuation.get(
            "eta_values"
        )
        if not explicit_values:
            _require_config_values(
                cfg,
                (
                    "continuation.eta_grid_mode",
                    "continuation.eta_min",
                    "continuation.eta_max",
                    "continuation.n_eta",
                ),
                context="Continuation grid",
            )
        period_based = continuation.get("use_period_based_times")
        if period_based is True:
            _require_config_values(
                cfg,
                (
                    "continuation.periods_transient",
                    "continuation.periods_keep",
                ),
                context="Period-based continuation",
            )
        elif period_based is False:
            _require_config_values(
                cfg,
                ("continuation.t_transient", "continuation.t_keep"),
                context="Time-based continuation",
            )
        else:
            raise ValueError(
                "Continuation requires explicit "
                "continuation.use_period_based_times."
            )

    if cfg.get("run_final_simulation") or cfg.get("run_attractor_only"):
        _require_config_values(
            cfg,
            ("final_simulation.t_final", "final_simulation.t_burn"),
            context="Final simulation",
        )

    if cfg.get("run_sphere_tests") or cfg.get("run_hiddenness_tests"):
        _require_config_values(
            cfg,
            (
                "sphere_tests.radii",
                "sphere_tests.samples_initial",
                "sphere_tests.t_final",
                "sphere_tests.t_burn",
                "sphere_tests.h",
                "hiddenness.required_radii",
            ),
            context="Neighborhood verification",
        )

    if cfg.get("run_basin_slices"):
        _require_config_values(
            cfg,
            (
                "basin.planes",
                "basin.grid_n",
                "basin.t_final",
                "basin.t_burn",
                "basin.h",
            ),
            context="Basin calculation",
        )

    if cfg.get("run_bifurcation"):
        _require_config_values(
            cfg,
            (
                "bifurcation.parameter",
                "bifurcation.values.min",
                "bifurcation.values.max",
                "bifurcation.values.n",
                "bifurcation.initial_condition",
                "bifurcation.discard_time",
                "bifurcation.sample_time",
                "bifurcation.h",
            ),
            context="Bifurcation calculation",
        )


def _validate(cfg: Dict[str, Any]) -> None:
    """Raise ValueError / UserWarning for invalid combinations."""
    from hidden_attractors.integrations.selector import validate_integrator_compatibility

    integrator = cfg.get("integrator")
    q = cfg.get("q")

    # Validate integrator × q compatibility (only if q is specified)
    if q is not None and integrator is not None:
        validate_integrator_compatibility(integrator, float(q))

    # Transfer / continuation / dynamics modes
    valid_transfer = {"integer", "fractional", "published_integer_laplace", "fractional_spectral"}
    valid_modes = {"integer", "fractional"}
    
    val_tm = cfg.get("transfer_mode")
    if val_tm is not None and val_tm not in valid_transfer:
        raise ValueError(f"Invalid transfer_mode: '{val_tm}'. Must be one of {valid_transfer}.")
        
    for key in ("seed_mode", "continuation_mode"):
        val = cfg.get(key)
        if val is not None and val not in valid_modes:
            raise ValueError(f"Invalid {key}: '{val}'. Must be one of {valid_modes}.")

    dm = cfg.get("dynamics_mode")
    if dm is not None and dm not in {"integer", "fractional", "system"}:
        raise ValueError(f"Invalid dynamics_mode: '{dm}'.")

    # Route-separation validation
    if cfg.get("machado_enabled") and cfg.get("transfer_mode") != "fractional":
        raise ValueError("Generalised Machado Describing Function is only supported when transfer_mode is 'fractional'.")

    if cfg.get("transfer_mode") == "integer":
        if cfg.get("seed_mode") == "fractional" or cfg.get("continuation_mode") == "fractional":
            raise ValueError("Invalid mode mixture: transfer_mode is 'integer' but seed_mode or continuation_mode is 'fractional'. All order modes must be consistent.")
        if q is not None and q < 1.0:
            import warnings
            warnings.warn(
                "You are using transfer_mode='integer' on a fractional system (q < 1). "
                "The transfer calculation and the fractional dynamics therefore "
                "use different order contracts; report them separately.",
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
    system_id = cfg.get("system_id")
    is_arctan = bool(
        system_id and ("arctan" in system_id or "wu2023" in system_id)
    )
    
    invalid_for_nonsmooth = {"m", "n", "a1", "a2", "rho"}
    invalid_for_arctan = {"m", "n", "m0", "m1"}
    
    if "m" in cfg or "n" in cfg:
        raise ValueError("Legacy parameter keys 'm' and 'n' are no longer supported. Please use 'a1', 'a2', 'rho' for arctan model, or 'm0', 'm1' for nonsmooth model.")
        
    if is_arctan:
        for k in invalid_for_arctan:
            if k in cfg:
                raise ValueError(f"Parameter '{k}' is invalid for arctan system '{system_id}'. Allowed parameters: alpha, beta, gamma, a1, a2, rho.")
    elif system_id:
        for k in invalid_for_nonsmooth:
            if k in cfg:
                raise ValueError(f"Parameter '{k}' is invalid for nonsmooth system '{system_id}'. Allowed parameters: alpha, beta, gamma, m0, m1.")

    _validate_explicit_stage_contracts(cfg)


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
    system_id = cfg.get("system_id") or "configuration"
    resolved = os.path.join("outputs", system_id, run_id)
    cfg["output_dir"] = resolved
    return resolved


def _normalize_memory_config(flat: Dict[str, Any]) -> None:
    """Normalize and infer memory_mode / memory_policy before defaults are applied."""
    mp = flat.get("memory_policy")
    if mp == "full_history":
        flat["memory_policy"] = "full_caputo"
        mp = "full_caputo"
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
            
        if mw_time is not None:
            h = flat.get("h")
            if h is None:
                raise ValueError(
                    "memory_window_time requires an explicit integration step h."
                )
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


_CLI_NESTED_MAPPINGS = {
    "df_order": "seed.df_order",
    "transfer_mode": "seed.transfer_mode",
    "q_seed": "seed.q_seed",
    "continuation_order": "continuation.continuation_order",
    "q_continuation": "continuation.q_continuation",
    "dynamics_order": "dynamics.dynamics_order",
    "q_dynamics": "dynamics.q_dynamics",
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
        if k in _CLI_NESTED_MAPPINGS:
            mapped_overrides[_CLI_NESTED_MAPPINGS[k]] = v
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

    # A global q override must not leave the normalized seed, continuation and
    # dynamics contracts pinned to the q value loaded from the YAML file.
    if "q" in mapped_overrides:
        q_value = mapped_overrides["q"]
        if "seed.q_seed" not in mapped_overrides:
            _set_nested(cfg, "seed.q_seed", q_value)
            cfg["q_seed"] = q_value
        if "continuation.q_continuation" not in mapped_overrides:
            _set_nested(cfg, "continuation.q_continuation", q_value)
        if "dynamics.q_dynamics" not in mapped_overrides:
            _set_nested(cfg, "dynamics.q_dynamics", q_value)
            cfg["q_dynamics"] = q_value

    # Keep the legacy summary key synchronized with the explicit seed contract.
    if "seed.transfer_mode" in mapped_overrides:
        cfg["transfer_mode"] = mapped_overrides["seed.transfer_mode"]

    _normalize_memory_config(cfg)
    cfg = _normalize(cfg)
    _validate(cfg)
    return cfg


def resolve_seed_transfer_contract(config: Dict[str, Any], system: Any) -> Dict[str, Any]:
    """Resolve the explicit contract for describing function and transfer function evaluation.

    Parameters
    ----------
    config : dict
        Normalized config dictionary.
    system : object
        System object from the systems registry.

    Returns
    -------
    dict
        SeedTransferContract dictionary.
    """
    # Resolve order and modes
    q = config.get("q")
    if q is None:
        raise ValueError(
            "Seed-transfer evaluation requires an explicit system order q."
        )

    seed_sec = config.get("seed") or {}
    df_order = seed_sec.get("df_order")
    if df_order is None:
        raise ValueError(
            "Seed-transfer evaluation requires explicit seed.df_order "
            "(or seed_mode before normalization)."
        )

    q_seed = seed_sec.get("q_seed")
    if q_seed is None:
        q_seed = 1.0 if df_order == "integer" else q

    transfer_mode = seed_sec.get("transfer_mode")
    if transfer_mode is None:
        raise ValueError(
            "Seed-transfer evaluation requires explicit seed.transfer_mode "
            "(or transfer_mode before normalization)."
        )
    if transfer_mode == "integer":
        transfer_mode = "published_integer_laplace"
    elif transfer_mode == "fractional":
        transfer_mode = "fractional_spectral"

    frequency_rule = "lambda=jomega" if df_order == "integer" else "lambda=(jomega)^q"

    legacy_source_fields = []
    if "seed_mode" in config:
        legacy_source_fields.append("seed_mode")
    if "transfer_mode" in config:
        legacy_source_fields.append("transfer_mode")

    return {
        "df_order": df_order,
        "transfer_mode": transfer_mode,
        "q_seed": float(q_seed),
        "lambda_frequency_rule": frequency_rule,
        "legacy_source_fields": legacy_source_fields
    }
