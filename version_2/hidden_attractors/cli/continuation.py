"""CLI commands for numerical continuation under the unified hidden-attractors CLI.

Stability: internal
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Sequence, Any, Dict, List

import numpy as np

from ..workflows.config_loader import load_config, apply_cli_overrides, resolve_seed_transfer_contract
from ..reproducibility import collect_run_metadata, collect_lure_metadata, collect_seed_metadata
from ..systems import get_system
from ..seed_generation.chua import chua_matrices, psi_sigma
from ..continuation.continuation_fractional import run_fractional_continuation
from ..continuation.continuation_integer import run_integer_continuation
from ..integrations.selector import integrate, validate_integrator_compatibility


def load_seeds_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    seeds = []
    if not csv_path.exists():
        return seeds
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seeds.append({
                "candidate_id": row["candidate_id"],
                "family": row["family"],
                "centered_or_biased": row["centered_or_biased"],
                "A": float(row["A"]),
                "sigma0": float(row["sigma0"]),
                "omega": float(row["omega"]),
                "q": float(row["q"]),
                "harmonic_residual": float(row["harmonic_residual"]),
                "rho_H": float(row["rho_H"]),
                "x0": json.loads(row["x0"]),
                "reconstruction_metadata": json.loads(row["reconstruction_metadata"]),
                "source_config": row["source_config"],
            })
    return seeds


def _resolve_explicit_fractional_q(
    config: dict[str, Any],
    *,
    system: Any,
    seeds: list[dict[str, Any]],
    path_parameters: dict[str, Any] | None = None,
) -> float:
    """Resolve ``q`` only from inputs already supplied by the caller."""

    q_value: Any = config.get("q")
    system_config = config.get("system")
    if q_value is None and isinstance(system_config, dict):
        q_value = system_config.get("q")
    if q_value is None and path_parameters and "q" in path_parameters:
        q_range = path_parameters["q"]
        if isinstance(q_range, dict):
            q_value = q_range.get("start")
    if q_value is None and system is not None and hasattr(system, "parameters"):
        q_value = system.parameters.get("q")
    if q_value is None and seeds:
        seed_orders = {float(seed["q"]) for seed in seeds if seed.get("q") is not None}
        if len(seed_orders) == 1:
            q_value = seed_orders.pop()
    if q_value is None:
        raise ValueError(
            "Fractional continuation requires an explicit q value in "
            "continuation.q_continuation, a q path, the system configuration, "
            "or a consistent seed file."
        )
    return float(q_value)


def _require_value(mapping: dict[str, Any], key: str, label: str) -> Any:
    """Return a caller-supplied value or raise a contract-focused error."""

    value = mapping.get(key)
    if value is None:
        raise ValueError(f"Continuation requires an explicit {label}.")
    return value


def _resolve_continuation_times(
    continuation: dict[str, Any],
) -> tuple[float, float]:
    """Resolve transient/kept durations only from an explicit time contract."""

    use_periods = continuation.get("use_period_based_times")
    has_periods = (
        continuation.get("periods_transient") is not None
        and continuation.get("periods_keep") is not None
    )
    has_times = (
        continuation.get("t_transient") is not None
        and continuation.get("t_keep") is not None
    )

    if use_periods is True or (use_periods is None and has_periods and not has_times):
        if not has_periods:
            raise ValueError(
                "Period-based continuation requires explicit "
                "continuation.periods_transient and continuation.periods_keep."
            )
        transient = float(continuation["periods_transient"]) * 2.0 * np.pi
        kept = float(continuation["periods_keep"]) * 2.0 * np.pi
    elif use_periods is False or (use_periods is None and has_times and not has_periods):
        if not has_times:
            raise ValueError(
                "Time-based continuation requires explicit "
                "continuation.t_transient and continuation.t_keep."
            )
        transient = float(continuation["t_transient"])
        kept = float(continuation["t_keep"])
    else:
        raise ValueError(
            "Continuation requires one unambiguous explicit duration contract: "
            "periods_transient/periods_keep or t_transient/t_keep."
        )

    if not np.isfinite(transient) or not np.isfinite(kept):
        raise ValueError("Continuation durations must be finite.")
    if transient <= 0.0 or kept <= 0.0:
        raise ValueError("Continuation durations must be strictly positive.")
    return transient, kept


def _require_seed_gain(seed: dict[str, Any]) -> float:
    metadata = seed.get("reconstruction_metadata")
    if not isinstance(metadata, dict) or metadata.get("gain") is None:
        raise ValueError(
            "Each continuation seed requires an explicit "
            "reconstruction_metadata.gain value."
        )
    return float(metadata["gain"])


def run_scalar_continuation(
    argv: Sequence[str] | None = None
) -> None:
    parser = argparse.ArgumentParser(prog="hidden-attractors continuation run")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to an explicit YAML continuation contract",
    )
    parser.add_argument("-s", "--seed-file", type=str, help="Path to CSV seeds file")
    parser.add_argument("-o", "--output-dir", type=str, help="Directory to save output files")
    parser.add_argument("--lambda-values", type=str, help="Comma-separated list of lambda/eta values")
    
    # Explicit CLI options for continuation
    parser.add_argument("--continuation-order", type=str, choices=["integer", "fractional"], help="continuation order type")
    parser.add_argument("--q-continuation", type=float, help="continuation order")
    parser.add_argument("--integrator", type=str, help="integrator name")
    parser.add_argument("--h", type=float, help="step size")
    parser.add_argument("--memory-policy", type=str, choices=["full_history", "full_caputo", "finite_window", "none"], help="Caputo memory policy")
    parser.add_argument("--memory-mode", type=str, choices=["full", "window", "none"], help="Caputo memory mode")
    parser.add_argument("--memory-window-time", type=float, help="window size in seconds")
    parser.add_argument("--memory-window-steps", type=int, help="window size in steps")
    parser.add_argument("--use-c-backend", action="store_true", default=None, help="use compiled C/Numba backend")
    parser.add_argument("--no-c-backend", action="store_false", dest="use_c_backend", help="do not use compiled C/Numba backend")
    parser.add_argument("--allow-python-fallback", action="store_true", default=None, help="allow fallback to Python")
    parser.add_argument("--no-python-fallback", action="store_false", dest="allow_python_fallback", help="do not allow fallback to Python")

    args, extra_args = parser.parse_known_args(argv)
    
    if args.config:
        config_path = Path(args.config)
        config = load_config(config_path)
    else:
        config_path = None
        config = {}

    # Build overrides dictionary
    overrides = {}
    for key in ("continuation_order", "q_continuation", "integrator", "h",
                 "memory_policy", "memory_mode", "memory_window_time",
                 "memory_window_steps", "use_c_backend", "allow_python_fallback"):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val

    config = apply_cli_overrides(config, overrides)
        
    output_dir = args.output_dir or config.get("output_dir") or "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Resolve seeds
    seeds = []
    if args.seed_file:
        seeds = load_seeds_from_csv(Path(args.seed_file))
    elif config_path:
        # Check if seeds.csv exists in the config's output directory
        possible_seeds = Path(output_dir) / "seeds.csv"
        if possible_seeds.exists():
            seeds = load_seeds_from_csv(possible_seeds)
            
    if not seeds:
        print("Error: No seeds found. Please provide --seed-file or run seed generation first.")
        sys.exit(1)
        
    system_id = str(_require_value(config, "system_id", "system_id"))
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
    
    # Resolve continuation properties
    continuation_cfg = config.get("continuation") or {}
    continuation_order = str(
        _require_value(
            continuation_cfg,
            "continuation_order",
            "continuation.continuation_order",
        )
    )
    q_continuation = continuation_cfg.get("q_continuation")
    
    # Ensure q_continuation is 1.0 for integer continuation, or less than 1.0 for fractional continuation
    if continuation_order == "integer":
        q_continuation = 1.0
        config["continuation"]["q_continuation"] = 1.0
    elif continuation_order == "fractional":
        if q_continuation is None:
            q_continuation = _resolve_explicit_fractional_q(
                config,
                system=system,
                seeds=seeds,
            )
            config["continuation"]["q_continuation"] = q_continuation
            
        if q_continuation >= 1.0:
            raise ValueError(f"For fractional continuation, q_continuation must be strictly less than 1.0. Got {q_continuation}.")
            
    integrator = str(_require_value(config, "integrator", "integrator"))
    h = float(_require_value(config, "h", "integration step h"))
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("Continuation integration step h must be finite and positive.")
    
    # Validate compatibility between integrator and order
    if continuation_order == "fractional":
        if integrator in ("rk4", "heun", "efork_q1"):
            raise ValueError(f"{integrator} only supports integer-order dynamics. Use abm or efork3 for fractional Caputo continuation.")
        validate_integrator_compatibility(integrator, q_continuation)
    else:
        # integer order
        if integrator in ("abm", "adm_wu2023"):
            raise ValueError(f"{integrator} requires 0<q<1. Use rk4, heun or efork_q1 for integer-order continuation.")
        validate_integrator_compatibility(integrator, 1.0)
        if integrator == "efork3":
            import warnings
            warnings.warn(
                "Integrator 'efork3' at q=1.0 redirects to the integer-order 'efork_q1' limit. "
                "For pure integer-order work, prefer 'rk4' or 'heun' which are simpler and faster.",
                UserWarning,
                stacklevel=2
            )

    # Check memory policy warnings for Caputo continuation
    memory_policy = str(
        _require_value(config, "memory_policy", "memory_policy")
    )
    memory_mode = str(_require_value(config, "memory_mode", "memory_mode"))
    history_carried = memory_policy != "none"
    if memory_policy == "none":
        history_carried = False
        print("Warning: last-state continuation in a Caputo system is a warm-start, not strict history-preserving continuation.")
        
    # Grid values
    if args.lambda_values:
        lambda_values = [float(v) for v in args.lambda_values.split(",") if v.strip()]
        config.setdefault("continuation", {})["lambda_values"] = lambda_values
    elif continuation_cfg.get("lambda_values"):
        lambda_values = [float(v) for v in continuation_cfg["lambda_values"]]
    else:
        raise ValueError(
            "Continuation requires explicit --lambda-values or "
            "continuation.lambda_values."
        )
    if not lambda_values or not np.all(np.isfinite(lambda_values)):
        raise ValueError("Continuation lambda values must be a non-empty finite list.")
    if np.any(np.diff(np.asarray(lambda_values, dtype=float)) <= 0.0):
        raise ValueError("Continuation lambda values must be strictly increasing.")
        
    continuation_mode = continuation_order
    
    trace_rows = []
    final_candidates = []
    summaries = []
    
    t_transient, t_keep = _resolve_continuation_times(continuation_cfg)
    div_threshold = float(
        _require_value(config, "divergence_norm", "divergence_norm")
    )
    if not np.isfinite(div_threshold) or div_threshold <= 0.0:
        raise ValueError("Continuation divergence_norm must be finite and positive.")
    
    for s_idx, s in enumerate(seeds):
        candidate_id = s["candidate_id"]
        x0 = np.array(s["x0"], dtype=float)
        k_gain = _require_seed_gain(s)
        
        print(f"Running scalar continuation for candidate {candidate_id}...")
        
        if continuation_mode == "integer":
            steps = run_integer_continuation(
                system=system,
                seed_x0=x0,
                k_gain=k_gain,
                lambda_values=lambda_values,
                h=h,
                t_transient=t_transient,
                t_keep=t_keep,
                div_threshold=div_threshold,
                integrator=integrator,
                early_stop_config=config.get("early_stop"),
                equilibria=[],
            )
        else:
            steps = run_fractional_continuation(
                system=system,
                seed_x0=x0,
                k_gain=k_gain,
                lambda_values=lambda_values,
                h=h,
                memory_mode=memory_mode,
                memory_window_length=config.get("memory_window_length"),
                div_threshold=div_threshold,
                integrator=integrator,
                use_c_backend=bool(config.get("use_c_backend", False)),
                t_transient=t_transient,
                t_keep=t_keep,
                early_stop_config=config.get("early_stop"),
                equilibria=[],
                require_c_backend=bool(
                    continuation_cfg.get("require_c_backend", False)
                ),
                allow_python_fallback=bool(
                    config.get("allow_python_fallback", False)
                ),
                q=q_continuation,
            )
            
        # Record trace
        for idx, step in enumerate(steps):
            x_out = step["x_out"]
            x_val = float(x_out[0]) if len(x_out) > 0 else 0.0
            y_val = float(x_out[1]) if len(x_out) > 1 else 0.0
            z_val = float(x_out[2]) if len(x_out) > 2 else 0.0
            
            trace_rows.append({
                "candidate_id": candidate_id,
                "step_index": idx,
                "eta": step["lambda_value"],
                "q": q_continuation,
                "h": h,
                "memory_policy": memory_policy,
                "continuation_mode": continuation_mode,
                "x": x_val,
                "y": y_val,
                "z": z_val,
                "status": step["status"],
                "residual_or_distance": float(step.get("max_norm", 0.0)),
                "failure_reason": step.get("early_stop_reason", ""),
            })
            
        last_step = steps[-1] if steps else None
        if last_step and last_step["status"] == "ok":
            final_candidates.append({
                "candidate_id": candidate_id,
                "family": s["family"],
                "centered_or_biased": s["centered_or_biased"],
                "A": s["A"],
                "sigma0": s["sigma0"],
                "omega": s["omega"],
                "q": q_continuation,
                "eta": last_step["lambda_value"],
                "x": float(last_step["x_out"][0]),
                "y": float(last_step["x_out"][1]),
                "z": float(last_step["x_out"][2]),
                "status": last_step["status"],
            })
            
        summaries.append({
            "candidate_id": candidate_id,
            "success": bool(last_step and last_step["status"] == "ok"),
            "final_status": last_step["status"] if last_step else "failed",
            "steps_completed": len(steps),
        })
        
    # Write output files
    # 1. continuation_trace.csv
    trace_path = Path(output_dir) / "continuation_trace.csv"
    with open(trace_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "step_index", "eta", "q", "h", "memory_policy", "continuation_mode", "x", "y", "z", "status", "residual_or_distance", "failure_reason"])
        for r in trace_rows:
            w.writerow([
                r["candidate_id"], r["step_index"], r["eta"], r["q"], r["h"],
                r["memory_policy"], r["continuation_mode"],
                r["x"], r["y"], r["z"],
                r["status"], r["residual_or_distance"], r["failure_reason"]
            ])
            
    # 2. continuation_summary.json
    summary_path = Path(output_dir) / "continuation_summary.json"
    summary_data = {
        "system_id": system_id,
        "q": q_continuation,
        "memory_policy": memory_policy,
        "history_carried": history_carried,
        "memory_window_time": config.get("memory_window_time"),
        "memory_window_steps": config.get("memory_window_steps"),
        "candidates": summaries,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    # 3. final_candidates.csv
    final_path = Path(output_dir) / "final_candidates.csv"
    with open(final_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "family", "centered_or_biased", "A", "sigma0", "omega", "q", "eta", "x", "y", "z", "status"])
        for fc in final_candidates:
            w.writerow([
                fc["candidate_id"], fc["family"], fc["centered_or_biased"],
                fc["A"], fc["sigma0"], fc["omega"], fc["q"], fc["eta"],
                fc["x"], fc["y"], fc["z"], fc["status"]
            ])
            
    # 4. run_metadata.json, effective_config.json, effective_config.yaml
    first_seed = seeds[0] if seeds else None
    run_meta = collect_run_metadata(
        run_id=config.get("run_id", "auto_continuation"),
        workflow="continuation_run",
        system=system_id,
        q=q_continuation,
        h=h,
        t_final=t_transient + t_keep,
        t_burn=t_transient,
        memory_mode=memory_mode,
        integrator_name=integrator,
        integrator_backend="native" if config.get("use_c_backend", False) else "python",
        caputo=continuation_order == "fractional",
        parameters=system.parameters,
        lure=collect_lure_metadata(
            system.lure,
            transfer_convention=str(
                config.get("transfer_convention") or "not_specified"
            ),
            harmonic_condition=str(
                config.get("harmonic_condition") or "not_specified"
            ),
        ),
        seed=collect_seed_metadata(first_seed, source="continuation_run") if first_seed else None,
        random_seed=config.get("random_seed"),
        random_seed_policy=config.get("random_seed_policy") or "not_specified",
    )
    
    from ..reproducibility import metadata_to_jsonable
    meta_jsonable = metadata_to_jsonable(run_meta)

    # Resolve contracts
    seed_contract = resolve_seed_transfer_contract(config, system)
    seed_transfer_contract = {
        "df_order": seed_contract["df_order"],
        "transfer_mode": seed_contract["transfer_mode"],
        "q_seed": float(seed_contract["q_seed"]),
        "frequency_rule": seed_contract["lambda_frequency_rule"]
    }

    continuation_contract = {
        "continuation_order": continuation_order,
        "q_continuation": float(q_continuation),
        "integrator": integrator,
        "memory_policy": config.get("memory_policy"),
        "history_carried": history_carried
    }

    dynamics_contract = {
        "dynamics_order": config["dynamics"].get("dynamics_order"),
        "q_dynamics": float(config["dynamics"].get("q_dynamics")) if config["dynamics"].get("q_dynamics") is not None else None,
        "integrator": integrator
    }

    meta_jsonable["seed_transfer_contract"] = seed_transfer_contract
    meta_jsonable["continuation_contract"] = continuation_contract
    meta_jsonable["dynamics_contract"] = dynamics_contract

    meta_path = Path(output_dir) / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_jsonable, f, indent=2)

    config["seed_transfer_contract"] = seed_transfer_contract
    config["continuation_contract"] = continuation_contract
    config["dynamics_contract"] = dynamics_contract

    from ..workflows.config_loader import save_effective_config
    save_effective_config(config, output_dir)

    eff_json_path = Path(output_dir) / "effective_config.json"
    with open(eff_json_path, "w", encoding="utf-8") as f:
        def _clean(obj: Any) -> Any:
            if hasattr(obj, "tolist"):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(v) for v in obj]
            return obj
        json.dump(_clean(dict(config)), f, indent=2)
        
    print(f"Continuation completed. Summary written to {summary_path}")

def run_multiparameter_continuation(
    argv: Sequence[str] | None = None
) -> None:
    parser = argparse.ArgumentParser(prog="hidden-attractors continuation multiparameter")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to an explicit YAML continuation contract",
    )
    parser.add_argument("-p", "--path", type=str, help="Path to JSON path definition file")
    parser.add_argument("-s", "--seed-file", type=str, help="Path to CSV seeds file")
    parser.add_argument("-o", "--output-dir", type=str, help="Directory to save output files")
    parser.add_argument("--lambda-values", type=str, help="Comma-separated list of lambda/eta values")
    
    # Explicit options
    parser.add_argument("--continuation-order", type=str, choices=["integer", "fractional"], help="continuation order type")
    parser.add_argument("--q-continuation", type=float, help="continuation order")
    parser.add_argument("--integrator", type=str, help="integrator name")
    parser.add_argument("--h", type=float, help="step size")
    parser.add_argument("--memory-policy", type=str, choices=["full_history", "full_caputo", "finite_window", "none", "carry_window"], help="Caputo memory policy")
    parser.add_argument("--memory-mode", type=str, choices=["full", "window", "none"], help="Caputo memory mode")
    parser.add_argument("--memory-window-time", type=float, help="window size in seconds")
    parser.add_argument("--memory-window-steps", type=int, help="window size in steps")
    parser.add_argument("--use-c-backend", action="store_true", default=None, help="use compiled C/Numba backend")
    parser.add_argument("--no-c-backend", action="store_false", dest="use_c_backend", help="do not use compiled C/Numba backend")
    parser.add_argument("--allow-python-fallback", action="store_true", default=None, help="allow fallback to Python")
    parser.add_argument("--no-python-fallback", action="store_false", dest="allow_python_fallback", help="do not allow fallback to Python")

    args, extra_args = parser.parse_known_args(argv)
    
    if args.config:
        config_path = Path(args.config)
        config = load_config(config_path)
    else:
        config_path = None
        config = {}

    # Build overrides dictionary
    overrides = {}
    for key in ("continuation_order", "q_continuation", "integrator", "h",
                 "memory_policy", "memory_mode", "memory_window_time",
                 "memory_window_steps", "use_c_backend", "allow_python_fallback"):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val

    config = apply_cli_overrides(config, overrides)
    
    if args.lambda_values:
        parsed_lambda_values = [float(v) for v in args.lambda_values.split(",") if v.strip()]
        config.setdefault("continuation", {})["lambda_values"] = parsed_lambda_values
        
    output_dir = args.output_dir or config.get("output_dir") or "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse path parameters from config or JSON file
    path_config = {}
    if args.path:
        with open(args.path, "r", encoding="utf-8") as f:
            path_config = json.load(f)
    elif "continuation" in config:
        path_config = config["continuation"]
        
    steps_count = int(
        _require_value(path_config, "steps", "continuation.steps")
    )
    if steps_count < 2:
        raise ValueError("continuation.steps must be at least 2.")
    parameters_def = _require_value(
        path_config,
        "parameters",
        "continuation.parameters",
    )
    if not isinstance(parameters_def, dict) or not parameters_def:
        raise ValueError(
            "continuation.parameters must be a non-empty mapping of explicit paths."
        )
    for parameter_name, parameter_range in parameters_def.items():
        if not isinstance(parameter_range, dict):
            raise ValueError(
                f"continuation.parameters.{parameter_name} must be a mapping."
            )
        _require_value(
            parameter_range,
            "start",
            f"continuation.parameters.{parameter_name}.start",
        )
        _require_value(
            parameter_range,
            "end",
            f"continuation.parameters.{parameter_name}.end",
        )
    t_final_step = float(
        _require_value(
            path_config,
            "t_step",
            "continuation.t_step",
        )
    )
    if not np.isfinite(t_final_step) or t_final_step <= 0.0:
        raise ValueError("continuation.t_step must be finite and strictly positive.")
    div_threshold = float(
        _require_value(config, "divergence_norm", "divergence_norm")
    )
    if not np.isfinite(div_threshold) or div_threshold <= 0.0:
        raise ValueError("Continuation divergence_norm must be finite and positive.")
    integrator = str(_require_value(config, "integrator", "integrator"))
    h = float(_require_value(config, "h", "integration step h"))
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("Continuation integration step h must be finite and positive.")
    
    # Check overrides or path_config for memory policy
    if args.memory_policy:
        memory_policy = args.memory_policy
    else:
        memory_policy = path_config.get("memory_policy")
        if memory_policy is None:
            memory_policy = config.get("memory_policy")
    if memory_policy is None:
        raise ValueError(
            "Multiparameter continuation requires an explicit memory_policy."
        )
    memory_window_time = path_config.get("memory_window_time")
    if memory_policy in {"carry_window", "finite_window"}:
        if memory_window_time is None:
            raise ValueError(
                "Windowed multiparameter continuation requires explicit "
                "continuation.memory_window_time."
            )
        memory_window_length = int(round(float(memory_window_time) / h))
        if memory_window_length < 1:
            raise ValueError(
                "continuation.memory_window_time must span at least one step."
            )
        integration_memory_mode = "window"
    elif memory_policy in {"full_history", "full_caputo"}:
        memory_window_length = None
        integration_memory_mode = "full"
    elif memory_policy == "none":
        memory_window_length = None
        integration_memory_mode = "none"
    else:
        raise ValueError(
            "Unsupported multiparameter memory_policy: "
            f"{memory_policy!r}."
        )
    
    # Resolve seeds
    seeds = []
    if args.seed_file:
        seeds = load_seeds_from_csv(Path(args.seed_file))
    else:
        possible_seeds = Path(output_dir) / "seeds.csv"
        if possible_seeds.exists():
            seeds = load_seeds_from_csv(possible_seeds)
            
    if not seeds:
        print("Error: No seeds found for multiparameter continuation.")
        sys.exit(1)
        
    system_id = str(_require_value(config, "system_id", "system_id"))
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
    
    continuation_cfg = config.get("continuation") or {}
    continuation_order = str(
        _require_value(
            continuation_cfg,
            "continuation_order",
            "continuation.continuation_order",
        )
    )
    q_continuation = continuation_cfg.get("q_continuation")
    
    # Ensure q_continuation is 1.0 for integer continuation, or less than 1.0 for fractional continuation
    if continuation_order == "integer":
        q_continuation = 1.0
        config["continuation"]["q_continuation"] = 1.0
    elif continuation_order == "fractional":
        if q_continuation is None:
            q_continuation = _resolve_explicit_fractional_q(
                config,
                system=system,
                seeds=seeds,
                path_parameters=parameters_def,
            )
            config["continuation"]["q_continuation"] = q_continuation
            
        if q_continuation >= 1.0:
            raise ValueError(f"For fractional continuation, q_continuation must be strictly less than 1.0. Got {q_continuation}.")
            
    # Validate compatibility between integrator and order
    if continuation_order == "fractional":
        if integrator in ("rk4", "heun", "efork_q1"):
            raise ValueError(f"{integrator} only supports integer-order dynamics. Use abm or efork3 for fractional Caputo continuation.")
        validate_integrator_compatibility(integrator, q_continuation)
    else:
        # integer order
        if integrator in ("abm", "adm_wu2023"):
            raise ValueError(f"{integrator} requires 0<q<1. Use rk4, heun or efork_q1 for integer-order continuation.")
        validate_integrator_compatibility(integrator, 1.0)
        if integrator == "efork3":
            import warnings
            warnings.warn(
                "Integrator 'efork3' at q=1.0 redirects to the integer-order 'efork_q1' limit. "
                "For pure integer-order work, prefer 'rk4' or 'heun' which are simpler and faster.",
                UserWarning,
                stacklevel=2
            )
    
    # Distinguish causal continuation from sweep
    is_causal = memory_policy in ("carry_window", "full_history", "full_caputo", "finite_window")
    continuation_type = "continuation_causal_caputo" if is_causal else "independent_parameter_sweep"
    
    if not is_causal:
        print("Warning: last-state continuation in a Caputo system is a warm-start, not strict history-preserving continuation.")
        
    tau_grid = np.linspace(0.0, 1.0, steps_count)
    
    trace_rows = []
    final_candidates = []
    summaries = []
    
    # Run path for each seed
    for s in seeds:
        candidate_id = s["candidate_id"]
        x_in = np.array(s["x0"], dtype=float).copy()
        k_gain = _require_seed_gain(s)
        
        print(f"Running multi-parameter continuation for candidate {candidate_id} (type: {continuation_type})...")
        
        hist_t = None
        hist_x = None
        
        steps_ok = True
        
        for idx, tau in enumerate(tau_grid):
            # Compute parameters at tau
            current_params = dict(system.parameters)
            current_q = float(s["q"])
            current_eta = float(tau) # default eta changes from 0 to 1
            
            for param_name, param_range in parameters_def.items():
                start_val = float(param_range["start"])
                end_val = float(param_range["end"])
                val = start_val + tau * (end_val - start_val)
                if param_name == "q":
                    current_q = val
                elif param_name == "eta":
                    current_eta = val
                else:
                    current_params[param_name] = val
                    
            # Set up deformed vector field
            # Chua specific lure split
            from ..models.chua import chua_parameters
            required_parameters = (
                "model",
                "alpha",
                "beta",
                "gamma",
                "m0",
                "m1",
                "a1",
                "a2",
                "rho",
            )
            missing_parameters = [
                name for name in required_parameters if name not in current_params
            ]
            if missing_parameters:
                raise ValueError(
                    "The current registered system does not provide the explicit "
                    "parameters required by this Chua continuation route: "
                    + ", ".join(missing_parameters)
                )
            cp = chua_parameters(
                model=str(current_params["model"]),
                alpha=float(current_params["alpha"]),
                beta=float(current_params["beta"]),
                gamma=float(current_params["gamma"]),
                m0=float(current_params["m0"]),
                m1=float(current_params["m1"]),
                a1=float(current_params["a1"]),
                a2=float(current_params["a2"]),
                rho=float(current_params["rho"]),
            )
            P_mat, b_vec, c_vec = chua_matrices(cp)
            sat_gain = float(cp.m0 - cp.m1)
            
            def psi_deformed(sigma: float) -> float:
                value = float(sigma)
                if cp.model == "arctan":
                    return float(cp.a2 * np.arctan(cp.rho * value))
                return float(sat_gain * np.clip(value, -1.0, 1.0))
                
            P0 = P_mat + k_gain * np.outer(b_vec, c_vec)
            
            def rhs_deformed(t_val, x_val):
                sigma = float(c_vec @ x_val)
                delta = psi_deformed(sigma) - k_gain * sigma
                return P0 @ x_val + current_eta * b_vec * delta
                
            # Integrate this step
            try:
                t_tr, x_tr, status = integrate(
                    rhs=rhs_deformed,
                    x0=x_in,
                    q=current_q,
                    h=h,
                    t_final=t_final_step,
                    integrator=integrator,
                    memory_mode=integration_memory_mode,
                    memory_window_length=memory_window_length,
                    divergence_norm=div_threshold,
                    history_times=hist_t,
                    history_states=hist_x,
                    use_c_backend=bool(config.get("use_c_backend", False)),
                    allow_python_fallback=bool(
                        config.get("allow_python_fallback", False)
                    ),
                )
            except Exception as e:
                status = f"error: {e}"
                
            x_out = x_tr[-1].copy() if status == "ok" else x_in.copy()
            x_val = float(x_out[0]) if len(x_out) > 0 else 0.0
            y_val = float(x_out[1]) if len(x_out) > 1 else 0.0
            z_val = float(x_out[2]) if len(x_out) > 2 else 0.0
            
            trace_rows.append({
                "candidate_id": candidate_id,
                "step_index": idx,
                "eta": current_eta,
                "q": current_q,
                "h": h,
                "memory_policy": memory_policy,
                "continuation_mode": continuation_type,
                "x": x_val,
                "y": y_val,
                "z": z_val,
                "status": status,
                "residual_or_distance": float(np.linalg.norm(x_out)),
                "failure_reason": "" if status == "ok" else status,
            })
            
            if status != "ok":
                steps_ok = False
                break
                
            # Update history and x_in for next step
            if is_causal:
                hist_t = t_tr - t_tr[-1]
                hist_x = x_tr
            else:
                hist_t = None
                hist_x = None
                
            x_in = x_out
            
        if steps_ok:
            final_candidates.append({
                "candidate_id": candidate_id,
                "family": s["family"],
                "centered_or_biased": s["centered_or_biased"],
                "A": s["A"],
                "sigma0": s["sigma0"],
                "omega": s["omega"],
                "q": current_q,
                "eta": current_eta,
                "x": float(x_in[0]),
                "y": float(x_in[1]),
                "z": float(x_in[2]),
                "status": "ok",
            })
            
        summaries.append({
            "candidate_id": candidate_id,
            "success": steps_ok,
            "final_status": "ok" if steps_ok else "failed",
            "steps_completed": len(tau_grid) if steps_ok else idx,
        })
        
    # Write output files
    # 1. continuation_trace.csv
    trace_path = Path(output_dir) / "continuation_trace.csv"
    with open(trace_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "step_index", "eta", "q", "h", "memory_policy", "continuation_mode", "x", "y", "z", "status", "residual_or_distance", "failure_reason"])
        for r in trace_rows:
            w.writerow([
                r["candidate_id"], r["step_index"], r["eta"], r["q"], r["h"],
                r["memory_policy"], r["continuation_mode"],
                r["x"], r["y"], r["z"],
                r["status"], r["residual_or_distance"], r["failure_reason"]
            ])
            
    # 2. continuation_summary.json
    summary_path = Path(output_dir) / "continuation_summary.json"
    summary_data = {
        "system_id": system_id,
        "memory_policy": memory_policy,
        "history_carried": is_causal,
        "memory_window_time": path_config.get("memory_window_time"),
        "memory_window_steps": (
            int(round(float(path_config["memory_window_time"]) / h))
            if path_config.get("memory_window_time") is not None
            else None
        ),
        "candidates": summaries,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    # 3. final_candidates.csv
    final_path = Path(output_dir) / "final_candidates.csv"
    with open(final_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "family", "centered_or_biased", "A", "sigma0", "omega", "q", "eta", "x", "y", "z", "status"])
        for fc in final_candidates:
            w.writerow([
                fc["candidate_id"], fc["family"], fc["centered_or_biased"],
                fc["A"], fc["sigma0"], fc["omega"], fc["q"], fc["eta"],
                fc["x"], fc["y"], fc["z"], fc["status"]
            ])
            
    # 4. run_metadata.json, effective_config.json, effective_config.yaml
    first_seed = seeds[0] if seeds else None
    run_meta = collect_run_metadata(
        run_id=config.get("run_id", "auto_multiparameter"),
        workflow="continuation_multiparameter",
        system=system_id,
        q=float(first_seed["q"]),
        h=h,
        t_final=t_final_step * steps_count,
        t_burn=t_final_step,
        memory_mode=integration_memory_mode,
        integrator_name=integrator,
        integrator_backend="native" if config.get("use_c_backend", False) else "python",
        caputo=continuation_order == "fractional",
        parameters=system.parameters,
        lure=collect_lure_metadata(
            system.lure,
            transfer_convention=str(
                config.get("transfer_convention") or "not_specified"
            ),
            harmonic_condition=str(
                config.get("harmonic_condition") or "not_specified"
            ),
        ),
        seed=collect_seed_metadata(first_seed, source="continuation_multiparameter") if first_seed else None,
        random_seed=config.get("random_seed"),
        random_seed_policy=config.get("random_seed_policy") or "not_specified",
    )
    
    from ..reproducibility import metadata_to_jsonable
    meta_jsonable = metadata_to_jsonable(run_meta)

    # Resolve contracts
    seed_contract = resolve_seed_transfer_contract(config, system)
    seed_transfer_contract = {
        "df_order": seed_contract["df_order"],
        "transfer_mode": seed_contract["transfer_mode"],
        "q_seed": float(seed_contract["q_seed"]),
        "frequency_rule": seed_contract["lambda_frequency_rule"]
    }

    continuation_contract = {
        "continuation_order": continuation_order,
        "q_continuation": float(q_continuation),
        "integrator": integrator,
        "memory_policy": config.get("memory_policy"),
        "history_carried": is_causal
    }

    dynamics_contract = {
        "dynamics_order": config["dynamics"].get("dynamics_order"),
        "q_dynamics": float(config["dynamics"].get("q_dynamics")) if config["dynamics"].get("q_dynamics") is not None else None,
        "integrator": integrator
    }

    meta_jsonable["seed_transfer_contract"] = seed_transfer_contract
    meta_jsonable["continuation_contract"] = continuation_contract
    meta_jsonable["dynamics_contract"] = dynamics_contract

    meta_path = Path(output_dir) / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_jsonable, f, indent=2)

    config["seed_transfer_contract"] = seed_transfer_contract
    config["continuation_contract"] = continuation_contract
    config["dynamics_contract"] = dynamics_contract

    from ..workflows.config_loader import save_effective_config
    save_effective_config(config, output_dir)

    eff_json_path = Path(output_dir) / "effective_config.json"
    with open(eff_json_path, "w", encoding="utf-8") as f:
        def _clean(obj: Any) -> Any:
            if hasattr(obj, "tolist"):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(v) for v in obj]
            return obj
        json.dump(_clean(dict(config)), f, indent=2)
        
    print(f"Multiparameter continuation completed. Summary written to {summary_path}")
