"""CLI commands for seed generation under the unified hidden-attractors CLI.

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
from typing import Sequence, Any, Dict

import numpy as np

from ..workflows.config_loader import load_config, apply_cli_overrides, resolve_seed_transfer_contract
from ..reproducibility import collect_run_metadata, collect_lure_metadata, collect_seed_metadata
from ..systems import get_system
from ..seed_generation.lure import (
    find_lure_omega_gain_candidates,
    solve_lure_amplitude_from_gain,
    build_lure_fractional_seed,
    lure_transfer_function,
    fourier_coefficients_lure,
    reconstruct_biased_lure_seed_from_system,
)


def _nested_value(config: dict[str, Any], dotted_path: str) -> Any:
    """Return one explicitly configured value, or ``None`` when absent."""

    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _raw_has_path(config: dict[str, Any], dotted_path: str) -> bool:
    """Return whether a value was written explicitly in the source mapping."""

    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return value is not None


def _require_explicit_source_policies(
    raw_config: dict[str, Any],
    override_keys: set[str],
) -> None:
    """Reject policy values that could otherwise come from loader defaults."""

    required_sources = {
        "experiment.random_seed": "random_seed",
        "integrator.memory_mode": "memory_mode",
        "integrator.memory_policy": "memory_policy",
        "integrator.use_c_backend": "use_c_backend",
        "integrator.allow_python_fallback": "allow_python_fallback",
    }
    missing = [
        source_path
        for source_path, override_key in required_sources.items()
        if not _raw_has_path(raw_config, source_path)
        and not _raw_has_path(raw_config, override_key)
        and override_key not in override_keys
    ]
    if missing:
        raise ValueError(
            "Seed generation requires explicit reproducibility/backend policies; "
            "missing: " + ", ".join(missing)
        )
    raw_memory_mode = _nested_value(raw_config, "integrator.memory_mode")
    raw_memory_policy = _nested_value(raw_config, "integrator.memory_policy")
    finite_window_requested = (
        raw_memory_mode == "window"
        or raw_memory_policy == "finite_window"
        or override_keys.intersection({"memory_mode", "memory_policy"})
        and (
            _nested_value(raw_config, "integrator.memory_window_time") is not None
            or "memory_window_time" in override_keys
        )
    )
    if (
        finite_window_requested
        and not _raw_has_path(raw_config, "integrator.memory_window_time")
        and "memory_window_time" not in override_keys
    ):
        raise ValueError(
            "Finite-window seed metadata requires explicit "
            "integrator.memory_window_time."
        )


def _seed_execution_contract(
    config: dict[str, Any],
    *,
    biased: bool,
) -> dict[str, Any]:
    """Validate and return the explicit numerical contract for seed generation."""

    required = [
        "system_id",
        "seed.df_order",
        "seed.transfer_mode",
        "seed.q_seed",
        "omega_min",
        "omega_max",
        "grid_size_omega",
        "seed_theta",
        "h",
        "integrator",
        "memory_mode",
        "memory_policy",
        "use_c_backend",
        "allow_python_fallback",
        "final_simulation.t_final",
        "final_simulation.t_burn",
        "random_seed",
        "seed.calculation.harmonics",
        "seed.calculation.quadrature_points",
        "seed.calculation.spectral_denominator_floor",
        "seed.metadata.transfer_convention",
        "seed.metadata.harmonic_condition",
        "seed.metadata.random_seed_policy",
    ]
    if biased:
        required.extend(
            [
                "amplitude_min",
                "amplitude_max",
                "grid_size_amplitude",
                "seed.biased_search.sigma0_min",
                "seed.biased_search.sigma0_max",
                "seed.biased_search.sigma0_grid_size",
                "seed.biased_search.sigma0_exclusion_radius",
                "seed.biased_search.dc_residual_max",
                "seed.biased_search.harmonic_residual_max",
                "seed.biased_search.sigma_division_floor",
                "seed.biased_search.duplicate_omega_tolerance",
                "seed.biased_search.duplicate_amplitude_tolerance",
                "seed.biased_search.duplicate_sigma0_tolerance",
                "seed.biased_search.max_candidates",
            ]
        )
    missing = [path for path in required if _nested_value(config, path) is None]
    if missing:
        raise ValueError(
            "Seed generation requires explicit configuration values; missing: "
            + ", ".join(missing)
        )

    contract = {path: _nested_value(config, path) for path in required}
    q = float(contract["seed.q_seed"])
    omega_min = float(contract["omega_min"])
    omega_max = float(contract["omega_max"])
    frequency_grid_size = int(contract["grid_size_omega"])
    theta = float(contract["seed_theta"])
    h = float(contract["h"])
    t_final = float(contract["final_simulation.t_final"])
    t_burn = float(contract["final_simulation.t_burn"])
    harmonics = int(contract["seed.calculation.harmonics"])
    quadrature_points = int(contract["seed.calculation.quadrature_points"])
    denominator_floor = float(contract["seed.calculation.spectral_denominator_floor"])

    if not 0.0 < q <= 1.0:
        raise ValueError("seed.q_seed must satisfy 0 < q_seed <= 1.")
    if not 0.0 < omega_min < omega_max:
        raise ValueError("omega bounds must satisfy 0 < omega_min < omega_max.")
    if frequency_grid_size < 2:
        raise ValueError("grid_size_omega must be at least 2.")
    if not np.isfinite(theta):
        raise ValueError("seed_theta must be finite.")
    if h <= 0.0:
        raise ValueError("h must be positive.")
    if not 0.0 <= t_burn < t_final:
        raise ValueError("final_simulation times must satisfy 0 <= t_burn < t_final.")
    if harmonics < 2:
        raise ValueError("seed.calculation.harmonics must be at least 2.")
    if quadrature_points < 2:
        raise ValueError("seed.calculation.quadrature_points must be at least 2.")
    if denominator_floor <= 0.0:
        raise ValueError("seed.calculation.spectral_denominator_floor must be positive.")

    memory_mode = str(contract["memory_mode"])
    memory_policy = str(contract["memory_policy"])
    if memory_mode == "window" or memory_policy == "finite_window":
        memory_window = _nested_value(config, "memory_window_time")
        if memory_window is None or float(memory_window) <= 0.0:
            raise ValueError(
                "Finite-window seed metadata requires explicit positive memory_window_time."
            )

    if biased:
        amplitude_min = float(contract["amplitude_min"])
        amplitude_max = float(contract["amplitude_max"])
        amplitude_grid_size = int(contract["grid_size_amplitude"])
        sigma0_min = float(contract["seed.biased_search.sigma0_min"])
        sigma0_max = float(contract["seed.biased_search.sigma0_max"])
        sigma0_grid_size = int(contract["seed.biased_search.sigma0_grid_size"])
        if not 0.0 < amplitude_min < amplitude_max:
            raise ValueError(
                "amplitude bounds must satisfy 0 < amplitude_min < amplitude_max."
            )
        if amplitude_grid_size < 2 or sigma0_grid_size < 2:
            raise ValueError("biased-search grid sizes must be at least 2.")
        if not sigma0_min < sigma0_max:
            raise ValueError("sigma0 bounds must satisfy sigma0_min < sigma0_max.")
        positive_paths = (
            "seed.biased_search.sigma_division_floor",
            "seed.biased_search.duplicate_omega_tolerance",
            "seed.biased_search.duplicate_amplitude_tolerance",
            "seed.biased_search.duplicate_sigma0_tolerance",
        )
        nonnegative_paths = (
            "seed.biased_search.sigma0_exclusion_radius",
            "seed.biased_search.dc_residual_max",
            "seed.biased_search.harmonic_residual_max",
        )
        if any(float(contract[path]) <= 0.0 for path in positive_paths):
            raise ValueError("biased-search numerical tolerances must be positive.")
        if any(float(contract[path]) < 0.0 for path in nonnegative_paths):
            raise ValueError("biased-search radii and residual limits must be non-negative.")
        if int(contract["seed.biased_search.max_candidates"]) < 1:
            raise ValueError("seed.biased_search.max_candidates must be positive.")

    return contract


def compute_rho_H_for_lure(
    system: Any,
    q: float,
    omega: float,
    amplitude: float,
    sigma0: float,
    gain: float,
    harmonics: int,
    quadrature_points: int,
    denominator_floor: float,
) -> tuple[float, dict[str, Any]]:
    # 1. Fourier coefficients of nonlinearity under psi(sigma0 + A cos(theta))
    fourier = fourier_coefficients_lure(
        amplitude=amplitude,
        sigma0=sigma0,
        system=system,
        harmonics=harmonics,
        n_quad=quadrature_points,
    )
    coeffs = fourier["coefficients"]
    Y1 = complex(coeffs[1]["Y"])
    
    # 2. Transfer function at fundamental frequency
    W1 = lure_transfer_function(omega, q, system)
    denom = abs(W1) * abs(Y1) + denominator_floor
    
    # 3. Sum of higher order harmonics response
    higher = 0.0
    for k in range(2, harmonics + 1):
        Yk = complex(coeffs[k]["Y"])
        Wk = lure_transfer_function(k * omega, q, system)
        higher += abs(Wk) * abs(Yk)
        
    rho_H = float(higher / denom)
    return rho_H, fourier

def search_biased_seeds(
    system: Any,
    q: float,
    wmin: float,
    wmax: float,
    nscan: int,
    A_min: float,
    A_max: float,
    sigma0_min: float,
    sigma0_max: float,
    amplitude_grid_size: int,
    sigma0_grid_size: int,
    sigma0_exclusion_radius: float,
    dc_residual_max: float,
    harmonic_residual_max: float,
    sigma_division_floor: float,
    duplicate_omega_tolerance: float,
    duplicate_amplitude_tolerance: float,
    duplicate_sigma0_tolerance: float,
    max_candidates: int,
    harmonics: int,
    quadrature_points: int,
    denominator_floor: float,
    config_path: Path,
    theta: float,
) -> list[dict[str, Any]]:
    pairs = find_lure_omega_gain_candidates(
        q=q,
        system=system,
        wmin=wmin,
        wmax=wmax,
        nscan=nscan,
        compatible_only=False,
    )
    
    candidates = []
    pmat = np.asarray(system.matrix, dtype=float)
    bvec = np.asarray(system.input_vector, dtype=float)
    cvec = np.asarray(system.output_vector, dtype=float)
    W0 = float((cvec.reshape(1, -1) @ np.linalg.solve(pmat, bvec.reshape(-1, 1)))[0, 0])
    
    A_vals = np.linspace(A_min, A_max, amplitude_grid_size)
    sigma0_vals = np.linspace(sigma0_min, sigma0_max, sigma0_grid_size)
    sigma0_vals = [s for s in sigma0_vals if abs(s) > sigma0_exclusion_radius]
    
    for omega, gain in pairs:
        W1 = lure_transfer_function(omega, q, system)
        for A in A_vals:
            for sigma0 in sigma0_vals:
                rho_H, fourier = compute_rho_H_for_lure(
                    system=system,
                    q=q,
                    omega=omega,
                    amplitude=A,
                    sigma0=sigma0,
                    gain=gain,
                    harmonics=harmonics,
                    quadrature_points=quadrature_points,
                    denominator_floor=denominator_floor,
                )
                y_mean = float(fourier["y_mean"])
                Y1 = complex(fourier["coefficients"][1]["Y"])
                N1 = Y1 / A
                N0 = (
                    y_mean / sigma0
                    if abs(sigma0) > sigma_division_floor
                    else 0.0
                )
                
                res_dc = abs(sigma0 + W0 * y_mean)
                res_h = abs(1.0 + W1 * N1)
                total_res = res_dc + res_h
                
                if res_dc < dc_residual_max and res_h < harmonic_residual_max:
                    candidates.append({
                        "omega": float(omega),
                        "gain": float(gain),
                        "A": float(A),
                        "sigma0": float(sigma0),
                        "residual_dc": float(res_dc),
                        "residual_h": float(res_h),
                        "total_res": float(total_res),
                        "N1": N1,
                        "N0": N0,
                        "rho_H": rho_H,
                    })
                    
    candidates.sort(key=lambda x: x["total_res"])
    unique_candidates = []
    for cand in candidates:
        is_dup = False
        for uc in unique_candidates:
            if abs(cand["omega"] - uc["omega"]) < duplicate_omega_tolerance and \
               abs(cand["A"] - uc["A"]) < duplicate_amplitude_tolerance and \
               abs(cand["sigma0"] - uc["sigma0"]) < duplicate_sigma0_tolerance:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(cand)
            if len(unique_candidates) >= max_candidates:
                break
                
    final_candidates = []
    for idx, uc in enumerate(unique_candidates):
        try:
            biased_seed = reconstruct_biased_lure_seed_from_system(
                q=q,
                system=system,
                amplitude=uc["A"],
                sigma0=uc["sigma0"],
                omega=uc["omega"],
                theta=theta,
            )
            final_candidates.append({
                "candidate_id": f"biased_classical_b{idx}",
                "family": "lure_classical_biased",
                "centered_or_biased": "biased",
                "A": uc["A"],
                "sigma0": uc["sigma0"],
                "omega": uc["omega"],
                "q": q,
                "harmonic_residual": uc["residual_h"],
                "rho_H": uc["rho_H"],
                "x0": biased_seed.seed.tolist(),
                "reconstruction_metadata": {
                    "gain": uc["gain"],
                    "residual_dc": uc["residual_dc"],
                    "residual_h": uc["residual_h"],
                    "N1": [uc["N1"].real, uc["N1"].imag],
                    "N0": uc["N0"],
                    "theta": theta,
                },
                "source_config": str(config_path),
            })
        except Exception as e:
            print(f"Warning: failed to reconstruct biased seed for b{idx}: {e}")
            
    return final_candidates

def run_seed_generation(
    centered_or_biased: str,
    argv: Sequence[str] | None = None
) -> None:
    parser = argparse.ArgumentParser(prog=f"hidden-attractors seed lure-{centered_or_biased}")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("-o", "--output-dir", type=str, help="Directory to save output files")
    
    # Explicit CLI options for seed
    parser.add_argument("--df-order", type=str, choices=["integer", "fractional"], help="describing function order type")
    parser.add_argument("--transfer-mode", type=str, choices=["published_integer_laplace", "fractional_spectral"], help="transfer mode concrete type")
    parser.add_argument("--q-seed", type=float, help="order used for seeds")
    parser.add_argument("--integrator", type=str, help="integrator name")
    parser.add_argument("--h", type=float, help="integration step size")
    parser.add_argument("--memory-policy", type=str, choices=["full_history", "finite_window", "none"], help="Caputo memory policy")
    parser.add_argument("--memory-window-time", type=float, help="memory window length in seconds")
    parser.add_argument("--use-c-backend", action="store_true", default=None, help="use compiled C/Numba backend")
    parser.add_argument("--no-c-backend", action="store_false", dest="use_c_backend", help="do not use compiled C/Numba backend")
    parser.add_argument("--allow-python-fallback", action="store_true", default=None, help="allow fallback to Python")
    parser.add_argument("--no-python-fallback", action="store_false", dest="allow_python_fallback", help="do not allow fallback to Python")

    args, extra_args = parser.parse_known_args(argv)
    
    config_path = Path(args.config)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("Seed configuration must be a YAML mapping.")

    config = load_config(config_path)

    # Build overrides dictionary
    from .run import parse_dynamic_overrides

    overrides = parse_dynamic_overrides(list(extra_args))
    for key in ("df_order", "transfer_mode", "q_seed", "integrator", "h",
                 "memory_policy", "memory_window_time", "use_c_backend",
                 "allow_python_fallback"):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val

    _require_explicit_source_policies(raw_config, set(overrides))
    if overrides:
        config = apply_cli_overrides(config, overrides)

    execution = _seed_execution_contract(
        config,
        biased=centered_or_biased == "biased",
    )
    system_id = str(execution["system_id"])
    system = get_system(system_id)
    lure_sys = system.lure
    if lure_sys is None:
        print(f"Error: System '{system_id}' does not have a registered Lur'e decomposition.")
        sys.exit(1)

    # Resolve seed transfer contract
    contract = resolve_seed_transfer_contract(config, system)
    df_order = contract["df_order"]
    q = contract["q_seed"]
    transfer_mode = contract["transfer_mode"]

    if df_order == "fractional":
        print("Warning: Fourier/Nyquist/describing-function calculations are interpreted through the Weyl-Caputo bridge. The harmonic solution is a seed-generation approximation, not proof of an exact Caputo periodic orbit.")

    wmin = float(execution["omega_min"])
    wmax = float(execution["omega_max"])
    nscan = int(execution["grid_size_omega"])
    theta = float(execution["seed_theta"])
    harmonics = int(execution["seed.calculation.harmonics"])
    quadrature_points = int(execution["seed.calculation.quadrature_points"])
    denominator_floor = float(
        execution["seed.calculation.spectral_denominator_floor"]
    )
    
    output_dir = args.output_dir or config.get("output_dir") or "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    candidates = []
    
    if centered_or_biased == "centered":
        pairs = find_lure_omega_gain_candidates(
            q=q,
            system=lure_sys,
            wmin=wmin,
            wmax=wmax,
            nscan=nscan,
            compatible_only=True,
        )
        for idx, (omega, gain) in enumerate(pairs):
            try:
                A = solve_lure_amplitude_from_gain(gain, lure_sys, method="classic")
                x0, vector, matched = build_lure_fractional_seed(
                    q=q,
                    system=lure_sys,
                    omega=omega,
                    gain=gain,
                    amplitude=A,
                    theta=theta,
                )
                W1 = lure_transfer_function(omega, q, lure_sys)
                N_A = complex(lure_sys.describing_function(A))
                harmonic_residual = abs(1.0 + W1 * N_A)
                
                rho_H, _ = compute_rho_H_for_lure(
                    system=lure_sys,
                    q=q,
                    omega=omega,
                    amplitude=A,
                    sigma0=0.0,
                    gain=gain,
                    harmonics=harmonics,
                    quadrature_points=quadrature_points,
                    denominator_floor=denominator_floor,
                )
                candidates.append({
                    "candidate_id": f"centered_classical_b{idx}",
                    "family": "lure_classical_centered",
                    "centered_or_biased": "centered",
                    "A": float(A),
                    "sigma0": 0.0,
                    "omega": float(omega),
                    "q": float(q),
                    "harmonic_residual": float(harmonic_residual),
                    "rho_H": float(rho_H),
                    "x0": x0.tolist(),
                    "reconstruction_metadata": {
                        "gain": float(gain),
                        "matched_eigenvalue": [float(matched.real), float(matched.imag)],
                        "eigenvector": [[float(val.real), float(val.imag)] for val in vector],
                        "theta": theta,
                    },
                    "source_config": str(config_path),
                })
            except Exception as e:
                print(f"Warning: failed to build centered candidate b{idx} (omega={omega:.4f}): {e}")
    else:
        print("Note: Biased describing function (BDF) is a first-harmonic approximation to expand seed searches; it is not a proof of attractor existence or hiddenness.")
        A_min = float(execution["amplitude_min"])
        A_max = float(execution["amplitude_max"])
        sigma0_min = float(execution["seed.biased_search.sigma0_min"])
        sigma0_max = float(execution["seed.biased_search.sigma0_max"])
        candidates = search_biased_seeds(
            system=lure_sys,
            q=q,
            wmin=wmin,
            wmax=wmax,
            nscan=nscan,
            A_min=A_min,
            A_max=A_max,
            sigma0_min=sigma0_min,
            sigma0_max=sigma0_max,
            amplitude_grid_size=int(execution["grid_size_amplitude"]),
            sigma0_grid_size=int(
                execution["seed.biased_search.sigma0_grid_size"]
            ),
            sigma0_exclusion_radius=float(
                execution["seed.biased_search.sigma0_exclusion_radius"]
            ),
            dc_residual_max=float(
                execution["seed.biased_search.dc_residual_max"]
            ),
            harmonic_residual_max=float(
                execution["seed.biased_search.harmonic_residual_max"]
            ),
            sigma_division_floor=float(
                execution["seed.biased_search.sigma_division_floor"]
            ),
            duplicate_omega_tolerance=float(
                execution["seed.biased_search.duplicate_omega_tolerance"]
            ),
            duplicate_amplitude_tolerance=float(
                execution["seed.biased_search.duplicate_amplitude_tolerance"]
            ),
            duplicate_sigma0_tolerance=float(
                execution["seed.biased_search.duplicate_sigma0_tolerance"]
            ),
            max_candidates=int(execution["seed.biased_search.max_candidates"]),
            harmonics=harmonics,
            quadrature_points=quadrature_points,
            denominator_floor=denominator_floor,
            config_path=config_path,
            theta=theta,
        )
        
    # ── Write minimum outputs ──
    # 1. seed_generation_summary.json
    summary_path = Path(output_dir) / "seed_generation_summary.json"
    summary_data = {
        "system_id": system_id,
        "family": f"lure_classical_{centered_or_biased}",
        "q": q,
        "omega_min": wmin,
        "omega_max": wmax,
        "candidates_count": len(candidates),
        "candidates": candidates,
    }
    if centered_or_biased == "biased":
        summary_data["scientific_warning"] = (
            "The biased describing function is a first-harmonic approximation "
            "used to scan for candidate initial states. It does not establish the existence, "
            "stability, or hiddenness of any attractor."
        )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    # 2. harmonic_residuals.csv
    residuals_path = Path(output_dir) / "harmonic_residuals.csv"
    with open(residuals_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "family", "centered_or_biased", "A", "sigma0", "omega", "q", "harmonic_residual", "rho_H"])
        for c in candidates:
            w.writerow([
                c["candidate_id"], c["family"], c["centered_or_biased"],
                c["A"], c["sigma0"], c["omega"], c["q"],
                c["harmonic_residual"], c["rho_H"]
            ])
            
    # 3. seeds.csv
    seeds_path = Path(output_dir) / "seeds.csv"
    with open(seeds_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "family", "centered_or_biased", "A", "sigma0", "omega", "q", "harmonic_residual", "rho_H", "x0", "reconstruction_metadata", "source_config"])
        for c in candidates:
            w.writerow([
                c["candidate_id"], c["family"], c["centered_or_biased"],
                c["A"], c["sigma0"], c["omega"], c["q"],
                c["harmonic_residual"], c["rho_H"],
                json.dumps(c["x0"]), json.dumps(c["reconstruction_metadata"]),
                c["source_config"]
            ])
            
    # 4. run_metadata.json, effective_config.yaml, effective_config.json
    h_val = float(execution["h"])
    t_final = float(execution["final_simulation.t_final"])
    t_burn = float(execution["final_simulation.t_burn"])
    
    first_seed = candidates[0] if candidates else None
    run_meta = collect_run_metadata(
        run_id=str(config["run_id"]),
        workflow=f"seed_lure_{centered_or_biased}",
        system=system_id,
        q=q,
        h=h_val,
        t_final=t_final,
        t_burn=t_burn,
        memory_mode=str(execution["memory_mode"]),
        integrator_name=str(execution["integrator"]),
        integrator_backend=(
            "native" if bool(execution["use_c_backend"]) else "python"
        ),
        caputo=bool(q < 1.0),
        parameters=system.parameters,
        lure=collect_lure_metadata(
            lure_sys,
            transfer_convention=str(
                execution["seed.metadata.transfer_convention"]
            ),
            harmonic_condition=str(execution["seed.metadata.harmonic_condition"]),
        ),
        seed=collect_seed_metadata(first_seed, source=f"seed_lure_{centered_or_biased}") if first_seed else None,
        random_seed=int(execution["random_seed"]),
        random_seed_policy=str(execution["seed.metadata.random_seed_policy"]),
    )
    
    from ..reproducibility import metadata_to_jsonable
    meta_jsonable = metadata_to_jsonable(run_meta)

    # Resolve contracts
    seed_transfer_contract = {
        "df_order": contract["df_order"],
        "transfer_mode": contract["transfer_mode"],
        "q_seed": float(contract["q_seed"]),
        "frequency_rule": contract["lambda_frequency_rule"]
    }

    continuation_contract = {
        "continuation_order": config["continuation"].get("continuation_order"),
        "q_continuation": float(config["continuation"].get("q_continuation")) if config["continuation"].get("q_continuation") is not None else None,
        "integrator": config["integrator"],
        "memory_policy": config["memory_policy"],
        "history_carried": config["memory_policy"] != "none"
    }

    dynamics_contract = {
        "dynamics_order": config["dynamics"].get("dynamics_order"),
        "q_dynamics": float(config["dynamics"].get("q_dynamics")) if config["dynamics"].get("q_dynamics") is not None else None,
        "integrator": config["integrator"]
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
        
    print(f"Generated {len(candidates)} seeds. Summary: {summary_path}")

def lure_centered(argv: Sequence[str] | None = None) -> None:
    run_seed_generation("centered", argv)

def lure_biased(argv: Sequence[str] | None = None) -> None:
    run_seed_generation("biased", argv)
