"""CLI commands for seed generation under the unified hidden-attractors CLI.

Stability: internal
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Sequence, Any, Dict

import numpy as np

from ..workflows.config_loader import load_config
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

def compute_rho_H_for_lure(
    system: Any,
    q: float,
    omega: float,
    amplitude: float,
    sigma0: float,
    gain: float,
    K: int = 10,
    n_quad: int = 1024,
) -> tuple[float, dict[str, Any]]:
    # 1. Fourier coefficients of nonlinearity under psi(sigma0 + A cos(theta))
    fourier = fourier_coefficients_lure(
        amplitude=amplitude,
        sigma0=sigma0,
        system=system,
        harmonics=K,
        n_quad=n_quad,
    )
    coeffs = fourier["coefficients"]
    Y1 = complex(coeffs[1]["Y"])
    
    # 2. Transfer function at fundamental frequency
    W1 = lure_transfer_function(omega, q, system)
    denom = abs(W1) * abs(Y1) + 1.0e-14
    
    # 3. Sum of higher order harmonics response
    higher = 0.0
    for k in range(2, K + 1):
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
    config_path: Path,
    theta: float = 0.0,
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
    
    A_vals = np.linspace(A_min, A_max, 50)
    sigma0_vals = np.linspace(sigma0_min, sigma0_max, 50)
    sigma0_vals = [s for s in sigma0_vals if abs(s) > 1e-3]
    
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
                    K=10,
                    n_quad=1024,
                )
                y_mean = float(fourier["y_mean"])
                Y1 = complex(fourier["coefficients"][1]["Y"])
                N1 = Y1 / A
                N0 = y_mean / sigma0 if abs(sigma0) > 1e-12 else 0.0
                
                res_dc = abs(sigma0 + W0 * y_mean)
                res_h = abs(1.0 + W1 * N1)
                total_res = res_dc + res_h
                
                if res_dc < 0.2 and res_h < 0.2:
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
            if abs(cand["omega"] - uc["omega"]) < 1e-2 and \
               abs(cand["A"] - uc["A"]) < 0.1 and \
               abs(cand["sigma0"] - uc["sigma0"]) < 0.1:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(cand)
            if len(unique_candidates) >= 10:
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
    parser.add_argument("-c", "--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("-p", "--preset", type=str, help="Select a built-in config preset")
    parser.add_argument("-o", "--output-dir", type=str, help="Directory to save output files")
    
    args, extra_args = parser.parse_known_args(argv)
    
    if args.preset:
        from .run import PRESETS, find_example_config
        filename = PRESETS.get(args.preset)
        if not filename:
            print(f"Error: Preset '{args.preset}' not recognized. Available: {list(PRESETS.keys())}")
            sys.exit(1)
        config_path = find_example_config(filename)
    elif args.config:
        config_path = Path(args.config)
    else:
        print("Error: Must provide --config or --preset.")
        sys.exit(1)
        
    config = load_config(config_path)
    
    system_id = config.get("system_id", "chua_fractional_saturation")
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
    lure_sys = system.lure
    if lure_sys is None:
        print(f"Error: System '{system_id}' does not have a registered Lur'e decomposition.")
        sys.exit(1)
        
    q = float(config.get("q_seed") or config.get("q") or system.parameters.get("q", 1.0))
    wmin = float(config.get("omega_min", 1e-4))
    wmax = float(config.get("omega_max", 10.0))
    nscan = int(config.get("grid_size_omega", 20000))
    
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
                    theta=float(config.get("seed_theta", 0.0)),
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
                    K=10,
                    n_quad=1024,
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
                        "theta": float(config.get("seed_theta", 0.0)),
                    },
                    "source_config": str(config_path),
                })
            except Exception as e:
                print(f"Warning: failed to build centered candidate b{idx} (omega={omega:.4f}): {e}")
    else:
        print("Note: Biased describing function (BDF) is a first-harmonic approximation to expand seed searches; it is not a proof of attractor existence or hiddenness.")
        A_min = float(config.get("amplitude_min", 0.1))
        A_max = float(config.get("amplitude_max", 20.0))
        sigma0_min = float(config.get("biased_search", {}).get("sigma0_min", -8.0))
        sigma0_max = float(config.get("biased_search", {}).get("sigma0_max", 8.0))
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
            config_path=config_path,
            theta=float(config.get("seed_theta", 0.0)),
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
            
    # 4. run_metadata.json
    h_val = float(config.get("h", 0.01))
    t_final = float(config.get("final_simulation", {}).get("t_final", 500.0))
    t_burn = float(config.get("final_simulation", {}).get("t_burn", 120.0))
    
    first_seed = candidates[0] if candidates else None
    run_meta = collect_run_metadata(
        run_id=config.get("run_id", "auto_seed"),
        workflow=f"seed_lure_{centered_or_biased}",
        system=system_id,
        q=q,
        h=h_val,
        t_final=t_final,
        t_burn=t_burn,
        memory_mode=config.get("memory_mode", "full"),
        integrator_name=config.get("integrator", "efork3"),
        integrator_backend="native" if config.get("use_c_backend", True) else "python",
        caputo=True,
        parameters=system.parameters,
        lure=collect_lure_metadata(lure_sys, transfer_convention="standard", harmonic_condition="1_minus_WN"),
        seed=collect_seed_metadata(first_seed, source=f"seed_lure_{centered_or_biased}") if first_seed else None,
        random_seed=config.get("random_seed"),
        random_seed_policy=config.get("random_seed_policy", "fixed_reproducible"),
    )
    
    meta_path = Path(output_dir) / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        # Custom dict serialisation for metadata to handle dataclasses nested
        from ..reproducibility import metadata_to_jsonable
        json.dump(metadata_to_jsonable(run_meta), f, indent=2)
        
    print(f"Generated {len(candidates)} seeds. Summary: {summary_path}")

def lure_centered(argv: Sequence[str] | None = None) -> None:
    run_seed_generation("centered", argv)

def lure_biased(argv: Sequence[str] | None = None) -> None:
    run_seed_generation("biased", argv)
