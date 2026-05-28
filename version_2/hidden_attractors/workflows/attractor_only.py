"""Direct attractor simulation workflow.

Stability: experimental

This workflow:
  - Integrates a registered chaotic system directly from given initial conditions.
  - Computes diagnostic metrics (FFT, zero-one test, entropy).
  - Saves time series and attractor trajectories to CSV.
  - Generates premium 3D phase space, projection, and time series plots.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hidden_attractors.systems import get_system
from hidden_attractors.integrations.selector import integrate
from hidden_attractors.plotting.dynamics import (
    plot_phase_space,
    plot_phase_projections,
    plot_time_series,
)
from hidden_attractors.workflows.config_loader import save_effective_config


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _compute_diagnostics(
    times: np.ndarray, states: np.ndarray, t_burn: float, h: float
) -> Dict[str, Any]:
    """Compute basic diagnostic metrics on the post-burn portion of a trajectory."""
    n_burn = max(0, int(math.ceil(t_burn / h)))
    if n_burn >= len(times):
        n_burn = 0

    tail_t = times[n_burn:]
    tail_x = states[n_burn:]

    if len(tail_x) < 4:
        return {
            "warning": "trajectory_too_short_for_diagnostics",
            "n_burn_steps": n_burn,
            "n_tail_steps": len(tail_x),
        }

    x, y, z = tail_x[:, 0], tail_x[:, 1], tail_x[:, 2]
    norms = np.linalg.norm(tail_x, axis=1)

    diag: Dict[str, Any] = {
        "n_burn_steps": n_burn,
        "n_tail_steps": len(tail_x),
        "t_burn_actual": float(tail_t[0]),
        "t_final": float(tail_t[-1]),
        "has_nan": bool(np.any(~np.isfinite(tail_x))),
        "range_x": [float(x.min()), float(x.max())],
        "range_y": [float(y.min()), float(y.max())],
        "range_z": [float(z.min()), float(z.max())],
        "max_norm": float(norms.max()),
        "mean_norm": float(norms.mean()),
    }

    if diag["has_nan"]:
        diag["classification"] = "invalid"
        return diag

    # Dominant frequency (FFT on x)
    try:
        n_fft = len(x)
        fft_amp = np.abs(np.fft.rfft(x - x.mean()))
        freqs = np.fft.rfftfreq(n_fft, d=h)
        # Skip DC (index 0)
        dom_idx = int(np.argmax(fft_amp[1:]) + 1)
        dominant_freq = float(freqs[dom_idx])
        peak_ratio = float(fft_amp[dom_idx] / (fft_amp[1:].mean() + 1e-15))
        diag["dominant_frequency"] = dominant_freq
        diag["peak_ratio"] = peak_ratio

        # Spectral entropy
        psd = fft_amp[1:] ** 2
        psd_norm = psd / (psd.sum() + 1e-30)
        ent = float(-np.sum(psd_norm * np.log(psd_norm + 1e-30)))
        max_ent = math.log(len(psd_norm))
        diag["spectral_entropy"] = ent
        diag["spectral_entropy_normalized"] = float(ent / max_ent) if max_ent > 0 else 0.0
    except Exception as exc:
        diag["fft_warning"] = str(exc)
        peak_ratio = 1.0

    # 0–1 test (approximate)
    try:
        diag["zero_one_test"] = _zero_one_test_approx(x)
    except Exception:
        diag["zero_one_test"] = None

    # Preliminary classification
    norm_max = diag["max_norm"]
    if norm_max > 1e6:
        cls = "diverged"
    elif diag["has_nan"]:
        cls = "invalid"
    elif diag.get("zero_one_test") is not None and diag["zero_one_test"] > 0.8:
        cls = "chaotic_like"
    elif diag.get("peak_ratio", 1.0) > 20.0 and diag.get("spectral_entropy_normalized", 1.0) < 0.3:
        cls = "periodic_like"
    elif diag.get("spectral_entropy_normalized", 0.0) > 0.6:
        cls = "chaotic_like"
    else:
        cls = "unclassified"

    diag["classification"] = cls
    return diag


def _zero_one_test_approx(x: np.ndarray, n_samples: int = 100) -> float:
    """Approximate 0-1 test for chaos (Gottwald & Melbourne 2004)."""
    rng = np.random.default_rng(42)

    # Downsample if array is too long
    max_points = 5000
    if len(x) > max_points:
        step = int(math.ceil(len(x) / max_points))
        x = x[::step]

    N = len(x)
    if N < 100:
        return float("nan")

    c_vals = rng.uniform(math.pi / 5.0, 4.0 * math.pi / 5.0, size=min(n_samples, 20))
    K_vals = []

    for c in c_vals:
        p = np.cumsum(x * np.cos(np.arange(N) * c))
        q_arr = np.cumsum(x * np.sin(np.arange(N) * c))
        n_max = N // 10
        if n_max < 5:
            continue
        ns = np.arange(1, n_max + 1)
        M = np.array([
            np.mean((p[n:] - p[:-n]) ** 2 + (q_arr[n:] - q_arr[:-n]) ** 2)
            for n in ns
        ])
        if len(ns) < 3:
            continue
        slope = np.polyfit(np.log(ns + 1e-10), np.log(M + 1e-10), 1)[0]
        K_vals.append(float(np.clip(slope, 0.0, 2.0) / 1.0))

    if not K_vals:
        return float("nan")
    return float(np.median(K_vals))


# ---------------------------------------------------------------------------
# Output savers
# ---------------------------------------------------------------------------

def _save_csv(path: Path, times: np.ndarray, states: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        # Header dynamically matches state dimensions
        cols = ["t"] + [f"x{i}" if i > 0 else "x" for i in range(states.shape[1])]
        # Chua circuit standard names for 3D system
        if states.shape[1] == 3:
            cols = ["t", "x", "y", "z"]
        writer.writerow(cols)
        for i in range(len(times)):
            writer.writerow([times[i]] + list(states[i]))


# ---------------------------------------------------------------------------
# Core simulation runner
# ---------------------------------------------------------------------------

def run_attractor_only_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the simulate_attractor_only workflow (Ruta B)."""
    import dataclasses

    system_id = config.get("system_id", "chua_fractional_saturation")
    integrator = config.get("integrator", "efork3")
    q = config.get("q")
    h = float(config.get("h", 0.001))
    
    # Calculate N or t_final
    if "N" in config and config["N"] is not None:
        N = int(config["N"])
        t_final = N * h
    else:
        t_final = float(config.get("final_simulation", {}).get("t_final", 500.0))
        N = int(math.ceil(t_final / h))

    t_burn = float(config.get("final_simulation", {}).get("t_burn", 120.0))
    div_norm = float(config.get("final_simulation", {}).get("divergence_norm", 120.0))
    output_dir = Path(config.get("output_dir", "outputs"))
    sci_label = config.get("scientific_label", "Direct attractor simulation workflow")

    # Load chaotic system and parameters
    system = get_system(system_id)
    
    # Merge custom parameters from config
    system_params = {}
    for p_name in ["alpha", "beta", "gamma", "m", "n", "m0", "m1", "a1", "a2", "rho"]:
        if p_name in config and config[p_name] is not None:
            system_params[p_name] = config[p_name]
    
    if system_params:
        merged_params = dict(system.parameters)
        merged_params.update(system_params)
        system = dataclasses.replace(system, parameters=merged_params)

    # Use the system default q if not specified
    if q is None:
        q = float(system.parameters.get("q", 0.99))
    else:
        q = float(q)

    # Resolve initial conditions
    ics = config.get("initial_conditions", {})
    if not ics:
        single_ic = config.get("final_simulation", {}).get("initial_condition")
        if single_ic is not None:
            ics = {"x0": single_ic}
        else:
            raise ValueError(
                "No initial conditions found in config. "
                "Provide 'initial_conditions' dict or 'simulation.initial_condition' vector."
            )

    # Resolve memory settings
    memory_mode = config.get("memory_mode", "full")
    memory_window_length = config.get("memory_window_length") or config.get("memory_window_steps")

    # Print workflow header
    print()
    print("=" * 72)
    print(" simulate_attractor_only - version_2")
    print("=" * 72)
    print(f"  system           = {system_id}")
    print(f"  integrator       = {integrator}")
    print(f"  q                = {q}")
    print(f"  h                = {h}")
    print(f"  N                = {N}  (t_final = {t_final:.1f})")
    if memory_mode == "window":
        print(f"  Memory mode      = window ({memory_window_length} steps)")
    else:
        print(f"  Memory mode      = full (from t=0)")
    print(f"  t_burn           = {t_burn}")
    print(f"  divergence_norm  = {div_norm}")
    print(f"  output_dir       = {output_dir}")
    print(f"  Initial cond.    = {list(ics.keys())}")
    print(f"  Scientific label = {sci_label}")
    print("=" * 72)

    # Get equilibria for early stopping
    equilibria = list(system.equilibrium_points().values())
    early_stop_config = config.get("early_stop", {})

    all_results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for ic_label, x0 in ics.items():
        x0_arr = np.asarray(x0, dtype=float)
        
        t0_wall = time.perf_counter()
        # Integrate using the selector
        times, states, status = integrate(
            rhs=system.rhs,
            x0=x0_arr,
            q=q,
            h=h,
            t_final=t_final,
            integrator=integrator,
            memory_mode=memory_mode,
            memory_window_length=memory_window_length,
            divergence_norm=div_norm,
            system=system,
            use_c_backend=config.get("use_c_backend", True),
            allow_python_fallback=config.get("allow_python_fallback", True),
            early_stop_config=early_stop_config,
            equilibria=equilibria,
        )
        elapsed = time.perf_counter() - t0_wall

        result: Dict[str, Any] = {
            "ic_label": ic_label,
            "x0": x0_arr.tolist(),
            "integrator": integrator,
            "status": status,
            "elapsed_s": round(elapsed, 3),
            "steps_completed": len(times) - 1,
            "t_final_reached": float(times[-1]),
            "hidden_verified": False,
        }

        # Save time series CSV
        if config.get("save_timeseries", True):
            ts_path = output_dir / f"{ic_label}_timeseries.csv"
            _save_csv(ts_path, times, states)
            result["timeseries_csv"] = str(ts_path)
            # Legacy alias compat
            if len(ics) == 1:
                import shutil
                shutil.copy2(ts_path, output_dir / "final_timeseries.csv")

        # Save attractor CSV (post-burn)
        n_burn = max(0, int(math.ceil(t_burn / h)))
        if config.get("save_attractor", True) and n_burn < len(times):
            att_path = output_dir / f"{ic_label}_attractor.csv"
            _save_csv(att_path, times[n_burn:], states[n_burn:])
            result["attractor_csv"] = str(att_path)
            # Legacy alias compat
            if len(ics) == 1:
                import shutil
                shutil.copy2(att_path, output_dir / "final_attractor.csv")

        # Compute diagnostics
        if config.get("diagnostics", True) and len(times) > 1:
            diag = _compute_diagnostics(times, states, t_burn, h)
            result["diagnostics"] = diag
            result["classification"] = diag.get("classification", "unknown")

        # Generate plots
        if config.get("plot_enabled", True) and status in ("ok", "diverged", "diverged_early", "converged_equilibrium_early"):
            try:
                figures_dir = output_dir / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)
                
                # Combine times and states into a single trajectory array
                trajectory = np.column_stack([times, states])
                
                if config.get("plot_attractors", True):
                    # 3D phase space
                    plot_phase_space(
                        trajectory[n_burn:],
                        figures_dir / f"{ic_label}_3d.png",
                        title=f"{ic_label} Phase Space ({integrator})",
                    )
                    # Projections
                    plot_phase_projections(
                        trajectory[n_burn:],
                        figures_dir / f"{ic_label}_projections.png",
                        title=f"{ic_label} Projections",
                    )
                    
                if config.get("plot_timeseries", True):
                    plot_time_series(
                        trajectory,
                        figures_dir / f"{ic_label}_timeseries.png",
                        title=f"{ic_label} Time Series",
                    )
                
                result["figures_dir"] = str(figures_dir)
            except Exception as exc:
                result["plot_warning"] = str(exc)

        print(
            f"  [{ic_label}] integrator={integrator}  status={status}"
            f"  steps={result['steps_completed']}  t={result['t_final_reached']:.1f}"
            f"  class={result.get('classification', '?')}  [{elapsed:.1f}s]"
        )
        all_results.append(result)

    # Save summary and effective config
    summary = {
        "workflow_mode": "simulate_attractor_only",
        "system_id": system_id,
        "integrator": integrator,
        "scientific_label": sci_label,
        "hidden_verified": False,
        "q": q,
        "h": h,
        "N": N,
        "t_burn": t_burn,
        "divergence_norm": div_norm,
        "params": system_params,
        "initial_conditions": {k: v for k, v in ics.items()},
        "results": all_results,
        "n_completed": sum(1 for r in all_results if r["status"] == "ok"),
        "n_diverged": sum(1 for r in all_results if r["status"] == "diverged"),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    
    save_effective_config(config, str(output_dir))
    print(f"\n  Summary saved -> {summary_path}")

    return summary
