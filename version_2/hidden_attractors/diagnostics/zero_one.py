"""0-1 chaos-test diagnostic runner.

Stability: experimental
"""

from __future__ import annotations

import csv
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional

from hidden_attractors.systems import get_system
from hidden_attractors.integrations.selector import integrate
from hidden_attractors.analysis.zero_one import zero_one_test
from hidden_attractors.analysis.spectral import infer_step
from hidden_attractors.plotting.zero_one import plot_zero_one_phase_styled
from hidden_attractors.reproducibility import collect_run_metadata, write_run_metadata
from hidden_attractors.workflows.config_loader import load_config, apply_cli_overrides


def run_zero_one_diagnostic(
    times: np.ndarray,
    states: np.ndarray,
    observable: str,
    output_dir: Path,
    *,
    t_burn: float = 0.0,
    n_c: int = 100,
    c_min: float = 0.1,
    c_max: float = 3.04159,
    seed: int = 12345,
    threshold_chaotic: float = 0.85,
    threshold_regular: float = 0.15,
    system_id: str = "unknown",
    metadata_base: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Execute 0-1 test on a trajectory time-series and write outputs."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Infer step
    h = infer_step(times, fallback=0.01)
    
    # Detrend/pre-burn portion
    n_burn = max(0, int(np.ceil(t_burn / h)))
    if n_burn >= len(times):
        n_burn = 0
        
    tail_t = times[n_burn:]
    tail_states = states[n_burn:]
    
    # Resolve observable coordinate index
    coord_map = {"x": 0, "y": 1, "z": 2}
    coord_idx = coord_map.get(observable.lower(), 0)
    if coord_idx >= tail_states.shape[1]:
        coord_idx = 0
        
    signal = tail_states[:, coord_idx]
    
    # Generate c values in (c_min, c_max)
    rng = np.random.default_rng(seed)
    # We clip c values to avoid resonance at 0, pi
    c_array = rng.uniform(max(0.01, c_min), min(np.pi - 0.01, c_max), size=n_c)
    
    # Run test
    result = zero_one_test(
        signal,
        c_values=c_array,
        random_seed=seed,
        detrend=True,
        normalize=True,
    )
    
    k_median = result["K_median"]
    
    # Classification based on configurable thresholds
    if k_median >= threshold_chaotic:
        classification = "chaotic"
    elif k_median <= threshold_regular:
        classification = "regular"
    else:
        classification = "inconclusive"
        
    # Write zero_one_c_values.csv
    # We need K for each c. Let's compute them or extract them.
    # Note that zero_one_test in analysis doesn't return list of K per c, but we can compute it here:
    from hidden_attractors.analysis.zero_one import _k_for_c, _prepare_signal
    prep_sig = _prepare_signal(signal, detrend=True, normalize=True, max_samples=None)
    k_values = [_k_for_c(prep_sig, c_val) for c_val in c_array]
    
    c_values_path = output_dir / "zero_one_c_values.csv"
    with open(c_values_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["c_value", "K_c"])
        for c_val, k_val in zip(c_array, k_values):
            writer.writerow([c_val, k_val])
            
    # For zero_one_displacement.csv, let's find the c value closest to the median K, and write its displacement
    diffs = np.abs(np.array(k_values) - k_median)
    best_idx = int(np.argmin(diffs))
    best_c = float(c_array[best_idx])
    
    # Compute displacement for best_c
    index = np.arange(1, prep_sig.size + 1, dtype=float)
    p_values = np.cumsum(prep_sig * np.cos(index * best_c))
    q_values = np.cumsum(prep_sig * np.sin(index * best_c))
    max_lag = min(max(20, prep_sig.size // 10), 500)
    lags = np.arange(1, max_lag + 1, dtype=int)
    displacement = np.empty(max_lag, dtype=float)
    mean_signal = float(np.mean(prep_sig))
    denominator = max(1.0 - float(np.cos(best_c)), np.finfo(float).eps)
    for offset, lag in enumerate(range(1, max_lag + 1)):
        dp = p_values[lag:] - p_values[:-lag]
        dq = q_values[lag:] - q_values[:-lag]
        raw = float(np.mean(dp * dp + dq * dq))
        oscillatory = mean_signal**2 * (1.0 - float(np.cos(lag * best_c))) / denominator
        displacement[offset] = raw - oscillatory
        
    displacement_path = output_dir / "zero_one_displacement.csv"
    with open(displacement_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lag", "displacement"])
        for lag_val, disp_val in zip(lags, displacement):
            writer.writerow([lag_val, disp_val])
            
    # Save phase plot for this best_c
    plot_path = output_dir / "zero_one_plot.png"
    plot_zero_one_phase_styled(prep_sig, best_c, plot_path, system_id=system_id)
    
    # Prepare warnings
    warnings = []
    if len(signal) < 1000:
        warnings.append("small_sample_size: 0-1 test is a finite-time diagnostic; small sample sizes may be inaccurate.")
    warnings.append("zero_one_does_not_certify_hiddenness: the 0-1 test is a supporting chaotic time-series diagnostic only and does not prove hiddenness.")
    
    summary_dict = {
        "analysis_type": "zero_one_test",
        "observable": observable,
        "n_samples": int(prep_sig.size),
        "n_c_values": int(n_c),
        "K_values": [float(k) for k in k_values],
        "K_median": float(k_median),
        "classification": classification,
        "finite_time_warning": True,
        "warnings": warnings,
        "c_values_csv": str(c_values_path),
        "displacement_csv": str(displacement_path),
        "plot_path": str(plot_path),
    }
    
    summary_json_path = output_dir / "zero_one_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
        
    # Write run metadata
    metadata_kwargs = {
        "run_id": str(output_dir.name),
        "workflow": "zero_one_diagnostic",
        "system": system_id,
        "q": 1.0,
        "h": h,
        "t_final": float(times[-1]),
        "t_burn": t_burn,
        "memory_mode": "not_applicable",
        "integrator_name": "zero_one",
        "integrator_backend": "python",
        "caputo": False,
        "parameters": {},
        "extra": {
            "observable": observable,
            "K_median": float(k_median),
            "classification": classification,
            "zero_one_does_not_prove_chaos": True,
        }
    }
    if metadata_base:
        metadata_kwargs.update(metadata_base)
        
    metadata = collect_run_metadata(**metadata_kwargs)
    write_run_metadata(output_dir / "run_metadata.json", metadata)
    
    return summary_dict


def run_zero_one_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate from config and run 0-1 test workflow."""
    import dataclasses
    
    system_id = config.get("system_id", "chua_fractional_saturation")
    integrator = config.get("integrator", "efork3")
    q = config.get("q")
    h = float(config.get("h", 0.001))
    
    zo_cfg = config.get("zero_one", {})
    observable = zo_cfg.get("observable", "x")
    
    t_final = float(zo_cfg.get("t_final", 500.0))
    t_burn = float(zo_cfg.get("t_burn", 120.0))
    div_norm = float(zo_cfg.get("divergence_norm", 120.0))
    output_dir = Path(config.get("output_dir", "outputs"))
    
    n_c = int(zo_cfg.get("n_c_values", 100))
    c_min = float(zo_cfg.get("c_min", 0.1))
    c_max = float(zo_cfg.get("c_max", 3.04159))
    seed = int(zo_cfg.get("seed", 20260517))
    
    threshold_chaotic = float(zo_cfg.get("threshold_chaotic", 0.85))
    threshold_regular = float(zo_cfg.get("threshold_regular", 0.15))
    
    # Load chaotic system and parameters
    system = get_system(system_id)
    system_params = {}
    for p_name in system.parameters:
        if p_name in config and config[p_name] is not None:
            system_params[p_name] = config[p_name]
        else:
            system_params[p_name] = system.parameters[p_name]
    system = dataclasses.replace(system, parameters=system_params)
    
    if q is None:
        q = float(system.parameters.get("q", 0.99))
    else:
        q = float(q)
        
    ics = zo_cfg.get("initial_condition", [0.1, 0.0, 0.0])
    x0_arr = np.asarray(ics, dtype=float)
    
    # Integrate using the selector
    times, states, status = integrate(
        rhs=system.rhs,
        x0=x0_arr,
        q=q,
        h=h,
        t_final=t_final,
        integrator=integrator,
        memory_mode=config.get("memory_mode", "full"),
        memory_window_length=config.get("memory_window_length") or config.get("memory_window_steps", 400),
        divergence_norm=div_norm,
        system=system,
        use_c_backend=config.get("use_c_backend", True),
        allow_python_fallback=config.get("allow_python_fallback", True),
        early_stop_config=config.get("early_stop", {}),
        equilibria=list(system.equilibrium_points().values()),
    )
    
    if status != "ok":
        raise RuntimeError(f"Simulation failed with status '{status}' before running 0-1 test.")
        
    metadata_base = {
        "q": q,
        "h": h,
        "t_final": t_final,
        "t_burn": t_burn,
        "memory_mode": config.get("memory_mode", "full"),
        "integrator_name": integrator,
        "integrator_backend": "python",
        "caputo": q < 1.0,
        "parameters": system_params,
    }
    
    return run_zero_one_diagnostic(
        times,
        states,
        observable,
        output_dir,
        t_burn=t_burn,
        n_c=n_c,
        c_min=c_min,
        c_max=c_max,
        seed=seed,
        threshold_chaotic=threshold_chaotic,
        threshold_regular=threshold_regular,
        system_id=system_id,
        metadata_base=metadata_base,
    )
