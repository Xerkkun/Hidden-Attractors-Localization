"""Lyapunov exponent calculation and convergence workflow.

Stability: experimental
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from hidden_attractors.systems import get_system
from hidden_attractors.analysis.lyapunov_api import compute_lyapunov_spectrum
from hidden_attractors.plotting.lyapunov import plot_lyapunov_convergence_styled
from hidden_attractors.reproducibility import collect_run_metadata, write_run_metadata
from hidden_attractors.workflows.config_loader import save_effective_config


def run_lyapunov_workflow(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the Lyapunov exponent estimation workflow.

    Parameters
    ----------
    config : Dict[str, Any]
        Normalized configuration dictionary.

    Returns
    -------
    summary : Dict[str, Any]
        Results summary.
    """
    system_id = config.get("system_id", "chua_fractional_saturation")
    integrator = config.get("integrator", "efork3")
    q = config.get("q")
    output_dir = Path(config.get("output_dir", "outputs"))
    
    lyap_cfg = config.get("lyapunov", {})
    method = lyap_cfg.get("method", "fractional_variational_abm_qr")
    
    t_final = float(lyap_cfg.get("t_final", 500.0))
    t_burn = float(lyap_cfg.get("t_burn", 100.0))
    h = float(lyap_cfg.get("h", 0.01))
    
    x0 = np.asarray(lyap_cfg.get("initial_condition", [0.1, 0.0, 0.0]), dtype=float)
    
    reorth_every = lyap_cfg.get("orthonormalization_interval", 10)
    
    system = get_system(system_id)
    if q is None:
        q = float(system.parameters.get("q", 0.99))
    else:
        q = float(q)

    # Reconstruct system with config parameters
    import dataclasses
    system_params = {}
    for p_name in system.parameters:
        if p_name in config and config[p_name] is not None:
            system_params[p_name] = config[p_name]
        else:
            system_params[p_name] = system.parameters[p_name]
    system = dataclasses.replace(system, parameters=system_params)

    print()
    print("=" * 72)
    print(" lyapunov calculation - version_2")
    print("=" * 72)
    print(f"  system           = {system_id}")
    print(f"  method           = {method}")
    print(f"  order (q)        = {q}")
    print(f"  t_final          = {t_final:.1f} (t_burn = {t_burn:.1f})")
    print(f"  step (h)         = {h}")
    print(f"  output_dir       = {output_dir}")
    print("=" * 72)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    memory_mode = config.get("memory_mode", "full")
    memory_window = config.get("memory_window_steps") or config.get("memory_window_length", 400)

    # Compute Lyapunov spectrum
    summary_obj = compute_lyapunov_spectrum(
        system=system,
        x0=x0,
        q=q,
        method=method,
        h=h,
        t_final=t_final,
        t_burn=t_burn,
        reorthonormalize_every=reorth_every,
        memory_mode=memory_mode,
        memory_window=memory_window,
    )
    
    result = summary_obj.result
    
    # Save Lyapunov spectrum CSV
    spectrum_path = output_dir / "lyapunov_spectrum.csv"
    with open(spectrum_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dimension_index", "exponent"])
        for idx, exp in enumerate(result.exponents):
            writer.writerow([idx, exp])
    print(f"  Spectrum saved -> {spectrum_path}")
            
    # Save Lyapunov convergence CSV
    convergence_path = output_dir / "lyapunov_convergence.csv"
    with open(convergence_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["t"] + [f"LE{idx}" for idx in range(result.exponents.size)]
        writer.writerow(header)
        for t_idx, t_val in enumerate(result.times):
            row = [t_val] + list(result.convergence[t_idx])
            writer.writerow(row)
    print(f"  Convergence data saved -> {convergence_path}")
            
    # Draw convergence plot
    plot_path = output_dir / "lyapunov_convergence.png"
    plot_lyapunov_convergence_styled(result, plot_path, system_id=system_id)
    print(f"  Convergence plot saved -> {plot_path}")

    # Determine largest exponent and chaos indicator
    largest_exp = float(np.max(result.exponents)) if result.exponents.size > 0 else None
    
    # Chaos indicator classification:
    # If there's a positive exponent (we use 0.005 threshold to be robust to noise)
    if largest_exp is not None:
        if largest_exp > 0.005:
            chaos_indicator = "positive_largest_exponent"
        else:
            chaos_indicator = "nonpositive"
    else:
        chaos_indicator = "inconclusive"
        
    if result.status != "ok":
        chaos_indicator = "inconclusive"
        
    warnings = list(summary_obj.warnings)
    if q < 1.0:
        warnings.append("finite_time_lyapunov_estimate: fractional Caputo lyapunov calculation is a finite time local estimate.")
        
    summary_dict = {
        "analysis_type": "lyapunov",
        "system_id": system_id,
        "q": q,
        "method": method,
        "status": "completed" if result.status == "ok" else "failed",
        "finite_time": q < 1.0 or bool(result.finite_time_local),
        "spectrum": [float(x) for x in result.exponents],
        "largest_exponent": largest_exp,
        "chaos_indicator": chaos_indicator,
        "warnings": warnings,
        "data_csv": str(spectrum_path),
        "convergence_csv": str(convergence_path),
        "plot_path": str(plot_path),
    }
    
    summary_json_path = output_dir / "lyapunov_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"  Summary saved -> {summary_json_path}")
        
    # Write run metadata
    metadata = collect_run_metadata(
        run_id=str(output_dir.name),
        workflow=f"lyapunov_workflow:{method}",
        system=system_id,
        q=q,
        h=h,
        t_final=t_final,
        t_burn=t_burn,
        memory_mode=memory_mode,
        memory_window_steps=memory_window,
        integrator_name=method,
        integrator_backend="python" if "python" in method else "native",
        caputo=q < 1.0,
        parameters=system_params,
        extra={"largest_exponent": largest_exp, "chaos_indicator": chaos_indicator},
    )
    write_run_metadata(output_dir / "run_metadata.json", metadata)
    print(f"  Metadata saved -> {output_dir / 'run_metadata.json'}")
    
    save_effective_config(config, str(output_dir))
    return summary_dict
