import sys
from pathlib import Path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import pytest
import numpy as np
import os
import shutil

def test_seed_plots_routed_to_diagnostic_directories(monkeypatch):
    import src.workflows.centered_lure_df_workflow as workflow_mod
    
    saved_dirs_and_prefixes = []
    
    def mock_plot_flexible_attractor_and_projections(*args, **kwargs):
        # Record where and under what prefix the plot was saved
        saved_dirs_and_prefixes.append((kwargs.get("output_dir"), kwargs.get("file_prefix")))
        
    monkeypatch.setattr(workflow_mod, "plot_flexible_attractor_and_projections", mock_plot_flexible_attractor_and_projections)
    monkeypatch.setattr(workflow_mod, "plot_timeseries_data", lambda *args, **kwargs: None)
    
    # Mock find_harmonic_candidates to return three candidate seeds
    monkeypatch.setattr(workflow_mod, "find_harmonic_candidates", lambda *args, **kwargs: [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)])
    
    # Mock build_lure_seed to return a mock vector
    monkeypatch.setattr(workflow_mod, "build_lure_seed", lambda *args, **kwargs: (np.array([0,0,0]), np.array([0,0,0])))
    
    # Mock integration to return distinct statuses for the three candidate seeds
    call_count = 0
    def mock_run_workflow_integration(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return np.array([0.0, 1.0]), np.array([[0,0,0], [1,1,1]]), "ok"
        elif call_count == 2:
            return np.array([0.0, 1.0]), np.array([[0,0,0], [1,1,1]]), "diverged_early"
        else:
            return np.array([0.0, 1.0]), np.array([[0,0,0], [1,1,1]]), "converged_equilibrium_early"
            
    monkeypatch.setattr(workflow_mod, "run_workflow_integration", mock_run_workflow_integration)
    
    # Mock workflow end to avoid full run
    def mock_run_fractional_continuation(*args, **kwargs):
        lambda_values = kwargs.get("lambda_values", [0.0])
        return [{"status": "ok", "x_out": np.array([0, 0, 0]), "lambda_value": lv} for lv in lambda_values]
    monkeypatch.setattr(workflow_mod, "run_fractional_continuation", mock_run_fractional_continuation)
    monkeypatch.setattr(workflow_mod, "_save_continuation_trace", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow_mod, "_build_summary_dict", lambda *args, **kwargs: {"notes": "test"})
    monkeypatch.setattr(workflow_mod, "_save_summary", lambda *args, **kwargs: None)
    
    config = {
        "system_id": "chua_fractional_saturation",
        "output_dir": "outputs/test_seed_plots_routing",
        "integrator": "abm",
        "dynamics_mode": "fractional",
        "seed_mode": "fractional",
        "continuation_mode": "fractional",
        "memory_policy": "full_caputo",
        "memory_mode": "full",
        "transfer_convention": "standard",
        "harmonic_condition": "1_minus_WN",
        "plot_enabled": True,
        "max_seed_candidates_to_plot": 3,
        "run_hiddenness_tests": False,
        "run_basin_slices": False,
        "run_sphere_tests": False
    }
    
    from src.workflows.centered_lure_df_workflow import run_centered_lure_df_workflow
    run_centered_lure_df_workflow(config)
    
    # Clean up output dir if created
    if os.path.exists("outputs/test_seed_plots_routing"):
        shutil.rmtree("outputs/test_seed_plots_routing", ignore_errors=True)
        
    assert len(saved_dirs_and_prefixes) == 3
    
    # Candidate 0: "ok" -> outputs/test_seed_plots_routing and prefix: seed_candidate_00
    assert "test_seed_plots_routing" in str(saved_dirs_and_prefixes[0][0])
    assert "diagnostics" not in str(saved_dirs_and_prefixes[0][0])
    assert saved_dirs_and_prefixes[0][1] == "seed_candidate_00"
    
    # Candidate 1: "diverged_early" -> outputs/test_seed_plots_routing/diagnostics/diverged_seeds and prefix: seed_diverged_01
    assert "diverged_seeds" in str(saved_dirs_and_prefixes[1][0])
    assert saved_dirs_and_prefixes[1][1] == "seed_diverged_01"
    
    # Candidate 2: "converged_equilibrium_early" -> outputs/test_seed_plots_routing/diagnostics/equilibrium_converged_seeds and prefix: seed_converged_02
    assert "equilibrium_converged_seeds" in str(saved_dirs_and_prefixes[2][0])
    assert saved_dirs_and_prefixes[2][1] == "seed_converged_02"
