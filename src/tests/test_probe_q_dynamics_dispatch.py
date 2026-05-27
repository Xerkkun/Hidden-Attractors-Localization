import sys
from pathlib import Path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import pytest
import numpy as np
import warnings

def test_run_neighborhood_probe_warns_when_q_dynamics_effective_none():
    from src.verification.hiddenness import run_neighborhood_probe
    from src.systems.chua_saturation import ChuaSaturationSystem
    
    system = ChuaSaturationSystem(q=0.9)
    # We expect a warning since q_dynamics_effective is None
    with pytest.warns(UserWarning, match="q_dynamics_effective is omitted"):
        try:
            run_neighborhood_probe(
                system=system,
                x0=np.array([1, 1, 1]),
                transfer_mode="fractional",
                integrator="heun", # will throw ValueError when resolved
                t_final=1.0,
                t_burn=0.5,
                h=0.01,
                ref_tail=np.array([[0,0,0]]),
                stable_equilibria=[],
                dynamics_mode="integer" # fallback forces active_q = 1.0 which works with Heun
            )
        except Exception:
            # We only care that the warning was raised before execution fails/completes
            pass

def test_sphere_sweep_propagates_q_dynamics_effective(monkeypatch):
    from src.verification.sphere_tests import run_sphere_probe_sweep
    from src.systems.chua_saturation import ChuaSaturationSystem
    
    system = ChuaSaturationSystem(q=0.88)
    
    called_payloads = []
    def mock_run_single_sphere_probe(payload):
        called_payloads.append(payload)
        idx = payload[0]
        return idx, {
            "x0": [0,0,0], "destination": "stable_equilibrium", "final_state": [0,0,0],
            "trajectory": np.array([[0,0,0]]), "status": "ok", "distance_to_target": 0.0, "distance_to_equilibrium": 0.0
        }
        
    monkeypatch.setattr("src.verification.sphere_tests.run_single_sphere_probe", mock_run_single_sphere_probe)
    monkeypatch.setattr("src.verification.sphere_tests.plot_sphere_test_results", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.verification.sphere_tests._print_and_save_hiddenness_tables", lambda *args, **kwargs: None)
    
    config = {
        "system_id": "chua_fractional_saturation",
        "transfer_mode": "fractional",
        "integrator": "abm",
        "equilibrium_tol": 0.5,
        "divergence_norm": 120.0,
        "target_match_metric": "centroid_distance",
        "target_match_tol": 0.5,
        "dynamics_mode": "fractional",
        "memory_mode": "full",
        "memory_window_length": None,
        "sphere_tests": {
            "enabled": True,
            "radii": [1e-3],
            "samples_initial": 2,
            "samples_growth_factor": 1.0,
            "random_seed": 42
        }
    }
    
    import os
    import shutil
    out_dir = "outputs/test_sphere_sweep_dispatch"
    os.makedirs(out_dir, exist_ok=True)
    try:
        run_sphere_probe_sweep(
            system=system,
            config=config,
            equilibria={"E0": np.array([0,0,0])},
            stable_eqs=[],
            ref_tail=np.array([[0,0,0]]),
            output_dir=out_dir,
            q_dynamics_effective=0.88
        )
    finally:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
    
    assert len(called_payloads) == 2
    # The last element in the payload tuple (index 19) is q_dynamics_effective
    assert called_payloads[0][19] == 0.88

def test_basin_slice_propagates_q_dynamics_effective(monkeypatch):
    from src.verification.basins import generate_basin_slice
    from src.systems.chua_saturation import ChuaSaturationSystem
    
    system = ChuaSaturationSystem(q=0.88)
    
    called_args = []
    def mock_classify_point_worker(args):
        called_args.append(args)
        return 0, 0, 0
        
    monkeypatch.setattr("src.verification.basins._classify_point_worker", mock_classify_point_worker)
    
    generate_basin_slice(
        plane="xy",
        system=system,
        transfer_mode="fractional",
        integrator="abm",
        ref_tail=np.array([[0,0,0]]),
        stable_eqs=[],
        fixed_values={"z": 0.0},
        grid_n=2,
        workers=1,
        q_dynamics_effective=0.88
    )
    
    assert len(called_args) == 4
    # The last element in the args tuple (index 20) is q_dynamics_effective
    assert called_args[0][20] == 0.88
