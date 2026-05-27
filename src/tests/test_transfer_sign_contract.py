import sys
from pathlib import Path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import numpy as np
import pytest
from src.lure.transfer import W_eval
from src.lure.nyquist import find_harmonic_candidates
from src.systems.chua_saturation import ChuaSaturationSystem

def test_transfer_convention_signs():
    P = np.array([[-1.0, 0.0, 0.0],
                  [0.0, -2.0, 0.0],
                  [0.0, 0.0, -3.0]])
    b = np.array([1.0, 1.0, 1.0])
    r = np.array([1.0, 2.0, 3.0])
    omega = 2.0
    q = 0.95
    
    w_std = W_eval(omega, q, "fractional", P, b, r, transfer_convention="standard")
    w_opp = W_eval(omega, q, "fractional", P, b, r, transfer_convention="opposite_sign")
    
    assert np.allclose(w_std, -w_opp)

def test_harmonic_candidates_receives_conventions(monkeypatch):
    system = ChuaSaturationSystem(q=0.98)
    
    called_args = {}
    from src.lure import nyquist
    original_w_eval = nyquist.W_eval
    
    def mock_w_eval(*args, **kwargs):
        called_args["transfer_convention"] = kwargs.get("transfer_convention")
        return original_w_eval(*args, **kwargs)
        
    monkeypatch.setattr(nyquist, "W_eval", mock_w_eval)
    
    find_harmonic_candidates(
        system=system,
        transfer_mode="fractional",
        seed_strategy="nyquist_df",
        omega_min=1.0,
        omega_max=2.0,
        amplitude_min=1.0,
        amplitude_max=2.0,
        grid_size_omega=3,
        grid_size_amplitude=3,
        root_refinement=False,
        q=0.98,
        transfer_convention="opposite_sign",
        harmonic_condition="1_plus_WN"
    )
    
    assert called_args.get("transfer_convention") == "opposite_sign"
