import sys
from pathlib import Path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import numpy as np
import pytest
from src.systems.registry import get_system_by_id
from src.lure.nyquist import find_harmonic_candidates
from src.lure.seeds import build_lure_seed, build_modal_lure_seed
from src.systems.chua_arctan import ChuaArctanSystem

def test_chua_saturation_candidates_q1():
    sys_int = get_system_by_id("chua_integer_saturation")
    
    # 1. Verification of candidates against reference (Mathematica/version_2)
    # alpha = 8.4562, beta = 12.0732, gamma = 0.0052, m0 = -0.1768, m1 = -1.1468
    # Using imw_gain strategy
    candidates = find_harmonic_candidates(
        system=sys_int,
        transfer_mode="integer",
        seed_strategy="imw_gain",
        omega_min=0.5,
        omega_max=4.0,
        q=1.0
    )
    
    assert len(candidates) >= 2
    
    # Candidate 1:
    A0, omega0, k = candidates[0]
    assert np.isclose(omega0, 2.039186939959, rtol=1e-4)
    assert np.isclose(k, 0.209867354515, rtol=1e-4)
    assert np.isclose(A0, 5.856145086257, rtol=1e-4)
    
    # Candidate 2:
    A0_2, omega0_2, k_2 = candidates[1]
    assert np.isclose(omega0_2, 3.245287288346, rtol=1e-4)
    assert np.isclose(k_2, 0.959688184945, rtol=1e-4)
    assert np.isclose(A0_2, 1.044921079893, rtol=1e-4)

def test_chua_saturation_candidates_q09998():
    sys_frac = get_system_by_id("chua_fractional_saturation")
    
    candidates = find_harmonic_candidates(
        system=sys_frac,
        transfer_mode="fractional",
        seed_strategy="imw_gain",
        omega_min=0.5,
        omega_max=4.0,
        q=0.9998
    )
    
    assert len(candidates) >= 1
    A0, omega0, k = candidates[0]
    assert np.isclose(omega0, 2.040286051080, rtol=1e-4)
    assert np.isclose(k, 0.210022792962, rtol=1e-4)
    assert np.isclose(A0, 5.851767785486, rtol=1e-4)

def test_modal_vs_closed_form_seed():
    sys_int = get_system_by_id("chua_integer_saturation")
    A0, omega0, k = 5.856145086257, 2.039186939959, 0.209867354515
    
    # For q=1, modal and closed_form should match
    seed_modal, _ = build_lure_seed(
        sys_int, A0, omega0, k, q=1.0, transfer_mode="integer", seed_construction="modal"
    )
    seed_closed, _ = build_lure_seed(
        sys_int, A0, omega0, k, q=1.0, transfer_mode="integer", seed_construction="closed_form_integer"
    )
    
    expected_seed = np.array([5.856145086257, 0.369331578247, -8.366536168329])
    assert np.allclose(seed_modal, expected_seed, atol=1e-4)
    assert np.allclose(seed_closed, expected_seed, atol=1e-4)
    
    # For q=0.9998, closed_form integer throws error
    sys_frac = get_system_by_id("chua_fractional_saturation")
    A0_f, omega0_f, k_f = 5.851767785486, 2.040286051080, 0.210022792962
    
    # Closed form integer should raise ValueError
    with pytest.raises(ValueError):
        build_lure_seed(
            sys_frac, A0_f, omega0_f, k_f, q=0.9998, transfer_mode="fractional", seed_construction="closed_form_integer"
        )
        
    # Modal should work
    seed_modal_f, _ = build_lure_seed(
        sys_frac, A0_f, omega0_f, k_f, q=0.9998, transfer_mode="fractional", seed_construction="modal"
    )
    expected_modal_f = np.array([5.851767785486, 0.370408600307, -8.360972934420])
    assert np.allclose(seed_modal_f, expected_modal_f, atol=1e-4)

def test_modal_seed_residuals():
    sys_frac = get_system_by_id("chua_fractional_saturation")
    A0, omega0, k = 5.851767785486, 2.040286051080, 0.210022792962
    
    _, v_norm, matched_ev, target_lam = build_modal_lure_seed(
        sys_frac, A0, omega0, k, q=0.9998, transfer_mode="fractional"
    )
    
    # Normalisation residual
    norm_res = np.abs(sys_frac.r.astype(complex) @ v_norm - 1.0)
    assert norm_res < 1e-12
    
    # Modal residual
    P0 = sys_frac.P.astype(complex) + k * np.outer(sys_frac.b.astype(complex), sys_frac.r.astype(complex))
    modal_res = np.linalg.norm(P0 @ v_norm - matched_ev * v_norm)
    assert modal_res < 1e-12
    
    # Closer than tolerance
    assert np.abs(matched_ev - target_lam) < 1e-3

def test_arctan_describing_function_equivalence():
    sys_arctan = ChuaArctanSystem(describing_function_mode="closed_form")
    sys_arctan_quad = ChuaArctanSystem(describing_function_mode="quadrature")
    
    for A in [0.5, 1.0, 2.5, 10.0]:
        val_closed = sys_arctan.describing_function(A)
        val_quad = sys_arctan_quad.describing_function(A)
        assert np.abs(val_closed - val_quad) < 1e-5
