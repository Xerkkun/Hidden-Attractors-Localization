import sys
from pathlib import Path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import numpy as np
import pytest
from src.verification.stability import classify_equilibrium_stability
from src.systems.chua_saturation import ChuaSaturationSystem

def test_stability_margin_stable(monkeypatch):
    system = ChuaSaturationSystem(q=0.9)
    eq_pt = np.zeros(3)
    
    # We want a positive margin: angles > q * pi / 2
    # threshold = 0.9 * pi / 2 = 1.4137
    # Let's set eigenvalue arguments to e.g. 1.5, which is > 1.4137 + tol
    eigvals = np.array([
        -1.0 + 10.0j,
        -1.0 - 10.0j,
        -5.0
    ]) # angle for -1+10j is arctan(10/-1) + pi = 1.6708, which is > 1.4137
    
    monkeypatch.setattr(np.linalg, "eigvals", lambda J: eigvals)
    
    res = classify_equilibrium_stability(system, eq_pt, tol=1e-8)
    assert res["stability_class"] == "stable"
    assert res["matignon_margin"] > 0
    assert res["stable"] is True

def test_stability_margin_unstable(monkeypatch):
    system = ChuaSaturationSystem(q=0.9)
    eq_pt = np.zeros(3)
    
    # We want a negative margin: at least one angle < q * pi / 2
    # threshold = 0.9 * pi / 2 = 1.4137
    # Let's use eigenvalue 1.0 + 0.1j (angle = 0.1, which is < 1.4137)
    eigvals = np.array([
        1.0 + 0.1j,
        1.0 - 0.1j,
        -5.0
    ])
    
    monkeypatch.setattr(np.linalg, "eigvals", lambda J: eigvals)
    
    res = classify_equilibrium_stability(system, eq_pt, tol=1e-8)
    assert res["stability_class"] == "unstable"
    assert res["matignon_margin"] < 0
    assert res["stable"] is False

def test_stability_margin_marginal(monkeypatch):
    system = ChuaSaturationSystem(q=0.9)
    eq_pt = np.zeros(3)
    
    threshold = 0.9 * np.pi / 2.0
    
    # We want angle exactly equal to the threshold (marginal/border case)
    # eigenvalue = exp(j * threshold) = cos(threshold) + j * sin(threshold)
    # let's set eigenvalues with one having angle extremely close to threshold:
    # let's say angle = threshold + 1e-10 (which is < tol=1e-8)
    target_angle = threshold + 1e-10
    eig = np.cos(target_angle) + 1j * np.sin(target_angle)
    eigvals = np.array([
        eig,
        np.conj(eig),
        -5.0
    ])
    
    monkeypatch.setattr(np.linalg, "eigvals", lambda J: eigvals)
    
    res = classify_equilibrium_stability(system, eq_pt, tol=1e-8)
    assert res["stability_class"] == "marginal_or_inconclusive"
    assert abs(res["matignon_margin"]) < 1e-8
    assert res["stable"] is False
