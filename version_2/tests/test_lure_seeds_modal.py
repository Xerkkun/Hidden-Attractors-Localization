"""Numerical regression tests for the modal seed construction in src/lure/seeds.py.

These tests verify that:
  1. For q=1  the modal seed matches the reference values.
  2. For q=0.9998 the modal seed matches the reference values.
  3. For q=1  the modal and closed-form seeds agree.
  4. For q=0.9998 the closed-form seed DIFFERS from the correct modal seed.
  5. Requesting closed_form_integer with q<1 raises ValueError.
  6. The matched eigenvalue is close to lambda0, and r^T v == 1.
  7. seed_neg == -seed_pos.

Reference parameters (Chua non-smooth with saturation):
    alpha=8.4562, beta=12.0732, gamma=0.0052, m0=-0.1768, m1=-1.1468

Reference q=1 (branch 1):
    omega0 = 2.03918693995900095839600543459
    a0     = 5.85614508625736042399812946778
    k      = 0.209867354515083823715425816146
    seed   = [5.85614508625736, 0.369331578246, -8.366536168329]

Reference q=0.9998 (branch 1):
    omega0 = 2.040286051080
    a0     = 5.851767785486
    k      = 0.210022792962
    seed   = [5.851767785486, 0.370408600307, -8.360972934420]

Closed-form integer seed for q=0.9998 (WRONG value for diagnostic):
    [5.851767785486, 0.369965103164, -8.362475658982]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure src is importable from the repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import importlib

from typing import Any
from hidden_attractors.systems import get_system
import dataclasses
from hidden_attractors.lure.seeds import build_lure_seed, build_closed_form_integer_seed, build_modal_lure_seed, _lambda_from_frequency

def get_system_by_id(system_id: str, **kwargs) -> Any:
    name_map = {
        "chua_integer_saturation": "chua-nonsmooth",
        "chua_fractional_saturation": "chua-nonsmooth",
        "chua_fractional_arctan": "chua-arctan",
        "chua_arctan_wu2023": "fractional-chua-arctan-wu2023",
    }
    normalized_sys_id = name_map.get(system_id, system_id)
    system = get_system(normalized_sys_id)
    
    # Merge overrides and build adapter attributes
    merged_params = dict(system.parameters)
    merged_params.update(kwargs)
    system = dataclasses.replace(system, parameters=merged_params)
    
    if "q" in kwargs:
        q_val = kwargs["q"]
    else:
        if system_id == "chua_fractional_saturation":
            q_val = 0.9998
        elif system_id == "chua_fractional_arctan":
            q_val = 0.995
        else:
            q_val = 1.0
            
    object.__setattr__(system, "q", q_val)
    for k, v in merged_params.items():
        try:
            object.__setattr__(system, k, v)
        except AttributeError:
            pass
            
    if system.lure is not None:
        object.__setattr__(system, "P", system.lure.matrix)
        object.__setattr__(system, "b", system.lure.input_vector)
        object.__setattr__(system, "r", system.lure.output_vector)
        object.__setattr__(system, "describing_function", system.lure.describing_function)
        object.__setattr__(system, "psi", system.lure.nonlinearity)
    object.__setattr__(system, "evaluate_rhs", lambda x: system.evaluate(x))
    return system

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

_ALPHA = 8.4562
_BETA = 12.0732
_GAMMA = 0.0052
_M0 = -0.1768
_M1 = -1.1468

# q = 1 branch 1
_REF_Q1 = dict(
    omega0=2.03918693995900095839600543459,
    a0=5.85614508625736042399812946778,
    k=0.209867354515083823715425816146,
    seed=np.array([5.85614508625736, 0.369331578246, -8.366536168329]),
)

# q = 0.9998 branch 1
_REF_Q0998 = dict(
    omega0=2.040286051080,
    a0=5.851767785486,
    k=0.210022792962,
    seed=np.array([5.851767785486, 0.370408600307, -8.360972934420]),
)

# What the closed-form integer formula produces for q=0.9998 (incorrect)
_WRONG_Q0998 = np.array([5.851767785486, 0.369965103164, -8.362475658982])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sys_integer():
    return get_system_by_id("chua_integer_saturation")


@pytest.fixture(scope="module")
def sys_fractional():
    return get_system_by_id("chua_fractional_saturation")  # q=0.9998 by default


# ---------------------------------------------------------------------------
# Test 1: modal seed for q=1 matches reference
# ---------------------------------------------------------------------------

def test_seed_integer_q1_modal_reference(sys_integer):
    """Modal seed for q=1 must match reference values with atol=5e-10."""
    ref = _REF_Q1
    seed_pos, seed_neg = build_lure_seed(
        sys_integer,
        A0=ref["a0"],
        omega0=ref["omega0"],
        k=ref["k"],
        seed_sign_convention="kuznetsov",
        q=1.0,
        transfer_mode="integer",
        theta=0.0,
        seed_construction="modal",
    )
    np.testing.assert_allclose(
        seed_pos, ref["seed"], atol=5e-10,
        err_msg="Modal seed for q=1 does not match reference.",
    )
    np.testing.assert_allclose(
        seed_neg, -ref["seed"], atol=5e-10,
        err_msg="seed_neg for q=1 is not -seed_pos.",
    )


# ---------------------------------------------------------------------------
# Test 2: modal seed for q=0.9998 matches reference
# ---------------------------------------------------------------------------

def test_seed_fractional_q0998_modal_reference(sys_fractional):
    """Modal seed for q=0.9998 must match reference values with atol=1e-8."""
    ref = _REF_Q0998
    seed_pos, seed_neg = build_lure_seed(
        sys_fractional,
        A0=ref["a0"],
        omega0=ref["omega0"],
        k=ref["k"],
        seed_sign_convention="kuznetsov",
        q=0.9998,
        transfer_mode="fractional",
        theta=0.0,
        seed_construction="modal",
    )
    np.testing.assert_allclose(
        seed_pos, ref["seed"], atol=1e-8,
        err_msg="Modal seed for q=0.9998 does not match reference.",
    )
    np.testing.assert_allclose(
        seed_neg, -ref["seed"], atol=1e-8,
        err_msg="seed_neg for q=0.9998 is not -seed_pos.",
    )


# ---------------------------------------------------------------------------
# Test 3: modal == closed-form for q=1
# ---------------------------------------------------------------------------

def test_modal_equals_closed_form_for_q1(sys_integer):
    """For q=1 the modal and closed-form seeds must agree within 1e-6."""
    ref = _REF_Q1
    seed_modal, _ = build_lure_seed(
        sys_integer, ref["a0"], ref["omega0"], ref["k"],
        q=1.0, transfer_mode="integer", seed_construction="modal",
    )
    seed_cf, _ = build_closed_form_integer_seed(
        sys_integer, ref["a0"], ref["omega0"], ref["k"],
    )
    np.testing.assert_allclose(
        seed_modal, seed_cf, atol=1e-6,
        err_msg="Modal and closed-form seeds disagree for q=1.",
    )


# ---------------------------------------------------------------------------
# Test 4: closed-form differs from modal for q=0.9998
# ---------------------------------------------------------------------------

def test_closed_form_differs_from_modal_for_q0998(sys_fractional):
    """The closed-form integer seed must differ from the correct modal seed
    for q=0.9998, and the modal seed must match the reference."""
    ref = _REF_Q0998
    seed_modal, _ = build_lure_seed(
        sys_fractional, ref["a0"], ref["omega0"], ref["k"],
        q=0.9998, transfer_mode="fractional", seed_construction="modal",
    )
    # The closed-form produces a different (wrong) result
    diff = np.linalg.norm(seed_modal - _WRONG_Q0998)
    assert diff > 1e-4, (
        f"Expected closed-form and modal seeds to differ for q=0.9998, "
        f"but ||modal - closed_form|| = {diff:.3e} is too small."
    )
    # Modal seed must still match reference
    np.testing.assert_allclose(
        seed_modal, ref["seed"], atol=1e-8,
        err_msg="Modal seed for q=0.9998 deviated from reference in closed-form comparison test.",
    )


# ---------------------------------------------------------------------------
# Test 5: closed_form_integer raises ValueError for q<1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q,transfer_mode", [
    (0.9998, "fractional"),
    (0.9998, "auto"),
    (0.95,   "fractional"),
])
def test_closed_form_raises_for_fractional(sys_fractional, q, transfer_mode):
    """seed_construction='closed_form_integer' must raise ValueError for q<1."""
    ref = _REF_Q0998
    with pytest.raises(ValueError, match="closed_form_integer"):
        build_lure_seed(
            sys_fractional, ref["a0"], ref["omega0"], ref["k"],
            q=q,
            transfer_mode=transfer_mode,
            seed_construction="closed_form_integer",
        )


# ---------------------------------------------------------------------------
# Test 6: eigenvector residual and normalisation
# ---------------------------------------------------------------------------

def test_modal_eigenvector_residual(sys_fractional):
    """Matched eigenvalue must be close to lambda0, and r^T v = 1 exactly."""
    ref = _REF_Q0998
    q = 0.9998
    X_seed, v, ev_matched, lam0 = build_modal_lure_seed(
        sys_fractional,
        A0=ref["a0"],
        omega0=ref["omega0"],
        k=ref["k"],
        q=q,
        transfer_mode="fractional",
        theta=0.0,
    )
    # Matched eigenvalue must be close to (i*omega0)^q
    assert abs(ev_matched - lam0) < 1e-8, (
        f"Matched eigenvalue {ev_matched} too far from lambda0={lam0}. "
        f"Difference = {abs(ev_matched - lam0):.3e}"
    )
    # Normalisation: r^T v == 1
    r = sys_fractional.r.astype(complex)
    scale = r @ v
    assert abs(scale - 1.0) < 1e-12, (
        f"r^T v = {scale} but expected 1.0 (error = {abs(scale - 1.0):.3e})"
    )


# ---------------------------------------------------------------------------
# Test 7: seed_neg == -seed_pos for both constructions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("construction,q,tm", [
    ("modal",              1.0,    "integer"),
    ("modal",              0.9998, "fractional"),
    ("closed_form_integer",1.0,    "integer"),
])
def test_seed_neg_is_minus_seed_pos(sys_integer, sys_fractional, construction, q, tm):
    """seed_neg must equal -seed_pos for all valid construction modes."""
    system = sys_integer if q == 1.0 else sys_fractional
    ref = _REF_Q1 if q == 1.0 else _REF_Q0998
    seed_pos, seed_neg = build_lure_seed(
        system, ref["a0"], ref["omega0"], ref["k"],
        q=q, transfer_mode=tm, seed_construction=construction,
    )
    np.testing.assert_array_equal(
        seed_neg, -seed_pos,
        err_msg=f"seed_neg != -seed_pos for construction={construction!r}, q={q}.",
    )
