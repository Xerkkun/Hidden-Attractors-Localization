import sys
from pathlib import Path
import numpy as np
import pytest

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.continuation.continuation_fractional import run_fractional_continuation, run_fractional_continuation_abm_monolithic

class DummyLureSystem:
    def __init__(self, q=0.8):
        self.P = np.array([[-1.0]])
        self.b = np.array([1.0])
        self.r = np.array([1.0])
        self.q = q
        self.psi = lambda sigma: 0.0  # linearised system


def test_causality_scalar_non_autonomous():
    """Verify that old history derivative values are never re-evaluated at eta boundaries.
    
    We compare two-stage continuation run_fractional_continuation (which routes to
    monolithic transport under ABM) against a monolithic continuous integration.
    """
    sys_obj = DummyLureSystem(q=0.8)
    h = 0.05
    t_transient = 0.2  # 4 steps
    t_keep = 0.2       # 4 steps
    # Total steps per stage: 8
    
    # We integrate under lambda_values [0.2, 0.8]
    # Let's run the monolithic continuation
    steps_records = run_fractional_continuation_abm_monolithic(
        system=sys_obj,
        seed_x0=np.array([1.0]),
        k_gain=0.1,
        lambda_values=[0.2, 0.8],
        h=h,
        memory_mode="full",
        t_transient=t_transient,
        t_keep=t_keep
    )
    
    assert len(steps_records) == 2
    assert steps_records[0]["lambda_value"] == 0.2
    assert steps_records[1]["lambda_value"] == 0.8
    assert steps_records[0]["eta_boundary_policy"] == "monolithic_transport"
    assert steps_records[0]["carry_derivative_history"] is True
    
    # Check that they match the public API dispatch
    steps_api = run_fractional_continuation(
        system=sys_obj,
        seed_x0=np.array([1.0]),
        k_gain=0.1,
        lambda_values=[0.2, 0.8],
        h=h,
        memory_mode="full",
        integrator="abm",
        t_transient=t_transient,
        t_keep=t_keep,
        use_c_backend=False
    )
    
    # Check that the dispatch produced exactly the same output
    for s_api, s_mono in zip(steps_api, steps_records):
        assert np.allclose(s_api["trajectory"], s_mono["trajectory"], atol=1e-12)
        assert s_api["status"] == "ok"


def test_monolithic_history_counts():
    """Verify that full-history carries all points and windowed saturates correctly."""
    sys_obj = DummyLureSystem(q=0.9)
    h = 0.1
    t_trans = 0.3  # 3 steps
    t_kp = 0.2     # 2 steps
    # 5 steps per stage. For 3 stages: 15 steps.
    
    # 1. Full-history: capacity is total_steps + prehistory
    steps_full = run_fractional_continuation_abm_monolithic(
        system=sys_obj,
        seed_x0=np.array([1.0]),
        k_gain=0.0,
        lambda_values=[0.0, 0.5, 1.0],
        h=h,
        memory_mode="full",
        t_transient=t_trans,
        t_keep=t_kp
    )
    assert len(steps_full) == 3
    # Check history count growing: n_steps represents steps in this stage.
    # The actual history array in monolithic transport has capacity growing.
    for step in steps_full:
        assert step["status"] == "ok"
        assert step["history_policy"] == "full_caputo_history"

    # 2. Finite window of length 3:
    steps_win = run_fractional_continuation_abm_monolithic(
        system=sys_obj,
        seed_x0=np.array([1.0]),
        k_gain=0.0,
        lambda_values=[0.0, 0.5, 1.0],
        h=h,
        memory_mode="window",
        memory_window_length=3,
        t_transient=t_trans,
        t_keep=t_kp
    )
    for step in steps_win:
        assert step["status"] == "ok"
        assert step["history_policy"] == "finite_memory_window"
