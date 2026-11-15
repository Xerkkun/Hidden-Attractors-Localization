"""Published rail anchoring uses physical time and retains legacy execution."""
from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pytest

from hidden_attractors.analysis.lyapunov_cloned import (
    LEGACY_EXECUTION_PROTOCOL, REFERENCE_RAIL_PROTOCOL, compute_cloned_dynamics_spectrum,
)
from hidden_attractors.integrations.abm_reference import integrate_reference_abm


def test_integer_linear_exact_reference_uses_reference_step_not_clone_index():
    reference_times = np.arange(9) * .05
    reference_states = np.exp(-reference_times[:, None])
    result = compute_cloned_dynamics_spectrum(
        lambda t, x: -x, np.ones(1), [1], .01, .1, 1, 4, 1e-4,
        execution_protocol=REFERENCE_RAIL_PROTOCOL,
        reference_times=reference_times, reference_states=reference_states,
        return_history=True,
    )
    assert result.status == "ok"
    history = result.method_metadata["history"]
    for block, record in enumerate(history):
        assert record["reference_anchor_time"] == pytest.approx(block * .1)
        np.testing.assert_array_equal(record["reference_anchor_state"], reference_states[block * 2])
    assert result.exponents[0] == pytest.approx(-1, abs=.001)
    assert result.method_metadata["reference_rail"]["provided_memory_policy_verified_by_solver"] is False


def test_fractional_reference_is_full_history_while_clone_fiducials_restart():
    rhs = lambda t, x: -.3 * x + .1
    times, states, status = integrate_reference_abm(rhs, np.array([.6]), [.7], .01, .6, convolution_block_size=7)
    assert status == "ok"
    rail = compute_cloned_dynamics_spectrum(
        rhs, np.array([.6]), [.7], .01, .2, 1, 3, 1e-4,
        execution_protocol=REFERENCE_RAIL_PROTOCOL, reference_times=times,
        reference_states=states, return_history=True)
    legacy = compute_cloned_dynamics_spectrum(
        rhs, np.array([.6]), [.7], .01, .2, 1, 3, 1e-4,
        execution_protocol=LEGACY_EXECUTION_PROTOCOL, return_history=True)
    assert rail.status == legacy.status == "ok"
    history = rail.method_metadata["history"]
    np.testing.assert_array_equal(history[2]["reference_anchor_state"], states[40])
    assert abs(history[1]["fiducial_state"][0] - states[40, 0]) > .001
    assert abs(legacy.method_metadata["history"][1]["fiducial_state"][0] - history[2]["reference_anchor_state"][0]) > .001
    assert rail.method_metadata["clone_lower_limit"] == "each_block_start"


def test_grid_misalignment_is_rejected_without_explicit_interpolation():
    times = np.arange(4) * .05
    kwargs = dict(rhs=lambda t, x: np.ones(1), x0=np.zeros(1), orders=[1], h=.01,
                  t_clone=.03, n_clones=1, k_blocks=5, delta=1e-4,
                  execution_protocol=REFERENCE_RAIL_PROTOCOL,
                  reference_times=times, reference_states=times[:, None])
    with pytest.raises(ValueError, match="clone boundary misses"):
        compute_cloned_dynamics_spectrum(**kwargs)
    result = compute_cloned_dynamics_spectrum(**kwargs, reference_sampling="linear", return_history=True)
    assert result.status == "ok"
    assert result.method_metadata["history"][1]["reference_anchor_state"][0] == pytest.approx(.03)
    assert result.method_metadata["reference_rail"]["sampling"] == "linear"


def test_reference_failure_is_reported_before_clone_accumulation():
    result = compute_cloned_dynamics_spectrum(
        lambda t, x: np.ones(1), np.zeros(1), [.5], .01, .2, 1, 2, 1e-4,
        execution_protocol=REFERENCE_RAIL_PROTOCOL, reference_h=.01, divergence_norm=.2)
    assert result.status == "reference_numerical_failure"
    assert len(result.times) == 0
    assert np.all(np.isnan(result.exponents))
    assert result.method_metadata["reference_rail"]["status"] == "diverged"


def test_new_rail_generation_matches_separately_computed_fixed_t0_reference():
    kwargs = dict(rhs=lambda t, x: -.1 * x, x0=np.ones(1), orders=[.9], h=.01,
                  t_clone=.1, n_clones=1, k_blocks=4, delta=1e-4,
                  execution_protocol=REFERENCE_RAIL_PROTOCOL, reference_h=.05,
                  reference_convolution_block_size=3, return_history=True)
    result = compute_cloned_dynamics_spectrum(**kwargs)
    assert result.status == "ok"
    reference = result.method_metadata["reference_rail"]
    assert reference["samples"] == 9
    assert reference["memory_protocol"] == "fixed_lower_limit_full_history"
    assert reference["fft_blocks_restart_memory"] is False
    for block, record in enumerate(result.method_metadata["history"]):
        np.testing.assert_array_equal(record["reference_anchor_state"], result.reference_states[block * 2])


def test_runner_consumes_financial_reference_contract_and_preserves_legacy_override(monkeypatch):
    validation_python = Path(__file__).resolve().parents[1] / "validation/python"
    sys.path.insert(0, str(validation_python))
    import cloned_dynamics_benchmarks as benchmarks
    captured = []

    def fake_solver(*args, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(exponents=np.array([.1, -.01, -1]), status="ok")

    monkeypatch.setattr(benchmarks, "compute_cloned_dynamics_spectrum", fake_solver)
    case = {"system": {"kind": "financial", "parameters": {"a": 1, "b": .15, "c": 1}},
            "method_id": benchmarks.METHOD_ID,
            "integration": {"h": .05, "T_ref": 5000, "N": 100000,
                            "h_clone": .01, "t_clone": 10, "K": 500, "delta": .001,
                            "memory_protocol": "published_block_restart"},
            "expected": {"tolerance_abs_target": .05, "tolerance_abs_initial": .15,
                         "sign_pattern_required": True}}
    row = {"orders": [1, 1, 1], "type": "Comm", "K01": 1, "lyapunov": [.1, -.01, -1]}
    current, _ = benchmarks.run_benchmark_row(case, row)
    legacy, _ = benchmarks.run_benchmark_row(case, row, execution_protocol=LEGACY_EXECUTION_PROTOCOL)
    assert current["execution_protocol"] == REFERENCE_RAIL_PROTOCOL
    assert captured[0]["reference_h"] == .05
    assert captured[0]["h"] == .01
    assert legacy["execution_protocol"] == LEGACY_EXECUTION_PROTOCOL
    assert "reference_h" not in captured[1]
