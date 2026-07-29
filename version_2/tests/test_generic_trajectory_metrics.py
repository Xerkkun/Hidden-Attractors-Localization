from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.analysis import (
    RobustnessCase,
    compute_trajectory_metrics,
    state_view,
    trajectory_metrics_for_system,
)


def test_robustness_case_requires_an_explicit_neutral_contract() -> None:
    with pytest.raises(TypeError):
        RobustnessCase("case")

    baseline = RobustnessCase(
        case_id="baseline",
        q=0.9,
        h=0.02,
        Lm=3.0,
        t_final=20.0,
        t_burn=2.0,
    )
    comparison = RobustnessCase(
        case_id="comparison",
        q=0.9,
        h=0.01,
        Lm=3.0,
        t_final=20.0,
        t_burn=2.0,
    )
    assert comparison.as_dict(baseline)["h_change_pct"] == pytest.approx(-50.0)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"case_id": ""}, "case_id"),
        ({"q": 0.0}, "q"),
        ({"h": 0.0}, "h"),
        ({"Lm": -1.0}, "Lm"),
        ({"t_final": 0.0}, "t_final"),
        ({"t_burn": 20.0}, "t_burn"),
    ],
)
def test_robustness_case_rejects_invalid_values(changes, message) -> None:
    values = {
        "case_id": "case",
        "q": 0.9,
        "h": 0.02,
        "Lm": 3.0,
        "t_final": 20.0,
        "t_burn": 2.0,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        RobustnessCase(**values)


def test_explicit_metrics_support_two_dimensional_states() -> None:
    times = np.linspace(0.0, 2.0, 201)
    states = np.column_stack((np.sin(times), np.cos(times)))

    result = compute_trajectory_metrics(times, states, t_start=1.0)

    assert result["dimension"] == 2
    assert result["n_samples"] == 201
    assert result["n_tail_samples"] == 101
    assert result["bounded"] is True
    assert result["evidence_status"] == "finite_time_trajectory_diagnostic"
    assert "range_0" in result and "range_1" in result
    assert "range_2" not in result


def test_timed_three_column_trajectory_does_not_count_time_as_state() -> None:
    times = np.linspace(0.0, 2.0, 201)
    trajectory = np.column_stack((times, np.sin(times), np.cos(times)))
    equilibria = {"origin": np.zeros(2)}

    result = trajectory_metrics_for_system(
        trajectory,
        equilibria=equilibria,
        h=0.01,
        t_start=1.0,
    )

    assert result["dimension"] == 2


def test_four_dimensional_pure_state_array_is_not_sliced_by_shape() -> None:
    states = np.arange(40, dtype=float).reshape(10, 4)

    assert state_view(states).shape == (10, 4)
    assert state_view(states, has_time=True).shape == (10, 3)


def test_generic_metrics_reject_ambiguous_or_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_trajectory_metrics(
            np.array([0.0, 0.0]),
            np.zeros((2, 1)),
        )
    with pytest.raises(ValueError, match="shape"):
        compute_trajectory_metrics(
            np.array([0.0, 1.0]),
            np.zeros((3, 1)),
        )
