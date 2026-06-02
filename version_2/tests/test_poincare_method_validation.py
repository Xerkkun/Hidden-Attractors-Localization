"""Synthetic validation for Poincare crossing detection and interpolation."""

from __future__ import annotations

import math

import numpy as np

from hidden_attractors.analysis.poincare import (
    ALLOWED_INTERPRETATION_LABELS,
    detect_poincare_crossings,
    summarize_poincare_points,
    write_poincare_outputs,
)


def _sine_trajectory(h: float = 0.07, periods: int = 5) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(-0.4, periods * 2.0 * math.pi + 0.4, h)
    return times, np.column_stack((np.sin(times), np.cos(times)))


def test_positive_sine_crossings_are_linearly_interpolated() -> None:
    h = 0.07
    times, trajectory = _sine_trajectory(h=h)
    result = detect_poincare_crossings(
        times,
        trajectory,
        direction="positive",
        derivative_mode="finite_difference_diagnostic",
        burn_time=0.1,
    )
    expected = np.arange(1, 5) * 2.0 * math.pi
    assert result.crossing_count >= expected.size
    assert np.allclose(result.points[: expected.size, 0], 0.0, atol=1.0e-10)
    assert np.allclose(result.points[: expected.size, 1], 1.0, atol=2.0e-3)
    assert np.all(np.abs(result.crossing_times[: expected.size] - expected) < h**2)
    assert set(result.crossing_directions) == {"positive"}


def test_negative_sine_crossings_are_not_confused_with_positive_crossings() -> None:
    h = 0.07
    times, trajectory = _sine_trajectory(h=h)
    result = detect_poincare_crossings(
        times,
        trajectory,
        direction="negative",
        derivative_mode="finite_difference_diagnostic",
        burn_time=0.1,
    )
    expected = math.pi + np.arange(0, 5) * 2.0 * math.pi
    assert np.all(np.abs(result.crossing_times[: expected.size] - expected) < h**2)
    assert np.allclose(result.points[: expected.size, 1], -1.0, atol=2.0e-3)
    assert set(result.crossing_directions) == {"negative"}


def test_no_crossings_status() -> None:
    times = np.linspace(0.0, 10.0, 1001)
    trajectory = np.column_stack((1.0 + 0.1 * np.sin(times), np.cos(times)))
    result = detect_poincare_crossings(
        times,
        trajectory,
        derivative_mode="finite_difference_diagnostic",
    )
    assert result.crossing_count == 0
    assert result.status == "no_crossings"


def test_integer_rhs_direction_filters_crossings() -> None:
    times, trajectory = _sine_trajectory()

    def oscillator_rhs(_time: float, state: np.ndarray) -> np.ndarray:
        return np.array([state[1], -state[0]])

    positive = detect_poincare_crossings(times, trajectory, rhs=oscillator_rhs)
    negative = detect_poincare_crossings(
        times,
        trajectory,
        direction="negative",
        derivative_mode="integer_rhs",
        rhs=oscillator_rhs,
    )
    assert positive.crossing_count > 0
    assert np.all(positive.points[:, 1] > 0.0)
    assert negative.crossing_count > 0
    assert np.all(negative.points[:, 1] < 0.0)


def test_fractional_mode_is_geometric_and_does_not_require_rhs() -> None:
    times, trajectory = _sine_trajectory()
    result = detect_poincare_crossings(
        times,
        trajectory,
        direction="positive_geometric_crossing",
        derivative_mode="geometric_fractional",
    )
    assert result.crossing_count > 0
    assert result.section_metadata["caputo_geometric_crossing"] is True
    assert result.section_metadata["exact_poincare_map"] is False
    assert result.section_metadata["uses_classical_rhs_direction"] is False


def test_linear_interpolation_hits_section_exactly() -> None:
    result = detect_poincare_crossings(
        [0.0, 1.0],
        [[-2.0, 4.0], [3.0, 9.0]],
        derivative_mode="finite_difference_diagnostic",
    )
    assert result.crossing_count == 1
    assert abs(result.points[0, 0]) < 1.0e-15
    assert abs(result.crossing_times[0] - 0.4) < 1.0e-15


def test_minimum_crossing_separation_filters_close_crossings() -> None:
    result = detect_poincare_crossings(
        [0.0, 0.2, 0.4, 0.6, 0.8],
        [[-1.0], [1.0], [-1.0], [1.0], [-1.0]],
        derivative_mode="finite_difference_diagnostic",
        min_crossing_separation=0.5,
    )
    assert result.crossing_count == 1
    assert result.section_metadata["filtered_by_min_crossing_separation"] == 1


def test_outputs_never_claim_false_certification(tmp_path) -> None:
    result = detect_poincare_crossings(
        [0.0, 1.0],
        [[-1.0, 0.0], [1.0, 1.0]],
        derivative_mode="geometric_fractional",
    )
    write_poincare_outputs(tmp_path, result)
    combined = " ".join(path.read_text(encoding="utf-8") for path in tmp_path.iterdir())
    assert '"chaos_verified": true' not in combined
    assert '"hidden_verified": true' not in combined
    assert '"periodic_orbit_exact": true' not in combined
    assert '"caputo_periodic_orbit_exact": true' not in combined
    assert summarize_poincare_points(result.points)["interpretation_label"] in ALLOWED_INTERPRETATION_LABELS
