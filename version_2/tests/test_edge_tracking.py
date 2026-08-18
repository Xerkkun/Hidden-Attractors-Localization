from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.verification.destination_classifier import DestinationClassification
from hidden_attractors.verification.edge_tracking import (
    EdgeDestination,
    EdgeTrackingConfig,
    ScaledCylindricalGeometry,
    ScaledEuclideanGeometry,
    edge_destination_from_classification,
    track_edge_bracket,
)


def test_scaled_euclidean_geometry_uses_declared_scale() -> None:
    geometry = ScaledEuclideanGeometry((2.0, 4.0))
    left = np.array([0.0, 0.0])
    right = np.array([2.0, 4.0])

    assert geometry.name == "scaled_euclidean"
    assert geometry.distance(left, right) == pytest.approx(np.sqrt(2.0))
    assert np.allclose(geometry.midpoint(left, right), [1.0, 2.0])


def test_cylindrical_midpoint_uses_short_arc_across_cut() -> None:
    geometry = ScaledCylindricalGeometry(
        scale=(1.0, np.pi),
        periodic_indices=(1,),
        periods=(2.0 * np.pi,),
    )
    left = np.array([0.0, 3.0])
    right = np.array([0.0, -3.0])
    midpoint = geometry.midpoint(left, right)

    assert midpoint[1] == pytest.approx(np.pi)
    assert geometry.distance(left, midpoint) == pytest.approx(
        0.5 * geometry.distance(left, right)
    )


def test_cylindrical_midpoint_rejects_antipodal_chart() -> None:
    geometry = ScaledCylindricalGeometry(
        scale=(1.0, np.pi),
        periodic_indices=(1,),
        periods=(2.0 * np.pi,),
    )

    with pytest.raises(ValueError, match="antipodal"):
        geometry.midpoint([0.0, 0.0], [0.0, np.pi])


def test_edge_tracking_on_cylinder_refines_across_phase_cut() -> None:
    geometry = ScaledCylindricalGeometry(
        scale=(1.0, np.pi),
        periodic_indices=(1,),
        periods=(2.0 * np.pi,),
    )

    def evaluator(state: np.ndarray, _context: object) -> EdgeDestination:
        return EdgeDestination.terminal("upper" if np.sin(state[1]) >= 0.0 else "lower")

    result = track_edge_bracket(
        [0.0, 3.0],
        [0.0, -3.0],
        evaluator=evaluator,
        geometry=geometry,
        bracket_id="phase-cut",
        config=EdgeTrackingConfig(tolerance=1.0e-8, max_iterations=40),
    )

    assert result.status == "converged"
    candidate_angle = result.candidate[1]
    assert abs(((candidate_angle - np.pi) + np.pi) % (2.0 * np.pi) - np.pi) <= 1.0e-7
    assert abs(candidate_angle) > 3.0


def test_edge_bisection_converges_and_confirms_b1_b2_endpoints() -> None:
    geometry = ScaledEuclideanGeometry((1.0,))

    def evaluator(state: np.ndarray, _context: object) -> EdgeDestination:
        return EdgeDestination.terminal("left" if state[0] < 0.3 else "right")

    result = track_edge_bracket(
        [-1.0],
        [1.0],
        evaluator=evaluator,
        geometry=geometry,
        bracket_id="synthetic-transition",
        config=EdgeTrackingConfig(tolerance=1.0e-7, max_iterations=40),
    )

    assert result.status == "converged"
    assert result.converged
    assert len(result.confirmations) == 4
    assert result.left_destination == "left"
    assert result.right_destination == "right"
    assert result.final_width <= 1.0e-7
    assert abs(result.candidate[0] - 0.3) <= 1.0e-7
    assert result.method == "initial_data_boundary_bisection"
    assert result.finite_resolution_only is True


def test_max_iterations_reports_midpoint_of_final_bracket() -> None:
    geometry = ScaledEuclideanGeometry((1.0,))

    def evaluator(state: np.ndarray, _context: object) -> EdgeDestination:
        return EdgeDestination.terminal("left" if state[0] < 0.3 else "right")

    result = track_edge_bracket(
        [-1.0],
        [1.0],
        evaluator=evaluator,
        geometry=geometry,
        bracket_id="one-refinement",
        config=EdgeTrackingConfig(tolerance=1.0e-12, max_iterations=1),
    )

    assert result.status == "max_iterations"
    assert result.final_left == pytest.approx((0.0,))
    assert result.final_right == pytest.approx((1.0,))
    assert result.candidate == pytest.approx((0.5,))


def test_ambiguous_midpoint_stops_after_three_refinements_without_forcing_label() -> None:
    geometry = ScaledEuclideanGeometry((1.0,))

    def evaluator(state: np.ndarray, _context: object) -> EdgeDestination:
        if abs(state[0]) <= 1.0e-14:
            return EdgeDestination.ambiguous(reason="synthetic separatrix")
        return EdgeDestination.terminal("left" if state[0] < 0.0 else "right")

    result = track_edge_bracket(
        [-1.0],
        [1.0],
        evaluator=evaluator,
        geometry=geometry,
        bracket_id="synthetic-ambiguous",
        config=EdgeTrackingConfig(tolerance=1.0e-12, max_iterations=20),
    )

    assert result.status == "ambiguous_limit"
    assert len(result.iterations) == 3
    assert [item.ambiguous_streak for item in result.iterations] == [1, 2, 3]
    assert result.iterations[0].event == "ambiguous_midpoint_refined"
    assert result.iterations[1].event == "ambiguous_midpoint_refined"
    assert result.iterations[2].event == "ambiguous_midpoint"
    assert result.candidate == pytest.approx((0.0,))


def test_endpoint_destination_must_be_consistent_across_budgets() -> None:
    geometry = ScaledEuclideanGeometry((1.0,))

    def evaluator(state: np.ndarray, context: object) -> EdgeDestination:
        budget = getattr(context, "budget_level")
        if state[0] < 0.0:
            return EdgeDestination.terminal("left" if budget == "B1" else "right")
        return EdgeDestination.terminal("right")

    result = track_edge_bracket(
        [-1.0],
        [1.0],
        evaluator=evaluator,
        geometry=geometry,
        bracket_id="inconsistent",
    )

    assert result.status == "rejected_endpoint_inconsistent"
    assert result.iterations == ()


def test_third_destination_stops_binary_edge_refinement() -> None:
    geometry = ScaledEuclideanGeometry((1.0,))

    def evaluator(state: np.ndarray, _context: object) -> EdgeDestination:
        if state[0] < -0.1:
            return EdgeDestination.terminal("basin_a")
        if state[0] > 0.1:
            return EdgeDestination.terminal("basin_b")
        return EdgeDestination.terminal("basin_c")

    result = track_edge_bracket(
        [-1.0],
        [1.0],
        evaluator=evaluator,
        geometry=geometry,
        bracket_id="three-basins",
    )

    assert result.status == "third_destination"
    assert result.iterations[-1].midpoint_outcome.label == "basin_c"


def test_multiple_transitions_on_segment_are_not_arbitrarily_selected() -> None:
    geometry = ScaledEuclideanGeometry((1.0,))

    def evaluator(state: np.ndarray, _context: object) -> EdgeDestination:
        value = state[0]
        if abs(value) <= 1.0e-14:
            return EdgeDestination.ambiguous(reason="central unresolved datum")
        if value < -0.75 or 0.0 < value < 0.75:
            return EdgeDestination.terminal("basin_a")
        return EdgeDestination.terminal("basin_b")

    result = track_edge_bracket(
        [-1.0],
        [1.0],
        evaluator=evaluator,
        geometry=geometry,
        bracket_id="multiple-crossings",
    )

    assert result.status == "multiple_transitions"
    assert result.iterations[-1].event == "multiple_transitions"


def test_inherited_caputo_requires_named_admissible_history_family() -> None:
    with pytest.raises(ValueError, match="admissible_history_family_id"):
        EdgeTrackingConfig(data_semantics="admissible_history_family_parameter")

    contract = EdgeTrackingConfig(
        data_semantics="admissible_history_family_parameter",
        admissible_history_family_id="common_past_scalar_perturbation_v1",
    )
    assert contract.admissible_history_family_id == "common_past_scalar_perturbation_v1"


def test_unified_classifier_adapter_keeps_transient_unresolved() -> None:
    classification = DestinationClassification(
        label="transient",
        destination_id="transient:unresolved",
        subtype="long_transient",
        confidence=0.75,
        is_ambiguous=False,
        reasons=("horizon insufficient",),
        metrics={"tail_drift": 0.2},
        evidence_status="finite_time",
    )
    outcome = edge_destination_from_classification(classification)

    assert outcome.label == "transient:unresolved"
    assert outcome.resolved is False
    assert outcome.evaluation_ok is True
    assert outcome.metadata["classifier_confidence"] == pytest.approx(0.75)


def test_unified_classifier_adapter_prefers_specific_destination_id() -> None:
    outcome = edge_destination_from_classification(
        {
            "label": "equilibrium",
            "destination_id": "equilibrium:E_focus",
            "confidence": 0.99,
            "is_ambiguous": False,
        }
    )

    assert outcome.label == "equilibrium:E_focus"
    assert outcome.resolved is True
    assert outcome.metadata["classifier_label"] == "equilibrium"
