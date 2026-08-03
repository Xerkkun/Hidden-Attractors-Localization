"""Isolated mathematical tests for integer-order SALI/GALI/LDI."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hidden_attractors.analysis.alignment_indices as alignment_module
from hidden_attractors.analysis.alignment_indices import (
    AlignmentIndexResult,
    alignment_indices_from_tangent_history,
    generalized_alignment_index,
    integer_flow_alignment_indices,
    integer_map_alignment_indices,
    integer_system_alignment_indices,
    linear_dependence_index,
    smaller_alignment_index,
)
from hidden_attractors.systems.base import ChaoticSystem


def _unit_columns(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=0)


def _generic_deviations() -> np.ndarray:
    return _unit_columns(
        np.array(
            [
                [1.0, 1.0, -0.5],
                [0.5, -1.0, 1.0],
                [-0.25, 0.75, 1.0],
            ]
        )
    )


def test_rotation_history_uses_public_samples_vectors_dimension_layout() -> None:
    angles = np.linspace(0.0, 1.25, 6)
    samples = []
    for angle in angles:
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        samples.append(np.stack((rotation[:, 0], rotation[:, 1]), axis=0))

    result = alignment_indices_from_tangent_history(
        np.stack(samples),
        coordinates=angles,
        gali_orders=(2,),
        backend="numpy",
    )

    assert isinstance(result, AlignmentIndexResult)
    assert result.status == "ok"
    assert result.initial_deviations.shape == (2, 2)
    assert result.final_deviations.shape == (2, 2)
    assert np.allclose(result.sali, np.sqrt(2.0), atol=2.0e-15)
    assert np.allclose(result.gali[:, 0], 1.0, atol=2.0e-15)
    assert not np.any(result.censored)
    assert result.metadata["history_layout"] == "samples_vectors_dimension"
    assert result.volume_method == "svd_product"


def test_sali_gali2_identity_and_ldi_gali_equivalence() -> None:
    vectors = _generic_deviations()
    first = vectors[:, 0]
    second = vectors[:, 1]
    sali = smaller_alignment_index(vectors)
    gali2 = generalized_alignment_index(vectors, order=2, backend="numpy")

    assert 2.0 * gali2 == pytest.approx(
        np.linalg.norm(first + second) * np.linalg.norm(first - second),
        rel=2.0e-15,
        abs=2.0e-15,
    )
    assert linear_dependence_index(vectors, order=3, backend="numpy") == pytest.approx(
        generalized_alignment_index(vectors, order=3, backend="numpy"),
        rel=2.0e-15,
        abs=2.0e-15,
    )


def test_log_gali_survives_linear_volume_underflow_and_rank_is_minus_infinity() -> None:
    full_rank_tiny = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.0, 1.0e-200, 0.0],
            [0.0, 0.0, 1.0e-200],
        ]
    )
    underflow = alignment_indices_from_tangent_history(
        full_rank_tiny.T[None, :, :],
        gali_orders=(3,),
        backend="numpy",
    )

    assert underflow.gali[0, 0] == 0.0
    assert np.isfinite(underflow.log_gali[0, 0])
    assert underflow.log_gali[0, 0] < -900.0
    assert underflow.censored[0, 0]

    rank_deficient = np.array([[1.0, 1.0], [0.0, 0.0]])
    singular = alignment_indices_from_tangent_history(
        rank_deficient.T[None, :, :],
        gali_orders=(2,),
        backend="numpy",
    )
    assert singular.gali[0, 0] == 0.0
    assert singular.log_gali[0, 0] == -np.inf
    assert singular.censored[0, 0]


def test_numba_householder_backend_matches_numpy_svd_reference() -> None:
    if not alignment_module.NUMBA_AVAILABLE:
        pytest.skip("Numba is not importable")
    vectors = _generic_deviations()
    try:
        numba_value = generalized_alignment_index(vectors, order=3, backend="numba")
    except RuntimeError:
        pytest.skip("Numba alignment backend is not operational")
    numpy_value = generalized_alignment_index(vectors, order=3, backend="numpy")

    assert numba_value == pytest.approx(numpy_value, rel=2.0e-13, abs=2.0e-15)
    history = np.stack((vectors.T, (np.diag([2.0, 1.0, 0.5]) @ vectors).T))
    numpy_result = alignment_indices_from_tangent_history(
        history,
        gali_orders=(2, 3),
        backend="numpy",
    )
    numba_result = alignment_indices_from_tangent_history(
        history,
        gali_orders=(2, 3),
        backend="numba",
    )
    assert np.allclose(numba_result.gali, numpy_result.gali, rtol=2.0e-13, atol=2.0e-15)
    assert np.allclose(numba_result.log_gali, numpy_result.log_gali, rtol=2.0e-13, atol=2.0e-15)
    assert numba_result.volume_method == "householder_qr_log_volume"


def test_integer_rotation_map_keeps_exact_alignment_indices() -> None:
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    result = integer_map_alignment_indices(
        lambda state: rotation @ state,
        lambda _state: rotation,
        np.zeros(2),
        iterations=12,
        initial_deviations=np.eye(2),
        gali_orders=(2,),
        backend="numpy",
    )

    assert result.status == "ok"
    assert np.allclose(result.sali, np.sqrt(2.0), atol=2.0e-15)
    assert np.allclose(result.gali[:, 0], 1.0, atol=2.0e-15)
    assert np.array_equal(result.coordinates, np.arange(13, dtype=float))


def test_hyperbolic_map_uses_exact_jacobian_recurrence_without_qr() -> None:
    matrix = np.diag([2.0, 1.0, 0.5])
    initial = _generic_deviations()
    result = integer_map_alignment_indices(
        lambda state: matrix @ state,
        lambda _state: matrix,
        np.zeros(3),
        iterations=18,
        initial_deviations=initial,
        gali_orders=(2, 3),
        backend="numpy",
    )

    expected_sali = []
    expected_gali = []
    for iteration in result.coordinates.astype(int):
        exact = _unit_columns(np.linalg.matrix_power(matrix, iteration) @ initial)
        expected_sali.append(smaller_alignment_index(exact, normalize=False))
        expected_gali.append(
            [
                generalized_alignment_index(exact, order=2, backend="numpy", normalize=False),
                generalized_alignment_index(exact, order=3, backend="numpy", normalize=False),
            ]
        )

    assert result.status == "ok"
    assert np.allclose(result.sali, expected_sali, rtol=2.0e-13, atol=2.0e-15)
    assert np.allclose(result.gali, expected_gali, rtol=3.0e-12, atol=2.0e-15)
    sali_slope = np.polyfit(result.coordinates[-8:], result.log_sali[-8:], 1)[0]
    gali3_slope = np.polyfit(result.coordinates[-8:], result.log_gali[-8:, 1], 1)[0]
    assert sali_slope == pytest.approx(-np.log(2.0), abs=2.0e-3)
    assert gali3_slope == pytest.approx(-np.log(8.0), abs=3.0e-3)
    assert result.orthonormalization == "none_during_evolution"


def test_diagonal_flow_matches_exact_normalized_variational_solution() -> None:
    matrix = np.diag([0.4, -0.2, -0.8])
    initial = _generic_deviations()
    result = integer_flow_alignment_indices(
        lambda state: matrix @ state,
        lambda _state: matrix,
        np.zeros(3),
        t_final=4.0,
        renormalization_time=0.25,
        initial_deviations=initial,
        gali_orders=(2, 3),
        rtol=1.0e-11,
        atol=1.0e-13,
        max_step=0.05,
        backend="numpy",
    )

    expected_sali = []
    expected_gali = []
    rates = np.diag(matrix)
    for time in result.coordinates:
        exact = _unit_columns(np.exp(rates * time)[:, None] * initial)
        expected_sali.append(smaller_alignment_index(exact, normalize=False))
        expected_gali.append(
            [
                generalized_alignment_index(exact, order=2, backend="numpy", normalize=False),
                generalized_alignment_index(exact, order=3, backend="numpy", normalize=False),
            ]
        )

    assert result.status == "ok"
    assert result.coordinates[-1] == pytest.approx(4.0)
    assert np.allclose(result.sali, expected_sali, rtol=2.0e-10, atol=2.0e-12)
    assert np.allclose(result.gali, expected_gali, rtol=3.0e-10, atol=2.0e-12)
    assert result.metadata["jacobian_source"] == "analytic"


def test_multi_particle_linear_map_matches_variational_method() -> None:
    matrix = np.diag([1.8, 0.9, 0.4])
    initial = _generic_deviations()
    common = dict(
        iterations=14,
        initial_deviations=initial,
        gali_orders=(2, 3),
        backend="numpy",
    )
    variational = integer_map_alignment_indices(
        lambda state: matrix @ state,
        lambda _state: matrix,
        np.zeros(3),
        method="variational",
        **common,
    )
    multi_particle = integer_map_alignment_indices(
        lambda state: matrix @ state,
        None,
        np.zeros(3),
        method="multi_particle",
        **common,
    )

    assert multi_particle.status == "ok"
    assert np.allclose(multi_particle.sali, variational.sali, rtol=3.0e-9, atol=2.0e-12)
    assert np.allclose(multi_particle.gali, variational.gali, rtol=3.0e-8, atol=2.0e-12)
    assert multi_particle.metadata["deviation_size"] == pytest.approx(np.sqrt(np.finfo(float).eps))
    assert multi_particle.metadata["renormalization_interval_iterations"] == 1


def test_multi_particle_linear_flow_matches_variational_and_counts_physical_calls() -> None:
    matrix = np.diag([0.3, -0.1, -0.7])
    initial = _generic_deviations()
    common = dict(
        t_final=1.5,
        renormalization_time=0.25,
        initial_deviations=initial,
        gali_orders=(2, 3),
        rtol=1.0e-11,
        atol=1.0e-13,
        max_step=0.05,
        backend="numpy",
    )
    variational = integer_flow_alignment_indices(
        lambda state: matrix @ state,
        lambda _state: matrix,
        np.zeros(3),
        method="variational",
        **common,
    )
    multi_particle = integer_flow_alignment_indices(
        lambda state: matrix @ state,
        None,
        np.zeros(3),
        method="multi_particle",
        **common,
    )

    assert multi_particle.status == "ok"
    assert np.allclose(multi_particle.sali, variational.sali, rtol=2.0e-8, atol=2.0e-11)
    assert np.allclose(multi_particle.gali, variational.gali, rtol=3.0e-8, atol=2.0e-11)
    assert multi_particle.metadata["rhs_calls"] == 4 * multi_particle.metadata["solver_nfev"]
    assert any("finite neighboring trajectories" in warning for warning in multi_particle.methodological_warnings)


def test_componentwise_relative_finite_difference_matches_analytic_at_mixed_scales() -> None:
    matrix = np.diag([0.9, 0.5])
    initial = np.array([[1.0, 1.0], [1.0, -1.0]])
    common = dict(
        iterations=5,
        initial_deviations=initial,
        gali_orders=(2,),
        backend="numpy",
    )
    analytic = integer_map_alignment_indices(
        lambda state: matrix @ state,
        lambda _state: matrix,
        np.array([1.0e12, 1.0e-12]),
        **common,
    )
    finite_difference = integer_map_alignment_indices(
        lambda state: matrix @ state,
        None,
        np.array([1.0e12, 1.0e-12]),
        jacobian_eps=1.0e-6,
        **common,
    )

    assert finite_difference.status == "ok"
    assert np.allclose(finite_difference.sali, analytic.sali, rtol=2.0e-8, atol=2.0e-10)
    assert np.allclose(finite_difference.gali, analytic.gali, rtol=2.0e-8, atol=2.0e-10)
    assert finite_difference.metadata["jacobian_source"] == "central_relative_componentwise"


def test_system_wrapper_applies_parameter_overrides_to_map_and_jacobian() -> None:
    system = ChaoticSystem(
        name="alignment-parameter-override-map",
        dimension=2,
        kind="map",
        rhs=lambda state, parameters: np.array(
            [parameters["a"] * state[0], parameters["b"] * state[1]]
        ),
        jacobian=lambda _state, parameters: np.diag([parameters["a"], parameters["b"]]),
        parameters={"a": 0.5, "b": 0.25},
    )
    result = integer_system_alignment_indices(
        system,
        np.ones(2),
        iterations=3,
        initial_deviations=np.eye(2),
        gali_orders=(2,),
        parameters={"a": 2.0, "b": 1.0},
        backend="numpy",
    )

    assert result.status == "ok"
    assert np.allclose(result.final_state, [8.0, 1.0])
    assert result.metadata["jacobian_source"] == "analytic"


def test_supplied_initial_vectors_are_never_qr_orthogonalized_during_evolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.diag([1.2, 0.8])

    def forbidden_qr(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("QR must not be applied to evolved SALI/GALI directions")

    monkeypatch.setattr(np.linalg, "qr", forbidden_qr)
    result = integer_map_alignment_indices(
        lambda state: matrix @ state,
        lambda _state: matrix,
        np.zeros(2),
        iterations=4,
        initial_deviations=np.array([[1.0, 1.0], [0.25, -0.5]]),
        gali_orders=(2,),
        backend="numpy",
    )
    assert result.status == "ok"


def test_default_initial_directions_are_generic_and_seed_deterministic() -> None:
    matrix = np.diag([1.1, 0.9, 0.7])
    options = dict(
        iterations=2,
        gali_orders=(2, 3),
        seed=17,
        backend="numpy",
    )
    first = integer_map_alignment_indices(
        lambda state: matrix @ state,
        lambda _state: matrix,
        np.zeros(3),
        **options,
    )
    second = integer_map_alignment_indices(
        lambda state: matrix @ state,
        lambda _state: matrix,
        np.zeros(3),
        **options,
    )

    assert np.array_equal(first.initial_deviations, second.initial_deviations)
    assert np.allclose(first.initial_deviations.T @ first.initial_deviations, np.eye(3))
    assert not np.allclose(first.initial_deviations, np.eye(3))


def test_configuration_errors_and_fractional_orders_are_rejected_explicitly() -> None:
    with pytest.raises(ValueError, match="samples, n_vectors, dimension"):
        alignment_indices_from_tangent_history(np.ones((2, 3)))
    with pytest.raises(ValueError, match="n_vectors cannot exceed"):
        alignment_indices_from_tangent_history(np.ones((4, 3, 2)))
    zero_history = np.ones((2, 2, 2))
    zero_history[1, 0] = 0.0
    with pytest.raises(ValueError, match="zero"):
        alignment_indices_from_tangent_history(zero_history, backend="numpy")
    with pytest.raises(ValueError, match="order cannot exceed"):
        generalized_alignment_index(np.ones((2, 3)), order=3, backend="numpy")
    with pytest.raises(ValueError, match="only for q=1"):
        alignment_indices_from_tangent_history(np.repeat(np.eye(2)[None, :, :], 2, axis=0), q=0.99)
    with pytest.raises(ValueError, match="only for q=1"):
        integer_flow_alignment_indices(
            lambda state: -state,
            lambda _state: -np.eye(2),
            np.ones(2),
            t_final=1.0,
            q=np.array([1.0, 0.95]),
        )
    with pytest.raises(ValueError, match="only for q=1"):
        integer_map_alignment_indices(
            lambda state: state,
            lambda _state: np.eye(2),
            np.ones(2),
            iterations=2,
            q=0.9,
        )
    fractional_system = SimpleNamespace(
        kind="map",
        evaluate=lambda state: np.asarray(state),
        jacobian=lambda state: np.eye(2),
        jacobian_matrix=lambda state: np.eye(2),
        metadata={"q": [1.0, 0.8]},
    )
    with pytest.raises(ValueError, match="only for q=1"):
        integer_system_alignment_indices(fractional_system, np.ones(2), iterations=2)


def test_result_keeps_dissipative_warning_and_has_no_chaos_label() -> None:
    stable = integer_flow_alignment_indices(
        lambda state: np.diag([-0.2, -1.0]) @ state,
        lambda _state: np.diag([-0.2, -1.0]),
        np.zeros(2),
        t_final=1.0,
        initial_deviations=np.array([[1.0, 1.0], [1.0, -1.0]]),
        backend="numpy",
    )

    assert not hasattr(stable, "chaotic")
    assert not hasattr(stable, "chaos_verified")
    assert any("dissipative" in warning for warning in stable.methodological_warnings)
    assert any("not, by themselves" in warning for warning in stable.methodological_warnings)
