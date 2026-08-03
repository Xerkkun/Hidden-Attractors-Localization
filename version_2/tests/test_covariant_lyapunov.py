from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hidden_attractors.analysis.covariant_lyapunov as clv_module
from hidden_attractors.analysis.covariant_lyapunov import (
    CovariantAngleResult,
    CovariantLyapunovResult,
    CovariantQRHistoryResult,
    covariant_lyapunov_angles,
    integer_covariant_vectors_from_qr_history,
    integer_flow_covariant_lyapunov_vectors,
    integer_map_covariant_lyapunov_vectors,
    integer_system_covariant_lyapunov_vectors,
)
from hidden_attractors.systems.base import ChaoticSystem


MAP_O = np.array([[3.0, -4.0], [4.0, 3.0]]) / 5.0
MAP_T = np.array([[4.0, 1.0], [0.0, 2.0]])
MAP_A = MAP_O @ MAP_T @ MAP_O.T
MAP_TERMINAL = np.array([[1.0, 1.0 / 3.0], [0.0, 1.0]])
MAP_EXACT = np.column_stack(
    (
        np.array([3.0, 4.0]) / 5.0,
        np.array([-11.0, 2.0]) / (5.0 * np.sqrt(5.0)),
    )
)

FLOW_O = np.array([[1.0, 8.0, 4.0], [8.0, 1.0, -4.0], [-4.0, 4.0, -7.0]]) / 9.0
FLOW_G = np.array([[1.0, 1.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
FLOW_B = FLOW_O @ FLOW_G @ FLOW_O.T
FLOW_EXACT = np.column_stack(
    (
        np.array([1.0, 8.0, -4.0]) / 9.0,
        np.array([7.0, -7.0, 8.0]) / (9.0 * np.sqrt(2.0)),
        np.array([-5.0, -4.0, -16.0]) / (3.0 * np.sqrt(33.0)),
    )
)


def _line_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    cosine = float(np.clip(abs(first @ second), 0.0, 1.0))
    return float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))


def _maximum_line_error(history: np.ndarray, exact_columns: np.ndarray) -> float:
    return max(
        _line_distance(history[sample, vector], exact_columns[:, vector])
        for sample in range(history.shape[0])
        for vector in range(history.shape[1])
    )


def test_qr_history_reconstructs_nonnormal_map_eigendirections() -> None:
    steps = 80
    q_history = np.repeat(MAP_O[None, :, :], steps + 1, axis=0)
    r_history = np.repeat(MAP_T[None, :, :], steps, axis=0)

    result = integer_covariant_vectors_from_qr_history(
        q_history,
        r_history,
        terminal_coefficients=MAP_TERMINAL,
        backend="numpy",
    )

    assert isinstance(result, CovariantQRHistoryResult)
    assert result.vectors.shape == (steps + 1, 2, 2)
    assert result.coefficients.shape == (steps + 1, 2, 2)
    assert _maximum_line_error(result.vectors[:41], MAP_EXACT) < 2.0e-7
    np.testing.assert_allclose(np.linalg.norm(result.vectors, axis=2), 1.0, atol=2.0e-15)
    np.testing.assert_allclose(np.tril(result.coefficients, -1), 0.0, atol=1.0e-15)


def test_numba_backward_sweep_matches_numpy_reference() -> None:
    if not clv_module.NUMBA_AVAILABLE:
        pytest.skip("Numba is not importable")
    steps = 48
    q_history = np.repeat(MAP_O[None, :, :], steps + 1, axis=0)
    r_history = np.repeat(MAP_T[None, :, :], steps, axis=0)
    reference = integer_covariant_vectors_from_qr_history(
        q_history,
        r_history,
        terminal_coefficients=MAP_TERMINAL,
        backend="numpy",
    )
    try:
        compiled = integer_covariant_vectors_from_qr_history(
            q_history,
            r_history,
            terminal_coefficients=MAP_TERMINAL,
            backend="numba",
        )
    except RuntimeError:
        pytest.skip("Numba CLV backend is not operational")

    np.testing.assert_allclose(compiled.vectors, reference.vectors, rtol=2.0e-14, atol=2.0e-15)
    np.testing.assert_allclose(
        compiled.coefficients, reference.coefficients, rtol=2.0e-14, atol=2.0e-15
    )


def test_integer_map_clvs_are_covariant_and_not_the_qr_basis() -> None:
    result = integer_map_covariant_lyapunov_vectors(
        lambda state: MAP_A @ state,
        lambda _state: MAP_A,
        np.zeros(2),
        iterations=16,
        forward_transient_iterations=4,
        backward_transient_iterations=60,
        initial_basis=MAP_O,
        terminal_coefficients=MAP_TERMINAL,
        backend="numpy",
    )

    assert isinstance(result, CovariantLyapunovResult)
    assert result.status == "ok"
    assert result.vectors.shape == (17, 2, 2)
    assert result.sampled_states.shape == (17, 2)
    np.testing.assert_array_equal(result.coordinates, np.arange(17.0))
    np.testing.assert_allclose(result.exponents, np.log([4.0, 2.0]), atol=2.0e-15)
    assert _maximum_line_error(result.vectors, MAP_EXACT) < 2.0e-7
    assert _line_distance(result.vectors[0, 1], MAP_O[:, 1]) > 0.1
    for sample in range(result.vectors.shape[0] - 1):
        for vector in range(2):
            assert (
                _line_distance(
                    MAP_A @ result.vectors[sample, vector],
                    result.vectors[sample + 1, vector],
                )
                < 2.0e-8
            )
    assert result.metadata["auto_transient_stopping"] is False
    assert result.metadata["transient_reference_doi"] == "10.1016/j.physd.2026.135237"


def test_integer_flow_matches_constant_nonnormal_generator() -> None:
    step = float(np.log(2.0))
    result = integer_flow_covariant_lyapunov_vectors(
        lambda state: FLOW_B @ state,
        lambda _state: FLOW_B,
        np.zeros(3),
        t_final=6.0 * step,
        forward_transient_time=2.0 * step,
        backward_transient_time=45.0 * step,
        qr_interval=step,
        initial_basis=FLOW_O,
        terminal_coefficients=np.array(
            [[1.0, 1.0 / 3.0, -1.0 / 5.0], [0.0, 1.0, 2.0 / 7.0], [0.0, 0.0, 1.0]]
        ),
        rtol=2.0e-11,
        atol=1.0e-13,
        max_step=step / 8.0,
        backend="numpy",
    )

    assert result.status == "ok"
    assert result.vectors.shape == (7, 3, 3)
    np.testing.assert_allclose(result.exponents, [1.0, 0.0, -1.0], atol=3.0e-11)
    assert _maximum_line_error(result.vectors, FLOW_EXACT) < 3.0e-7
    propagator = np.array(
        [
            [67.0 / 54.0, -1.0 / 54.0, -49.0 / 216.0],
            [34.0 / 27.0, 41.0 / 27.0, -35.0 / 54.0],
            [-8.0 / 27.0, -16.0 / 27.0, 20.0 / 27.0],
        ]
    )
    for sample in range(result.vectors.shape[0] - 1):
        for vector in range(3):
            assert _line_distance(
                propagator @ result.vectors[sample, vector],
                result.vectors[sample + 1, vector],
            ) < 2.0e-8


def test_map_finite_difference_jacobian_matches_analytic() -> None:
    options = dict(
        iterations=5,
        forward_transient_iterations=2,
        backward_transient_iterations=20,
        initial_basis=MAP_O,
        terminal_coefficients=MAP_TERMINAL,
        backend="numpy",
    )
    analytic = integer_map_covariant_lyapunov_vectors(
        lambda state: MAP_A @ state,
        lambda _state: MAP_A,
        np.array([1.25, -0.75]),
        **options,
    )
    numerical = integer_map_covariant_lyapunov_vectors(
        lambda state: MAP_A @ state,
        None,
        np.array([1.25, -0.75]),
        jacobian_eps=1.0e-6,
        **options,
    )

    assert analytic.status == numerical.status == "ok"
    np.testing.assert_allclose(numerical.exponents, analytic.exponents, rtol=2.0e-7, atol=2.0e-8)
    assert max(
        _line_distance(numerical.vectors[i, j], analytic.vectors[i, j])
        for i in range(analytic.vectors.shape[0])
        for j in range(2)
    ) < 2.0e-6
    assert numerical.metadata["jacobian_source"] == "central_relative_componentwise"


def test_system_wrapper_applies_map_parameter_overrides() -> None:
    system = ChaoticSystem(
        name="CLV parameter override map",
        dimension=2,
        kind="map",
        rhs=lambda state, parameters: np.array(
            [parameters["a"] * state[0], parameters["b"] * state[1]]
        ),
        jacobian=lambda _state, parameters: np.diag(
            [parameters["a"], parameters["b"]]
        ),
        parameters={"a": 0.5, "b": 0.25},
    )
    result = integer_system_covariant_lyapunov_vectors(
        system,
        np.ones(2),
        iterations=4,
        backward_transient_iterations=4,
        initial_basis=np.eye(2),
        terminal_coefficients=np.eye(2),
        parameters={"a": 4.0, "b": 2.0},
        backend="numpy",
    )

    assert result.status == "ok"
    np.testing.assert_allclose(result.exponents, np.log([4.0, 2.0]), atol=2.0e-15)
    np.testing.assert_allclose(result.final_state, [256.0, 16.0])
    np.testing.assert_allclose(result.future_state, [65536.0, 256.0])


def test_angles_use_unoriented_lines_stable_subspaces_and_rolling_windows() -> None:
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    diagonal = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    sample = np.stack((e1, -e2, diagonal))
    history = np.repeat(sample[None, :, :], 4, axis=0)
    result = covariant_lyapunov_angles(
        history,
        coordinates=np.arange(4.0),
        pairs=((0, 1), (0, 2), (1, 2)),
        subspaces=(((0, 1), (2,)), ((0,), (1,))),
        window=3,
    )

    assert isinstance(result, CovariantAngleResult)
    np.testing.assert_allclose(
        result.pair_angles,
        np.tile([np.pi / 2.0, np.pi / 4.0, np.pi / 4.0], (4, 1)),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(result.subspace_angles[:, 0], 0.0, atol=2.0e-15)
    np.testing.assert_allclose(result.subspace_angles[:, 1], np.pi / 2.0, atol=2.0e-15)
    np.testing.assert_allclose(result.window_coordinates, [1.0, 2.0])
    assert result.window_pair_angles.shape == (2, 3)
    oriented = covariant_lyapunov_angles(
        np.array([[[1.0, 0.0], [-1.0, 0.0]]]),
        pairs=((0, 1),),
        unoriented=False,
    )
    assert oriented.pair_angles[0, 0] == pytest.approx(np.pi)


def test_configuration_fractional_shape_singular_and_memory_errors_are_explicit() -> None:
    with pytest.raises(ValueError, match="only for q=1"):
        integer_map_covariant_lyapunov_vectors(
            lambda state: state,
            lambda _state: np.eye(2),
            np.ones(2),
            iterations=2,
            q=0.95,
        )
    with pytest.raises(ValueError, match="orthonormal_bases"):
        integer_covariant_vectors_from_qr_history(np.ones((2, 2)), np.ones((1, 2, 2)))
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        integer_covariant_vectors_from_qr_history(
            np.repeat(np.eye(2)[None, :, :], 2, axis=0),
            np.array([[[1.0, 0.0], [0.0, 1.0e-30]]]),
            backend="numpy",
        )
    with pytest.raises(MemoryError, match="max_workspace_bytes"):
        integer_map_covariant_lyapunov_vectors(
            lambda state: state,
            lambda _state: np.eye(4),
            np.ones(4),
            iterations=100,
            backward_transient_iterations=100,
            max_workspace_bytes=64,
        )
    singular_map = integer_map_covariant_lyapunov_vectors(
        lambda state: np.array([state[0], 0.0]),
        lambda _state: np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.ones(2),
        iterations=2,
        backend="numpy",
    )
    assert singular_map.status == "singular_cocycle"
    assert singular_map.vectors.shape == (0, 2, 2)


def test_declared_fractional_system_is_rejected_before_evaluation() -> None:
    system = SimpleNamespace(
        kind="flow",
        q=np.array([1.0, 0.9]),
        evaluate=lambda _state: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    with pytest.raises(ValueError, match="only for q=1"):
        integer_system_covariant_lyapunov_vectors(system, np.ones(2), t_final=1.0)


def test_nearly_degenerate_spectrum_keeps_subspace_warning_without_chaos_label() -> None:
    result = integer_map_covariant_lyapunov_vectors(
        lambda state: 2.0 * state,
        lambda _state: 2.0 * np.eye(2),
        np.zeros(2),
        iterations=3,
        backward_transient_iterations=3,
        initial_basis=np.eye(2),
        terminal_coefficients=np.eye(2),
        backend="numpy",
    )

    assert result.status == "ok"
    assert result.metadata["near_degenerate_finite_time_spectrum"] is True
    assert any("nonunique" in warning for warning in result.methodological_warnings)
    assert not hasattr(result, "is_chaotic")

