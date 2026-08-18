"""Focused algebraic contracts for the experimental geometric engine."""

from __future__ import annotations

from dataclasses import replace
from math import gamma, pi, sqrt

import numpy as np
import pytest

from hidden_attractors.geometry import (
    IncompatiblePartitionError,
    NondifferentiablePointError,
    chua_nonsmooth_partition,
    connecting_curve_residual,
    connecting_minor_pairs,
    connecting_minors,
    critical_surface_values,
    evaluate_differential_geometry,
    fractional_perpetual_startup_residual,
    generate_symmetry_images,
    normalized_perpetual_residual,
    perpetual_point_residual,
    sign_flip_symmetry,
    translation_symmetry,
    validate_affine_symmetry,
)
from hidden_attractors.models.chua import jacobian_nonsmooth, rhs_nonsmooth
from hidden_attractors.systems import ChaoticSystem, get_system
from hidden_attractors.systems.modified_van_der_pol_duffing import mavpd_2023_system


def _polynomial_system(*, analytic: bool = True) -> ChaoticSystem:
    def rhs(state, _parameters):
        x, y = np.asarray(state, dtype=float)
        return np.array([x * x + y, x - 3.0 * y], dtype=float)

    def jacobian(state, _parameters):
        x, _y = np.asarray(state, dtype=float)
        return np.array([[2.0 * x, 1.0], [1.0, -3.0]], dtype=float)

    return ChaoticSystem(
        name="test-polynomial-flow",
        dimension=2,
        rhs=rhs,
        jacobian=jacobian if analytic else None,
        kind="flow",
    )


def test_f_jacobian_and_jacobian_field_are_kept_distinct() -> None:
    system = _polynomial_system()
    state = np.array([2.0, -1.0])
    result = evaluate_differential_geometry(system, state)
    expected_field = np.array([3.0, 5.0])
    expected_jacobian = np.array([[4.0, 1.0], [1.0, -3.0]])
    assert np.allclose(result.field, expected_field)
    assert np.allclose(result.jacobian, expected_jacobian)
    assert np.allclose(result.jacobian_field, expected_jacobian @ expected_field)
    assert result.jacobian_determinant == pytest.approx(-13.0)
    assert result.jacobian_source == "analytic"


def test_auto_mode_uses_scaled_central_difference_only_when_needed() -> None:
    system = _polynomial_system(analytic=False)
    state = np.array([2.0, -1.0])
    result = evaluate_differential_geometry(system, state, jacobian_mode="auto")
    assert result.jacobian_source == "finite_difference"
    assert np.allclose(result.jacobian, [[4.0, 1.0], [1.0, -3.0]], rtol=2.0e-9, atol=2.0e-9)
    with pytest.raises(ValueError, match="no analytic Jacobian"):
        evaluate_differential_geometry(system, state, jacobian_mode="analytic")


def test_geometry_rejects_discrete_maps() -> None:
    map_system = ChaoticSystem(
        name="test-map",
        dimension=1,
        rhs=lambda state, _parameters: np.asarray(state, dtype=float),
        kind="map",
    )
    with pytest.raises(ValueError, match="kind='flow'"):
        evaluate_differential_geometry(map_system, np.array([1.0]), jacobian_mode="auto")


def test_critical_surface_values_are_raw_local_residuals() -> None:
    values = critical_surface_values(_polynomial_system(), np.array([2.0, -1.0]))
    assert values.velocity_component(0) == pytest.approx(3.0)
    assert values.acceleration_component(1) == pytest.approx(-12.0)
    assert values.jacobian_determinant == pytest.approx(-13.0)


def test_mavpd_exact_perpetual_points_and_connecting_inclusion() -> None:
    parameters = {"gamma": 0.1, "delta": 100.0, "rho": 200.0, "xi": 3.1}
    system = mavpd_2023_system(parameters)
    u = sqrt(parameters["gamma"] / 3.0)
    v = 2.0 * parameters["delta"] * parameters["gamma"] * u / (
        3.0 * (parameters["rho"] - parameters["delta"])
    )
    point = np.array([u, v, u - parameters["xi"] * v])

    pp = perpetual_point_residual(system, point, acceleration_tolerance=1.0e-10)
    connecting = connecting_curve_residual(system, point, residual_tolerance=1.0e-10)
    local = evaluate_differential_geometry(system, point)

    assert pp.is_candidate
    assert not pp.equilibrium_excluded
    assert pp.residual_norm < 1.0e-11
    assert pp.raw_residual_norm == pp.residual_norm
    assert pp.normalized_residual <= pp.residual_norm
    assert local.jacobian_determinant == pytest.approx(0.0, abs=1.0e-10)
    assert connecting.is_candidate
    assert np.linalg.norm(connecting.minors) < 1.0e-10


def test_perpetual_point_residual_excludes_equilibria() -> None:
    system = ChaoticSystem(
        name="linear-equilibrium",
        dimension=2,
        rhs=lambda state, _parameters: np.asarray(state, dtype=float),
        jacobian=lambda _state, _parameters: np.eye(2),
    )
    pp = perpetual_point_residual(system, np.zeros(2))
    connecting = connecting_curve_residual(system, np.zeros(2))
    assert pp.residual_norm == 0.0
    assert pp.equilibrium_excluded
    assert not pp.is_candidate
    assert connecting.equilibrium_excluded
    assert not connecting.is_candidate
    assert np.isinf(connecting.normalized_residual)


def test_picard_caputo_startup_scaling_and_q1_limit() -> None:
    system = _polynomial_system()
    state = np.array([2.0, -1.0])
    local = evaluate_differential_geometry(system, state)

    q1 = fractional_perpetual_startup_residual(system, state, 1.0)
    assert np.allclose(q1.first_coefficient, local.field)
    assert np.allclose(q1.startup_acceleration, local.jacobian_field)
    assert q1.startup_acceleration_norm == pytest.approx(local.acceleration_norm)
    assert q1.jacobian_field_norm == pytest.approx(local.acceleration_norm)
    assert q1.normalized_residual == pytest.approx(normalized_perpetual_residual(local))

    order = 0.8
    fractional = fractional_perpetual_startup_residual(system, state, order)
    assert np.allclose(fractional.first_coefficient, local.field / gamma(order + 1.0))
    assert np.allclose(
        fractional.second_coefficient,
        local.jacobian_field / gamma(2.0 * order + 1.0),
    )
    assert np.allclose(fractional.startup_acceleration, 2.0 * fractional.second_coefficient)
    assert fractional.startup_acceleration_norm == pytest.approx(
        np.linalg.norm(fractional.startup_acceleration)
    )
    # The Gamma-dependent startup coefficient is retained as a raw diagnostic,
    # while candidate acceptance uses the same geometric PP residual for all q.
    assert fractional.residual_norm != pytest.approx(q1.residual_norm)
    assert fractional.normalized_residual == pytest.approx(q1.normalized_residual)
    assert "not_global_caputo" in fractional.evidence_scope
    with pytest.raises(ValueError, match="0 < q <= 1"):
        fractional_perpetual_startup_residual(system, state, 0.0)


def test_perpetual_residual_uses_exact_scaled_formula_under_time_rescaling() -> None:
    base = _polynomial_system()
    state = np.array([2.0, -1.0])
    factor = 7.0

    def rhs(value, parameters):
        return factor * base.rhs(value, parameters)

    def jacobian(value, parameters):
        assert base.jacobian is not None
        return factor * base.jacobian(value, parameters)

    rescaled = ChaoticSystem(
        name="time-rescaled-polynomial-flow",
        dimension=base.dimension,
        rhs=rhs,
        jacobian=jacobian,
    )
    base_local = evaluate_differential_geometry(base, state)
    scaled_local = evaluate_differential_geometry(rescaled, state)
    base_pp = perpetual_point_residual(base, state, acceleration_tolerance=1.0)
    scaled_pp = perpetual_point_residual(rescaled, state, acceleration_tolerance=1.0)

    expected_base = np.linalg.norm(base_local.jacobian_field) / (
        1.0 + np.linalg.norm(base_local.jacobian, ord=2) * np.linalg.norm(base_local.field)
    )
    expected_scaled = np.linalg.norm(scaled_local.jacobian_field) / (
        1.0 + np.linalg.norm(scaled_local.jacobian, ord=2) * np.linalg.norm(scaled_local.field)
    )
    assert base_pp.normalized_residual == pytest.approx(expected_base)
    assert scaled_pp.normalized_residual == pytest.approx(expected_scaled)
    assert normalized_perpetual_residual(base_local) == pytest.approx(expected_base)
    assert scaled_pp.raw_residual_norm == pytest.approx(
        factor**2 * base_pp.raw_residual_norm
    )
    assert scaled_pp.jacobian_norm == pytest.approx(factor * base_pp.jacobian_norm)
    assert scaled_pp.field_norm == pytest.approx(factor * base_pp.field_norm)

    # A tolerance between the normalized and raw values proves that the gate
    # is applied to the protocol residual, not to the dimensional raw norm.
    acceptance_tolerance = 1.01 * scaled_pp.normalized_residual
    accepted = perpetual_point_residual(
        rescaled,
        state,
        acceleration_tolerance=acceptance_tolerance,
    )
    assert accepted.raw_residual_norm > acceptance_tolerance
    assert accepted.is_candidate


def test_connecting_minors_have_declared_lexicographic_order() -> None:
    field = np.array([1.0, 2.0, 3.0])
    acceleration = 2.0 * field
    assert connecting_minor_pairs(3) == ((0, 1), (0, 2), (1, 2))
    assert np.array_equal(connecting_minors(field, acceleration), np.zeros(3))
    nonparallel = connecting_minors(field, np.array([0.0, 1.0, 0.0]))
    assert np.array_equal(nonparallel, np.array([1.0, 0.0, -3.0]))


@pytest.mark.parametrize(
    ("state", "region"),
    [
        (np.array([-2.0, 0.3, -0.4]), "left"),
        (np.array([0.0, 0.3, -0.4]), "inner"),
        (np.array([2.0, 0.3, -0.4]), "right"),
    ],
)
def test_chua_pwl_partition_matches_registered_field_and_regional_jacobian(state, region) -> None:
    system = get_system("chua-nonsmooth")
    partition = chua_nonsmooth_partition(system.parameters)
    resolution = partition.resolve(state)
    result = evaluate_differential_geometry(system, state, partition=partition)
    assert resolution.status == "interior"
    assert resolution.region_names == (region,)
    assert result.region == region
    assert np.allclose(partition.field_at(state), rhs_nonsmooth(state))
    assert np.allclose(result.field, system.evaluate(state))
    assert np.allclose(result.jacobian, jacobian_nonsmooth(state))


@pytest.mark.parametrize(("x", "surface"), [(-1.0, "x=-1"), (1.0, "x=+1")])
def test_chua_switch_has_continuous_field_but_nonunique_jacobian(x, surface) -> None:
    system = get_system("chua-nonsmooth")
    partition = chua_nonsmooth_partition(system.parameters)
    state = np.array([x, 0.2, -0.3])
    resolution = partition.resolve(state)
    assert resolution.status == "switching"
    assert resolution.switching_names == (surface,)
    assert len(partition.adjacent_jacobians(state)) == 2
    assert np.allclose(partition.field_at(state), system.evaluate(state))
    with pytest.raises(NondifferentiablePointError, match="no unique classical Jacobian"):
        partition.jacobian_at(state)
    with pytest.raises(NondifferentiablePointError):
        evaluate_differential_geometry(system, state, partition=partition)


def test_chua_switch_is_rejected_without_requiring_explicit_partition() -> None:
    system = get_system("chua-nonsmooth")
    with pytest.raises(NondifferentiablePointError, match="no unique classical Jacobian"):
        evaluate_differential_geometry(system, np.array([1.0, 0.2, -0.3]))


def test_chua_pwl_partition_is_autoassociated_away_from_switches() -> None:
    system = get_system("chua-nonsmooth")
    result = evaluate_differential_geometry(system, np.array([0.25, 0.2, -0.3]))
    assert result.jacobian_source == "regional_affine"
    assert result.region == "inner"


def test_declared_symmetry_validator_and_one_step_images() -> None:
    system = mavpd_2023_system({"xi": 3.1})
    inversion = sign_flip_symmetry((-1, -1, -1), name="central_inversion")
    points = np.array(
        [
            [0.2, 0.1, -0.3],
            [-0.5, 0.7, 0.4],
            [1.0, -0.2, 0.8],
        ]
    )
    report = validate_affine_symmetry(system, inversion, points)
    assert report.passed
    assert report.max_absolute_residual < 1.0e-12

    wrong = sign_flip_symmetry((-1, -1, 1), name="wrong")
    assert not validate_affine_symmetry(system, wrong, points).passed

    images = generate_symmetry_images(points[:1], (inversion,))
    assert len(images) == 2
    assert {image.transform_name for image in images} == {"identity", "central_inversion"}


def test_translation_images_are_one_step_not_infinite_group_closure() -> None:
    translation = translation_symmetry((0.0, 2.0 * pi), name="pll_plus_2pi")
    images = generate_symmetry_images(np.array([0.1, 0.2]), (translation,))
    assert len(images) == 2
    assert np.allclose(images[1].state, [0.1, 0.2 + 2.0 * pi])


def test_parameter_override_requires_rebuilt_pwl_partition() -> None:
    system = get_system("chua-nonsmooth")
    partition = chua_nonsmooth_partition(system.parameters)
    with pytest.raises(IncompatiblePartitionError, match="rebuild the partition"):
        evaluate_differential_geometry(
            system,
            np.zeros(3),
            parameters={"alpha": 9.0},
            partition=partition,
        )


def test_partition_built_with_different_alpha_is_rejected_even_if_fields_coincide_at_origin() -> None:
    system = get_system("chua-nonsmooth")
    mismatched_parameters = dict(system.parameters)
    mismatched_parameters["alpha"] = float(system.parameters["alpha"]) + 1.0
    mismatched = chua_nonsmooth_partition(mismatched_parameters)
    with pytest.raises(IncompatiblePartitionError, match="parameter 'alpha'"):
        evaluate_differential_geometry(system, np.zeros(3), partition=mismatched)


def test_rebuilt_partition_accepts_matching_parameter_override() -> None:
    system = get_system("chua-nonsmooth")
    active_alpha = float(system.parameters["alpha"]) + 1.0
    active_parameters = dict(system.parameters)
    active_parameters["alpha"] = active_alpha
    partition = chua_nonsmooth_partition(active_parameters)
    state = np.array([0.25, 0.2, -0.3])
    result = evaluate_differential_geometry(
        system,
        state,
        parameters={"alpha": active_alpha},
        partition=partition,
    )
    assert result.region == "inner"
    assert result.jacobian_source == "regional_affine"
    assert np.allclose(result.field, system.evaluate(state, {"alpha": active_alpha}))


def test_bound_partition_cannot_silently_replace_registered_rhs() -> None:
    system = get_system("chua-nonsmooth")
    partition = chua_nonsmooth_partition(system.parameters)
    left = partition.regions[0]
    altered_matrix = left.matrix.copy()
    altered_matrix[0, 0] += 0.5
    tampered = replace(
        partition,
        regions=(replace(left, matrix=altered_matrix),) + partition.regions[1:],
    )
    with pytest.raises(IncompatiblePartitionError, match="coefficients do not match"):
        evaluate_differential_geometry(
            system,
            np.array([-2.0, 0.2, -0.3]),
            partition=tampered,
        )


def test_zero_field_cannot_hide_tampered_inner_jacobian_with_apparently_valid_binding() -> None:
    system = get_system("chua-nonsmooth")
    partition = chua_nonsmooth_partition(system.parameters)
    inner = partition.regions[1]
    zero_inner = replace(inner, matrix=np.zeros_like(inner.matrix))
    altered = replace(
        partition,
        regions=(partition.regions[0], zero_inner, partition.regions[2]),
    )
    # Even if an artifact rewrites its apparent coefficient binding alongside
    # the matrix, the registered analytic Jacobian remains an independent check.
    forged_binding = replace(
        altered.binding,
        coefficient_fingerprint=altered.coefficient_fingerprint,
    )
    apparently_bound = replace(altered, binding=forged_binding)
    assert np.array_equal(apparently_bound.field_at(np.zeros(3)), np.zeros(3))
    with pytest.raises(IncompatiblePartitionError, match="regional Jacobian"):
        evaluate_differential_geometry(
            system,
            np.zeros(3),
            partition=apparently_bound,
        )


def test_public_geometry_contracts_reject_nonfinite_inputs() -> None:
    system = _polynomial_system()
    with pytest.raises(ValueError, match="finite"):
        evaluate_differential_geometry(system, np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="non-smooth"):
        chua_nonsmooth_partition({"model": "arctan"})
    with pytest.raises(ValueError, match="finite"):
        translation_symmetry((0.0, np.inf))
