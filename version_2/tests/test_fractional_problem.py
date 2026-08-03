from __future__ import annotations

import numpy as np
import pytest
from numba import njit
from scipy.special import gamma

from hidden_attractors.fractional import (
    FRACTIONAL_DERIVATIVES,
    FRACTIONAL_METHODS,
    FractionalProblem,
    get_fractional_reference,
    solve_fractional_problem,
    solve_fractional_system,
)
from hidden_attractors.systems.expressions import (
    ExpressionSystemDefinition,
    compile_expression_system,
)


@njit
def _constant_rhs(time, state, parameters):
    return np.ones_like(state) * parameters[0] + 0.0 * time


def _constant_python_rhs(time, state, parameters):
    return np.ones_like(state) * parameters[0] + 0.0 * time


def _constant_mapping_rhs(time, state, parameters):
    return np.ones_like(state) * parameters["forcing"] + 0.0 * time


def test_problem_normalizes_complete_contract_and_references() -> None:
    problem = FractionalProblem(
        derivative="Caputo",
        method="caputo_abm_pece",
        orders=[0.6, 0.8],
        initial_state=[1.0, -1.0],
        step=0.01,
        t_span=(2.0, 3.0),
        memory_policy="block_restart",
        problem_id="componentwise_reference",
    )
    assert problem.derivative == "caputo"
    assert problem.order_mode == "componentwise"
    assert problem.lower_terminal == 2.0
    assert problem.initial_condition_kind == "classical"
    assert "diethelm_ford_freed2004" in problem.reference_keys
    metadata = problem.as_metadata()
    assert metadata["claims"] == "finite_numerical_trajectory_only"
    assert metadata["orders"] == [0.6, 0.8]
    assert metadata["initial_state"] == [1.0, -1.0]
    assert metadata["method_options"] == {}
    assert FractionalProblem.from_mapping(problem.to_mapping()) == problem


def test_every_registry_reference_resolves() -> None:
    keys = {
        key
        for item in (*FRACTIONAL_DERIVATIVES.values(), *FRACTIONAL_METHODS.values())
        for key in item.references
    }
    assert keys
    for key in keys:
        assert get_fractional_reference(key).url.startswith("https://")


def test_problem_rejects_wrong_initial_condition_semantics_and_missing_window() -> None:
    with pytest.raises(ValueError, match="requires initial_condition_kind"):
        FractionalProblem(
            "grunwald_letnikov",
            "gl_explicit_discrete",
            0.7,
            [1.0],
            0.01,
            (0.0, 1.0),
            initial_condition_kind="classical",
            allow_experimental=True,
        )
    with pytest.raises(ValueError, match="history_window"):
        FractionalProblem(
            "caputo",
            "caputo_abm_pece",
            0.7,
            [1.0],
            0.01,
            (0.0, 1.0),
            memory_policy="finite_window",
        )
    with pytest.raises(ValueError, match="integer number of fixed steps"):
        FractionalProblem(
            "caputo",
            "caputo_abm_pece",
            0.7,
            [1.0],
            0.3,
            (0.0, 1.0),
        )


def test_nonsingular_kernel_definitions_are_research_gated() -> None:
    problem = FractionalProblem(
        "caputo_fabrizio",
        "cf_predictor_corrector",
        0.8,
        [1.0],
        0.01,
        (0.0, 1.0),
    )
    assert problem.derivative_definition.implementation_status == "research_required"
    assert "diethelm_garrappa_giusti_stynes2020" in problem.reference_keys
    with pytest.raises(NotImplementedError, match="research_required"):
        problem.validate_executable()


def test_problem_does_not_mistake_a_sampled_operator_for_a_solver() -> None:
    problem = FractionalProblem(
        "riemann_liouville",
        "gl_direct",
        0.6,
        [1.0],
        0.01,
        (0.0, 1.0),
        allow_experimental=True,
    )
    with pytest.raises(NotImplementedError, match="sampled operator"):
        problem.validate_executable()


def test_problem_metadata_records_options_but_config_rejects_callable_values() -> None:
    def marker(time: float) -> float:
        return 0.8 + 0.0 * time

    problem = FractionalProblem(
        "caputo",
        "caputo_abm_pece",
        0.8,
        [1.0],
        0.1,
        (0.0, 1.0),
        method_options={"corrector_iterations": 2},
        kernel_parameters={"audit_callback": marker},
    )
    metadata = problem.as_metadata()
    assert metadata["method_options"] == {"corrector_iterations": 2}
    assert "marker" in metadata["kernel_parameters"]["audit_callback"]
    with pytest.raises(TypeError, match="non-serializable"):
        problem.to_mapping()


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("method_options", {"corrector_iterations": 2}, "method_options"),
        ("kernel_parameters", {"normalization": "custom"}, "kernel_parameters"),
    ],
)
def test_execution_rejects_options_that_no_solver_consumes(
    field_name: str,
    field_value: dict[str, object],
    message: str,
) -> None:
    kwargs = {field_name: field_value}
    problem = FractionalProblem(
        "caputo",
        "caputo_abm_pece",
        0.5,
        [0.0],
        0.01,
        (0.0, 0.1),
        **kwargs,
    )
    with pytest.raises(NotImplementedError, match=message):
        solve_fractional_problem(problem, _constant_python_rhs, [1.0])


def test_experimental_gl_requires_opt_in_then_solves_constant_forcing() -> None:
    blocked = FractionalProblem(
        "caputo",
        "gl_explicit_discrete",
        0.5,
        [2.0],
        0.001,
        (0.0, 1.0),
    )
    with pytest.raises(PermissionError, match="allow_experimental"):
        solve_fractional_problem(blocked, _constant_rhs, [1.0])

    problem = FractionalProblem(
        "caputo",
        "gl_explicit_discrete",
        0.5,
        [2.0],
        0.001,
        (0.0, 1.0),
        allow_experimental=True,
    )
    result = solve_fractional_problem(problem, _constant_rhs, [1.0])
    assert result.status == "ok"
    assert result.states[-1, 0] == pytest.approx(2.0 + 1.0 / gamma(1.5), rel=4e-4)
    assert result.metadata["reference_keys"] == list(problem.reference_keys)


def test_fractional_problem_forwards_gl_divergence_limit() -> None:
    problem = FractionalProblem(
        "caputo",
        "gl_explicit_discrete",
        1.0,
        [0.0],
        0.1,
        (0.0, 1.0),
        allow_experimental=True,
    )
    result = solve_fractional_problem(
        problem,
        _constant_rhs,
        [1.0],
        divergence_norm=0.25,
    )
    assert result.status == "diverged"
    assert result.times.size == result.states.shape[0] == 4
    assert result.metadata["divergence_norm"] == pytest.approx(0.25)


def test_gl_python_fallback_matches_numba_path_for_gui_callable() -> None:
    problem = FractionalProblem(
        "caputo",
        "gl_explicit_discrete",
        [0.55, 0.8],
        [0.2, -0.3],
        0.01,
        (0.0, 0.5),
        allow_experimental=True,
    )
    compiled = solve_fractional_problem(problem, _constant_rhs, [0.4])
    fallback = solve_fractional_problem(
        problem,
        _constant_mapping_rhs,
        {"forcing": 0.4},
        use_acceleration=False,
    )
    # Mapping-valued parameters exercise the GUI/Python callable lane.
    assert compiled.backend == "numba"
    assert fallback.backend == "python_numpy"
    assert np.allclose(compiled.states, fallback.states, rtol=2e-14, atol=2e-14)


def test_gl_problem_preserves_internal_rhs_typeerror() -> None:
    problem = FractionalProblem(
        "caputo",
        "gl_explicit_discrete",
        0.5,
        [0.0],
        0.01,
        (0.0, 0.1),
        allow_experimental=True,
    )
    calls = 0

    def broken_rhs(time, state, parameters):
        nonlocal calls
        del time, state, parameters
        calls += 1
        raise TypeError("gl-internal-typeerror")

    result = solve_fractional_problem(
        problem,
        broken_rhs,
        {"unused": True},
        use_acceleration=False,
    )
    assert result.status == "solver_exception:TypeError:gl-internal-typeerror"
    assert calls == 1


def test_caputo_problem_does_not_mask_internal_rhs_typeerror() -> None:
    problem = FractionalProblem(
        "caputo",
        "caputo_abm_pece",
        0.5,
        [0.0],
        0.01,
        (0.0, 0.1),
    )

    def broken_rhs(time, state, parameters):
        del time, state, parameters
        raise TypeError("caputo-internal-typeerror")

    with pytest.raises(TypeError, match="caputo-internal-typeerror"):
        solve_fractional_problem(
            problem,
            broken_rhs,
            {"unused": True},
            use_acceleration=False,
        )


def test_caputo_abm_problem_uses_same_structured_result_contract() -> None:
    problem = FractionalProblem(
        "caputo",
        "caputo_abm_pece",
        0.5,
        [0.0],
        0.005,
        (1.0, 2.0),
    )
    result = solve_fractional_problem(
        problem,
        _constant_python_rhs,
        [1.0],
        use_acceleration=False,
        divergence_norm=None,
    )
    assert result.times[0] == pytest.approx(1.0)
    assert result.times[-1] == pytest.approx(2.0)
    assert result.states[-1, 0] == pytest.approx(1.0 / gamma(1.5), rel=2e-3)
    assert result.metadata["derivative"] == "caputo"
    assert result.backend == "python_numpy"
    assert result.metadata["backend_info"]["used_c_backend"] is False
    assert result.metadata["backend_info"]["rhs_source"] == "python_native"


def test_componentwise_caputo_block_restart_executes_without_hidden_options() -> None:
    problem = FractionalProblem(
        "caputo",
        "caputo_abm_pece",
        [0.5, 0.7],
        [0.0, 0.0],
        0.01,
        (0.0, 0.1),
        memory_policy="block_restart",
    )
    result = solve_fractional_problem(
        problem,
        _constant_python_rhs,
        [1.0],
        use_acceleration=False,
        divergence_norm=None,
    )
    assert result.status == "ok"
    assert result.states.shape == (11, 2)
    assert result.backend == "python_numpy"
    assert result.metadata["backend_info"]["implementation"] == (
        "componentwise_block_restart"
    )


@pytest.mark.parametrize(
    ("order", "expected_backend"),
    [(0.7, "python_numpy"), (1.0, "python_efork_q1")],
)
def test_efork_backend_metadata_is_exact(
    order: float,
    expected_backend: str,
) -> None:
    problem = FractionalProblem(
        "caputo",
        "efork3",
        order,
        [0.0],
        0.01,
        (0.0, 0.1),
    )
    result = solve_fractional_problem(
        problem,
        _constant_python_rhs,
        [1.0],
        use_acceleration=False,
        divergence_norm=None,
    )
    assert result.status == "ok"
    assert result.backend == expected_backend
    assert result.metadata["backend_info"]["used_c_backend"] is False


def test_expression_defined_system_uses_fractional_gui_adapter() -> None:
    system = compile_expression_system(
        ExpressionSystemDefinition(
            name="constant-no-code",
            kind="flow",
            variables=("x",),
            parameters={"forcing": 1.0},
            equations=("forcing",),
            initial_state=(0.0,),
        )
    )
    problem = FractionalProblem(
        "caputo",
        "gl_explicit_discrete",
        0.5,
        [0.0],
        0.002,
        (0.0, 1.0),
        allow_experimental=True,
    )
    result = solve_fractional_system(problem, system)
    assert result.backend == "python_numpy"
    assert result.states[-1, 0] == pytest.approx(1.0 / gamma(1.5), rel=8e-4)
    assert result.metadata["system_name"] == "constant-no-code"
    assert result.metadata["adapter"] == "hafo_system"
