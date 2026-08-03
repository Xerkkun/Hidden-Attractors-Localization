from __future__ import annotations

import json

import numpy as np
import pytest

from hidden_attractors.analysis.contracts import (
    FRACTIONAL_TRAJECTORY_WARNING,
    AnalysisResult,
    PrehistorySpec,
    TrajectoryInput,
)
from hidden_attractors.simulation import SimulationResult


def _integer_trajectory(**overrides: object) -> TrajectoryInput:
    values: dict[str, object] = {
        "t": np.linspace(0.0, 1.0, 6),
        "x": np.column_stack(
            (np.linspace(0.0, 1.0, 6), np.linspace(1.0, 0.0, 6))
        ),
        "system_kind": "integer_flow",
        "projection": ("position", "velocity"),
        "solver_and_tolerances": {"method": "rk4", "step": 0.2},
    }
    values.update(overrides)
    return TrajectoryInput(**values)


def _fractional_trajectory(**overrides: object) -> TrajectoryInput:
    values: dict[str, object] = {
        "t": np.linspace(2.0, 3.0, 6),
        "x": np.linspace(1.0, 2.0, 6),
        "system_kind": "fractional_continuous",
        "derivative_definition": "caputo",
        "order": 0.85,
        "memory_policy": "full_history",
        "lower_terminal_and_prehistory": PrehistorySpec(
            kind="point_initial_value",
            lower_terminal=2.0,
        ),
        "solver_and_tolerances": {"method": "abm_pece", "step": 0.2},
    }
    values.update(overrides)
    return TrajectoryInput(**values)


def test_integer_contract_is_uniform_readonly_and_detached() -> None:
    t = np.linspace(0.0, 1.0, 6)
    x = np.column_stack((t, 1.0 - t))
    trajectory = _integer_trajectory(t=t, x=x)
    t[0] = 99.0
    x[0, 0] = 99.0

    assert trajectory.sample_count == 6
    assert trajectory.dimension == 2
    assert trajectory.sampled_uniformly
    assert trajectory.uniform_step == pytest.approx(0.2)
    assert trajectory.t[0] == 0.0
    assert trajectory.x[0, 0] == 0.0
    assert trajectory.t.flags.c_contiguous
    assert trajectory.x.flags.c_contiguous
    assert not trajectory.t.flags.writeable
    assert not trajectory.x.flags.writeable
    assert trajectory.scientific_warnings == ()


@pytest.mark.parametrize(
    ("time_coordinate", "accepted"),
    [("iteration_index", True), ("user_defined", True), ("physical_time", False)],
)
def test_integer_map_requires_explicit_discrete_coordinate(
    time_coordinate: str,
    accepted: bool,
) -> None:
    kwargs = dict(
        t=np.arange(5, dtype=float),
        x=np.arange(5, dtype=float),
        system_kind="integer_map",
        time_coordinate=time_coordinate,
    )
    if accepted:
        assert TrajectoryInput(**kwargs).time_coordinate == time_coordinate
    else:
        with pytest.raises(ValueError, match="integer_map"):
            TrajectoryInput(**kwargs)


def test_irregular_grid_is_detected_and_uniform_claim_is_checked() -> None:
    t = np.array([0.0, 0.1, 0.21, 0.33])
    trajectory = TrajectoryInput(
        t=t,
        x=np.arange(t.size),
        system_kind="sampled_data",
    )
    assert not trajectory.sampled_uniformly
    assert trajectory.uniform_step is None
    assert "irregular" in trajectory.scientific_warnings[0]
    with pytest.raises(ValueError, match="contradicts"):
        TrajectoryInput(
            t=t,
            x=np.arange(t.size),
            system_kind="sampled_data",
            sampled_uniformly=True,
        )


def test_fractional_contract_requires_operator_order_memory_and_prehistory() -> None:
    trajectory = _fractional_trajectory()
    assert trajectory.derivative_definition == "caputo"
    assert trajectory.order == pytest.approx(0.85)
    assert trajectory.memory_policy == "full_history"
    assert trajectory.lower_terminal_and_prehistory.kind == "point_initial_value"
    assert trajectory.lower_terminal_and_prehistory.coverage == (2.0, 2.0)
    assert FRACTIONAL_TRAJECTORY_WARNING in trajectory.scientific_warnings

    base = dict(
        t=[0.0, 1.0],
        x=[1.0, 1.5],
        system_kind="fractional_continuous",
        memory_policy="full_history",
        lower_terminal_and_prehistory={
            "kind": "point_initial_value",
            "lower_terminal": 0.0,
        },
    )
    with pytest.raises(ValueError, match="derivative_definition"):
        TrajectoryInput(**base, order=0.9)
    with pytest.raises(ValueError, match="order specification"):
        TrajectoryInput(**base, derivative_definition="caputo")
    with pytest.raises(ValueError, match="memory_policy"):
        TrajectoryInput(
            **{key: value for key, value in base.items() if key != "memory_policy"},
            derivative_definition="caputo",
            order=0.9,
        )


def test_fractional_contract_does_not_hide_unknown_prehistory() -> None:
    trajectory = _fractional_trajectory(
        lower_terminal_and_prehistory=PrehistorySpec(
            kind="unknown",
            lower_terminal=0.0,
            coverage=(0.0, 2.0),
        )
    )
    assert trajectory.lower_terminal_and_prehistory.kind == "unknown"
    assert any("explicitly unknown" in warning for warning in trajectory.scientific_warnings)


def test_sampled_prehistory_is_readonly_and_roundtrips() -> None:
    source_times = np.array([-1.0, -0.5, 0.0])
    source_values = np.column_stack((source_times, source_times**2))
    history = PrehistorySpec(
        kind="sampled",
        lower_terminal=-1.0,
        sample_times=source_times,
        sample_values=source_values,
        metadata={"source": "declared fixture"},
    )
    source_times[0] = -99.0
    source_values[0, 0] = -99.0

    assert history.coverage == (-1.0, 0.0)
    assert history.sample_times[0] == -1.0
    assert history.sample_values[0, 0] == -1.0
    assert not history.sample_times.flags.writeable
    assert not history.sample_values.flags.writeable
    restored = PrehistorySpec.from_mapping(history.to_serializable())
    assert restored.kind == history.kind
    np.testing.assert_array_equal(restored.sample_times, history.sample_times)
    np.testing.assert_array_equal(restored.sample_values, history.sample_values)


def test_analytic_prehistory_requires_coverage_and_reference() -> None:
    history = PrehistorySpec(
        kind="analytic_reference",
        lower_terminal=-2.0,
        coverage=(-2.0, 0.0),
        analytic_reference="examples.history:constant_one",
    )
    assert history.analytic_reference.endswith("constant_one")
    with pytest.raises(ValueError, match="requires"):
        PrehistorySpec(kind="analytic_reference", lower_terminal=-2.0)


def test_not_applicable_history_rejects_hidden_data() -> None:
    with pytest.raises(ValueError, match="cannot carry"):
        PrehistorySpec(kind="not_applicable", lower_terminal=0.0)
    with pytest.raises(ValueError, match="not_applicable"):
        _fractional_trajectory(
            lower_terminal_and_prehistory=PrehistorySpec(kind="not_applicable")
        )
    with pytest.raises(ValueError, match="integer or generic"):
        _integer_trajectory(
            lower_terminal_and_prehistory=PrehistorySpec(
                kind="point_initial_value",
                lower_terminal=0.0,
            )
        )


def test_component_supports_index_and_projection_label() -> None:
    trajectory = _integer_trajectory()
    np.testing.assert_array_equal(
        trajectory.component(1),
        trajectory.component("velocity"),
    )
    assert not trajectory.component(0).flags.writeable
    with pytest.raises(KeyError):
        trajectory.component("missing")
    with pytest.raises(TypeError):
        trajectory.component(1.0)
    with pytest.raises(IndexError):
        trajectory.component(4)


def test_projection_labels_are_complete_and_unique() -> None:
    with pytest.raises(ValueError, match="one label"):
        _integer_trajectory(projection=("only_one",))
    with pytest.raises(ValueError, match="unique"):
        _integer_trajectory(projection=("same", "same"))


def test_trajectory_fingerprint_is_deterministic_and_semantic() -> None:
    first = _integer_trajectory()
    second = _integer_trajectory()
    changed_projection = _integer_trajectory(projection=("q", "velocity"))
    changed_solver = _integer_trajectory(
        solver_and_tolerances={"method": "rk4", "step": 0.1}
    )

    assert len(first.fingerprint()) == 64
    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() != changed_projection.fingerprint()
    assert first.fingerprint() != changed_solver.fingerprint()


def test_trajectory_serialization_is_strict_json() -> None:
    trajectory = _fractional_trajectory(
        order=np.array([0.85]),
        metadata={"seed": 7, "tags": ["integer-compatible", "fractional"]},
    )
    payload = trajectory.to_serializable()
    rendered = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert '"derivative_definition": "caputo"' in rendered
    assert payload["lower_terminal_and_prehistory"]["kind"] == "point_initial_value"


def test_integer_simulation_result_adapter_preserves_map_semantics() -> None:
    simulation = SimulationResult(
        times=np.arange(4, dtype=float),
        states=np.arange(8, dtype=float).reshape(4, 2),
        status="ok",
        system_name="fixture_map",
        system_kind="map",
        method="map_iteration",
        parameters={"r": 3.8},
        requested_steps=3,
        completed_steps=3,
        metadata={"claims": "trajectory_only"},
    )
    trajectory = TrajectoryInput.from_simulation_result(
        simulation,
        projection=("u", "v"),
    )
    assert trajectory.system_kind == "integer_map"
    assert trajectory.time_coordinate == "iteration_index"
    assert trajectory.memory_policy == "not_applicable"
    assert trajectory.metadata["system_name"] == "fixture_map"
    assert trajectory.solver_and_tolerances["method"] == "map_iteration"


def test_fractional_simulation_adapter_preserves_both_time_coordinates() -> None:
    simulation = SimulationResult(
        times=np.array([1.0, 1.1, 1.21]),
        states=np.array([[1.0], [0.9], [0.8]]),
        status="ok",
        system_name="fixture_fractional",
        system_kind="flow",
        method="caputo_hadamard_abm_pece",
        parameters={"gain": 2.0},
        step_size=np.log(1.1),
        requested_steps=2,
        completed_steps=2,
        metadata={
            "fractional_problem": {
                "derivative": "caputo_hadamard",
                "orders": [0.8],
                "memory_policy": "full_history",
                "lower_terminal": 1.0,
                "initial_condition_kind": "classical",
                "history_window": None,
                "method_options": {"corrector_iterations": 1},
                "kernel_parameters": {},
                "reference_keys": ["kilbas2006"],
            }
        },
        integrator_times=np.array([0.0, np.log(1.1), 2.0 * np.log(1.1)]),
        grid_coordinate="log_t_over_lower_terminal",
        backend="python",
        backend_info={"requested": "python"},
    )
    trajectory = TrajectoryInput.from_simulation_result(simulation)
    assert trajectory.system_kind == "fractional_continuous"
    assert trajectory.time_coordinate == "physical_time"
    assert not trajectory.sampled_uniformly
    assert trajectory.derivative_definition == "caputo_hadamard"
    assert trajectory.order == (0.8,)
    assert trajectory.lower_terminal_and_prehistory.lower_terminal == 1.0
    np.testing.assert_allclose(
        trajectory.solver_and_tolerances["integration_coordinate_times"],
        simulation.integrator_times,
    )
    assert (
        trajectory.solver_and_tolerances["grid_coordinate"]
        == "log_t_over_lower_terminal"
    )


def test_analysis_result_is_immutable_serializable_and_traceable() -> None:
    trajectory = _integer_trajectory()
    counts = np.array([1, 2, 3], dtype=np.uint64)
    result = AnalysisResult(
        method="bandt_pompe_permutation_entropy",
        values={"entropy": 0.75, "counts": counts},
        parameters={"embedding_dimension": 3, "delay": 1},
        backend="hafo_numba",
        status="finite_numerical_diagnostic",
        trajectory_fingerprint=trajectory.fingerprint(),
        package_versions_and_hashes={"hafo": "1.1.0"},
        warnings=("Finite-sample estimate; not a chaos certificate.",),
        references=("10.1103/PhysRevLett.88.174102",),
    )
    counts[0] = 99

    assert result.values["counts"][0] == 1
    assert not result.values["counts"].flags.writeable
    assert result.references == ("10.1103/PhysRevLett.88.174102",)
    payload = result.to_serializable()
    assert payload["values"]["counts"] == [1, 2, 3]
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("backend", "gpu_magic", "backend"),
        ("status", "proven_chaotic", "status"),
        ("trajectory_fingerprint", "abc", "SHA-256"),
    ],
)
def test_analysis_result_rejects_unregistered_contract_tokens(
    field: str,
    value: str,
    match: str,
) -> None:
    kwargs: dict[str, object] = {
        "method": "test_method",
        "values": {"value": 1.0},
        "parameters": {},
        "backend": "hafo_python",
        "status": "experimental",
        "trajectory_fingerprint": "0" * 64,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        AnalysisResult(**kwargs)


def test_contract_rejects_nonfinite_samples_and_metadata() -> None:
    with pytest.raises(ValueError, match="finite"):
        _integer_trajectory(x=np.array([[0.0, 1.0], [np.nan, 2.0]]))
    with pytest.raises(TypeError, match="non-finite"):
        _integer_trajectory(metadata={"bad": float("inf")})
    with pytest.raises(TypeError, match="non-finite"):
        AnalysisResult(
            method="test_method",
            values={"value": np.array([np.nan])},
            parameters={},
            backend="hafo_python",
            status="experimental",
            trajectory_fingerprint="0" * 64,
        )
