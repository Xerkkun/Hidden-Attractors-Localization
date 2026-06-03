from __future__ import annotations

import numpy as np

from hidden_attractors.reproducibility import (
    collect_lure_metadata,
    collect_run_metadata,
    extract_run_metadata,
    metadata_to_jsonable,
    validate_hiddenness_promotion_metadata,
    validate_run_metadata,
)


def test_numpy_arrays_are_jsonable() -> None:
    assert metadata_to_jsonable({"x": np.array([1.0, 2.0])}) == {"x": [1.0, 2.0]}


def test_collected_metadata_records_scipy_and_default_tolerances() -> None:
    metadata = metadata_to_jsonable(
        collect_run_metadata(
            run_id="pytest",
            workflow="pytest",
            system="test",
            q=0.9,
            h=0.01,
            t_final=10.0,
            t_burn=1.0,
            memory_mode="full",
            integrator_name="abm",
            integrator_backend="python",
            caputo=True,
        )
    )
    assert metadata["software"]["scipy_version"]
    assert metadata["tolerances"]["lyapunov_positive_tol"] > 0.0
    assert validate_run_metadata(metadata) == []


def test_tolerances_are_required(valid_run_metadata) -> None:
    metadata = metadata_to_jsonable(valid_run_metadata)
    metadata.pop("tolerances")
    assert "tolerances must be an object" in validate_run_metadata(metadata)


def test_continuation_eta_path_is_required_when_used(valid_run_metadata) -> None:
    metadata = metadata_to_jsonable(valid_run_metadata)
    metadata["continuation"].update({"used": True, "final_eta": 1.0, "eta_path": []})
    assert "continuation.eta_path is required when continuation.used=true" in validate_run_metadata(metadata)


def test_metadata_aliases_are_read_and_writing_is_canonical(valid_run_metadata) -> None:
    assert extract_run_metadata({"run_metadata": valid_run_metadata}) == valid_run_metadata
    assert extract_run_metadata({"reproducibility_metadata": valid_run_metadata}) == valid_run_metadata


def test_full_and_window_memory_are_explicit(valid_run_metadata) -> None:
    full = metadata_to_jsonable(valid_run_metadata)
    assert validate_run_metadata(full) == []
    window = metadata_to_jsonable(valid_run_metadata)
    window["numerical_contract"]["memory"].update(
        {"mode": "finite_window", "M": 100, "memory_window_steps": 100, "memory_window_time": 1.0, "is_full_caputo": False}
    )
    assert validate_run_metadata(window) == []
    assert validate_hiddenness_promotion_metadata(window)


def test_callable_lure_psi_is_stored_as_text() -> None:
    class Lure:
        matrix = np.array([[-1.0]])
        input_vector = np.array([1.0])
        output_vector = np.array([1.0])

        @staticmethod
        def nonlinearity(value: float) -> float:
            return value

    lure = metadata_to_jsonable(
        collect_lure_metadata(Lure(), transfer_convention="P-sI", harmonic_condition="1+kW=0")
    )
    assert isinstance(lure["scalar_nonlinearity"], str)
    assert "nonlinearity" in lure["scalar_nonlinearity"]


def test_incomplete_metadata_blocks_strong_label(valid_run_metadata) -> None:
    metadata = metadata_to_jsonable(valid_run_metadata)
    metadata["software"]["git_commit"] = "unknown"
    assert any("known software.git_commit" in error for error in validate_hiddenness_promotion_metadata(metadata))
