from __future__ import annotations

import numpy as np

from tools.reconstruct_c590_search_provenance import (
    EXPECTED_C590_INTEGER_SEED,
    EXPECTED_C590_PARAMETERS,
    EXPECTED_EXTRACTED_SEEDS,
    EXPECTED_FINAL_SEED,
    EXPECTED_GLOBAL_PARAMETERS,
    EXPECTED_GLOBAL_SEED,
    generate_local_bank,
    reconstruct,
)
from tools.rerun_c590_discovery import (
    _aggregate_stage_regression,
    _bounded_case,
    _portable_path,
)


def test_c590_parameter_and_seed_provenance_is_exact() -> None:
    payload = reconstruct()
    selection = payload["local_exploration"]["selection"]
    refinement = payload["fractional_refinement"]["seed_extraction"]

    assert selection["zero_based_index"] == 590
    assert selection["parameters"] == EXPECTED_C590_PARAMETERS
    assert np.array_equal(selection["integer_order_seed"], EXPECTED_C590_INTEGER_SEED)
    assert refinement["selected_zero_based_index"] == 9
    assert refinement["source_indices"][0] == 30001
    assert refinement["source_times"][0] == 150.00499999993608
    assert refinement["selected_source_time"] == 239.99999999985423
    assert refinement["selected_source_time_nominal"] == 240.0
    assert np.array_equal(
        refinement["source_states"],
        EXPECTED_EXTRACTED_SEEDS,
    )
    assert np.array_equal(
        refinement["source_states"][9],
        EXPECTED_FINAL_SEED,
    )
    assert np.array_equal(refinement["selected_seed"], EXPECTED_FINAL_SEED)
    assert payload["verification"]["global_max_abs_error"] == 0.0
    assert payload["verification"]["local_max_abs_error"] == 0.0


def test_local_bank_preserves_archived_floating_operation_order() -> None:
    parameters, seeds = generate_local_bank(
        EXPECTED_GLOBAL_PARAMETERS,
        EXPECTED_GLOBAL_SEED,
    )
    archived_x = (
        EXPECTED_GLOBAL_SEED[0]
        * EXPECTED_GLOBAL_PARAMETERS["rho"]
        / parameters["rho"]
    )

    assert np.array_equal(seeds[:, 0], archived_x)
    assert seeds[603, 0] == 8.602648964531289
    assert seeds[709, 0] == 7.517753871283924

    reassociated_x = EXPECTED_GLOBAL_SEED[0] * (
        EXPECTED_GLOBAL_PARAMETERS["rho"] / parameters["rho"]
    )
    assert seeds[603, 0] != reassociated_x[603]
    assert seeds[709, 0] != reassociated_x[709]


def test_boundedness_requires_status_and_declared_norm_limit() -> None:
    assert _bounded_case({"status": "ok", "max_norm": 49.9})
    assert not _bounded_case({"status": "ok", "max_norm": 270.395})
    assert not _bounded_case({"status": "diverged", "max_norm": 18.0})


def test_aggregate_stage_regression_propagates_mismatch() -> None:
    status, matched = _aggregate_stage_regression(
        [
            {"status": "completed"},
            {
                "status": "completed_recorded_regression_mismatch",
                "recorded_regression_matched": False,
            },
        ]
    )

    assert status == "completed_recorded_regression_mismatch"
    assert matched is False


def test_portable_path_is_relative_and_uses_posix_separators(
    tmp_path,
) -> None:
    root = tmp_path / "artifact"
    source = root / "integer_local" / "screen.json"

    assert _portable_path(source, root) == "integer_local/screen.json"
