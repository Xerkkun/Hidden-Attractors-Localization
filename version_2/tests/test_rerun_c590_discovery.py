from __future__ import annotations

import json

import numpy as np

from tools.reconstruct_c590_search_provenance import (
    EXPECTED_C590_INTEGER_SEED,
    EXPECTED_GLOBAL_SEED,
    GLOBAL_INDEX,
    LOCAL_INDEX,
)
from tools.rerun_c590_discovery import (
    EXECUTABLE_STAGES,
    build_plan,
    execute_banks,
)


def test_plan_marks_recovered_integer_route_executable() -> None:
    plan = build_plan("integer-route")

    assert plan["overall_status"] == "ready_to_execute"
    assert plan["blocked_stages"] == []
    assert tuple(row["stage"] for row in plan["stages"]) == EXECUTABLE_STAGES
    assert all(row["status"] == "executable" for row in plan["stages"])


def test_full_plan_includes_executable_caputo_seed9_route() -> None:
    plan = build_plan("all")

    assert plan["overall_status"] == "ready_to_execute"
    assert plan["blocked_stages"] == []
    by_name = {row["stage"]: row for row in plan["stages"]}
    assert by_name["integer-global"]["contract"]["cases"] == 2400
    assert by_name["integer-local"]["contract"]["cases"] == 1000
    assert by_name["variational-shortlist"]["contract"]["candidates"] == 30
    caputo = by_name["caputo-seed9"]
    assert caputo["status"] == "executable"
    assert caputo["contract"]["q_scan"]["q_values"] == [
        0.9995,
        0.9998,
        0.9999,
        0.99995,
    ]
    assert caputo["contract"]["seed_extraction"]["count"] == 16
    assert caputo["contract"]["seed_extraction"][
        "recorded_source_indices"
    ] == [30001, *range(32000, 60001, 2000)]
    assert caputo["contract"]["seed_extraction"][
        "recorded_source_times"
    ][0] == 150.00499999993608
    assert caputo["contract"]["seed_extraction"][
        "recorded_source_times"
    ][9] == 239.99999999985423
    assert caputo["contract"]["seed_extraction"][
        "recorded_source_times"
    ][-1] == 299.9999999997997
    assert caputo["contract"]["seed_extraction"][
        "recorded_source_states"
    ][9] == [
        5.864244979081692,
        1.5847111486491057,
        3.2155806477633915,
    ]
    assert caputo["contract"]["survivor_long_audit"][
        "recorded_passing_indices"
    ] == [9]


def test_execute_banks_persists_exact_selected_rows(tmp_path) -> None:
    result = execute_banks(tmp_path)

    provenance = json.loads(
        (tmp_path / "search_provenance.json").read_text(encoding="utf-8")
    )
    with np.load(tmp_path / "search_banks.npz") as banks:
        assert banks["global_parameters"].shape == (2400, 6)
        assert banks["global_target_seeds"].shape == (2400, 3)
        assert banks["local_parameters"].shape == (1000, 6)
        assert banks["local_target_seeds"].shape == (1000, 3)
        assert np.array_equal(
            banks["global_target_seeds"][GLOBAL_INDEX],
            EXPECTED_GLOBAL_SEED,
        )
        assert np.array_equal(
            banks["local_target_seeds"][LOCAL_INDEX],
            EXPECTED_C590_INTEGER_SEED,
        )

    assert result["status"] == "completed"
    assert provenance["verification"]["global_max_abs_error"] == 0.0
    assert provenance["verification"]["local_max_abs_error"] == 0.0
