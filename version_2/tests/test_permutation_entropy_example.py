from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from examples import permutation_entropy_integer_fractional as example


@pytest.fixture(scope="module")
def example_record() -> dict[str, Any]:
    record = example.run_example()
    json.dumps(record, allow_nan=False)
    return record


def test_example_uses_public_integer_fractional_and_trajectory_contracts(
    example_record: dict[str, Any],
) -> None:
    record = example_record
    assert record["scope"] == "finite_sample_empirical_ordinal_pattern_diagnostic"
    contract = record["analysis_contract"]
    assert contract == {
        "observable": "position",
        "embedding_dimension": 4,
        "delay_samples": 2,
        "window_order": "forward_x_s_plus_k_delay",
        "tie_policy": "stable_index",
        "ordinal_encoding": "zero_based_lexicographic_lehmer_rank",
        "log_base": 2.0,
        "estimator": "relative_frequency_plugin_shannon_entropy",
        "backend": "python",
    }

    integer = record["integer"]
    fractional = record["fractional"]
    assert integer["trajectory_status"] == fractional["trajectory_status"] == "ok"
    assert integer["declared_order"] == 1.0
    assert integer["derivative"] == "ordinary_first_derivative"
    assert fractional["declared_order"] == 0.85
    assert fractional["derivative"] == "caputo"
    assert fractional["method"] == "caputo_abm_pece"
    assert fractional["memory_policy"] == "full_history"

    integer_trajectory = integer["trajectory"]
    fractional_trajectory = fractional["trajectory"]
    assert integer_trajectory["sample_count"] == 81
    assert fractional_trajectory["sample_count"] == 81
    assert integer_trajectory["projection"] == ["position", "velocity"]
    assert fractional_trajectory["projection"] == ["position", "velocity"]
    assert integer_trajectory["system_kind"] == "integer_flow"
    assert integer_trajectory["derivative_definition"] is None
    assert integer_trajectory["order"] is None
    assert integer_trajectory["memory_policy"] == "not_applicable"
    assert integer_trajectory["prehistory_kind"] == "not_applicable"
    assert integer_trajectory["solver_method"] == "rk4"
    assert fractional_trajectory["system_kind"] == "fractional_continuous"
    assert fractional_trajectory["derivative_definition"] == "caputo"
    assert fractional_trajectory["order"] == [0.85, 0.85]
    assert fractional_trajectory["memory_policy"] == "full_history"
    assert fractional_trajectory["prehistory_kind"] == "point_initial_value"
    assert fractional_trajectory["lower_terminal"] == 0.0
    assert fractional_trajectory["solver_method"] == "caputo_abm_pece"
    for trajectory in (integer_trajectory, fractional_trajectory):
        assert trajectory["sampled_uniformly"] is True
        assert trajectory["uniform_step"] == pytest.approx(0.05)
        assert len(trajectory["fingerprint"]) == 64
        assert trajectory["metadata"]["selected_observable"] == "position"
        assert "finite scalar-projection" in trajectory["metadata"][
            "evidence_boundary"
        ]
    assert integer_trajectory["fingerprint"] != fractional_trajectory["fingerprint"]


def test_example_entropy_records_are_dense_finite_and_traceable(
    example_record: dict[str, Any],
) -> None:
    record = example_record
    integer = record["integer"]["permutation_entropy"]
    fractional = record["fractional"]["permutation_entropy"]

    for result, expected_kind, expected_derivative, expected_memory in (
        (integer, "integer_flow", None, "not_applicable"),
        (fractional, "fractional_continuous", "caputo", "full_history"),
    ):
        counts = np.asarray(result["counts"], dtype=np.uint64)
        probabilities = np.asarray(result["probabilities"], dtype=float)
        assert result["sample_count"] == 81
        assert result["total_windows"] == 75
        assert result["valid_windows"] == 75
        assert result["tied_windows"] == 0
        assert result["omitted_windows"] == 0
        assert result["possible_patterns"] == math.factorial(4)
        assert counts.shape == probabilities.shape == (24,)
        assert int(counts.sum()) == result["valid_windows"]
        np.testing.assert_allclose(
            probabilities,
            counts.astype(float) / float(result["valid_windows"]),
            rtol=0.0,
            atol=0.0,
        )
        assert probabilities.sum() == pytest.approx(1.0, abs=1e-15)
        positive = probabilities[probabilities > 0.0]
        expected_entropy = float(-np.sum(positive * np.log2(positive)))
        assert result["entropy"] == pytest.approx(expected_entropy, abs=1e-15)
        assert result["maximum_entropy"] == pytest.approx(math.log2(24), abs=1e-15)
        assert result["normalized_entropy"] == pytest.approx(
            expected_entropy / math.log2(24),
            abs=1e-15,
        )
        assert 0.0 <= result["normalized_entropy"] <= 1.0
        assert result["requested_backend"] == result["backend"] == "python"
        assert result["projection"] == "position"
        assert result["trajectory_system_kind"] == expected_kind
        assert result["derivative_definition"] == expected_derivative
        assert result["memory_policy"] == expected_memory
        assert result["status"] == "ok"
        assert result["evidence_scope"] == record["scope"]
        assert result["analysis_envelope"] == {
            "method": "bandt_pompe_permutation_entropy",
            "backend": "hafo_python",
            "status": "finite_numerical_diagnostic",
            "evidence_scope": record["scope"],
            "trajectory_fingerprint": result["trajectory_fingerprint"],
        }
        warnings = " ".join(result["warnings"]).lower()
        assert "finite-sample" in warnings
        assert "not by itself proof of chaos" in warnings

    assert integer["counts"] != fractional["counts"]
    assert integer["counts"] == [
        13,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        58,
    ]
    assert fractional["counts"] == [
        16,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        55,
    ]
    assert integer["trajectory_fingerprint"] != fractional["trajectory_fingerprint"]
    assert any(
        "hereditary state" in warning.lower()
        for warning in fractional["warnings"]
    )


def test_example_states_the_evidence_boundary(
    example_record: dict[str, Any],
) -> None:
    claims = example_record["claims"].lower()
    comparison = example_record["comparison_policy"].lower()

    assert "finite" in claims
    assert "not entropy rates" in claims
    assert "not proof of chaos" in claims
    assert "hiddenness" in claims
    assert "not the complete hereditary state" in claims
    assert "not asserted to be physically equivalent" in comparison


def test_main_prints_strict_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {
        "scope": "finite_sample_empirical_ordinal_pattern_diagnostic",
        "claims": "not proof of chaos, attraction, or hiddenness",
    }
    monkeypatch.setattr(example, "run_example", lambda: payload)

    example.main()

    assert json.loads(capsys.readouterr().out) == payload
