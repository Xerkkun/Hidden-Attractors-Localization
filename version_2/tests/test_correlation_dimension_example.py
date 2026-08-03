from __future__ import annotations

import json

import numpy as np
import pytest

from examples import correlation_dimension_integer_fractional as example


def test_integer_and_fractional_example_is_small_deterministic_and_explicit() -> None:
    record = example.run_example()

    assert record["scope"] == "finite_sample_empirical_trajectory_diagnostic"
    contract = record["analysis_contract"]
    assert contract["radii"] == example.RADII.tolist()
    assert contract["fit_radius_range"] == list(example.FIT_RADIUS_RANGE)
    assert contract["fit_range_selection"] == (
        "explicit_caller_supplied_inclusive_range"
    )
    assert contract["theiler_window_samples"] == 2
    assert contract["threshold"] == "distance_strictly_less_than_radius"

    integer = record["integer"]
    fractional = record["fractional"]
    assert integer["trajectory_status"] == fractional["trajectory_status"] == "ok"
    assert integer["order"] == 1.0
    assert integer["derivative"] == "ordinary_first_derivative"
    assert fractional["order"] == 0.85
    assert fractional["derivative"] == "caputo"
    assert fractional["memory_policy"] == "full_history"
    assert integer["samples_supplied"] == fractional["samples_supplied"] == 71
    assert integer["feature_dimension"] == fractional["feature_dimension"] == 2
    assert integer["eligible_pairs"] == fractional["eligible_pairs"] == 2346
    assert integer["backend"] == fractional["backend"] == "python"
    assert integer["fit_indices"] == fractional["fit_indices"] == [3, 4, 5, 6]

    assert integer["counts"] == [0, 0, 0, 58, 142, 300, 537, 855, 1298]
    assert fractional["counts"] == [0, 0, 7, 61, 164, 324, 553, 878, 1316]
    for result in (integer, fractional):
        sums = np.asarray(result["correlation_sums"], dtype=float)
        fit_sums = sums[np.asarray(result["fit_indices"], dtype=int)]
        assert np.all((fit_sums > 0.0) & (fit_sums < 1.0))
        assert np.isfinite(result["dimension_slope"])
        assert np.isfinite(result["intercept"])
        assert 0.0 <= result["r_squared"] <= 1.0
        assert result["regression_standard_error"] >= 0.0
        assert result["evidence_scope"] == record["scope"]
        assert "hereditary state" in result["fractional_state_caveat"]


def test_example_keeps_the_finite_evidence_boundary() -> None:
    record = example.run_example()
    claims = record["claims"].lower()

    assert "finite" in claims
    assert "not proof of chaos" in claims
    assert "hiddenness" in claims
    assert "not the complete hereditary state" in claims
    assert "q=1" in record["integer"]["projection"]
    assert "caputo q=0.85" in record["fractional"]["projection"].lower()


def test_main_prints_the_run_record_as_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {
        "scope": "finite_sample_empirical_trajectory_diagnostic",
        "claims": "not proof of chaos or hiddenness",
    }
    monkeypatch.setattr(example, "run_example", lambda: payload)

    example.main()

    assert json.loads(capsys.readouterr().out) == payload
