from __future__ import annotations

import json

import pytest

from examples import multi_term_caputo_relaxation as example


@pytest.fixture(scope="module")
def record() -> dict[str, object]:
    payload = example.run_example(use_acceleration=False)
    json.dumps(payload, allow_nan=False)
    return payload


def test_example_preserves_multi_term_equation_semantics(
    record: dict[str, object],
) -> None:
    assert record["model"] == "forced_multi_scale_caputo_relaxation"
    assert record["method"] == "multi_term_caputo_l1"
    assert record["underlying_method"] == "distributed_order_caputo_l1"
    assert record["status"] == "ok"
    assert record["memory_policy"] == "full_history"
    assert record["canonical_orders"] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert record["canonical_coefficients"] == pytest.approx([0.4, 0.7, 0.75])
    assert record["coefficient_sum"] == pytest.approx(1.85)
    assert record["normalization"] == "none"
    assert record["measure_kind"] == "finite_discrete_atomic_order_measure"
    assert record["continuous_order_quadrature_used"] is False
    assert record["duplicate_terms_coalesced"] == 1
    assert record["zero_terms_removed"] == 1
    assert record["alpha_one_handling"] == "exact_backward_euler_limit"


def test_example_matches_its_independent_affine_solution(
    record: dict[str, object],
) -> None:
    assert record["sample_count"] == example.N_STEPS + 1
    assert record["actual_upper_terminal"] == pytest.approx(0.5)
    assert record["maximum_absolute_error"] < 2.0e-12
    assert record["final_state"] == pytest.approx(
        record["exact_final_state"],
        abs=2.0e-12,
    )


def test_example_states_reuse_evidence_and_claim_boundaries(
    record: dict[str, object],
) -> None:
    assert record["implementation_reuse"] == (
        "distributed_order_combined_l1_kernel_without_solver_reconstruction"
    )
    assert len(record["references"]) == 4  # type: ignore[arg-type]
    assert len(record["scispace_paper_ids"]) == 4  # type: ignore[arg-type]
    claims = str(record["claims"]).lower()
    assert "finite" in claims
    assert "no general convergence" in claims
    assert "chaos" in claims
    assert "hiddenness" in claims


def test_main_prints_strict_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"status": "ok", "claims": "finite numerical evidence only"}
    monkeypatch.setattr(example, "run_example", lambda: payload)

    example.main()

    assert json.loads(capsys.readouterr().out) == payload
