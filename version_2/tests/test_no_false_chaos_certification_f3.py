"""F3 must not promote chaos, hiddenness, or diagnostics completion."""

from __future__ import annotations

import json
from pathlib import Path

from hidden_attractors.analysis.lyapunov_methods import LYAPUNOV_METHODS


ROOT = Path(__file__).resolve().parents[1]


def test_f3_registry_contains_no_positive_certification_claims() -> None:
    for method in (
        "fractional_cloned_dynamics_abm_gs_published",
        "fractional_cloned_dynamics_abm_qr",
    ):
        text = " ".join(LYAPUNOV_METHODS[method].warnings).lower()
        assert "chaos_verified: true" not in text
        assert "hidden_verified: true" not in text
        assert LYAPUNOV_METHODS[method].no_chaos_certification is True
        assert LYAPUNOV_METHODS[method].no_hiddenness_certification is True


def test_f3_summary_stays_pending_without_published_run() -> None:
    summary = json.loads(
        (
            ROOT
            / "validation"
            / "chaos_validation"
            / "lyapunov_methods"
            / "fractional_cloned_dynamics_abm_gs_published"
            / "validation_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["status"] == "published_benchmarks_not_run"
    assert summary["validated"] is False
    assert summary["validated_against_published_benchmarks"] is False
    assert summary["certifications"]["chaos_certified_by_this_pipeline"] is False
    assert summary["certifications"]["hiddenness_certified_by_this_pipeline"] is False


def test_diagnostics_stage_remains_partial() -> None:
    summary = json.loads(
        (ROOT / "validation" / "10_diagnostics" / "diagnostics_validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["status"] == "diagnostics_partial_current_protocol"
