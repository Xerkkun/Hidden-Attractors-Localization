"""Contract checks for the documented Fischer 2020 F3 reproduction limits."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHOD_DIR = (
    ROOT
    / "validation"
    / "chaos_validation"
    / "lyapunov_methods"
    / "fractional_cloned_dynamics_abm_gs_published"
)
SUMMARY_PATH = METHOD_DIR / "validation_summary.json"
LIMITATIONS_PATH = (
    METHOD_DIR / "discrepancy_diagnostics" / "fischer2020_reproduction_limitations.md"
)


def test_reproduction_limitations_document_exists_and_covers_missing_data() -> None:
    text = LIMITATIONS_PATH.read_text(encoding="utf-8").lower()
    required = {
        "clone initialization and restart protocol",
        "treatment of fractional memory",
        "abm predictor-corrector convention",
        "t_clone",
        "n_c",
        "h_c",
        "`k`",
        "orthonormalization convention",
        "log accumulation and ordering of exponents",
        "incommensurate fractional systems",
        "arithmetic precision, software, and internal tolerances",
        "jerk system implementation details",
        "exponential",
    }
    assert required <= {phrase for phrase in required if phrase in text}


def test_official_summary_tracks_closed_documented_discrepancies() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    diagnostics = summary["discrepancy_diagnostics"]
    closure = diagnostics["diagnostic_closure"]
    assert diagnostics["reproduction_limitations"] == (
        "discrepancy_diagnostics/fischer2020_reproduction_limitations.md"
    )
    assert closure["status"] == "closed_with_documented_discrepancies"
    assert closure["additional_sweeps_required_for_current_scope"] is False
    assert closure["validated_after_closure"] is False
    assert closure["validated_after_diagnostics"] is False
    assert summary["validated"] is False
    assert summary["validated_against_published_benchmarks"] is False
    serialized = json.dumps(summary)
    assert "chaos_verified" not in serialized
    assert "hidden_verified" not in serialized
