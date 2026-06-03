"""Freeze audit for new JSON artifacts and explicit legacy boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hidden_attractors.reproducibility import extract_run_metadata, validate_run_metadata


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("validation", "outputs", "artifacts")
LEGACY_JSON_PREFIXES = (
    "validation/00_manifest/",
    "validation/01_numerical_contract/",
    "validation/02_algebraic_validation/",
    "validation/03_seed_generation/",
    "validation/04_soft_precheck/",
    "validation/05_continuation/",
    "validation/06_post_continuation_filter/",
    "validation/07_dynamic_reference/",
    "validation/08_robustness/",
    "validation/09_hiddenness_tests/",
    "validation/10_diagnostics/",
    "validation/chaos_validation/dynamics_diagnostics/",
    "validation/chaos_validation/lyapunov_methods/",
    "validation/outputs/",
    "validation/reference_cases/",
)


def _json_files() -> list[Path]:
    return [
        path
        for root_name in SCAN_ROOTS
        if (root := ROOT / root_name).exists()
        for path in root.rglob("*.json")
    ]


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _legacy(path: Path) -> bool:
    relative = _relative(path)
    return any(relative.startswith(prefix) for prefix in LEGACY_JSON_PREFIXES)


def _find_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_find_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_find_key(item, key) for item in value)
    return False


def test_new_json_outputs_do_not_emit_obsolete_strong_flags() -> None:
    violations = []
    for path in _json_files():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if _legacy(path):
            continue
        if _find_key(payload, "chaos_verified"):
            violations.append(f"{_relative(path)} contains chaos_verified")
        if _find_key(payload, "hidden_verified"):
            gate = payload.get("candidate_gate", {}) if isinstance(payload, dict) else {}
            metadata = extract_run_metadata(payload if isinstance(payload, dict) else {})
            if not (
                gate.get("verdict") == "hiddenness_supported_under_tested_neighborhoods"
                and gate.get("missing_conditions") == []
                and metadata is not None
                and validate_run_metadata(metadata) == []
            ):
                violations.append(f"{_relative(path)} contains unsupported hidden_verified")
    assert violations == []


def test_new_reports_centralize_repeated_caution_text() -> None:
    phrases = ("does not certify chaos", "does not certify hiddenness", "not a proof")
    violations = []
    for path in _json_files():
        if _legacy(path):
            continue
        text = path.read_text(encoding="utf-8").lower()
        for phrase in phrases:
            if text.count(phrase) > 1:
                violations.append(f"{_relative(path)} repeats {phrase!r}")
    assert violations == []


def test_phase_f_freeze_summary_uses_positive_evidence_vocabulary() -> None:
    path = ROOT / "validation" / "chaos_validation" / "phase_F_closure" / "phase_F_closure_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["phase_F_frozen"] is True
    assert payload["evidence_layer"] == "finite_time_chaos_evidence"
    assert "chaos_verified" not in path.read_text(encoding="utf-8")


def test_f6_f7_phase_f_do_not_contain_obsolete_vocabulary() -> None:
    forbidden = [
        "chaos_verified",
        "hidden_verified",
        "non_certifying",
        "no_chaos_certification",
        "no_hiddenness_certification",
    ]
    files = [
        ROOT / "validation/chaos_validation/integrated_chaos_validator/integrated_chaos_summary.json",
        ROOT / "validation/chaos_validation/method_comparison/method_comparison_summary.json",
        ROOT / "validation/chaos_validation/phase_F_closure/phase_F_closure_summary.json",
    ]
    violations = []
    for path in files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for term in forbidden:
            if term.lower() in text:
                violations.append(f"{path.name} contains forbidden term: {term}")
    assert violations == []

