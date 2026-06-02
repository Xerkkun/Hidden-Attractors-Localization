"""F4 must never promote method validation or chaos/hiddenness certification."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
F4_ROOT = (
    PROJECT_ROOT
    / "validation"
    / "chaos_validation"
    / "lyapunov_methods"
    / "F4_internal_validation"
)


def _summaries() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in F4_ROOT.rglob("*.json")
    ]


def _walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_f4_certifications_remain_false() -> None:
    for summary in _summaries():
        certifications = summary.get("certifications")
        if certifications is None:
            continue
        assert certifications["chaos_certified_by_this_pipeline"] is False
        assert certifications["hiddenness_certified_by_this_pipeline"] is False
        assert certifications["fractional_lyapunov_validated_by_f4"] is False
        assert certifications["caputo_lyapunov_validated_by_f4"] is False
        assert certifications["published_decimal_reproduction_implies_method_validation"] is False


def test_f4_contains_no_positive_forbidden_claim() -> None:
    forbidden = {
        "chaos_verified",
        "hidden_verified",
        "fractional_lyapunov_validated",
        "caputo_lyapunov_validated",
        "poincare_proves_chaos",
    }
    for summary in _summaries():
        for key, value in _walk(summary):
            assert not (key in forbidden and value is True), f"{key}=true is forbidden in F4"


def test_f4_text_never_claims_published_validation() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in F4_ROOT.rglob("*.md")).lower()
    assert "chaos_verified: true" not in combined
    assert "hidden_verified: true" not in combined
    assert "caputo_lyapunov_validated: true" not in combined
