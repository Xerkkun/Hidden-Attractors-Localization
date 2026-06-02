"""Conservative F7 comparison helpers for Lyapunov and F5 diagnostics."""

from __future__ import annotations

from typing import Any


ALLOWED_COMPARISON_STATUSES = {
    "methods_agree_chaotic_candidate",
    "methods_agree_regular_candidate",
    "methods_mixed_inconclusive",
    "lyapunov_methods_conflict",
    "f5_diagnostics_conflict",
    "method_not_applicable",
    "method_not_available",
    "method_validation_pending",
    "insufficient_comparable_methods",
}


def classify_method_row(method: dict[str, Any]) -> str:
    """Classify one case-method row while preserving validation boundaries."""

    if not method["applicable"]:
        return "not_applicable"
    if method["lambda_max"] is None:
        return "unavailable"
    if not method["validated"]:
        return "validation_pending"
    return method["lambda_class"]


def compare_lyapunov_methods(rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Compare case-specific validated or explicitly acceptable spectra."""

    classified = [(row, classify_method_row(row)) for row in rows]
    comparable = [
        state
        for row, state in classified
        if row["validated"] and state in {"positive_lambda_max", "nonpositive_lambda_max", "near_zero_lambda_max"}
    ]
    warnings = []
    if any(state == "validation_pending" for _, state in classified):
        warnings.append("one or more available spectra belong to methods with validation pending")
    if "positive_lambda_max" in comparable and any(
        state in {"nonpositive_lambda_max", "near_zero_lambda_max"} for state in comparable
    ):
        return "lyapunov_methods_conflict", warnings
    if len(comparable) >= 2 and all(state == "positive_lambda_max" for state in comparable):
        return "lyapunov_consensus_positive", warnings
    if len(comparable) >= 2 and all(
        state in {"nonpositive_lambda_max", "near_zero_lambda_max"} for state in comparable
    ):
        return "lyapunov_consensus_nonpositive", warnings
    if any(row["applicable"] and not row["validated"] for row, _ in classified) and not comparable:
        return "method_validation_pending", warnings
    return "insufficient_comparable_methods", warnings


def compare_f5_diagnostics(
    *,
    boundedness: str | None,
    zero_one: str | None,
    psd_fft: str | None,
    poincare: str | None,
) -> tuple[str, str]:
    """Compare F5 indicators without treating boundedness as chaos evidence."""

    if None in {boundedness, zero_one, psd_fft, poincare}:
        return "insufficient_comparable_methods", "one or more F5 diagnostics are unavailable"
    if boundedness in {"unbounded_candidate", "nonfinite_trajectory"}:
        return "methods_mixed_inconclusive", f"boundedness={boundedness}"
    if (
        boundedness == "bounded_candidate"
        and zero_one == "zero_one_chaotic_candidate"
        and psd_fft == "broadband_spectrum"
        and poincare in {"cloud_like", "dispersed_cloud_like"}
    ):
        return "methods_agree_chaotic_candidate", "three F5 geometry indicators support a chaotic candidate"
    if (
        boundedness == "bounded_candidate"
        and zero_one == "zero_one_regular_candidate"
        and psd_fft in {"dominant_periodic_peak", "quasiperiodic_candidate"}
        and poincare in {"point_like_or_fixed_return", "finite_set_like", "curve_like"}
    ):
        return "methods_agree_regular_candidate", "three F5 geometry indicators support a regular candidate"
    if zero_one == "zero_one_regular_candidate" and poincare in {"cloud_like", "dispersed_cloud_like"}:
        return "f5_diagnostics_conflict", "zero-one is regular-like while Poincare geometry is cloud-like"
    return "methods_mixed_inconclusive", "F5 indicators do not form a consistent candidate label"


__all__ = [
    "ALLOWED_COMPARISON_STATUSES",
    "classify_method_row",
    "compare_f5_diagnostics",
    "compare_lyapunov_methods",
]
