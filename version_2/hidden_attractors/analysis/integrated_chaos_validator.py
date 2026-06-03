"""Conservative F6 integration of Lyapunov and trajectory diagnostics.

The functions in this module classify finite-time numerical evidence. They do
not certify chaos, hiddenness, or fractional Lyapunov methods.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .lyapunov_methods import LYAPUNOV_METHODS, LyapunovMethodInfo


ALLOWED_INTEGRATED_STATUSES = {
    "chaotic_candidate_numerically_supported",
    "regular_candidate_numerically_supported",
    "mixed_diagnostics_inconclusive",
    "insufficient_lyapunov_support",
    "insufficient_f5_support",
    "method_validation_pending",
    "numerical_failure",
    "not_evaluated",
}

CASE_Q = {
    "chua_integer_q1_reference": 1.0,
    "danca2017_chua_fractional_saturation_q09998": 0.9998,
    "wu2023_chua_fractional_arctan_q099": 0.99,
}

F4_INTEGER_CASE_ID = "chua_integer_q1_reference"


def method_is_applicable(method: LyapunovMethodInfo, q: float) -> bool:
    """Return whether a registered method may be compared at ``q``."""

    if method.method_id == "integer_qr_benettin":
        return abs(q - 1.0) <= 1.0e-12
    if method.method_id == "fractional_variational_abm_qr":
        return 0.0 < q < 1.0
    if method.method_id == "fractional_variational_dk2018_block_restart_abm_gs":
        return 0.0 < q < 1.0
    if method.method_id in {
        "fractional_cloned_dynamics_abm_gs_published",
        "fractional_cloned_dynamics_abm_qr",
    }:
        return 0.0 < q <= 1.0
    return False


def method_registry_rows(
    registry: Mapping[str, LyapunovMethodInfo] = LYAPUNOV_METHODS,
) -> list[dict[str, Any]]:
    """Return JSON-ready metadata for the implemented comparison methods."""

    method_ids = (
        "integer_qr_benettin",
        "fractional_variational_abm_qr",
        "fractional_variational_dk2018_block_restart_abm_gs",
        "fractional_cloned_dynamics_abm_gs_published",
        "fractional_cloned_dynamics_abm_qr",
    )
    return [asdict(registry[method_id]) for method_id in method_ids]


def classify_lambda_max(value: float | None, *, near_zero: float = 0.02) -> str:
    """Classify one finite-time largest exponent without promoting a proof."""

    if value is None:
        return "unavailable"
    if abs(value) < near_zero:
        return "near_zero_lambda_max"
    if value > 0.0:
        return "positive_lambda_max"
    return "nonpositive_lambda_max"


def normalize_lyapunov_case_evidence(
    *,
    case_id: str,
    q: float,
    f4_integer_rows: list[dict[str, Any]] | None = None,
    registry: Mapping[str, LyapunovMethodInfo] = LYAPUNOV_METHODS,
) -> list[dict[str, Any]]:
    """Build per-case method evidence.

    F4 integer Chua rows are the only case-specific Lyapunov spectra currently
    consumed. Published DK2018 and Fischer benchmark rows belong to different
    benchmark systems and must not be transplanted into an F5 Chua case.
    """

    spectra: dict[str, list[float]] = {}
    if case_id == F4_INTEGER_CASE_ID:
        for row in f4_integer_rows or []:
            values = row.get("computed_exponents")
            if isinstance(values, list):
                spectra[str(row["method_id"])] = [float(value) for value in values]

    output = []
    for method_id, method in registry.items():
        if method_id not in {
            "integer_qr_benettin",
            "fractional_variational_abm_qr",
            "fractional_variational_dk2018_block_restart_abm_gs",
            "fractional_cloned_dynamics_abm_gs_published",
            "fractional_cloned_dynamics_abm_qr",
        }:
            continue
        applicable = method_is_applicable(method, q)
        spectrum = spectra.get(method_id)
        lambda_max = max(spectrum) if spectrum else None
        output.append(
            {
                "method_id": method_id,
                "applicable": applicable,
                "implemented": method.implemented,
                "validated": method.validated,
                "validated_against_published_benchmarks": method.validated_against_published_benchmarks,
                "benchmark_status": method.benchmark_status,
                "q_support": method.q_support,
                "lambda_max": lambda_max,
                "full_spectrum": spectrum,
                "sign_pattern": _sign_pattern(spectrum),
                "finite_time_local": method.finite_time_local,
                "memory_protocol": method.memory_protocol,
                "warnings": list(method.warnings),
                "result_status": (
                    "not_applicable"
                    if not applicable
                    else "available_finite_time_local"
                    if spectrum is not None
                    else "unavailable_for_case"
                ),
                "lambda_class": classify_lambda_max(lambda_max),
            }
        )
    return output


def integrate_case_evidence(
    *,
    case_id: str,
    boundedness_status: str | None,
    zero_one_status: str | None,
    psd_fft_status: str | None,
    poincare_status: str | None,
    lyapunov_evidence: list[dict[str, Any]],
    f4_status: str,
) -> dict[str, Any]:
    """Apply explicit conservative F6 rules to one case."""

    statuses = {boundedness_status, zero_one_status, psd_fft_status, poincare_status}
    if None in statuses:
        return _decision(case_id, "insufficient_f5_support", "one or more F5 summaries are missing")
    if boundedness_status in {"unbounded_candidate", "nonfinite_trajectory"}:
        return _decision(case_id, "numerical_failure", f"boundedness={boundedness_status}")

    validated_positive = [
        item
        for item in lyapunov_evidence
        if item["applicable"]
        and item["validated"]
        and item["lambda_class"] == "positive_lambda_max"
    ]
    validated_nonpositive = [
        item
        for item in lyapunov_evidence
        if item["applicable"]
        and item["validated"]
        and item["lambda_class"] in {"nonpositive_lambda_max", "near_zero_lambda_max"}
    ]
    positive_support = sum(
        (
            bool(validated_positive),
            zero_one_status == "zero_one_chaotic_candidate",
            psd_fft_status == "broadband_spectrum",
            poincare_status in {"cloud_like", "dispersed_cloud_like"},
        )
    )
    regular_support = sum(
        (
            bool(validated_nonpositive),
            zero_one_status == "zero_one_regular_candidate",
            psd_fft_status in {"dominant_periodic_peak", "quasiperiodic_candidate"},
            poincare_status in {"point_like_or_fixed_return", "finite_set_like", "curve_like"},
        )
    )
    contradictory_f5 = (
        zero_one_status == "zero_one_regular_candidate"
        and poincare_status in {"cloud_like", "dispersed_cloud_like"}
    )
    pending_fractional_method = any(
        item["applicable"]
        and item["implemented"]
        and not item["validated"]
        and item["method_id"] != "fractional_variational_dk2018_block_restart_abm_gs"
        for item in lyapunov_evidence
    )

    if boundedness_status != "bounded_candidate":
        return _decision(case_id, "mixed_diagnostics_inconclusive", f"boundedness={boundedness_status}")
    if contradictory_f5 or psd_fft_status == "spectral_inconclusive":
        return _decision(
            case_id,
            "mixed_diagnostics_inconclusive",
            "F5 indicators conflict or PSD/FFT remains inconclusive",
        )
    if positive_support >= 2:
        return _decision(
            case_id,
            "chaotic_candidate_numerically_supported",
            f"bounded finite trajectory with {positive_support} non-certifying chaotic-support indicators",
        )
    if regular_support >= 2:
        return _decision(
            case_id,
            "regular_candidate_numerically_supported",
            f"bounded finite trajectory with {regular_support} non-certifying regular-support indicators",
        )
    if pending_fractional_method:
        return _decision(
            case_id,
            "method_validation_pending",
            f"decision depends on unvalidated fractional methods; F4={f4_status}",
        )
    return _decision(case_id, "insufficient_lyapunov_support", "no sufficient applicable Lyapunov support")


def _sign_pattern(spectrum: list[float] | None) -> list[str] | None:
    if spectrum is None:
        return None
    return ["positive" if value > 0.0 else "negative" if value < 0.0 else "zero" for value in spectrum]


def _decision(case_id: str, status: str, reason: str) -> dict[str, Any]:
    if status not in ALLOWED_INTEGRATED_STATUSES:
        raise ValueError(f"unsupported integrated status: {status}")
    
    mapping = {
        "chaotic_candidate_numerically_supported": "strong_chaos_evidence",
        "regular_candidate_numerically_supported": "regular_or_periodic_candidate",
        "numerical_failure": "unbounded_or_diverged",
        "mixed_diagnostics_inconclusive": "chaos_evidence_inconclusive",
        "insufficient_lyapunov_support": "chaos_evidence_inconclusive",
        "insufficient_f5_support": "chaos_evidence_inconclusive",
        "method_validation_pending": "chaos_evidence_inconclusive",
        "not_evaluated": "chaos_evidence_inconclusive",
    }
    level = mapping.get(status, "chaos_evidence_inconclusive")
    
    return {
        "case_id": case_id,
        "chaos_evidence_level": level,
        "legacy_internal_label": status,
        "decision_reason": reason,
        "hiddenness_evidence_level": "not_evaluated_by_this_stage",
    }


__all__ = [
    "ALLOWED_INTEGRATED_STATUSES",
    "CASE_Q",
    "classify_lambda_max",
    "integrate_case_evidence",
    "method_is_applicable",
    "method_registry_rows",
    "normalize_lyapunov_case_evidence",
]
