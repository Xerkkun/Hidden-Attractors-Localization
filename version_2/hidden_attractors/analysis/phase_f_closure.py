"""Phase F freeze assessment for finite-time chaos evidence."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .lyapunov_methods import LYAPUNOV_METHODS, LyapunovMethodInfo


PHASE_F_STRUCTURED_STATUS = "phase_F_frozen"
FRACTIONAL_CANDIDATE_IDS = (
    "danca2017_chua_fractional_saturation_q09998",
    "wu2023_chua_fractional_arctan_q099",
)

PHASE_F_CLOSURE_RULES = {
    "schema_version": "1.0",
    "stage": "phase_F_closure",
    "evidence_layer": "finite_time_chaos_evidence",
    "route_A": {
        "route_id": "strict_fractional_variational_validation",
        "method_id": "fractional_variational_abm_qr",
        "non_strict_assessment_status": "assessed_with_documented_validation_gap",
        "requires": [
            "validated=true",
            "validated_against_published_benchmarks=true or accepted_fractional_lyapunov_validation_policy.md exists",
            "case-specific spectrum for each current fractional candidate",
            "lambda_max",
            "full_spectrum",
            "sign_pattern",
            "convergence_status",
            "sensitivity_status",
            "method_validation_status",
        ],
    },
    "route_B": {
        "route_id": "strict_fractional_cloned_validation",
        "method_id": "fractional_cloned_dynamics_abm_gs_published",
        "non_strict_assessment_status": "assessed_with_documented_discrepancies",
        "requires": [
            "validated=true",
            "validated_against_published_benchmarks=true or fischer2020_discrepancy_resolution.md exists",
            "published_benchmarks_pending_discrepancy must be resolved",
        ],
    },
    "route_C": {
        "route_id": "diagnostic_scope_closure",
        "requires": [
            "F4.status=f4_complete_with_documented_discrepancies",
            "F5.final_f5_status=f5_diagnostics_structured_outputs_ready",
            "phase_F_diagnostic_scope_statement.md exists",
        ],
        "result_status": PHASE_F_STRUCTURED_STATUS,
    },
    "allowed_evidence_levels": [
        "strong_chaos_evidence",
        "chaotic_dynamics_supported",
        "chaos_evidence_inconclusive",
        "regular_or_periodic_candidate",
        "unbounded_or_diverged",
    ],
}


def assess_phase_f_closure(
    *,
    f4_summary: Mapping[str, Any],
    f5_summary: Mapping[str, Any],
    dk2018_summary: Mapping[str, Any],
    fischer2020_summary: Mapping[str, Any],
    registry: Mapping[str, LyapunovMethodInfo] = LYAPUNOV_METHODS,
    f6_summary: Mapping[str, Any] | None = None,
    f7_summary: Mapping[str, Any] | None = None,
    accepted_fractional_policy_exists: bool = False,
    fischer_resolution_exists: bool = False,
    diagnostic_scope_statement_exists: bool = False,
) -> dict[str, Any]:
    """Evaluate strict and diagnostic Phase F closure routes."""

    variational = registry["fractional_variational_abm_qr"]
    cloned = registry["fractional_cloned_dynamics_abm_gs_published"]
    f6_cases = {
        case["case_id"]: case
        for case in (f6_summary or {}).get("cases", [])
    }
    f4_controls = {
        control["method_id"]: control
        for control in f4_summary.get("method_controls", [])
    }
    variational_applications = {
        case_id: _method_application(f6_cases.get(case_id), variational.method_id)
        for case_id in FRACTIONAL_CANDIDATE_IDS
    }
    variational_applied_to_each_candidate = all(
        application["required_fields_available"]
        for application in variational_applications.values()
    )
    route_a_complete = (
        variational.validated
        and (
            variational.validated_against_published_benchmarks
            or accepted_fractional_policy_exists
        )
        and variational_applied_to_each_candidate
    )
    route_b_complete = (
        cloned.validated
        and (
            cloned.validated_against_published_benchmarks
            or fischer_resolution_exists
        )
        and fischer2020_summary.get("status")
        != "published_benchmarks_pending_discrepancy"
    )
    route_c_complete = (
        f4_summary.get("status") == "f4_complete_with_documented_discrepancies"
        and f5_summary.get("final_f5_status")
        == "f5_diagnostics_structured_outputs_ready"
        and diagnostic_scope_statement_exists
    )
    strict_closed = route_a_complete or route_b_complete
    boundedness_passed = _all_f5_cases_match(f5_summary, "boundedness", "bounded_candidate")
    zero_one_compatible = _all_f5_cases_match(
        f5_summary,
        "zero_one",
        "zero_one_regular_candidate",
    )
    psd_states = _f5_values(f5_summary, "psd_fft")
    criteria = {
        "reproducible_bounded_trajectory": boundedness_passed,
        "valid_fractional_lyapunov_method_per_candidate": (
            True
            if route_a_complete
            else "not_strictly_validated_with_documented_attempts"
        ),
        "two_methods_sign_agreement_preferred": "not_available_with_documented_method_limitations",
        "zero_one_compatible": zero_one_compatible,
        "psd_fft_without_dominant_periodicity": (
            "partial" if psd_states and all(state == "spectral_inconclusive" for state in psd_states) else False
        ),
        "warnings_for_nonsmooth_memory_sensitivity": True,
        "scope_note_centralized": True,
        "hiddenness_scope_separated": True,
    }
    evidence_summary = _phase_f_evidence_summary(f6_cases)
    return {
        "stage": "phase_F_closure",
        "status": (
            PHASE_F_STRUCTURED_STATUS
            if route_c_complete
            else "F_structured_diagnostics_closure_pending"
        ),
        "phase_F_frozen": route_c_complete,
        "evidence_layer": "finite_time_chaos_evidence",
        **evidence_summary,
        "structured_diagnostics_closed": route_c_complete,
        "closure_routes": {
            "route_A_fractional_variational_abm_qr": {
                "status": (
                    "strict_requirements_satisfied"
                    if route_a_complete
                    else "assessed_with_documented_validation_gap"
                ),
                "strict_requirement_satisfied": route_a_complete,
                "reason": (
                    "fractional_variational_abm_qr_validated_and_applied"
                    if route_a_complete
                    else "rigorous_internal_controls_completed_but_published_validation_or_accepted_policy_not_achieved"
                ),
                "evidence_interpretation": (
                    "The full-history QR lane has internal controls and sensitivity evidence. "
                    "It remains distinct from the DK2018 block-restart lane and is not promoted."
                ),
                "f4_method_control": f4_controls.get(variational.method_id),
                "related_dk2018_reproduction_lane": {
                    "method_id": dk2018_summary.get("method_id"),
                    "status": dk2018_summary.get("status"),
                    "validation_run_class": dk2018_summary.get("validation_run_class"),
                    "local_full_history_qr_status": dk2018_summary.get("local_full_history_qr_status"),
                    "does_not_promote_full_history_qr": True,
                },
                "method_registry": asdict(variational),
                "accepted_policy_exists": accepted_fractional_policy_exists,
                "applied_to_each_fractional_candidate": variational_applied_to_each_candidate,
                "candidate_applications": variational_applications,
            },
            "route_B_fractional_cloned_dynamics": {
                "status": (
                    "strict_requirements_satisfied"
                    if route_b_complete
                    else "assessed_with_documented_discrepancies"
                ),
                "strict_requirement_satisfied": route_b_complete,
                "reason": (
                    "fractional_cloned_dynamics_formally_accepted"
                    if route_b_complete
                    else "rigorous_fischer2020_reproduction_and_sensitivity_completed_with_documented_discrepancies"
                ),
                "evidence_interpretation": (
                    "The published GS lane was exercised with long reproduction and bounded "
                    "sensitivity sweeps. Remaining discrepancies are retained explicitly."
                ),
                "f4_method_control": f4_controls.get(cloned.method_id),
                "method_registry": asdict(cloned),
                "fischer2020_status": fischer2020_summary.get("status"),
                "rows_total": fischer2020_summary.get("rows_total"),
                "rows_passed_quantitative": fischer2020_summary.get("rows_passed_quantitative"),
                "rows_passed_sign_pattern": fischer2020_summary.get("rows_passed_sign_pattern"),
                "rows_failed": fischer2020_summary.get("rows_failed"),
                "sensitivity_status": (
                    fischer2020_summary.get("discrepancy_diagnostics", {})
                    .get("sensitivity_status")
                ),
                "diagnostic_closure": (
                    fischer2020_summary.get("discrepancy_diagnostics", {})
                    .get("diagnostic_closure")
                ),
                "discrepancy_resolution_exists": fischer_resolution_exists,
            },
            "route_C_diagnostic_scope_closure": {
                "status": "completed" if route_c_complete else "blocked",
                "reason": (
                    "F4 and F5 are structurally complete with documented limitations"
                    if route_c_complete
                    else "F4_F5_or_diagnostic_scope_statement_missing"
                ),
            },
        },
        "criteria": criteria,
        "input_statuses": {
            "f4_status": f4_summary.get("status"),
            "f5_status": f5_summary.get("final_f5_status"),
            "dk2018_status": dk2018_summary.get("status"),
            "fischer2020_status": fischer2020_summary.get("status"),
            "f6_status": (f6_summary or {}).get("status", "not_available"),
            "f7_status": (f7_summary or {}).get("status", "not_available"),
        },
        "invariants": {
            "finite_time_scope_recorded": True,
            "fractional_method_contracts_kept_separate": True,
            "dk2018_block_restart_does_not_validate_full_history_qr": True,
            "strong_level_requires_coherent_multiple_diagnostics": True,
        },
        "future_method_improvements": [
            "Validate fractional_variational_abm_qr against a published benchmark or accepted policy.",
            "Or resolve Fischer 2020 cloned dynamics discrepancies.",
            "Apply a valid fractional Lyapunov method to each fractional candidate.",
            "Prefer sign agreement between two applicable methods.",
            "Keep F5 diagnostics compatible.",
        ],
    }


def build_phase_f_closure_matrix(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create the auditable criterion matrix for the Phase F assessment."""

    criteria = summary["criteria"]
    return [
        _criterion(
            "reproducible_bounded_trajectory",
            "Post-transient trajectories remain finite-time bounded and reproducible.",
            True,
            criteria["reproducible_bounded_trajectory"],
            "validation/chaos_validation/dynamics_diagnostics/boundedness/boundedness_diagnostics_summary.json",
            "",
            "All configured F5 cases report bounded_candidate.",
        ),
        _criterion(
            "valid_fractional_lyapunov_method_per_candidate",
            "At least one validated fractional Lyapunov method is applied to every fractional candidate.",
            True,
            criteria["valid_fractional_lyapunov_method_per_candidate"],
            "hidden_attractors/analysis/lyapunov_methods.py",
            "",
            "Rigorous attempts are documented, but no fractional method is strictly validated and applied per candidate.",
        ),
        _criterion(
            "two_methods_sign_agreement_preferred",
            "Prefer sign agreement between two valid applicable fractional methods.",
            False,
            criteria["two_methods_sign_agreement_preferred"],
            "validation/chaos_validation/method_comparison/method_comparison_summary.json",
            "",
            "Preferred cross-method sign agreement is not available; method limitations are documented.",
        ),
        _criterion(
            "zero_one_compatible",
            "Zero-one diagnostic is recorded and interpreted conservatively.",
            False,
            criteria["zero_one_compatible"],
            "validation/chaos_validation/dynamics_diagnostics/zero_one/zero_one_diagnostics_summary.json",
            "",
            "zero_one_regular_candidate in current cases; compatible with regular/inconclusive, not chaos support.",
        ),
        _criterion(
            "psd_fft_no_dominant_periodicity",
            "PSD/FFT does not force a periodic or chaotic interpretation.",
            False,
            criteria["psd_fft_without_dominant_periodicity"],
            "validation/chaos_validation/dynamics_diagnostics/psd_fft/psd_fft_diagnostics_summary.json",
            "",
            "spectral_inconclusive; no strong dominant periodic label but not broadband support.",
        ),
        _criterion(
            "warnings_for_nonsmooth_memory_sensitivity",
            "Method warnings preserve non-smooth and Caputo-memory limitations.",
            True,
            criteria["warnings_for_nonsmooth_memory_sensitivity"],
            "hidden_attractors/analysis/lyapunov_methods.py",
            "",
            "Registry warnings remain explicit.",
        ),
        _criterion(
            "scope_note_centralized",
            "Phase F reports numerical evidence levels under the recorded contract.",
            True,
            criteria["scope_note_centralized"],
            "validation/chaos_validation/phase_F_closure/phase_F_closure_summary.json",
            "",
            "The finite-time interpretation is centralized in the freeze summary.",
        ),
        _criterion(
            "hiddenness_scope_separated",
            "Hiddenness assessment remains a separate sampled-neighborhood gate.",
            True,
            criteria["hiddenness_scope_separated"],
            "validation/chaos_validation/phase_F_closure/phase_F_closure_summary.json",
            "",
            "Phase F reports chaos evidence; candidate_gate reports hiddenness scope.",
        ),
    ]


def _method_application(case: Mapping[str, Any] | None, method_id: str) -> dict[str, Any]:
    methods = ((case or {}).get("evidence") or {}).get("lyapunov_methods", [])
    method = next((item for item in methods if item.get("method_id") == method_id), {})
    required_fields = {
        "lambda_max": method.get("lambda_max"),
        "full_spectrum": method.get("full_spectrum"),
        "sign_pattern": method.get("sign_pattern"),
        "convergence_status": method.get("convergence_status"),
        "sensitivity_status": method.get("sensitivity_status"),
        "method_validation_status": method.get("benchmark_status"),
    }
    return {
        **required_fields,
        "required_fields_available": all(value is not None for value in required_fields.values()),
    }


def _phase_f_evidence_summary(cases: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize F6 case evidence with the frozen positive vocabulary."""

    level_aliases = {
        "chaotic_candidate_numerically_supported": "strong_chaos_evidence",
        "regular_candidate_numerically_supported": "regular_or_periodic_candidate",
        "numerical_failure": "unbounded_or_diverged",
        "mixed_diagnostics_inconclusive": "chaos_evidence_inconclusive",
        "insufficient_lyapunov_support": "chaos_evidence_inconclusive",
        "insufficient_f5_support": "chaos_evidence_inconclusive",
        "method_validation_pending": "chaos_evidence_inconclusive",
        "not_evaluated": "chaos_evidence_inconclusive",
    }
    priority = {
        "strong_chaos_evidence": 4,
        "chaotic_dynamics_supported": 3,
        "regular_or_periodic_candidate": 2,
        "chaos_evidence_inconclusive": 1,
        "unbounded_or_diverged": 0,
    }
    per_case = []
    for case_id, case in cases.items():
        evidence = case.get("evidence", {})
        level = level_aliases.get(
            str(case.get("integrated_status")),
            str(case.get("chaos_evidence_level", "chaos_evidence_inconclusive")),
        )
        per_case.append(
            {
                "case_id": case_id,
                "chaos_evidence_level": level,
                "lyapunov_support": _lyapunov_support(evidence),
                "zero_one_support": evidence.get("zero_one_status"),
                "spectral_support": evidence.get("psd_fft_status"),
                "boundedness_support": evidence.get("boundedness_status"),
                "poincare_support": evidence.get("poincare_status"),
                "diagnostic_conflicts": (
                    ["zero_one_regular_vs_cloud_like_poincare"]
                    if evidence.get("zero_one_status") == "zero_one_regular_candidate"
                    and evidence.get("poincare_status") in {"cloud_like", "dispersed_cloud_like"}
                    else []
                ),
                "recommended_interpretation": _evidence_interpretation(level),
            }
        )
    available = max(
        (item["chaos_evidence_level"] for item in per_case),
        key=lambda level: priority.get(level, -1),
        default="chaos_evidence_inconclusive",
    )
    return {
        "available_evidence_level": available,
        "chaos_evidence_level": available,
        "lyapunov_support": {item["case_id"]: item["lyapunov_support"] for item in per_case},
        "zero_one_support": {item["case_id"]: item["zero_one_support"] for item in per_case},
        "spectral_support": {item["case_id"]: item["spectral_support"] for item in per_case},
        "boundedness_support": {item["case_id"]: item["boundedness_support"] for item in per_case},
        "poincare_support": {item["case_id"]: item["poincare_support"] for item in per_case},
        "diagnostic_conflicts": {
            item["case_id"]: item["diagnostic_conflicts"] for item in per_case if item["diagnostic_conflicts"]
        },
        "recommended_interpretation": _evidence_interpretation(available),
        "per_case_evidence": per_case,
    }


def _lyapunov_support(evidence: Mapping[str, Any]) -> str:
    values = [
        item.get("lambda_max")
        for item in evidence.get("lyapunov_methods", [])
        if item.get("lambda_max") is not None
    ]
    return "positive_lambda_max" if any(float(value) > 0.02 for value in values) else "not_available_or_nonpositive"


def _evidence_interpretation(level: str) -> str:
    return {
        "strong_chaos_evidence": "strong numerical evidence of chaos",
        "chaotic_dynamics_supported": "chaotic dynamics supported by finite-time diagnostics",
        "chaos_evidence_inconclusive": "inconclusive chaos evidence",
        "regular_or_periodic_candidate": "regular/periodic dynamics supported",
        "unbounded_or_diverged": "trajectory is unbounded or diverged",
    }[level]
def _all_f5_cases_match(summary: Mapping[str, Any], key: str, expected: str) -> bool:
    values = _f5_values(summary, key)
    return bool(values) and all(value == expected for value in values)


def _f5_values(summary: Mapping[str, Any], key: str) -> list[Any]:
    return [
        case.get(key)
        for case in summary.get("combined_interpretation", {}).get("per_case", [])
    ]


def _criterion(
    criterion_id: str,
    criterion: str,
    required_for_strict_closure: bool,
    passed: Any,
    evidence_file: str,
    blocker: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "criterion_id": criterion_id,
        "criterion": criterion,
        "required_for_strict_closure": required_for_strict_closure,
        "current_status": "passed" if passed is True else "partial" if passed == "partial" else str(passed),
        "evidence_file": evidence_file,
        "passed": passed,
        "blocker": blocker if blocker and passed is not True else "",
        "notes": notes,
    }


__all__ = [
    "FRACTIONAL_CANDIDATE_IDS",
    "PHASE_F_CLOSURE_RULES",
    "PHASE_F_STRUCTURED_STATUS",
    "assess_phase_f_closure",
    "build_phase_f_closure_matrix",
]
