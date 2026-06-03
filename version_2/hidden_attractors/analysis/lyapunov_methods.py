"""Method registry for Lyapunov exponent estimators.

F0 AUDIT — method metadata registry
=====================================
This module provides a static registry of all known Lyapunov exponent
methods in this codebase, including their implementation status,
validation status, scope, and bibliographic references.

The registry is **metadata only**.  It does not implement any numerical
methods; those live in :mod:`hidden_attractors.analysis.lyapunov`.

F0 status
---------
* ``integer_qr_benettin``: implemented and validated for q=1 only.
* Fractional methods: tracked independently by execution contract.

Evidence levels are interpreted under the recorded finite-time contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def _dk2018_published_validation_status() -> str:
    """Read DK2018 evidence conservatively; missing or malformed means pending."""

    summary = (
        Path(__file__).resolve().parents[2]
        / "validation"
        / "chaos_validation"
        / "lyapunov_methods"
        / "fractional_variational_dk2018_block_restart_abm_gs_published"
        / "validation_summary.json"
    )
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return "published_benchmarks_pending"
    status = data.get("status")
    if status in {
        "published_benchmarks_pending",
        "published_benchmarks_pending_reproduced_discrepancy",
        "published_quantitative_validated",
    }:
        return status
    return "published_benchmarks_pending"


_DK2018_PUBLISHED_STATUS = _dk2018_published_validation_status()
_DK2018_PUBLISHED_VALIDATED = _DK2018_PUBLISHED_STATUS == "published_quantitative_validated"
FINITE_TIME_SCOPE_WARNING = "Scope: finite-time numerical Lyapunov evidence under the recorded method contract."


@dataclass(frozen=True)
class LyapunovMethodInfo:
    """Metadata descriptor for a Lyapunov exponent estimation method.

    Attributes
    ----------
    method_id : str
        Canonical identifier (e.g., ``'integer_qr_benettin'``).
    derivative_model : str
        ``'integer'`` for q=1 ODE, ``'caputo'`` for fractional.
    q_support : str
        Description of supported fractional orders (e.g., ``'q=1 only'``).
    requires_jacobian : bool
        Whether the method needs a Jacobian (analytic or finite-difference).
    orthonormalization : str
        Orthonormalisation scheme (e.g., ``'qr'``, ``'gs'``, ``'none'``).
    finite_time_local : bool
        Whether results are finite-time local estimates.
    implemented : bool
        Whether the method is currently implemented in this codebase.
    validated : bool
        Whether the method has been validated against published benchmarks.
    references : tuple[str, ...]
        Bibliographic references.
    warnings : tuple[str, ...]
        Methodological warnings and scope limitations.
    """

    method_id: str
    derivative_model: str
    q_support: str
    requires_jacobian: bool
    orthonormalization: str
    finite_time_local: bool
    implemented: bool
    validated: bool
    references: tuple[str, ...]
    warnings: tuple[str, ...]
    validated_against_synthetic_tests: bool = True
    validated_against_published_benchmarks: bool = False
    benchmark_status: str = "published_benchmarks_pending"
    family: str = "integer"
    method_type: str = "variational"
    supports_q_less_than_1: bool = False
    supports_q_equal_1: bool = True
    supports_commensurate: bool = True
    supports_incommensurate: bool | str = False
    memory_protocol: str = "not_applicable"
    evidence_scope: str = "finite_time_method_evidence"
    hiddenness_scope: str = "not_evaluated_by_this_stage"



# ---------------------------------------------------------------------------
# Registry (F0)
# ---------------------------------------------------------------------------

LYAPUNOV_METHODS: dict[str, LyapunovMethodInfo] = {
    # ------------------------------------------------------------------
    # F0 — frozen, integer-order only
    # ------------------------------------------------------------------
    "integer_qr_benettin": LyapunovMethodInfo(
        method_id="integer_qr_benettin",
        derivative_model="integer",
        q_support="q=1 only",
        requires_jacobian=True,
        orthonormalization="qr",
        finite_time_local=True,
        implemented=True,
        validated=True,
        references=(
            "Benettin et al. 1980 — Lyapunov Characteristic Exponents"
            " for Smooth Dynamical Systems (Meccanica 15)",
            "Wolf et al. 1985 — Determining Lyapunov Exponents from a"
            " Time Series (Physica D 16)",
            "Danca & Kuznetsov 2018 — Matlab Code for Lyapunov Exponents"
            " of Fractional-Order Systems (Int. J. Bifurcation Chaos 28(5)):"
            " establishes that fractional Caputo spectra require extended-memory"
            " variational integration; integer QR is NOT valid for q<1.",
        ),
        warnings=(
            "This routine is not a validated Caputo fractional Lyapunov method."
            " It is restricted to q=1."
            " Fractional Caputo spectra require a dedicated extended-memory"
            " variational method.",
            "Finite-time local exponents: convergence depends on integration"
            " length and step size.",
            FINITE_TIME_SCOPE_WARNING,
        ),
        validated_against_synthetic_tests=True,
        validated_against_published_benchmarks=True,
        benchmark_status="validated_against_published_benchmarks",

    ),

    # ------------------------------------------------------------------
    # F2 — local fixed-lower-limit full-history method
    # ------------------------------------------------------------------
    "fractional_variational_abm_qr": LyapunovMethodInfo(
        method_id="fractional_variational_abm_qr",
        derivative_model="caputo",
        q_support="0 < q < 1",
        requires_jacobian=True,
        orthonormalization="qr",
        finite_time_local=True,
        implemented=True,
        validated=False,
        references=(
            "Danca & Kuznetsov 2018 — Matlab Code for Lyapunov Exponents"
            " of Fractional-Order Systems (Int. J. Bifurcation Chaos 28(5)):"
            " primary reference for the extended original–variational Caputo"
            " system with ABM predictor-corrector and QR reorthonormalisation.",
            "Benettin et al. 1980 — Lyapunov Characteristic Exponents (Meccanica 15).",
            "Wolf et al. 1985 — Determining Lyapunov Exponents from a Time Series"
            " (Physica D 16).",
        ),
        warnings=(
            "F2 — implemented, NOT yet validated against published benchmarks.",
            "validated_against_published_benchmarks: false.",
            "synthetic benchmark infrastructure available",
            "published benchmark validation pending unless completed",
            "Results are finite-time local Lyapunov exponent estimates, NOT asymptotic proofs.",
            "Caputo memory: history-aware QR transforms the entire stored variational"
            " history at each reorthonormalisation step (history_aware_qr=True).",
            "If history_aware_qr=False (block-restart), method is NOT full-memory"
            " Caputo-aware; label results accordingly.",
            FINITE_TIME_SCOPE_WARNING,
            "Not validated for non-smooth systems (e.g., Chua saturation);"
            " derivative undefined at switching surfaces.",
        ),
        validated_against_synthetic_tests=True,
        validated_against_published_benchmarks=False,
        benchmark_status="published_benchmarks_pending",
        family="fractional",
        supports_q_less_than_1=True,
        supports_q_equal_1=False,
        memory_protocol="fixed_lower_limit_full_history_qr",

    ),

    "fractional_variational_dk2018_block_restart_abm_gs": LyapunovMethodInfo(
        method_id="fractional_variational_dk2018_block_restart_abm_gs",
        derivative_model="caputo",
        q_support="0 < q < 1",
        requires_jacobian=True,
        orthonormalization="gs",
        finite_time_local=True,
        implemented=True,
        validated=_DK2018_PUBLISHED_VALIDATED,
        references=(
            "Danca & Kuznetsov 2018 — Matlab Code for Lyapunov Exponents"
            " of Fractional-Order Systems (Int. J. Bifurcation Chaos 28(5)):"
            " reproduction contract for block-restarted FDE12 integration and"
            " Gram-Schmidt renormalisation.",
        ),
        warnings=(
            "Published-value reproduction lane; distinct from fixed-lower-limit"
            " full-history Caputo QR.",
            "Passing this lane does not validate fractional_variational_abm_qr.",
            FINITE_TIME_SCOPE_WARNING,
        ),
        validated_against_synthetic_tests=True,
        validated_against_published_benchmarks=_DK2018_PUBLISHED_VALIDATED,
        benchmark_status=(
            "validated_against_published_benchmarks"
            if _DK2018_PUBLISHED_VALIDATED
            else _DK2018_PUBLISHED_STATUS
        ),
        family="fractional",
        supports_q_less_than_1=True,
        supports_q_equal_1=False,
        memory_protocol="dk2018_block_restart_abm_gs",
    ),

    "fractional_cloned_dynamics_abm": LyapunovMethodInfo(
        method_id="fractional_cloned_dynamics_abm",
        derivative_model="caputo",
        q_support="0 < q < 1",
        requires_jacobian=False,
        orthonormalization="none",
        finite_time_local=True,
        implemented=False,
        validated=False,
        references=(
            "Danca 2021 — Cloned dynamics approach for fractional Lyapunov"
            " exponents (reference for future phase only).",
        ),
        warnings=(
            "NOT implemented in F0.",
            "Cloned dynamics method: does not require Jacobian but needs"
            " multiple copies of the fractional system integrated with memory.",
        ),
        validated_against_synthetic_tests=False,
        validated_against_published_benchmarks=False,
        benchmark_status="not_implemented",

    ),

    # ------------------------------------------------------------------
    # F3 - Fischer et al. 2020 cloned-dynamics lanes
    # ------------------------------------------------------------------
    "fractional_cloned_dynamics_abm_gs_published": LyapunovMethodInfo(
        method_id="fractional_cloned_dynamics_abm_gs_published",
        derivative_model="caputo",
        q_support="0 < q <= 1; commensurate and component-wise incommensurate orders",
        requires_jacobian=False,
        orthonormalization="gram_schmidt",
        finite_time_local=True,
        implemented=True,
        validated=False,
        references=(
            "Fischer, Zourmba, and Mohamadou 2020 - Lyapunov exponents spectrum"
            " estimation of fractional order nonlinear systems using Cloned Dynamics"
            " (Applied Numerical Mathematics 154, 187-204;"
            " DOI: 10.1016/j.apnum.2020.03.027).",
        ),
        warnings=(
            "F3 - implemented, NOT yet validated against published benchmarks.",
            "Published reproduction lane with ABM predictor-corrector, modified"
            " Gram-Schmidt, and published_block_restart memory.",
            "No Jacobian or variational system is used.",
            "Finite-time local Lyapunov indicators only.",
            "Fractional results are not a full-memory Caputo-aware claim.",
            FINITE_TIME_SCOPE_WARNING,
        ),
        validated_against_synthetic_tests=True,
        validated_against_published_benchmarks=False,
        benchmark_status="published_benchmarks_pending_discrepancy",
        family="fractional",
        method_type="cloned_dynamics",
        supports_q_less_than_1=True,
        supports_q_equal_1=True,
        supports_commensurate=True,
        supports_incommensurate=True,
        memory_protocol="published_block_restart",
    ),

    "fractional_cloned_dynamics_abm_qr": LyapunovMethodInfo(
        method_id="fractional_cloned_dynamics_abm_qr",
        derivative_model="caputo",
        q_support="0 < q <= 1; commensurate and component-wise incommensurate orders",
        requires_jacobian=False,
        orthonormalization="qr",
        finite_time_local=True,
        implemented=True,
        validated=False,
        references=(
            "Internal QR variant based on Fischer, Zourmba, and Mohamadou 2020"
            " (DOI: 10.1016/j.apnum.2020.03.027).",
        ),
        warnings=(
            "F3 internal QR variant - experimental and NOT validated.",
            "Compare signs and trends against the published GS lane, not decimal agreement.",
            "No Jacobian or variational system is used.",
            "Finite-time local Lyapunov indicators only.",
            "Fractional results are not a full-memory Caputo-aware claim.",
            FINITE_TIME_SCOPE_WARNING,
        ),
        validated_against_synthetic_tests=True,
        validated_against_published_benchmarks=False,
        benchmark_status="internal_variant_pending",
        family="fractional",
        method_type="cloned_dynamics",
        supports_q_less_than_1=True,
        supports_q_equal_1=True,
        supports_commensurate=True,
        supports_incommensurate=True,
        memory_protocol="experimental_qr_block_restart",
    ),
}


__all__ = [
    "LyapunovMethodInfo",
    "LYAPUNOV_METHODS",
]
