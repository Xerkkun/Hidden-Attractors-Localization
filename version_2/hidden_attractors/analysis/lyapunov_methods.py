"""Internal method registry for Lyapunov exponent estimators.

This module records implementation status, validation scope, and
bibliographic references for the common dispatcher.

The registry is **metadata only**.  It does not implement any numerical
methods; those live in :mod:`hidden_attractors.analysis.lyapunov`.

Evidence levels are interpreted under the recorded finite-time contract.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    benchmark_status: str = "not_claimed"
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
# Dispatcher registry
# ---------------------------------------------------------------------------

LYAPUNOV_METHODS: dict[str, LyapunovMethodInfo] = {
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
        validated_against_published_benchmarks=False,
        benchmark_status="validated_against_exact_linear_controls_and_internal_crosschecks",

    ),

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
            "Implemented and covered by synthetic numerical tests.",
            "No published quantitative-validation claim is made for this method.",
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
        benchmark_status="synthetic_validation_only",
        family="fractional",
        supports_q_less_than_1=True,
        supports_q_equal_1=False,
        memory_protocol="fixed_lower_limit_full_history_qr",

    ),

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
            "Implemented numerical diagnostic with a recorded benchmark discrepancy.",
            "Published reproduction lane with ABM predictor-corrector, modified"
            " Gram-Schmidt, and published_block_restart memory.",
            "No Jacobian or variational system is used.",
            "Finite-time local Lyapunov indicators only.",
            "Fractional results are not a full-memory Caputo-aware claim.",
            FINITE_TIME_SCOPE_WARNING,
        ),
        validated_against_synthetic_tests=True,
        validated_against_published_benchmarks=False,
        benchmark_status="recorded_benchmark_discrepancy",
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
            "QR numerical diagnostic; no published validation claim.",
            "Compare signs and trends against the published GS lane, not decimal agreement.",
            "No Jacobian or variational system is used.",
            "Finite-time local Lyapunov indicators only.",
            "Fractional results are not a full-memory Caputo-aware claim.",
            FINITE_TIME_SCOPE_WARNING,
        ),
        validated_against_synthetic_tests=True,
        validated_against_published_benchmarks=False,
        benchmark_status="numerical_comparison_only",
        family="fractional",
        method_type="cloned_dynamics",
        supports_q_less_than_1=True,
        supports_q_equal_1=True,
        supports_commensurate=True,
        supports_incommensurate=True,
        memory_protocol="published_block_restart",
    ),
}


__all__ = [
    "LyapunovMethodInfo",
    "LYAPUNOV_METHODS",
]
