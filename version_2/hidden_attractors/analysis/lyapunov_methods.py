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
* All fractional methods: listed as future phases — NOT implemented in F0.

No claim is made about chaos or hiddenness certification.

    chaos_certified_by_this_pipeline: false
    hiddenness_certified_by_this_pipeline: false
"""

from __future__ import annotations

from dataclasses import dataclass


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
            "Does not certify chaos; does not certify hiddenness of attractors.",
            "chaos_certified_by_this_pipeline: false",
            "hiddenness_certified_by_this_pipeline: false",
        ),
    ),

    # ------------------------------------------------------------------
    # Future phase — NOT implemented in F0
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
            "Results are finite-time local Lyapunov exponent estimates, NOT asymptotic proofs.",
            "Caputo memory: history-aware QR transforms the entire stored variational"
            " history at each reorthonormalisation step (history_aware_qr=True).",
            "If history_aware_qr=False (block-restart), method is NOT full-memory"
            " Caputo-aware; label results accordingly.",
            "Does not certify chaos; does not certify hiddenness of attractors.",
            "chaos_certified_by_this_pipeline: false",
            "hiddenness_certified_by_this_pipeline: false",
            "Not validated for non-smooth systems (e.g., Chua saturation);"
            " derivative undefined at switching surfaces.",
        ),
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
    ),
}


__all__ = [
    "LyapunovMethodInfo",
    "LYAPUNOV_METHODS",
]
