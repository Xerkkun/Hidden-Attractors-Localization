"""Validation-only Lyapunov method metadata.

Entries in this registry describe reproducibility lanes used by the validation
suite. They are deliberately separate from the installed library registry
because they are not callable public methods.
"""

from __future__ import annotations

from hidden_attractors.analysis.lyapunov_methods import (
    FINITE_TIME_SCOPE_WARNING,
    LYAPUNOV_METHODS,
    LyapunovMethodInfo,
)


VALIDATION_LYAPUNOV_METHODS: dict[str, LyapunovMethodInfo] = {
    "fractional_variational_dk2018_block_restart_abm_gs": LyapunovMethodInfo(
        method_id="fractional_variational_dk2018_block_restart_abm_gs",
        derivative_model="caputo",
        q_support="0 < q < 1",
        requires_jacobian=True,
        orthonormalization="gs",
        finite_time_local=True,
        implemented=False,
        validated=False,
        references=(
            "Danca & Kuznetsov 2018 - Matlab Code for Lyapunov Exponents"
            " of Fractional-Order Systems (Int. J. Bifurcation Chaos 28(5)):"
            " reproduction contract for block-restarted FDE12 integration and"
            " Gram-Schmidt renormalisation.",
        ),
        warnings=(
            "Validation-only published-value reproduction contract; it is not"
            " dispatched by the public common API.",
            "Passing this lane does not validate fractional_variational_abm_qr.",
            FINITE_TIME_SCOPE_WARNING,
        ),
        validated_against_synthetic_tests=True,
        validated_against_published_benchmarks=False,
        benchmark_status="recorded_published_discrepancy",
        family="fractional",
        supports_q_less_than_1=True,
        supports_q_equal_1=False,
        memory_protocol="dk2018_block_restart_abm_gs",
    ),
}

ALL_VALIDATION_LYAPUNOV_METHODS: dict[str, LyapunovMethodInfo] = {
    **LYAPUNOV_METHODS,
    **VALIDATION_LYAPUNOV_METHODS,
}


__all__ = [
    "ALL_VALIDATION_LYAPUNOV_METHODS",
    "VALIDATION_LYAPUNOV_METHODS",
]
