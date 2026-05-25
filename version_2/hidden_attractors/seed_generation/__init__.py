"""Harmonic-balance seed generation for fractional Chua and generic Lur'e workflows.

Stability: experimental
    This sub-package is the library-facing version of the reusable mathematics
    that previously lived only in legacy scripts.  All numerical choices are
    explicit arguments; no ``HIDDEN_ATTRACTORS_*`` environment variables are
    read here.

    The API is useful and tested.  New parameters or helper functions may be
    added as additional system families are supported.

Sub-modules
-----------
core
    Shared dataclasses (:class:`HarmonicSeed`, :class:`BiasedHarmonicSeed`),
    ``validate_fractional_order``, ``fractional_iomega_power``.
chua
    Chua-specific describing functions, Machado family, biased/Fourier helpers,
    and harmonic-seed constructors.
lure
    System-independent Lur'e transfer function, frequency scan, and seed
    constructors operating on any :class:`~hidden_attractors.systems.LureSystem`.
"""

from __future__ import annotations

__api_tier__ = "experimental"

from .core import (
    BiasedHarmonicSeed,
    HarmonicSeed,
    complex_dtype,
    fractional_iomega_power,
    real_dtype,
    validate_fractional_order,
)
from .chua import (
    biased_describing_function,
    build_fractional_seed,
    build_linearized_matrix,
    chua_gain,
    chua_matrices,
    describing_function,
    find_harmonic_seed,
    find_omega_gain_candidates,
    format_seed_report,
    fourier_coefficients_psi,
    is_describing_gain_compatible,
    machado_describing_function,
    psi_sigma,
    reconstruct_biased_lure_seed,
    solve_amplitude_from_gain,
    solve_machado_amplitude_from_gain,
    transfer_function,
)
from .lure import (
    biased_lure_describing_function,
    build_lure_fractional_seed,
    build_lure_linearized_matrix,
    find_lure_harmonic_seed,
    find_lure_omega_gain_candidates,
    fourier_coefficients_lure,
    lure_describing_function,
    lure_machado_describing_function,
    lure_transfer_function,
    reconstruct_biased_lure_seed_from_system,
    solve_lure_amplitude_from_gain,
)
from .chua_arctan_wu2023 import (
    find_centered_arctan_wu2023_branches,
    format_arctan_wu2023_seed_report,
    transfer_function_arctan_wu2023,
)

__all__ = [
    # core
    "BiasedHarmonicSeed",
    "HarmonicSeed",
    "complex_dtype",
    "fractional_iomega_power",
    "real_dtype",
    "validate_fractional_order",
    # chua
    "biased_describing_function",
    "build_fractional_seed",
    "build_linearized_matrix",
    "chua_gain",
    "chua_matrices",
    "describing_function",
    "find_harmonic_seed",
    "find_omega_gain_candidates",
    "format_seed_report",
    "fourier_coefficients_psi",
    "is_describing_gain_compatible",
    "machado_describing_function",
    "psi_sigma",
    "reconstruct_biased_lure_seed",
    "solve_amplitude_from_gain",
    "solve_machado_amplitude_from_gain",
    "transfer_function",
    # lure
    "biased_lure_describing_function",
    "build_lure_fractional_seed",
    "build_lure_linearized_matrix",
    "find_lure_harmonic_seed",
    "find_lure_omega_gain_candidates",
    "fourier_coefficients_lure",
    "lure_describing_function",
    "lure_machado_describing_function",
    "lure_transfer_function",
    "reconstruct_biased_lure_seed_from_system",
    "solve_lure_amplitude_from_gain",
    "find_centered_arctan_wu2023_branches",
    "format_arctan_wu2023_seed_report",
    "transfer_function_arctan_wu2023",
]
