from .decomposition import validate_lure_decomposition
from .describing_function import (
    DescribingFunctionResult,
    N_quadrature,
    N_segmented_quadrature,
    evaluate_describing_function,
    evaluate_describing_function_batch,
    solve_amplitude_from_gain,
)
from .nyquist import find_harmonic_candidates, HarmonicCandidate, CandidateList
from .seeds import (
    build_closed_form_integer_seed,
    build_modal_lure_seed,
    build_lure_seed,
)
from .transfer import (
    validate_fractional_order,
    W_spectral,
    W_eval,
    W_precompute_spectral,
    W_eval_from_cache,
)

__all__ = [
    "validate_lure_decomposition",
    "DescribingFunctionResult",
    "N_quadrature",
    "N_segmented_quadrature",
    "evaluate_describing_function",
    "evaluate_describing_function_batch",
    "solve_amplitude_from_gain",
    "find_harmonic_candidates",
    "HarmonicCandidate",
    "CandidateList",
    "build_closed_form_integer_seed",
    "build_modal_lure_seed",
    "build_lure_seed",
    "validate_fractional_order",
    "W_spectral",
    "W_eval",
    "W_precompute_spectral",
    "W_eval_from_cache",
]
