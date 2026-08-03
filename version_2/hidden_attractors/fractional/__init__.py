"""Fractional-calculus definitions, methods, and numerical operators.

Stability: experimental

The package deliberately keeps the derivative definition separate from its
discretization, memory policy, and initial-condition convention.  This avoids
treating every fractional model as a Caputo initial-value problem.
"""

from .caputo_fabrizio import (
    CAPUTO_FABRIZIO_REFERENCES,
    CaputoFabrizioDerivativeResult,
    caputo_fabrizio_derivative,
    caputo_fabrizio_derivative_reference,
)
from .atangana_baleanu import (
    ABC_MAX_ANALYSED_ALPHA,
    ABC_REFERENCES,
    ABCWeightResult,
    AtanganaBaleanuDerivativeResult,
    abc_piecewise_linear_weights,
    atangana_baleanu_caputo_derivative,
    atangana_baleanu_caputo_derivative_reference,
    atangana_baleanu_normalization,
)
from .abc_solver import (
    ABC_PREDICTOR_CORRECTOR_REFERENCES,
    ABCPredictorCorrectorResult,
    abc_linear_product_weights,
    integrate_abc_predictor_corrector,
)
from .caputo_hadamard_solver import (
    CaputoHadamardSimulationResult,
    integrate_caputo_hadamard_abm,
)
from .contracts import (
    FRACTIONAL_DERIVATIVES,
    FRACTIONAL_METHODS,
    FractionalDerivativeDefinition,
    FractionalMethodDefinition,
    get_fractional_derivative,
    get_fractional_method,
    list_fractional_derivatives,
    list_fractional_methods,
    normalize_fractional_orders,
    validate_fractional_method,
)
from .convolution_quadrature import (
    CAPUTO_SHIFTED_INITIAL_CONDITION,
    RL_OPERATOR_ONLY_INITIAL_CONDITION,
    LubichConvolutionQuadratureResult,
    lubich_bdf_weights,
    lubich_convolution_quadrature,
)
from .conformable_solver import (
    CONFORMABLE_SOLVER_REFERENCES,
    ConformableSimulationResult,
    conformable_clock_from_time,
    integrate_conformable_rk4,
    physical_times_from_conformable_clock,
)
from .distributed_order import (
    DistributedOrderDerivativeResult,
    distributed_order_gl_derivative,
)
from .distributed_order_caputo_solver import (
    DISTRIBUTED_ORDER_CAPUTO_L1_REFERENCES,
    DistributedOrderCaputoResult,
    DistributedOrderCorrectorError,
    DistributedOrderInitialCompatibilityWarning,
    distributed_order_l1_weight,
    integrate_distributed_order_caputo_l1,
)
from .multi_term_caputo import (
    MULTI_TERM_CAPUTO_L1_REFERENCES,
    MULTI_TERM_CAPUTO_SCISPACE_PAPER_IDS,
    MultiTermCaputoResult,
    MultiTermCaputoTerms,
    canonicalize_multi_term_caputo_terms,
    integrate_multi_term_caputo_l1,
)
from .fast_grunwald_letnikov import (
    FAST_GL_REFERENCES,
    FFT_AUTO_THRESHOLD,
    FastGLDerivativeResult,
    fast_grunwald_letnikov_derivative,
    gl_linear_convolution_fft_length,
)
from .grunwald_letnikov import (
    FractionalDerivativeResult,
    grunwald_letnikov_derivative,
    grunwald_letnikov_weights,
)
from .gl_solver import (
    FractionalSimulationResult,
    integrate_gl_explicit,
    integrate_gl_explicit_numba,
)
from .hadamard import (
    CAPUTO_HADAMARD_INITIAL_CONDITION,
    HADAMARD_RL_OPERATOR_ONLY_INITIAL_CONDITION,
    HadamardConvolutionQuadratureResult,
    hadamard_convolution_quadrature,
)
from .native_grunwald_letnikov import (
    NativeGLBackendUnavailable,
    NativeGLBuildMetadata,
    NativeGLKernelError,
    NativeGLResult,
    NativeGLWeightsResult,
    NativeGrunwaldLetnikovBackend,
    native_grunwald_letnikov_convolution,
    native_grunwald_letnikov_derivative,
    native_grunwald_letnikov_weights,
)
from .references import (
    FRACTIONAL_REFERENCES,
    FractionalReference,
    get_fractional_reference,
)
from .sampled_operators import (
    OPERATOR_ONLY_INITIAL_CONDITION,
    SampledFractionalDerivativeResult,
    conformable_khalil_derivative,
    riemann_liouville_gl_derivative,
    tempered_grunwald_letnikov_derivative,
    variable_order_grunwald_letnikov_derivative,
)
from .tempered_caputo_solver import (
    TEMPERED_CAPUTO_ABM_REFERENCES,
    TemperedCaputoSimulationResult,
    integrate_tempered_caputo_abm,
)
from .tempered_convolution_quadrature import (
    TEMPERED_CAPUTO_INITIAL_CONDITION,
    TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION,
    TemperedConvolutionQuadratureResult,
    tempered_convolution_quadrature,
)
from .tempered_fast_history import (
    TEMPERED_FAST_HISTORY_REFERENCES,
    TemperedFastHistoryResult,
    tempered_fast_multistep_history,
)
from .variable_order_caputo_type3 import (
    VARIABLE_ORDER_CAPUTO_TYPE3_REFERENCES,
    VariableOrderCaputoType3Result,
    VariableOrderCorrectorError,
    VariableOrderInitialCompatibilityWarning,
    integrate_variable_order_caputo_type3_l1,
    variable_order_l1_weight,
)
from .problem import (
    FractionalProblem,
    FractionalProblemResult,
    solve_fractional_problem,
    solve_fractional_system,
)

__all__ = [
    "ABC_MAX_ANALYSED_ALPHA",
    "ABC_PREDICTOR_CORRECTOR_REFERENCES",
    "ABC_REFERENCES",
    "ABCPredictorCorrectorResult",
    "ABCWeightResult",
    "CAPUTO_FABRIZIO_REFERENCES",
    "CAPUTO_HADAMARD_INITIAL_CONDITION",
    "CAPUTO_SHIFTED_INITIAL_CONDITION",
    "CONFORMABLE_SOLVER_REFERENCES",
    "DISTRIBUTED_ORDER_CAPUTO_L1_REFERENCES",
    "MULTI_TERM_CAPUTO_L1_REFERENCES",
    "MULTI_TERM_CAPUTO_SCISPACE_PAPER_IDS",
    "FRACTIONAL_DERIVATIVES",
    "FRACTIONAL_METHODS",
    "FRACTIONAL_REFERENCES",
    "FAST_GL_REFERENCES",
    "FFT_AUTO_THRESHOLD",
    "CaputoFabrizioDerivativeResult",
    "CaputoHadamardSimulationResult",
    "ConformableSimulationResult",
    "AtanganaBaleanuDerivativeResult",
    "DistributedOrderDerivativeResult",
    "DistributedOrderCaputoResult",
    "DistributedOrderCorrectorError",
    "DistributedOrderInitialCompatibilityWarning",
    "MultiTermCaputoResult",
    "MultiTermCaputoTerms",
    "FastGLDerivativeResult",
    "FractionalDerivativeDefinition",
    "FractionalDerivativeResult",
    "FractionalMethodDefinition",
    "FractionalProblem",
    "FractionalProblemResult",
    "FractionalReference",
    "FractionalSimulationResult",
    "HADAMARD_RL_OPERATOR_ONLY_INITIAL_CONDITION",
    "HadamardConvolutionQuadratureResult",
    "LubichConvolutionQuadratureResult",
    "NativeGLBackendUnavailable",
    "NativeGLBuildMetadata",
    "NativeGLKernelError",
    "NativeGLResult",
    "NativeGLWeightsResult",
    "NativeGrunwaldLetnikovBackend",
    "OPERATOR_ONLY_INITIAL_CONDITION",
    "RL_OPERATOR_ONLY_INITIAL_CONDITION",
    "SampledFractionalDerivativeResult",
    "TEMPERED_CAPUTO_ABM_REFERENCES",
    "TEMPERED_CAPUTO_INITIAL_CONDITION",
    "TEMPERED_FAST_HISTORY_REFERENCES",
    "TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION",
    "TemperedCaputoSimulationResult",
    "TemperedConvolutionQuadratureResult",
    "TemperedFastHistoryResult",
    "VARIABLE_ORDER_CAPUTO_TYPE3_REFERENCES",
    "VariableOrderCaputoType3Result",
    "VariableOrderCorrectorError",
    "VariableOrderInitialCompatibilityWarning",
    "abc_piecewise_linear_weights",
    "abc_linear_product_weights",
    "atangana_baleanu_caputo_derivative",
    "atangana_baleanu_caputo_derivative_reference",
    "atangana_baleanu_normalization",
    "caputo_fabrizio_derivative",
    "caputo_fabrizio_derivative_reference",
    "canonicalize_multi_term_caputo_terms",
    "conformable_khalil_derivative",
    "conformable_clock_from_time",
    "distributed_order_gl_derivative",
    "distributed_order_l1_weight",
    "fast_grunwald_letnikov_derivative",
    "get_fractional_derivative",
    "get_fractional_method",
    "get_fractional_reference",
    "gl_linear_convolution_fft_length",
    "grunwald_letnikov_derivative",
    "grunwald_letnikov_weights",
    "hadamard_convolution_quadrature",
    "integrate_gl_explicit",
    "integrate_gl_explicit_numba",
    "integrate_caputo_hadamard_abm",
    "integrate_abc_predictor_corrector",
    "integrate_conformable_rk4",
    "integrate_distributed_order_caputo_l1",
    "integrate_multi_term_caputo_l1",
    "integrate_tempered_caputo_abm",
    "integrate_variable_order_caputo_type3_l1",
    "list_fractional_derivatives",
    "list_fractional_methods",
    "lubich_bdf_weights",
    "lubich_convolution_quadrature",
    "native_grunwald_letnikov_convolution",
    "native_grunwald_letnikov_derivative",
    "native_grunwald_letnikov_weights",
    "normalize_fractional_orders",
    "physical_times_from_conformable_clock",
    "riemann_liouville_gl_derivative",
    "solve_fractional_problem",
    "solve_fractional_system",
    "tempered_convolution_quadrature",
    "tempered_fast_multistep_history",
    "tempered_grunwald_letnikov_derivative",
    "validate_fractional_method",
    "variable_order_grunwald_letnikov_derivative",
    "variable_order_l1_weight",
]
