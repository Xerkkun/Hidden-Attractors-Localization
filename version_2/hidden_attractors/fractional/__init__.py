"""Fractional-calculus definitions, methods, and numerical operators.

Stability: experimental

The package deliberately keeps the derivative definition separate from its
discretization, memory policy, and initial-condition convention.  This avoids
treating every fractional model as a Caputo initial-value problem.
"""

from importlib import import_module
import sys
from types import ModuleType

_EXPORT_GROUPS = {
    ".caputo_fabrizio": ("CAPUTO_FABRIZIO_REFERENCES", "CaputoFabrizioDerivativeResult", "caputo_fabrizio_derivative", "caputo_fabrizio_derivative_reference"),
    ".atangana_baleanu": ("ABC_MAX_ANALYSED_ALPHA", "ABC_REFERENCES", "ABCWeightResult", "AtanganaBaleanuDerivativeResult", "abc_piecewise_linear_weights", "atangana_baleanu_caputo_derivative", "atangana_baleanu_caputo_derivative_reference", "atangana_baleanu_normalization"),
    ".abc_solver": ("ABC_PREDICTOR_CORRECTOR_REFERENCES", "ABCPredictorCorrectorResult", "abc_linear_product_weights", "integrate_abc_predictor_corrector"),
    ".caputo_hadamard_solver": ("CaputoHadamardSimulationResult", "integrate_caputo_hadamard_abm"),
    ".contracts": ("FRACTIONAL_DERIVATIVES", "FRACTIONAL_METHODS", "FractionalDerivativeDefinition", "FractionalMethodDefinition", "get_fractional_derivative", "get_fractional_method", "list_fractional_derivatives", "list_fractional_methods", "normalize_fractional_orders", "validate_fractional_method"),
    ".convolution_quadrature": ("CAPUTO_SHIFTED_INITIAL_CONDITION", "RL_OPERATOR_ONLY_INITIAL_CONDITION", "LubichConvolutionQuadratureResult", "lubich_bdf_weights", "lubich_convolution_quadrature"),
    ".conformable_solver": ("CONFORMABLE_SOLVER_REFERENCES", "ConformableSimulationResult", "conformable_clock_from_time", "integrate_conformable_rk4", "physical_times_from_conformable_clock"),
    ".distributed_order": ("DistributedOrderDerivativeResult", "distributed_order_gl_derivative"),
    ".distributed_order_caputo_solver": ("DISTRIBUTED_ORDER_CAPUTO_L1_REFERENCES", "DistributedOrderCaputoResult", "DistributedOrderCorrectorError", "DistributedOrderInitialCompatibilityWarning", "distributed_order_l1_weight", "integrate_distributed_order_caputo_l1"),
    ".multi_term_caputo": ("MULTI_TERM_CAPUTO_L1_REFERENCES", "MULTI_TERM_CAPUTO_SCISPACE_PAPER_IDS", "MultiTermCaputoResult", "MultiTermCaputoTerms", "canonicalize_multi_term_caputo_terms", "integrate_multi_term_caputo_l1"),
    ".fast_grunwald_letnikov": ("FAST_GL_REFERENCES", "FFT_AUTO_THRESHOLD", "FastGLDerivativeResult", "fast_grunwald_letnikov_derivative", "gl_linear_convolution_fft_length"),
    ".grunwald_letnikov": ("FractionalDerivativeResult", "grunwald_letnikov_derivative", "grunwald_letnikov_weights"),
    ".gl_solver": ("FractionalSimulationResult", "integrate_gl_explicit", "integrate_gl_explicit_numba"),
    ".hadamard": ("CAPUTO_HADAMARD_INITIAL_CONDITION", "HADAMARD_RL_OPERATOR_ONLY_INITIAL_CONDITION", "HadamardConvolutionQuadratureResult", "hadamard_convolution_quadrature"),
    ".native_grunwald_letnikov": ("NativeGLBackendUnavailable", "NativeGLBuildMetadata", "NativeGLKernelError", "NativeGLResult", "NativeGLWeightsResult", "NativeGrunwaldLetnikovBackend", "native_grunwald_letnikov_convolution", "native_grunwald_letnikov_derivative", "native_grunwald_letnikov_weights"),
    ".references": ("FRACTIONAL_REFERENCES", "FractionalReference", "get_fractional_reference"),
    ".sampled_operators": ("OPERATOR_ONLY_INITIAL_CONDITION", "SampledFractionalDerivativeResult", "conformable_khalil_derivative", "riemann_liouville_gl_derivative", "tempered_grunwald_letnikov_derivative", "variable_order_grunwald_letnikov_derivative"),
    ".tempered_caputo_solver": ("TEMPERED_CAPUTO_ABM_REFERENCES", "TemperedCaputoSimulationResult", "integrate_tempered_caputo_abm"),
    ".tempered_convolution_quadrature": ("TEMPERED_CAPUTO_INITIAL_CONDITION", "TEMPERED_RL_OPERATOR_ONLY_INITIAL_CONDITION", "TemperedConvolutionQuadratureResult", "tempered_convolution_quadrature"),
    ".tempered_fast_history": ("TEMPERED_FAST_HISTORY_REFERENCES", "TemperedFastHistoryResult", "tempered_fast_multistep_history"),
    ".variable_order_caputo_type3": ("VARIABLE_ORDER_CAPUTO_TYPE3_REFERENCES", "VariableOrderCaputoType3Result", "VariableOrderCorrectorError", "VariableOrderInitialCompatibilityWarning", "integrate_variable_order_caputo_type3_l1", "variable_order_l1_weight"),
    ".problem": ("FractionalProblem", "FractionalProblemResult", "solve_fractional_problem", "solve_fractional_system"),
}
_LAZY_EXPORTS = {
    name: module_name
    for module_name, names in _EXPORT_GROUPS.items()
    for name in names
}


def __getattr__(name: str):
    """Resolve a fractional-calculus symbol on first access."""

    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


class _LazyFractionalModule(ModuleType):
    """Keep same-named public callables stable after direct submodule imports."""

    def __getattribute__(self, name: str):
        value = ModuleType.__getattribute__(self, name)
        namespace = ModuleType.__getattribute__(self, "__dict__")
        module_name = namespace.get("_LAZY_EXPORTS", {}).get(name)
        if module_name is not None and isinstance(value, ModuleType):
            value = getattr(import_module(module_name, namespace["__name__"]), name)
            ModuleType.__setattr__(self, name, value)
        return value


sys.modules[__name__].__class__ = _LazyFractionalModule


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

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
