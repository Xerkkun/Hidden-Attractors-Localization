"""Registry contracts for fractional derivatives and numerical methods.

Stability: experimental

Registry entries are capability statements, not claims that mathematically
different derivatives are interchangeable.  ``implemented`` entries have
validated executable HAFO paths; ``planned``, ``research_required`` and
``theoretical_only`` entries remain non-executable through
:class:`FractionalProblem`.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np


IMPLEMENTATION_STATES = frozenset(
    {"implemented", "experimental", "planned", "research_required", "theoretical_only"}
)


@dataclass(frozen=True, slots=True)
class FractionalDerivativeDefinition:
    """Mathematical and computational contract for one derivative family."""

    name: str
    display_name: str
    kernel_family: str
    nonlocal_operator: bool
    initial_condition_semantics: str
    supported_order_interval: tuple[float, float]
    implementation_status: str
    compatible_methods: tuple[str, ...]
    notes: str
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.implementation_status not in IMPLEMENTATION_STATES:
            raise ValueError(f"Unknown implementation state: {self.implementation_status}")


@dataclass(frozen=True, slots=True)
class FractionalMethodDefinition:
    """Numerical-method contract independent of a particular vector field.

    ``derivative_families``, ``order_modes`` and ``memory_policies`` remain
    public summary fields.  ``supported_combinations`` closes the otherwise
    ambiguous Cartesian product between them: every entry is the exact
    ``(derivative_definition, order_mode, memory_policy)`` triple accepted by
    the corresponding executor.  When omitted, the Cartesian product is
    derived for backward compatibility with third-party registry records.
    """

    name: str
    display_name: str
    derivative_families: tuple[str, ...]
    order_modes: tuple[str, ...]
    memory_policies: tuple[str, ...]
    implementation_status: str
    accuracy_note: str
    references: tuple[str, ...] = ()
    execution_kind: str = "solver"
    supported_combinations: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.implementation_status not in IMPLEMENTATION_STATES:
            raise ValueError(f"Unknown implementation state: {self.implementation_status}")
        if self.execution_kind not in {"solver", "sampled_operator"}:
            raise ValueError(f"Unknown execution kind: {self.execution_kind}")
        combinations = self.supported_combinations
        if not combinations:
            combinations = tuple(
                (derivative, order_mode, memory_policy)
                for derivative in self.derivative_families
                for order_mode in self.order_modes
                for memory_policy in self.memory_policies
            )
        if len(set(combinations)) != len(combinations):
            raise ValueError(f"Method {self.name!r} contains duplicate supported combinations.")
        for combination in combinations:
            if len(combination) != 3:
                raise ValueError(
                    f"Method {self.name!r} supported combinations must contain "
                    "(derivative_definition, order_mode, memory_policy)."
                )
            derivative, order_mode, memory_policy = combination
            if derivative not in self.derivative_families:
                raise ValueError(
                    f"Method {self.name!r} combination references undeclared derivative "
                    f"{derivative!r}."
                )
            if order_mode not in self.order_modes:
                raise ValueError(
                    f"Method {self.name!r} combination references undeclared order mode "
                    f"{order_mode!r}."
                )
            if memory_policy not in self.memory_policies:
                raise ValueError(
                    f"Method {self.name!r} combination references undeclared memory policy "
                    f"{memory_policy!r}."
                )
        object.__setattr__(self, "supported_combinations", tuple(combinations))

    def supports_combination(
        self,
        derivative_definition: str,
        order_mode: str,
        memory_policy: str,
    ) -> bool:
        """Return whether the executor accepts one exact numerical contract."""

        return (derivative_definition, order_mode, memory_policy) in self.supported_combinations


_DERIVATIVES = {
    "caputo": FractionalDerivativeDefinition(
        "caputo",
        "Caputo",
        "singular_power_law",
        True,
        "integer-order initial values at the lower terminal",
        (0.0, 1.0),
        "implemented",
        (
            "caputo_abm_pece",
            "efork3",
            "gl_explicit_discrete",
            "gl_backward_euler_caputo",
            "gl_fft_offline",
            "convolution_quadrature",
        ),
        "Production FDE workflows use ABM/PECE or EFORK; CQ is a sampled operator.",
        ("caputo1967", "diethelm_ford_freed2004", "podlubny1999", "lubich1986"),
    ),
    "grunwald_letnikov": FractionalDerivativeDefinition(
        "grunwald_letnikov",
        "Grunwald-Letnikov",
        "binomial_power_law",
        True,
        "sample history and an explicit lower terminal; no implicit Caputo shift",
        (0.0, 1.0),
        "implemented",
        ("gl_direct", "gl_explicit_discrete", "gl_backward_euler_caputo", "gl_fft_offline"),
        "HAFO implements the direct discrete operator and a separately contracted explicit recurrence.",
        ("podlubny1999", "lubich1986"),
    ),
    "riemann_liouville": FractionalDerivativeDefinition(
        "riemann_liouville",
        "Riemann-Liouville",
        "singular_power_law",
        True,
        "fractional integral/derivative data, not automatically x(t0)=x0",
        (0.0, 1.0),
        "implemented",
        ("gl_direct", "gl_fft_offline", "convolution_quadrature"),
        "The GL discretization is exposed as an approximation only under its regularity assumptions.",
        ("podlubny1999", "lubich1986"),
    ),
    "caputo_fabrizio": FractionalDerivativeDefinition(
        "caputo_fabrizio",
        "Caputo-Fabrizio",
        "nonsingular_exponential",
        True,
        "operator-specific normalization and classical initial values",
        (0.0, 1.0),
        "research_required",
        ("cf_direct_recursive", "cf_predictor_corrector"),
        "The sampled operator has a dedicated recurrence; FDE solves remain research-gated.",
        ("caputo_fabrizio2015", "cao_wang_xu2020", "diethelm_garrappa_giusti_stynes2020"),
    ),
    "atangana_baleanu_caputo": FractionalDerivativeDefinition(
        "atangana_baleanu_caputo",
        "Atangana-Baleanu in Caputo sense",
        "nonsingular_mittag_leffler",
        True,
        "operator-specific normalization and classical initial values",
        (0.0, 1.0),
        "implemented",
        (
            "abc_sampled_convolution",
            "abc_predictor_corrector",
            "abc_fast_soe_predictor_corrector",
        ),
        (
            "The sampled operator and a conventional full-history FDE solver are "
            "available under explicit normalization and compatibility contracts; "
            "the fast SOE solver remains planned and the nonsingular-kernel "
            "interpretation remains scientifically contested."
        ),
        (
            "atangana_baleanu2016",
            "yadav_pandey_shukla2019",
            "lee_kim_jang2024",
            "diethelm_garrappa_giusti_stynes2020",
        ),
    ),
    "tempered_caputo": FractionalDerivativeDefinition(
        "tempered_caputo",
        "Tempered Caputo",
        "tempered_power_law",
        True,
        "classical initial values plus an explicit tempering parameter",
        (0.0, 1.0),
        "implemented",
        (
            "tempered_caputo_abm_pece_transform",
            "tempered_convolution_quadrature",
            "tempered_fast_multistep_history",
            "tempered_symbol_shift_cq",
        ),
        (
            "The exponential-conjugation ABM solver and BDF1/BDF2 sampled CQ "
            "operator are executable. CQ subtracts the point value in the "
            "conjugated coordinate and does not silently apply a -lambda**q*x "
            "normalization."
        ),
        (
            "li_deng_zhao2019",
            "sabzikar_meerschaert_chen2015",
            "chen_deng2015",
            "guo_zeng_turner_burrage_karniadakis2019",
            "podlubny1999",
        ),
    ),
    "tempered_riemann_liouville": FractionalDerivativeDefinition(
        "tempered_riemann_liouville",
        "Tempered Riemann-Liouville",
        "exponentially_tempered_power_law",
        True,
        "sample history from an explicit lower terminal; operator evaluation only",
        (0.0, 1.0),
        "implemented",
        (
            "tempered_gl_direct",
            "tempered_convolution_quadrature",
            "tempered_fast_multistep_history",
            "tempered_symbol_shift_cq",
        ),
        (
            "Implements the unnormalized exponential-conjugation definition "
            "through direct GL or BDF1/BDF2 CQ; it is not tempered Caputo and "
            "does not subtract lambda**q*x."
        ),
        (
            "sabzikar_meerschaert_chen2015",
            "chen_deng2015",
            "guo_zeng_turner_burrage_karniadakis2019",
            "lubich1986",
        ),
    ),
    "variable_order_caputo": FractionalDerivativeDefinition(
        "variable_order_caputo",
        "Variable-order Caputo",
        "time_varying_power_law",
        True,
        "classical initial values plus a validated order function q(t, x)",
        (0.0, 1.0),
        "planned",
        ("variable_order_pece",),
        "Order variation invalidates caches that assume stationary convolution weights.",
        ("samko_ross1993",),
    ),
    "caputo_variable_type3": FractionalDerivativeDefinition(
        "caputo_variable_type3",
        "Variable-order Caputo, Tavares type III",
        "current_time_variable_power_law",
        True,
        (
            "classical point value; smooth starts imply f(a,x0)=0, while "
            "nonsmooth starts must be declared and do not inherit smooth L1 accuracy"
        ),
        (0.0, 1.0),
        "implemented",
        ("vo_caputo_type3_l1",),
        (
            "Uses alpha(t_n) throughout the history kernel; it is not Tavares "
            "type I or II and contains no alpha-prime terms."
        ),
        (
            "tavares_almeida_torres2016",
            "samko_ross1993",
        ),
    ),
    "variable_order_grunwald_letnikov": FractionalDerivativeDefinition(
        "variable_order_grunwald_letnikov",
        "Variable-order Grunwald-Letnikov",
        "time_varying_binomial_power_law",
        True,
        "sample history and an explicit variable-order convention",
        (0.0, 1.0),
        "implemented",
        ("variable_order_gl_direct",),
        "Uses q(t_n) for every history weight at output time t_n; other variable-order definitions differ.",
        ("samko_ross1993", "lubich1986"),
    ),
    "caputo_distributed_order": FractionalDerivativeDefinition(
        "caputo_distributed_order",
        "Caputo distributed order",
        "positive_weighted_caputo_power_law",
        True,
        (
            "one classical point value for 0 < alpha <= 1; smooth starts with "
            "no atom at alpha=1 imply f(a,x0)=0"
        ),
        (0.0, 1.0),
        "implemented",
        ("distributed_order_caputo_l1",),
        (
            "Uses an explicit nonnegative discrete order measure. The alpha=1 "
            "atom is an exact backward-Euler term; signed measures remain "
            "operator-only."
        ),
        (
            "caputo_distributed_order2001",
            "diethelm_ford2009",
            "hu_liu_anh_turner2014",
            "lin_xu2007",
        ),
    ),
    "distributed_order": FractionalDerivativeDefinition(
        "distributed_order",
        "Distributed order",
        "weighted_order_distribution",
        True,
        "initial data and a normalized order-density or quadrature rule",
        (0.0, 1.0),
        "implemented",
        ("distributed_order_gl_direct", "distributed_order_quadrature"),
        "The sampled double quadrature is implemented; an FDE solver remains planned.",
        (
            "caputo_distributed_order2001",
            "diethelm_ford2009",
            "podlubny1999",
            "lubich1986",
        ),
    ),
    "conformable": FractionalDerivativeDefinition(
        "conformable",
        "Conformable derivative",
        "local_rescaling",
        False,
        "classical local initial values",
        (0.0, 1.0),
        "implemented",
        (
            "conformable_sampled_local",
            "local_ode_transform",
            "conformable_rk4_clock",
        ),
        (
            "The sampled local rescaling and a commensurate RK4 clock transform "
            "are implemented; both remain distinct from hereditary memory."
        ),
        ("khalil2014",),
    ),
    "hadamard_riemann_liouville": FractionalDerivativeDefinition(
        "hadamard_riemann_liouville",
        "Hadamard (Riemann--Liouville type)",
        "singular_logarithmic_power_law",
        True,
        "fractional logarithmic-history data from a strictly positive lower terminal",
        (0.0, 1.0),
        "implemented",
        ("hadamard_convolution_quadrature",),
        "BDF1/BDF2 CQ is evaluated on a grid uniform in log(t/a); operator only.",
        ("jarad_abdeljawad_baleanu2012", "yin_zhang_liu_li2024"),
    ),
    "caputo_hadamard": FractionalDerivativeDefinition(
        "caputo_hadamard",
        "Caputo--Hadamard",
        "singular_logarithmic_power_law",
        True,
        "classical point value at a strictly positive lower terminal",
        (0.0, 1.0),
        "implemented",
        ("hadamard_convolution_quadrature", "caputo_hadamard_abm_pece"),
        "CQ is an operator; uniform-log ABM/PECE provides an implemented commensurate solver.",
        (
            "jarad_abdeljawad_baleanu2012",
            "zheng2021_caputo_hadamard_transform",
            "green_liu_yan2021",
            "yin_zhang_liu_li2024",
        ),
    ),
}


_METHODS = {
    "caputo_abm_pece": FractionalMethodDefinition(
        "caputo_abm_pece",
        "Caputo Adams-Bashforth-Moulton PECE",
        ("caputo",),
        ("commensurate", "componentwise"),
        ("full_history", "finite_window", "block_restart"),
        "implemented",
        "Predictor-corrector; convergence and cost depend on order and memory policy.",
        ("diethelm_ford_freed2004", "li_tao2009"),
        supported_combinations=(
            ("caputo", "commensurate", "full_history"),
            ("caputo", "commensurate", "finite_window"),
            ("caputo", "componentwise", "block_restart"),
        ),
    ),
    "efork3": FractionalMethodDefinition(
        "efork3",
        "EFORK-3",
        ("caputo",),
        ("commensurate",),
        ("full_history", "finite_window"),
        "implemented",
        "Project-specific validated Caputo lane; truncated memory changes the numerical contract.",
        ("caputo1967", "ghoreishi_ghaffari_saad2023"),
        supported_combinations=(
            ("caputo", "commensurate", "full_history"),
            ("caputo", "commensurate", "finite_window"),
        ),
    ),
    "gl_direct": FractionalMethodDefinition(
        "gl_direct",
        "Direct Grunwald-Letnikov convolution",
        ("grunwald_letnikov", "riemann_liouville"),
        ("commensurate", "componentwise"),
        ("full_history", "finite_window"),
        "implemented",
        "First-order binomial-history approximation on an equally spaced grid.",
        ("podlubny1999", "lubich1986"),
        execution_kind="sampled_operator",
    ),
    "gl_backward_euler_caputo": FractionalMethodDefinition(
        "gl_backward_euler_caputo",
        "GL/backward-Euler approximation of Caputo",
        ("caputo", "grunwald_letnikov"),
        ("commensurate", "componentwise"),
        ("full_history", "finite_window"),
        "implemented",
        "Uses the GL derivative of x-x0; it is not the raw RL/GL initial-value convention.",
        ("podlubny1999",),
        execution_kind="sampled_operator",
    ),
    "gl_fft_offline": FractionalMethodDefinition(
        "gl_fft_offline",
        "Offline zero-padded FFT GL convolution",
        ("caputo", "grunwald_letnikov", "riemann_liouville"),
        ("commensurate", "componentwise"),
        ("full_history",),
        "implemented",
        "Batch O(d N log N) convolution; crossover is host-dependent and configurable.",
        ("lubich1986", "matusiak2020"),
        execution_kind="sampled_operator",
    ),
    "gl_explicit_discrete": FractionalMethodDefinition(
        "gl_explicit_discrete",
        "Explicit Grunwald-Letnikov recurrence",
        ("grunwald_letnikov", "caputo"),
        ("commensurate", "componentwise"),
        ("full_history", "finite_window"),
        "implemented",
        "Explicit lagged-RHS recurrence; raw GL and Caputo-shifted initializations remain distinct.",
        ("podlubny1999",),
    ),
    "convolution_quadrature": FractionalMethodDefinition(
        "convolution_quadrature",
        "Lubich BDF convolution quadrature",
        ("caputo", "riemann_liouville"),
        ("commensurate", "componentwise"),
        ("full_history",),
        "implemented",
        "BDF1/BDF2 sampled operator, direct or FFT; no FDE solve or starting corrections.",
        ("lubich1986", "lubich2004", "jin_li_zhou2017"),
        execution_kind="sampled_operator",
    ),
    "hadamard_convolution_quadrature": FractionalMethodDefinition(
        "hadamard_convolution_quadrature",
        "Hadamard BDF convolution quadrature on an exponential grid",
        ("hadamard_riemann_liouville", "caputo_hadamard"),
        ("commensurate", "componentwise"),
        ("full_history",),
        "implemented",
        "BDF1/BDF2 sampled operator in log(t/a), direct or FFT; no FDE solve or starting corrections.",
        ("lubich1986", "jarad_abdeljawad_baleanu2012", "yin_zhang_liu_li2024"),
        execution_kind="sampled_operator",
    ),
    "caputo_hadamard_abm_pece": FractionalMethodDefinition(
        "caputo_hadamard_abm_pece",
        "Caputo--Hadamard ABM/PECE on a uniform logarithmic grid",
        ("caputo_hadamard",),
        ("commensurate",),
        ("full_history",),
        "implemented",
        "Transforms to Caputo time u=log(t/a); graded meshes and fast history are not implemented.",
        (
            "diethelm_ford_freed2004",
            "zheng2021_caputo_hadamard_transform",
            "green_liu_yan2021",
        ),
    ),
    "cf_direct_recursive": FractionalMethodDefinition(
        "cf_direct_recursive",
        "Direct Caputo-Fabrizio exponential recurrence",
        ("caputo_fabrizio",),
        ("commensurate",),
        ("recursive_kernel",),
        "implemented",
        "Exact exponential-kernel interval integration for piecewise-linear sampled data; operator only.",
        (
            "caputo_fabrizio2015",
            "cao_wang_xu2020",
            "diethelm_garrappa_giusti_stynes2020",
        ),
        execution_kind="sampled_operator",
    ),
    "cf_predictor_corrector": FractionalMethodDefinition(
        "cf_predictor_corrector",
        "Caputo-Fabrizio predictor-corrector",
        ("caputo_fabrizio",),
        ("commensurate", "componentwise"),
        ("full_history", "recursive_kernel"),
        "planned",
        "Dedicated nonsingular exponential-kernel method.",
        ("caputo_fabrizio2015", "cao_wang_xu2020", "diethelm_garrappa_giusti_stynes2020"),
    ),
    "abc_predictor_corrector": FractionalMethodDefinition(
        "abc_predictor_corrector",
        "Conventional Atangana--Baleanu--Caputo predictor--corrector",
        ("atangana_baleanu_caputo",),
        ("commensurate",),
        ("full_history",),
        "implemented",
        (
            "Lee--Kim--Jang equations (9)--(14), with an HAFO implicit "
            "product-trapezoid fixed-point startup; conventional O(N^2) history, "
            "0 < alpha < 1, and no smooth-order guarantee for nonsmooth flows."
        ),
        (
            "atangana_baleanu2016",
            "lee_kim_jang2024",
            "diethelm_garrappa_giusti_stynes2020",
        ),
        supported_combinations=(("atangana_baleanu_caputo", "commensurate", "full_history"),),
    ),
    "abc_fast_soe_predictor_corrector": FractionalMethodDefinition(
        "abc_fast_soe_predictor_corrector",
        "Fast SOE Atangana--Baleanu--Caputo predictor--corrector",
        ("atangana_baleanu_caputo",),
        ("commensurate",),
        ("fast_history",),
        "planned",
        (
            "Separate sum-of-exponentials recurrence from Lee--Kim--Jang; "
            "the conventional O(N^2) executor must not be reported as this O(N) lane."
        ),
        (
            "atangana_baleanu2016",
            "lee_kim_jang2024",
            "diethelm_garrappa_giusti_stynes2020",
        ),
        supported_combinations=(("atangana_baleanu_caputo", "commensurate", "fast_history"),),
    ),
    "abc_sampled_convolution": FractionalMethodDefinition(
        "abc_sampled_convolution",
        "Atangana--Baleanu--Caputo sampled interval convolution",
        ("atangana_baleanu_caputo",),
        ("commensurate",),
        ("full_history",),
        "implemented",
        (
            "Piecewise-linear interval integration for 0 < alpha <= 1/2; "
            "direct Numba/Python or offline FFT. This is an operator, not an FDE solver."
        ),
        (
            "atangana_baleanu2016",
            "yadav_pandey_shukla2019",
            "diethelm_garrappa_giusti_stynes2020",
        ),
        execution_kind="sampled_operator",
    ),
    "tempered_caputo_abm_pece_transform": FractionalMethodDefinition(
        "tempered_caputo_abm_pece_transform",
        "Tempered-Caputo ABM/PECE by exponential conjugation",
        ("tempered_caputo",),
        ("commensurate",),
        ("full_history", "finite_window"),
        "implemented",
        (
            "Uses v=exp(lambda*(t-a))*x to derive exponentially damped ABM/PECE "
            "history weights, evaluated directly in physical state by C or Python; "
            "this is not the Li--Deng--Zhao Jacobi algorithm, and finite-window "
            "memory is a sliding restart that changes the numerical model."
        ),
        (
            "li_deng_zhao2019",
            "sabzikar_meerschaert_chen2015",
            "diethelm_ford_freed2004",
        ),
        supported_combinations=(
            ("tempered_caputo", "commensurate", "full_history"),
            ("tempered_caputo", "commensurate", "finite_window"),
        ),
    ),
    "tempered_convolution_quadrature": FractionalMethodDefinition(
        "tempered_convolution_quadrature",
        "Tempered BDF convolution quadrature by exponential conjugation",
        ("tempered_caputo", "tempered_riemann_liouville"),
        ("commensurate", "componentwise"),
        ("full_history",),
        "implemented",
        (
            "BDF1/BDF2 sampled operator with direct Python/Numba or offline "
            "zero-padded FFT convolution. Uses delta(exp(-lambda*h)*z)**q, "
            "not (delta(z)/h+lambda)**q; no starting corrections or streaming "
            "fast-history executor are claimed."
        ),
        (
            "lubich1986",
            "chen_deng2015",
            "guo_zeng_turner_burrage_karniadakis2019",
            "jin_li_zhou2017",
        ),
        execution_kind="sampled_operator",
    ),
    "tempered_fast_multistep_history": FractionalMethodDefinition(
        "tempered_fast_multistep_history",
        "Fast tempered fractional linear multistep history",
        ("tempered_caputo", "tempered_riemann_liouville"),
        ("commensurate", "componentwise"),
        ("fast_history",),
        "implemented",
        (
            "Executable real-axis Fast Method II for FBDF1 and the published "
            "second-order GNGF2 generator. Exact local history and the "
            "conjugated Caputo anchor are retained; all compressed finite-grid "
            "weights are calibrated against the direct recurrence. The "
            "requested tolerance controls compression, not CQ/FDE error. "
            "Fractional BDF2 remains outside this real-only backend because "
            "its branch convention is not silently inferred."
        ),
        ("guo_zeng_turner_burrage_karniadakis2019",),
        execution_kind="sampled_operator",
    ),
    "tempered_symbol_shift_cq": FractionalMethodDefinition(
        "tempered_symbol_shift_cq",
        "Tempered CQ by direct Laplace-symbol shift",
        ("tempered_caputo", "tempered_riemann_liouville"),
        ("commensurate", "componentwise"),
        ("full_history",),
        "planned",
        (
            "Separate discretization based on (delta(z)/h+lambda)**q. It is "
            "not a backend for exponential-conjugation CQ because its weights, "
            "startup behavior, and discrete initial anchor differ."
        ),
        ("lubich1986", "lubich2004", "sabzikar_meerschaert_chen2015"),
        execution_kind="sampled_operator",
    ),
    "tempered_gl_direct": FractionalMethodDefinition(
        "tempered_gl_direct",
        "Direct tempered GL convolution",
        ("tempered_riemann_liouville",),
        ("commensurate", "componentwise"),
        ("full_history",),
        "implemented",
        "Direct O(N^2 d) sampled operator with exponential-conjugation weights.",
        ("sabzikar_meerschaert_chen2015", "lubich1986"),
        execution_kind="sampled_operator",
    ),
    "variable_order_pece": FractionalMethodDefinition(
        "variable_order_pece",
        "Variable-order PECE",
        ("variable_order_caputo",),
        ("variable",),
        ("full_history",),
        "planned",
        "Weights must be recomputed consistently with the selected variable-order definition.",
        ("samko_ross1993",),
    ),
    "vo_caputo_type3_l1": FractionalMethodDefinition(
        "vo_caputo_type3_l1",
        "Variable-order Caputo type III L1 with Picard corrector",
        ("caputo_variable_type3",),
        ("variable",),
        ("full_history",),
        "implemented",
        (
            "Direct O(N^2 d) L1 history with alpha evaluated at the current "
            "time; Picard solution of the implicit discrete equation is an "
            "explicit HAFO adaptation and must report nonconvergence."
        ),
        (
            "tavares_almeida_torres2016",
            "fang_sun_wang2020",
        ),
        supported_combinations=(
            ("caputo_variable_type3", "variable", "full_history"),
        ),
    ),
    "variable_order_gl_direct": FractionalMethodDefinition(
        "variable_order_gl_direct",
        "Direct variable-order GL convolution",
        ("variable_order_grunwald_letnikov",),
        ("variable",),
        ("full_history",),
        "implemented",
        "Direct O(N^2 d) operator with q(t_n) frozen across each output-time history sum.",
        ("samko_ross1993", "lubich1986"),
        execution_kind="sampled_operator",
    ),
    "distributed_order_caputo_l1": FractionalMethodDefinition(
        "distributed_order_caputo_l1",
        "Combined-kernel L1 for Caputo distributed order",
        ("caputo_distributed_order",),
        ("distributed",),
        ("full_history",),
        "implemented",
        (
            "Precomputes one combined L1 kernel in O(R*N), then advances the "
            "implicit system with O(N^2*d) direct history and a reported "
            "Picard corrector."
        ),
        (
            "caputo_distributed_order2001",
            "diethelm_ford2009",
            "hu_liu_anh_turner2014",
            "lin_xu2007",
        ),
        supported_combinations=((
            "caputo_distributed_order",
            "distributed",
            "full_history",
        ),),
    ),
    "distributed_order_quadrature": FractionalMethodDefinition(
        "distributed_order_quadrature",
        "Distributed-order quadrature",
        ("distributed_order",),
        ("distributed",),
        ("full_history", "fast_history"),
        "planned",
        "Combines order-space quadrature with time-history discretization.",
        ("podlubny1999",),
    ),
    "distributed_order_gl_direct": FractionalMethodDefinition(
        "distributed_order_gl_direct",
        "Distributed-order direct GL quadrature",
        ("distributed_order",),
        ("distributed",),
        ("full_history", "finite_window"),
        "implemented",
        "Combines declared order-space quadrature with direct GL time-history sums.",
        ("diethelm_ford2009", "lubich1986", "podlubny1999"),
        execution_kind="sampled_operator",
    ),
    "local_ode_transform": FractionalMethodDefinition(
        "local_ode_transform",
        "Local ODE transformation",
        ("conformable",),
        ("commensurate", "componentwise"),
        ("none",),
        "theoretical_only",
        "This local model is intentionally excluded from hereditary-memory claims.",
        ("khalil2014",),
    ),
    "conformable_rk4_clock": FractionalMethodDefinition(
        "conformable_rk4_clock",
        "Conformable-clock classical RK4",
        ("conformable",),
        ("commensurate",),
        ("none",),
        "implemented",
        (
            "Transforms tau=(t-a)^q/q and applies fixed-step classical RK4; "
            "this is a local ODE solver without hereditary memory."
        ),
        ("khalil2014",),
        supported_combinations=(("conformable", "commensurate", "none"),),
    ),
    "conformable_sampled_local": FractionalMethodDefinition(
        "conformable_sampled_local",
        "Sampled conformable local rescaling",
        ("conformable",),
        ("commensurate", "componentwise"),
        ("none",),
        "implemented",
        "Evaluates (t-a)^(1-q) f'(t); it contains no hereditary state.",
        ("khalil2014",),
        execution_kind="sampled_operator",
    ),
}

FRACTIONAL_DERIVATIVES: Mapping[str, FractionalDerivativeDefinition] = (
    MappingProxyType(_DERIVATIVES)
)
FRACTIONAL_METHODS: Mapping[str, FractionalMethodDefinition] = MappingProxyType(_METHODS)


def list_fractional_derivatives(*, status: str | None = None) -> tuple[FractionalDerivativeDefinition, ...]:
    """Return derivative definitions, optionally filtered by implementation state."""

    if status is not None and status not in IMPLEMENTATION_STATES:
        raise ValueError(f"Unknown implementation state: {status}")
    return tuple(
        item for item in FRACTIONAL_DERIVATIVES.values()
        if status is None or item.implementation_status == status
    )


def list_fractional_methods(*, status: str | None = None) -> tuple[FractionalMethodDefinition, ...]:
    """Return numerical-method definitions, optionally filtered by state."""

    if status is not None and status not in IMPLEMENTATION_STATES:
        raise ValueError(f"Unknown implementation state: {status}")
    return tuple(
        item for item in FRACTIONAL_METHODS.values()
        if status is None or item.implementation_status == status
    )


def get_fractional_derivative(name: str) -> FractionalDerivativeDefinition:
    """Return a registered derivative definition by canonical name."""

    try:
        return FRACTIONAL_DERIVATIVES[str(name).strip().lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown fractional derivative: {name!r}") from exc


def get_fractional_method(name: str) -> FractionalMethodDefinition:
    """Return a registered numerical method by canonical name."""

    try:
        return FRACTIONAL_METHODS[str(name).strip().lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown fractional method: {name!r}") from exc


def normalize_fractional_orders(
    orders: float | Iterable[float] | np.ndarray,
    dimension: int,
    *,
    upper_bound: float = 1.0,
) -> np.ndarray:
    """Return one validated derivative order per state component."""

    dimension = int(dimension)
    if dimension < 1:
        raise ValueError("dimension must be a positive integer.")
    values = np.asarray(orders, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, dimension)
    if values.size != dimension:
        raise ValueError(
            f"orders must contain one value or {dimension} values; received {values.size}."
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0) or np.any(values > upper_bound):
        raise ValueError(f"orders must be finite and lie in (0, {upper_bound}].")
    return values


def validate_fractional_method(
    derivative: str,
    method: str,
    *,
    order_mode: str = "commensurate",
    memory_policy: str = "full_history",
    require_implemented: bool = True,
) -> FractionalMethodDefinition:
    """Validate a derivative/method/order/memory combination.

    This function rejects scientifically invalid combinations before a solver
    runs, such as applying Caputo ABM weights to an Atangana-Baleanu kernel.
    """

    derivative_info = get_fractional_derivative(derivative)
    method_info = get_fractional_method(method)
    if derivative_info.name not in method_info.derivative_families:
        raise ValueError(
            f"Method {method!r} is not registered for derivative {derivative!r}."
        )
    if order_mode not in method_info.order_modes:
        raise ValueError(f"Method {method!r} does not support order mode {order_mode!r}.")
    if memory_policy not in method_info.memory_policies:
        raise ValueError(
            f"Method {method!r} does not support memory policy {memory_policy!r}."
        )
    if not method_info.supports_combination(
        derivative_info.name,
        order_mode,
        memory_policy,
    ):
        raise ValueError(
            f"Method {method!r} does not support the exact combination "
            f"derivative_definition={derivative_info.name!r}, "
            f"order_mode={order_mode!r}, memory_policy={memory_policy!r}."
        )
    if require_implemented and method_info.implementation_status != "implemented":
        raise NotImplementedError(
            f"Method {method!r} is {method_info.implementation_status}, not implemented."
        )
    return method_info


__all__ = [
    "FRACTIONAL_DERIVATIVES",
    "FRACTIONAL_METHODS",
    "FractionalDerivativeDefinition",
    "FractionalMethodDefinition",
    "get_fractional_derivative",
    "get_fractional_method",
    "list_fractional_derivatives",
    "list_fractional_methods",
    "normalize_fractional_orders",
    "validate_fractional_method",
]
