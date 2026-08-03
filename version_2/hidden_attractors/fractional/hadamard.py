"""Hadamard and Caputo--Hadamard sampled convolution quadrature.

Stability: experimental

For a positive lower terminal ``a`` introduce logarithmic time

``u = log(t / a)``.

The dilation derivative ``delta = t*d/dt`` becomes ``d/du``.  Consequently,
left Hadamard and Caputo--Hadamard operators can be discretized by ordinary
Lubich convolution quadrature on a grid uniform in ``u``.  In physical time
this is an exponential grid ``t_n = a*exp(n*log_step)``.

This module evaluates a sampled operator; it is not an FDE solver.  BDF1 and
BDF2 are available through the canonical HAFO CQ implementation, including
direct Python, direct Numba, and zero-padded FFT backends.  No starting
corrections are applied.

References
----------
F. Jarad, T. Abdeljawad, and D. Baleanu, "Caputo-type modification of the
Hadamard fractional derivatives", Advances in Difference Equations (2012),
https://doi.org/10.1186/1687-1847-2012-142.
B. Yin, G. Zhang, Y. Liu, and H. Li, "Convolution quadrature for Hadamard
fractional calculus and correction methods for the subdiffusion with singular
source terms", Communications in Nonlinear Science and Numerical Simulation
138 (2024), https://doi.org/10.1016/j.cnsns.2024.108221.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .convolution_quadrature import (
    CAPUTO_SHIFTED_INITIAL_CONDITION,
    RL_OPERATOR_ONLY_INITIAL_CONDITION,
    lubich_convolution_quadrature,
)


HADAMARD_RL_OPERATOR_ONLY_INITIAL_CONDITION = "hadamard_operator_only_no_ivp"
"""Acknowledgement token for raw left Hadamard operator evaluation."""

CAPUTO_HADAMARD_INITIAL_CONDITION = "caputo_hadamard_point_value_shift"
"""Point-value shift used by the sampled Caputo--Hadamard convention."""

_DEFINITIONS = frozenset({"hadamard_riemann_liouville", "caputo_hadamard"})
_REFERENCES = (
    "https://doi.org/10.1186/1687-1847-2012-142",
    "https://doi.org/10.1016/j.cnsns.2024.108221",
)


@dataclass(frozen=True, slots=True)
class HadamardConvolutionQuadratureResult:
    """Structured finite-grid evaluation on an exponential physical grid."""

    values: np.ndarray
    times: np.ndarray
    log_times: np.ndarray
    orders: np.ndarray
    weights: np.ndarray
    definition: str
    bdf_order: int
    backend: str
    log_step: float
    lower_terminal: float
    initial_condition_semantics: str
    grid_kind: str
    transformation: str
    time_complexity: str
    working_memory: str
    startup_convention: str
    starting_corrections: str
    references: tuple[str, ...]
    scope: str = "sampled_fractional_operator_only_not_an_fde_solver"
    status: str = "finite_numerical_diagnostic"


def _real_positive_terminal(lower_terminal: float) -> float:
    if isinstance(lower_terminal, (bool, np.bool_)) or np.iscomplexobj(lower_terminal):
        raise TypeError("lower_terminal must be a positive real number.")
    terminal = float(lower_terminal)
    if not np.isfinite(terminal) or terminal <= 0.0:
        raise ValueError("Hadamard operators require lower_terminal > 0.")
    return terminal


def _sample_count(samples: np.ndarray) -> int:
    values = np.asarray(samples)
    if values.ndim not in (1, 2) or values.shape[0] < 1:
        raise ValueError(
            "samples must have shape (n_times,) or (n_times, dimension)."
        )
    return int(values.shape[0])


def _physical_and_log_grid(
    n_times: int,
    *,
    lower_terminal: float,
    log_step: float | None,
    times: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    if (log_step is None) == (times is None):
        raise ValueError("Pass exactly one of log_step or times.")

    if times is None:
        if isinstance(log_step, (bool, np.bool_)) or np.iscomplexobj(log_step):
            raise TypeError("log_step must be a positive real number.")
        normalized_step = float(log_step)
        if not np.isfinite(normalized_step) or normalized_step <= 0.0:
            raise ValueError("log_step must be finite and positive.")
        log_grid = normalized_step * np.arange(n_times, dtype=np.float64)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            physical_grid = np.exp(np.log(lower_terminal) + log_grid)
        physical_grid[0] = lower_terminal
        if not np.all(np.isfinite(physical_grid)) or (
            n_times > 1 and np.any(np.diff(physical_grid) <= 0.0)
        ):
            raise ValueError(
                "log_step does not produce a finite, strictly increasing physical grid."
            )
        return (
            np.ascontiguousarray(physical_grid),
            np.ascontiguousarray(log_grid),
            normalized_step,
            False,
        )

    if np.iscomplexobj(times):
        raise TypeError("times must be real-valued.")
    physical_grid = np.asarray(times, dtype=np.float64).reshape(-1)
    if physical_grid.size != n_times:
        raise ValueError("times length must match the number of sample rows.")
    if not np.all(np.isfinite(physical_grid)) or np.any(physical_grid <= 0.0):
        raise ValueError("times must contain only finite positive values.")
    if physical_grid[0] != lower_terminal:
        raise ValueError("times[0] must equal lower_terminal.")
    if n_times < 2:
        raise ValueError("times requires at least two samples to infer log_step.")
    if np.any(np.diff(physical_grid) <= 0.0):
        raise ValueError("times must be strictly increasing.")
    log_grid = np.log(physical_grid) - np.log(lower_terminal)
    log_grid[0] = 0.0
    differences = np.diff(log_grid)
    normalized_step = float(differences[0])
    if not np.isfinite(normalized_step) or normalized_step <= 0.0:
        raise ValueError("times must be strictly increasing in logarithmic time.")
    step_scale = max(1.0, abs(normalized_step))
    if not np.allclose(
        differences,
        normalized_step,
        rtol=2.0e-12,
        atol=64.0 * np.finfo(np.float64).eps * step_scale,
    ):
        raise ValueError(
            "Hadamard convolution quadrature requires a grid uniform in log(t/a)."
        )
    return (
        np.ascontiguousarray(physical_grid),
        np.ascontiguousarray(log_grid),
        normalized_step,
        True,
    )


def _definition_contract(
    definition: str,
    initial_condition_semantics: str,
) -> tuple[str, str, str]:
    normalized_definition = str(definition).strip().lower()
    if normalized_definition not in _DEFINITIONS:
        raise ValueError(f"definition must be one of {sorted(_DEFINITIONS)}.")
    normalized_semantics = str(initial_condition_semantics).strip().lower()
    if normalized_definition == "hadamard_riemann_liouville":
        required = HADAMARD_RL_OPERATOR_ONLY_INITIAL_CONDITION
        cq_definition = "riemann_liouville"
        cq_semantics = RL_OPERATOR_ONLY_INITIAL_CONDITION
    else:
        required = CAPUTO_HADAMARD_INITIAL_CONDITION
        cq_definition = "caputo_shifted"
        cq_semantics = CAPUTO_SHIFTED_INITIAL_CONDITION
    if normalized_semantics != required:
        raise ValueError(
            f"definition={normalized_definition!r} requires "
            f"initial_condition_semantics={required!r}."
        )
    return normalized_definition, cq_definition, cq_semantics


def hadamard_convolution_quadrature(
    samples: np.ndarray,
    orders: float | list[float] | tuple[float, ...] | np.ndarray,
    *,
    bdf_order: int = 1,
    definition: str = "hadamard_riemann_liouville",
    log_step: float | None = None,
    times: np.ndarray | None = None,
    lower_terminal: float,
    initial_condition_semantics: str,
    backend: str = "numba",
) -> HadamardConvolutionQuadratureResult:
    """Evaluate a left Hadamard-family operator on a logarithmic time grid.

    ``definition="hadamard_riemann_liouville"`` applies the raw terminal-
    truncated history.  ``definition="caputo_hadamard"`` applies the same CQ
    to ``x(t)-x(a)``.  Pass either ``log_step`` to construct an exponential
    physical grid or explicit ``times`` that are uniform in ``log(t/a)``.

    The BDF1/BDF2 coefficients, startup convention, and backends are exactly
    those of :func:`lubich_convolution_quadrature`.  In particular, this
    routine has no prehistory extrapolation and no starting corrections.
    """

    terminal = _real_positive_terminal(lower_terminal)
    n_times = _sample_count(samples)
    physical_grid, log_grid, normalized_step, use_explicit_grid = (
        _physical_and_log_grid(
            n_times,
            lower_terminal=terminal,
            log_step=log_step,
            times=times,
        )
    )
    normalized_definition, cq_definition, cq_semantics = _definition_contract(
        definition, initial_condition_semantics
    )

    cq_kwargs: dict[str, object]
    if use_explicit_grid:
        cq_kwargs = {"times": log_grid}
    else:
        cq_kwargs = {"step": normalized_step}
    cq_result = lubich_convolution_quadrature(
        samples,
        orders,
        bdf_order=bdf_order,
        definition=cq_definition,
        lower_terminal=0.0,
        initial_condition_semantics=cq_semantics,
        backend=backend,
        **cq_kwargs,
    )

    return HadamardConvolutionQuadratureResult(
        values=cq_result.values,
        times=physical_grid,
        log_times=cq_result.times,
        orders=cq_result.orders,
        weights=cq_result.weights,
        definition=normalized_definition,
        bdf_order=cq_result.bdf_order,
        backend=cq_result.backend,
        log_step=normalized_step,
        lower_terminal=terminal,
        initial_condition_semantics=str(initial_condition_semantics).strip().lower(),
        grid_kind="exponential_uniform_in_log_t_over_a",
        transformation="u=log(t/a); delta=t*d/dt maps to d/du",
        time_complexity=cq_result.time_complexity,
        working_memory=cq_result.working_memory,
        startup_convention=cq_result.startup_convention,
        starting_corrections=cq_result.starting_corrections,
        references=_REFERENCES,
    )


__all__ = [
    "CAPUTO_HADAMARD_INITIAL_CONDITION",
    "HADAMARD_RL_OPERATOR_ONLY_INITIAL_CONDITION",
    "HadamardConvolutionQuadratureResult",
    "hadamard_convolution_quadrature",
]
