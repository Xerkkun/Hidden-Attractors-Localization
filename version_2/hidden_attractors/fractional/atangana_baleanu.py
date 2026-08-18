r"""Experimental Atangana--Baleanu--Caputo sampled-data operator.

Stability: research_required / experimental

For ``0 < alpha <= 1/2`` this module evaluates the left-sided ABC operator

.. math::

   {}^{ABC}_{a}D_t^\alpha x(t)
   = \frac{B(\alpha)}{1-\alpha}\int_a^t x'(\tau)
     E_\alpha\!\left[-\frac{\alpha}{1-\alpha}
     (t-\tau)^\alpha\right]d\tau.

The restriction ``alpha <= 1/2`` is part of the numerical-method contract,
not of the abstract ABC definition.  It follows the finite-difference
approximation analysed by Yadav, Pandey, and Shukla (2019).  HAFO does not
silently extrapolate that analysis to the full interval ``(0, 1)``.

Samples are interpreted by piecewise-linear interpolation on a uniform grid.
Writing ``Delta x_j = x_j - x_{j-1}``, exact interval integration of the
Mittag--Leffler kernel gives

.. math::

   D_n = \frac{B(\alpha)}{1-\alpha}
         \sum_{j=1}^{n} \Delta x_j w_{n-j},

.. math::

   w_k = \frac{1}{h}\left[
     s E_{\alpha,2}(-\lambda s^\alpha)
   \right]_{s=kh}^{s=(k+1)h},
   \qquad \lambda=\frac{\alpha}{1-\alpha}.

The interval weights are evaluated directly from the defining convergent
series.  The implementation checks convergence, cancellation, positivity,
and monotonicity.  It raises an error when the requested grid leaves the
verified numerical domain instead of returning an unchecked approximation.
The history convolution can run in transparent Python, Numba, or offline FFT
mode.  FFT changes computational cost, not the mathematical discretization.

This operator is not a Caputo power-law derivative and it is not an FDE
solver.  Derivatives with nonsingular kernels have documented fundamental-
theorem and initial-compatibility objections; results therefore retain the
``research_required`` label.

References
----------
A. Atangana and D. Baleanu, "New fractional derivatives with nonlocal and
nonsingular kernel: theory and application to heat transfer model", Thermal
Science 20 (2016), 763--769, https://doi.org/10.2298/TSCI160111018A.

S. Yadav, R. K. Pandey, and A. K. Shukla, "Numerical approximations of
Atangana--Baleanu Caputo derivative and its application", Chaos, Solitons &
Fractals 118 (2019), 58--64,
https://doi.org/10.1016/j.chaos.2018.11.009.

K. Diethelm, R. Garrappa, A. Giusti, and M. Stynes, "Why fractional
derivatives with nonsingular kernels should not be used", Fractional Calculus
and Applied Analysis 23 (2020), 610--634,
https://doi.org/10.1515/fca-2020-0032.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

import numpy as np

from ._validation import sample_matrix as _validate_samples

try:  # Numba is a core dependency; the Python fallback documents the algorithm.
    from numba import njit
except ImportError:  # pragma: no cover - exercised only in reduced installations
    njit = None


ABC_MAX_ANALYSED_ALPHA = 0.5
ABC_REFERENCES = (
    "https://doi.org/10.2298/TSCI160111018A",
    "https://doi.org/10.1016/j.chaos.2018.11.009",
    "https://doi.org/10.1515/fca-2020-0032",
)


@dataclass(frozen=True, slots=True)
class ABCWeightResult:
    """Interval-averaged Mittag--Leffler weights and numerical diagnostics."""

    values: np.ndarray
    alpha: float
    step: float
    kernel_rate: float
    max_kernel_argument: float
    backend: str
    series_rtol: float
    series_atol: float
    max_series_terms: int
    max_terms_used: int
    max_cancellation_condition: float
    references: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AtanganaBaleanuDerivativeResult:
    """Structured evidence from one sampled ABC-operator evaluation."""

    values: np.ndarray
    times: np.ndarray
    alpha: float
    step: float
    lower_terminal: float
    normalization_value: float
    normalization_description: str
    definition: str
    method: str
    backend: str
    weight_backend: str
    kernel_rate: float
    max_kernel_argument: float
    max_terms_used: int
    max_cancellation_condition: float
    fft_length: int | None
    implementation_status: str
    stability: str
    status: str
    references: tuple[str, ...]
    semantics: Mapping[str, object]


def atangana_baleanu_normalization(alpha: float) -> float:
    r"""Return the commonly used normalization
    ``B(alpha)=1-alpha+alpha/Gamma(alpha)``.

    The function is provided as an explicit option.  It is not silently used
    by the operator because the ABC literature contains multiple normalization
    conventions.
    """

    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1].")
    if alpha == 0.0 or alpha == 1.0:
        return 1.0
    return float(1.0 - alpha + alpha / math.gamma(alpha))


def _validate_method_parameters(
    step: float,
    alpha: float,
    lower_terminal: float,
    series_rtol: float,
    series_atol: float,
    max_series_terms: int,
    max_cancellation_condition: float,
) -> tuple[float, float, float, float, float, int, float]:
    step = float(step)
    alpha = float(alpha)
    lower_terminal = float(lower_terminal)
    series_rtol = float(series_rtol)
    series_atol = float(series_atol)
    max_cancellation_condition = float(max_cancellation_condition)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be a finite positive number.")
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha > ABC_MAX_ANALYSED_ALPHA:
        raise ValueError(
            "The analysed ABC sampled method requires 0 < alpha <= 0.5."
        )
    if not np.isfinite(lower_terminal):
        raise ValueError("lower_terminal must be finite.")
    if not np.isfinite(series_rtol) or series_rtol <= 0.0:
        raise ValueError("series_rtol must be a finite positive number.")
    if not np.isfinite(series_atol) or series_atol < 0.0:
        raise ValueError("series_atol must be finite and nonnegative.")
    if isinstance(max_series_terms, (bool, np.bool_)):
        raise ValueError("max_series_terms must be an integer >= 8.")
    max_series_terms = int(max_series_terms)
    if max_series_terms < 8:
        raise ValueError("max_series_terms must be an integer >= 8.")
    if (
        not np.isfinite(max_cancellation_condition)
        or max_cancellation_condition <= 1.0
    ):
        raise ValueError(
            "max_cancellation_condition must be finite and greater than 1."
        )
    return (
        step,
        alpha,
        lower_terminal,
        series_rtol,
        series_atol,
        max_series_terms,
        max_cancellation_condition,
    )


def _evaluate_normalization(
    normalization: float | Callable[[float], float],
    alpha: float,
    normalization_name: str | None,
) -> tuple[float, str]:
    if callable(normalization):
        try:
            value = float(normalization(alpha))
        except Exception as exc:
            raise ValueError("normalization callable failed at alpha.") from exc
        callable_name = getattr(normalization, "__name__", type(normalization).__name__)
        description = normalization_name or f"callable:{callable_name}"
    else:
        if isinstance(normalization, (bool, np.bool_)):
            raise ValueError("normalization must be a positive scalar or callable.")
        value = float(normalization)
        if normalization_name is not None:
            description = str(normalization_name)
        elif value == 1.0:
            description = "B(alpha)=1 (documented library default)"
        else:
            description = f"constant B(alpha)={value:.17g}"
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("B(alpha) must evaluate to a finite positive number.")
    return value, description


def _interval_power_difference_log(lag: int, power: float) -> float:
    if lag == 0:
        return 0.0
    lag_float = float(lag)
    relative_increment = math.expm1(power * math.log1p(1.0 / lag_float))
    return power * math.log(lag_float) + math.log(relative_increment)


def _abc_weights_python(
    count: int,
    step: float,
    alpha: float,
    series_rtol: float,
    series_atol: float,
    max_series_terms: int,
) -> tuple[np.ndarray, int, float, int]:
    weights = np.empty(count, dtype=np.float64)
    if count == 0:
        return weights, 0, 1.0, -1
    kernel_rate = alpha / (1.0 - alpha)
    argument_scale = kernel_rate * step**alpha
    log_argument_scale = math.log(argument_scale)
    max_terms_used = 1
    max_condition = 1.0
    failed_lag = -1
    previous_weight = math.inf
    tolerance_guard = 256.0 * np.finfo(np.float64).eps

    for lag in range(count):
        total = 1.0
        absolute_sum = 1.0
        small_terms = 0
        converged = False
        terms_used = 1
        for series_index in range(1, max_series_terms):
            power = alpha * series_index + 1.0
            log_power_difference = _interval_power_difference_log(lag, power)
            log_magnitude = (
                series_index * log_argument_scale
                + log_power_difference
                - math.lgamma(alpha * series_index + 2.0)
            )
            if log_magnitude > math.log(np.finfo(np.float64).max):
                break
            magnitude = math.exp(log_magnitude)
            term = -magnitude if series_index % 2 else magnitude
            total += term
            absolute_sum += magnitude
            terms_used = series_index + 1
            if magnitude <= series_atol + series_rtol * abs(total):
                small_terms += 1
                if small_terms >= 2:
                    converged = True
                    break
            else:
                small_terms = 0
        if not converged or not math.isfinite(total):
            failed_lag = lag
            break
        condition = absolute_sum / max(abs(total), np.finfo(np.float64).tiny)
        if (
            total <= -tolerance_guard
            or total > 1.0 + tolerance_guard
            or total > previous_weight + tolerance_guard
        ):
            failed_lag = lag
            break
        weights[lag] = min(1.0, max(0.0, total))
        previous_weight = weights[lag]
        max_terms_used = max(max_terms_used, terms_used)
        max_condition = max(max_condition, condition)
    return weights, max_terms_used, max_condition, failed_lag


if njit is not None:

    @njit(cache=True, nogil=True)
    def _interval_power_difference_log_numba(lag: int, power: float) -> float:
        if lag == 0:
            return 0.0
        lag_float = float(lag)
        relative_increment = math.expm1(power * math.log1p(1.0 / lag_float))
        return power * math.log(lag_float) + math.log(relative_increment)


    @njit(cache=True, nogil=True)
    def _abc_weights_numba(
        count: int,
        step: float,
        alpha: float,
        series_rtol: float,
        series_atol: float,
        max_series_terms: int,
    ) -> tuple[np.ndarray, int, float, int]:
        weights = np.empty(count, dtype=np.float64)
        if count == 0:
            return weights, 0, 1.0, -1
        kernel_rate = alpha / (1.0 - alpha)
        argument_scale = kernel_rate * step**alpha
        log_argument_scale = math.log(argument_scale)
        max_terms_used = 1
        max_condition = 1.0
        failed_lag = -1
        previous_weight = math.inf
        tolerance_guard = 256.0 * np.finfo(np.float64).eps
        maximum_log = math.log(np.finfo(np.float64).max)
        tiny = np.finfo(np.float64).tiny

        for lag in range(count):
            total = 1.0
            absolute_sum = 1.0
            small_terms = 0
            converged = False
            terms_used = 1
            for series_index in range(1, max_series_terms):
                power = alpha * series_index + 1.0
                log_power_difference = _interval_power_difference_log_numba(
                    lag, power
                )
                log_magnitude = (
                    series_index * log_argument_scale
                    + log_power_difference
                    - math.lgamma(alpha * series_index + 2.0)
                )
                if log_magnitude > maximum_log:
                    break
                magnitude = math.exp(log_magnitude)
                term = -magnitude if series_index % 2 else magnitude
                total += term
                absolute_sum += magnitude
                terms_used = series_index + 1
                if magnitude <= series_atol + series_rtol * abs(total):
                    small_terms += 1
                    if small_terms >= 2:
                        converged = True
                        break
                else:
                    small_terms = 0
            if not converged or not math.isfinite(total):
                failed_lag = lag
                break
            condition = absolute_sum / max(abs(total), tiny)
            if (
                total <= -tolerance_guard
                or total > 1.0 + tolerance_guard
                or total > previous_weight + tolerance_guard
            ):
                failed_lag = lag
                break
            weights[lag] = min(1.0, max(0.0, total))
            previous_weight = weights[lag]
            max_terms_used = max(max_terms_used, terms_used)
            max_condition = max(max_condition, condition)
        return weights, max_terms_used, max_condition, failed_lag


    @njit(cache=True, nogil=True)
    def _abc_direct_convolution_numba(
        increments: np.ndarray,
        weights: np.ndarray,
        scale: float,
    ) -> np.ndarray:
        n_increments, dimension = increments.shape
        output = np.zeros((n_increments + 1, dimension), dtype=np.float64)
        for output_index in range(n_increments):
            for component in range(dimension):
                accumulated = 0.0
                for history_index in range(output_index + 1):
                    accumulated += (
                        increments[history_index, component]
                        * weights[output_index - history_index]
                    )
                output[output_index + 1, component] = scale * accumulated
        return output

else:  # pragma: no cover - used only if Numba is absent
    _abc_weights_numba = None
    _abc_direct_convolution_numba = None


def abc_piecewise_linear_weights(
    step: float,
    alpha: float,
    count: int,
    *,
    backend: str = "auto",
    series_rtol: float = 2.0e-14,
    series_atol: float = 2.0e-16,
    max_series_terms: int = 4096,
    max_cancellation_condition: float = 1.0e10,
) -> ABCWeightResult:
    """Return interval-averaged ABC kernel weights.

    ``count`` is the number of history intervals.  The first weight belongs to
    zero lag.  A failed convergence, positivity, monotonicity, or cancellation
    check raises :class:`ArithmeticError` with the first affected lag.
    """

    if isinstance(count, (bool, np.bool_)):
        raise ValueError("count must be a nonnegative integer.")
    count = int(count)
    if count < 0:
        raise ValueError("count must be a nonnegative integer.")
    (
        step,
        alpha,
        _,
        series_rtol,
        series_atol,
        max_series_terms,
        max_cancellation_condition,
    ) = _validate_method_parameters(
        step,
        alpha,
        0.0,
        series_rtol,
        series_atol,
        max_series_terms,
        max_cancellation_condition,
    )
    backend = str(backend).strip().lower()
    if backend not in {"auto", "numba", "python"}:
        raise ValueError("weight backend must be one of ['auto', 'numba', 'python'].")
    use_numba = backend in {"auto", "numba"} and _abc_weights_numba is not None
    if backend == "numba" and _abc_weights_numba is None:
        raise RuntimeError("The requested Numba backend is unavailable.")
    if use_numba:
        weights, max_terms_used, max_condition, failed_lag = _abc_weights_numba(
            count,
            step,
            alpha,
            series_rtol,
            series_atol,
            max_series_terms,
        )
        selected_backend = "numba"
    else:
        weights, max_terms_used, max_condition, failed_lag = _abc_weights_python(
            count,
            step,
            alpha,
            series_rtol,
            series_atol,
            max_series_terms,
        )
        selected_backend = "python"
    if failed_lag >= 0:
        maximum_argument = (
            alpha / (1.0 - alpha) * ((failed_lag + 1) * step) ** alpha
        )
        raise ArithmeticError(
            "Mittag-Leffler interval series failed its numerical checks at "
            f"lag {failed_lag} (kernel argument <= {maximum_argument:.8g}). "
            "Reduce the horizon/step ratio or select another validated method."
        )
    if max_condition > max_cancellation_condition:
        raise ArithmeticError(
            "Mittag-Leffler interval weights exceeded the declared cancellation "
            f"condition ({max_condition:.8g} > {max_cancellation_condition:.8g})."
        )
    kernel_rate = alpha / (1.0 - alpha)
    maximum_argument = (
        0.0 if count == 0 else kernel_rate * (count * step) ** alpha
    )
    return ABCWeightResult(
        values=weights,
        alpha=alpha,
        step=step,
        kernel_rate=kernel_rate,
        max_kernel_argument=float(maximum_argument),
        backend=selected_backend,
        series_rtol=series_rtol,
        series_atol=series_atol,
        max_series_terms=max_series_terms,
        max_terms_used=max_terms_used,
        max_cancellation_condition=float(max_condition),
        references=ABC_REFERENCES,
    )


def _abc_direct_convolution_python(
    increments: np.ndarray,
    weights: np.ndarray,
    scale: float,
) -> np.ndarray:
    output = np.zeros((increments.shape[0] + 1, increments.shape[1]), dtype=np.float64)
    for output_index in range(increments.shape[0]):
        for history_index in range(output_index + 1):
            output[output_index + 1] += (
                increments[history_index] * weights[output_index - history_index]
            )
    output[1:] *= scale
    return output


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value - 1).bit_length())


def _abc_fft_convolution(
    increments: np.ndarray,
    weights: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, int]:
    n_increments, dimension = increments.shape
    output = np.zeros((n_increments + 1, dimension), dtype=np.float64)
    if n_increments == 0:
        return output, 1
    fft_length = _next_power_of_two(2 * n_increments - 1)
    transformed_weights = np.fft.rfft(weights, n=fft_length)
    for component in range(dimension):
        transformed_values = np.fft.rfft(increments[:, component], n=fft_length)
        convolved = np.fft.irfft(
            transformed_values * transformed_weights, n=fft_length
        )[:n_increments]
        output[1:, component] = scale * convolved
    return output, fft_length


def _build_result(
    values: np.ndarray,
    *,
    was_vector: bool,
    n_times: int,
    alpha: float,
    step: float,
    lower_terminal: float,
    normalization_value: float,
    normalization_description: str,
    backend: str,
    weights: ABCWeightResult,
    fft_length: int | None,
    complexity: str,
) -> AtanganaBaleanuDerivativeResult:
    if was_vector:
        values = values[:, 0]
    return AtanganaBaleanuDerivativeResult(
        values=values,
        times=lower_terminal + step * np.arange(n_times, dtype=np.float64),
        alpha=alpha,
        step=step,
        lower_terminal=lower_terminal,
        normalization_value=normalization_value,
        normalization_description=normalization_description,
        definition="atangana_baleanu_caputo_mittag_leffler_kernel",
        method="abc_piecewise_linear_interval_convolution",
        backend=backend,
        weight_backend=weights.backend,
        kernel_rate=weights.kernel_rate,
        max_kernel_argument=weights.max_kernel_argument,
        max_terms_used=weights.max_terms_used,
        max_cancellation_condition=weights.max_cancellation_condition,
        fft_length=fft_length,
        implementation_status="research_required",
        stability="experimental",
        status="finite_sampled_data_diagnostic",
        references=ABC_REFERENCES,
        semantics={
            "grid": "uniform; sample n is at lower_terminal + n*step",
            "lower_terminal_sample": "samples[0] is x(lower_terminal)",
            "interpolation": "piecewise linear samples; constant derivative per interval",
            "interval_quadrature": "Mittag-Leffler E_(alpha,2) defining series",
            "value_at_lower_terminal": 0.0,
            "analysed_order_interval": "0 < alpha <= 0.5",
            "normalization_is_universal": False,
            "caputo_equivalence_claimed": False,
            "fde_solver": False,
            "complexity": complexity,
            "evidence_scope": (
                "finite sampled-data operator; not evidence of convergence, chaos, "
                "attraction, hiddenness, or equivalence with Caputo/RL"
            ),
        },
    )


def atangana_baleanu_caputo_derivative(
    samples: np.ndarray,
    step: float,
    alpha: float,
    *,
    lower_terminal: float = 0.0,
    normalization: float | Callable[[float], float] = 1.0,
    normalization_name: str | None = None,
    backend: str = "auto",
    series_rtol: float = 2.0e-14,
    series_atol: float = 2.0e-16,
    max_series_terms: int = 4096,
    max_cancellation_condition: float = 1.0e10,
) -> AtanganaBaleanuDerivativeResult:
    """Evaluate the ABC derivative of uniformly sampled scalar/vector data.

    ``backend='auto'`` selects the direct Numba path when available and the
    transparent Python path otherwise.  ``backend='fft'`` selects an offline
    batch convolution; no unbenchmarked automatic crossover is assumed.
    """

    array, was_vector = _validate_samples(samples)
    (
        step,
        alpha,
        lower_terminal,
        series_rtol,
        series_atol,
        max_series_terms,
        max_cancellation_condition,
    ) = _validate_method_parameters(
        step,
        alpha,
        lower_terminal,
        series_rtol,
        series_atol,
        max_series_terms,
        max_cancellation_condition,
    )
    normalization_value, normalization_description = _evaluate_normalization(
        normalization,
        alpha,
        normalization_name,
    )
    backend = str(backend).strip().lower()
    if backend not in {"auto", "numba", "python", "fft"}:
        raise ValueError("backend must be one of ['auto', 'numba', 'python', 'fft'].")
    use_numba = backend in {"auto", "numba"} and _abc_direct_convolution_numba is not None
    if backend == "numba" and _abc_direct_convolution_numba is None:
        raise RuntimeError("The requested Numba backend is unavailable.")
    weight_backend = "numba" if backend != "python" and _abc_weights_numba is not None else "python"
    weights = abc_piecewise_linear_weights(
        step,
        alpha,
        max(0, array.shape[0] - 1),
        backend=weight_backend,
        series_rtol=series_rtol,
        series_atol=series_atol,
        max_series_terms=max_series_terms,
        max_cancellation_condition=max_cancellation_condition,
    )
    increments = np.ascontiguousarray(np.diff(array, axis=0), dtype=np.float64)
    scale = normalization_value / (1.0 - alpha)
    if backend == "fft":
        values, fft_length = _abc_fft_convolution(increments, weights.values, scale)
        selected_backend = "numpy_fft_offline"
        complexity = (
            "O(n_times * series_terms + dimension * n_times * log(n_times))"
        )
    elif use_numba:
        values = _abc_direct_convolution_numba(increments, weights.values, scale)
        fft_length = None
        selected_backend = "numba_direct"
        complexity = "O(dimension * n_times**2)"
    else:
        values = _abc_direct_convolution_python(increments, weights.values, scale)
        fft_length = None
        selected_backend = "python_direct"
        complexity = "O(dimension * n_times**2)"
    return _build_result(
        values,
        was_vector=was_vector,
        n_times=array.shape[0],
        alpha=alpha,
        step=step,
        lower_terminal=lower_terminal,
        normalization_value=normalization_value,
        normalization_description=normalization_description,
        backend=selected_backend,
        weights=weights,
        fft_length=fft_length,
        complexity=complexity,
    )


def atangana_baleanu_caputo_derivative_reference(
    samples: np.ndarray,
    step: float,
    alpha: float,
    *,
    lower_terminal: float = 0.0,
    normalization: float | Callable[[float], float] = 1.0,
    normalization_name: str | None = None,
    series_rtol: float = 2.0e-14,
    series_atol: float = 2.0e-16,
    max_series_terms: int = 4096,
    max_cancellation_condition: float = 1.0e10,
) -> AtanganaBaleanuDerivativeResult:
    """Transparent direct-Python path used as an implementation oracle.

    This path is independent of the Numba and FFT convolutions, but it shares
    the mathematical interval series.  Analytic manufactured solutions remain
    necessary for independent mathematical validation.
    """

    return atangana_baleanu_caputo_derivative(
        samples,
        step,
        alpha,
        lower_terminal=lower_terminal,
        normalization=normalization,
        normalization_name=normalization_name,
        backend="python",
        series_rtol=series_rtol,
        series_atol=series_atol,
        max_series_terms=max_series_terms,
        max_cancellation_condition=max_cancellation_condition,
    )


__all__ = [
    "ABC_MAX_ANALYSED_ALPHA",
    "ABC_REFERENCES",
    "ABCWeightResult",
    "AtanganaBaleanuDerivativeResult",
    "abc_piecewise_linear_weights",
    "atangana_baleanu_caputo_derivative",
    "atangana_baleanu_caputo_derivative_reference",
    "atangana_baleanu_normalization",
]
