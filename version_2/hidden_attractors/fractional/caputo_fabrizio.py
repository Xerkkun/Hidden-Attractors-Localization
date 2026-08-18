r"""Experimental Caputo--Fabrizio derivative for uniformly sampled data.

Stability: research_required / experimental

For ``0 < alpha < 1`` this module uses the original exponential-kernel
definition

.. math::

   {}^{CF}D_{t_0}^{\alpha}x(t)
   = \frac{M(\alpha)}{1-\alpha}
     \int_{t_0}^{t}x'(\tau)
     \exp\!\left[-\frac{\alpha}{1-\alpha}(t-\tau)\right]d\tau.

``M(alpha)`` is always evaluated and recorded explicitly.  The library
default is the convention ``M(alpha) = 1``; callers studying another
normalization must pass it explicitly.  The first sample is interpreted as
``x(t0)`` and samples are assumed to lie on the uniform grid
``t0 + n * step``.  Linear interpolation of the samples makes ``x'``
constant on each interval.  Exact integration of the exponential kernel then
gives the stable recurrence

.. math::

   S_n = \rho S_{n-1} + x_n - x_{n-1},\qquad
   D_n = \frac{M(\alpha)(1-\rho)}{\alpha h}S_n,
   \quad \rho = \exp\!\left[-\frac{\alpha h}{1-\alpha}\right].

The fast path is therefore ``O(n_times * dimension)`` in time and
``O(dimension)`` in auxiliary storage.  ``caputo_fabrizio_derivative_reference``
evaluates the same interval formula directly in ``O(n_times**2 * dimension)``
and exists only as an independent validation oracle.

Endpoint extensions are explicit: ``alpha=0`` returns
``M(0) * (x(t)-x(t0))`` and ``alpha=1`` returns a backward difference, with
zero at the lower terminal because no left sample is present.  The integer
endpoint requires ``M(1)=1``.  For small positive ``alpha``, ``expm1`` avoids
loss of significance and the recurrence approaches the ``alpha=0`` branch.

This operator is *not* interchangeable with the singular-kernel Caputo
derivative.  In particular, nonsingular-kernel definitions have documented
fundamental-theorem and initial-compatibility concerns.  Results are labelled
finite sampled-data diagnostics and require research-level interpretation.

References
----------
M. Caputo and M. Fabrizio, "A New Definition of Fractional Derivative without
Singular Kernel", Progress in Fractional Differentiation and Applications 1
(2015), 73--85, https://doi.org/10.12785/pfda/010201.

K. Diethelm, R. Garrappa, A. Giusti, and M. Stynes, "Why fractional
derivatives with nonsingular kernels should not be used", Fractional Calculus
and Applied Analysis 23 (2020), 610--634,
https://doi.org/10.1515/fca-2020-0032.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from ._validation import sample_matrix as _validate_samples

try:  # Numba is a core dependency, but keeping the fallback makes the ABI clear.
    from numba import njit
except ImportError:  # pragma: no cover - exercised only in reduced installations
    njit = None


CAPUTO_FABRIZIO_REFERENCES = (
    "https://doi.org/10.12785/pfda/010201",
    "https://doi.org/10.1515/fca-2020-0032",
)


@dataclass(frozen=True, slots=True)
class CaputoFabrizioDerivativeResult:
    """Structured evidence from one sampled-data CF operator evaluation."""

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
    kernel_rate: float | None
    implementation_status: str
    stability: str
    status: str
    references: tuple[str, ...]
    semantics: Mapping[str, object]


def _cf_recurrence_python(
    samples: np.ndarray,
    decay: float,
    scale: float,
) -> np.ndarray:
    """Pure-Python/NumPy fallback for the sequential recurrence."""

    n_times, dimension = samples.shape
    output = np.zeros_like(samples)
    history = np.zeros(dimension, dtype=np.float64)
    for index in range(1, n_times):
        history = decay * history + samples[index] - samples[index - 1]
        output[index] = scale * history
    return output


if njit is not None:

    @njit(cache=True, nogil=True)
    def _cf_recurrence_numba(
        samples: np.ndarray,
        decay: float,
        scale: float,
    ) -> np.ndarray:
        n_times, dimension = samples.shape
        output = np.zeros_like(samples)
        history = np.zeros(dimension, dtype=np.float64)
        for index in range(1, n_times):
            for component in range(dimension):
                increment = samples[index, component] - samples[index - 1, component]
                history[component] = decay * history[component] + increment
                output[index, component] = scale * history[component]
        return output

else:  # pragma: no cover - used only if Numba is absent
    _cf_recurrence_numba = None


def _validate_grid(step: float, lower_terminal: float) -> tuple[float, float]:
    step = float(step)
    lower_terminal = float(lower_terminal)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be a finite positive number.")
    if not np.isfinite(lower_terminal):
        raise ValueError("lower_terminal must be finite.")
    return step, lower_terminal


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
            raise ValueError("normalization must be a positive finite scalar or callable.")
        value = float(normalization)
        if normalization_name is not None:
            description = str(normalization_name)
        elif value == 1.0:
            description = "M(alpha)=1 (documented library default)"
        else:
            description = f"constant M(alpha)={value:.17g}"
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("M(alpha) must evaluate to a finite positive number.")
    return value, description


def _prepare_inputs(
    samples: np.ndarray,
    step: float,
    alpha: float,
    lower_terminal: float,
    normalization: float | Callable[[float], float],
    normalization_name: str | None,
) -> tuple[np.ndarray, bool, float, float, float, float, str]:
    array, was_vector = _validate_samples(samples)
    step, lower_terminal = _validate_grid(step, lower_terminal)
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1].")
    normalization_value, normalization_description = _evaluate_normalization(
        normalization,
        alpha,
        normalization_name,
    )
    if alpha == 1.0 and not np.isclose(
        normalization_value,
        1.0,
        rtol=0.0,
        atol=8.0 * np.finfo(np.float64).eps,
    ):
        raise ValueError(
            "The explicit alpha=1 integer endpoint requires M(1)=1."
        )
    return (
        array,
        was_vector,
        step,
        alpha,
        lower_terminal,
        normalization_value,
        normalization_description,
    )


def _endpoint_values(
    samples: np.ndarray,
    step: float,
    alpha: float,
    normalization_value: float,
) -> np.ndarray | None:
    if alpha == 0.0:
        return normalization_value * (samples - samples[0])
    if alpha == 1.0:
        output = np.zeros_like(samples)
        output[1:] = np.diff(samples, axis=0) / step
        return output
    return None


def _kernel_parameters(
    step: float,
    alpha: float,
    normalization_value: float,
) -> tuple[float, float, float]:
    kernel_rate = alpha / (1.0 - alpha)
    exponent = -kernel_rate * step
    one_minus_decay = -np.expm1(exponent)
    decay = np.exp(exponent)
    scale = normalization_value * one_minus_decay / (alpha * step)
    return float(kernel_rate), float(decay), float(scale)


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
    method: str,
    backend: str,
    kernel_rate: float | None,
    complexity: str,
) -> CaputoFabrizioDerivativeResult:
    if was_vector:
        values = values[:, 0]
    times = lower_terminal + step * np.arange(n_times, dtype=np.float64)
    return CaputoFabrizioDerivativeResult(
        values=values,
        times=times,
        alpha=alpha,
        step=step,
        lower_terminal=lower_terminal,
        normalization_value=normalization_value,
        normalization_description=normalization_description,
        definition="caputo_fabrizio_exponential_kernel",
        method=method,
        backend=backend,
        kernel_rate=kernel_rate,
        implementation_status="research_required",
        stability="experimental",
        status="finite_sampled_data_diagnostic",
        references=CAPUTO_FABRIZIO_REFERENCES,
        semantics={
            "grid": "uniform; sample n is located at lower_terminal + n*step",
            "lower_terminal_sample": "samples[0] is x(lower_terminal)",
            "interpolation": "piecewise linear samples; constant derivative per interval",
            "interval_quadrature": "exact exponential-kernel integration",
            "value_at_lower_terminal": 0.0,
            "alpha_zero_extension": "M(0)*(x(t)-x(lower_terminal))",
            "alpha_one_extension": "backward difference; M(1) must equal 1",
            "caputo_equivalence_claimed": False,
            "complexity": complexity,
            "evidence_scope": (
                "finite sampled-data operator; not evidence of chaos, attraction, "
                "hiddenness, or equivalence with the Caputo derivative"
            ),
        },
    )


def caputo_fabrizio_derivative(
    samples: np.ndarray,
    step: float,
    alpha: float,
    *,
    lower_terminal: float = 0.0,
    normalization: float | Callable[[float], float] = 1.0,
    normalization_name: str | None = None,
    backend: str = "auto",
) -> CaputoFabrizioDerivativeResult:
    """Evaluate the left-sided CF derivative with an ``O(N*d)`` recurrence.

    Parameters
    ----------
    samples:
        Scalar or multicomponent samples with shape ``(n_times,)`` or
        ``(n_times, dimension)``.  The first row is the value at the lower
        terminal.
    step:
        Positive spacing of the assumed uniform grid.
    alpha:
        Order in ``[0, 1]``.  The endpoint extensions are documented in the
        module docstring and result semantics.
    lower_terminal:
        Time assigned to the first sample.  No prehistory before this terminal
        is inferred.
    normalization:
        Positive scalar or callable ``M(alpha)``.  The documented default is
        exactly ``M(alpha)=1``; this is a convention, not a universal identity.
    normalization_name:
        Optional provenance label for a callable or scalar normalization.
    backend:
        ``"auto"`` selects Numba when installed, ``"numba"`` requires it,
        and ``"python"`` selects the transparent NumPy/Python fallback.
    """

    (
        array,
        was_vector,
        step,
        alpha,
        lower_terminal,
        normalization_value,
        normalization_description,
    ) = _prepare_inputs(
        samples,
        step,
        alpha,
        lower_terminal,
        normalization,
        normalization_name,
    )
    backend = str(backend).strip().lower()
    if backend not in {"auto", "numba", "python"}:
        raise ValueError("backend must be one of ['auto', 'numba', 'python'].")

    endpoint = _endpoint_values(array, step, alpha, normalization_value)
    if endpoint is not None:
        endpoint_name = "alpha_zero_exact_extension" if alpha == 0.0 else "backward_difference"
        return _build_result(
            endpoint,
            was_vector=was_vector,
            n_times=array.shape[0],
            alpha=alpha,
            step=step,
            lower_terminal=lower_terminal,
            normalization_value=normalization_value,
            normalization_description=normalization_description,
            method=endpoint_name,
            backend="numpy_endpoint",
            kernel_rate=0.0 if alpha == 0.0 else None,
            complexity="O(n_times * dimension)",
        )

    kernel_rate, decay, scale = _kernel_parameters(
        step,
        alpha,
        normalization_value,
    )
    use_numba = backend in {"auto", "numba"} and _cf_recurrence_numba is not None
    if backend == "numba" and _cf_recurrence_numba is None:
        raise RuntimeError("The requested Numba backend is unavailable.")
    if use_numba:
        values = _cf_recurrence_numba(array, decay, scale)
        selected_backend = "numba"
    else:
        values = _cf_recurrence_python(array, decay, scale)
        selected_backend = "python"
    return _build_result(
        values,
        was_vector=was_vector,
        n_times=array.shape[0],
        alpha=alpha,
        step=step,
        lower_terminal=lower_terminal,
        normalization_value=normalization_value,
        normalization_description=normalization_description,
        method="cf_exponential_recurrence_piecewise_linear",
        backend=selected_backend,
        kernel_rate=kernel_rate,
        complexity="O(n_times * dimension)",
    )


def caputo_fabrizio_derivative_reference(
    samples: np.ndarray,
    step: float,
    alpha: float,
    *,
    lower_terminal: float = 0.0,
    normalization: float | Callable[[float], float] = 1.0,
    normalization_name: str | None = None,
) -> CaputoFabrizioDerivativeResult:
    """Direct ``O(N**2*d)`` validation oracle for the sampled CF operator.

    This intentionally sums every exact interval contribution independently;
    production workloads should use :func:`caputo_fabrizio_derivative`.
    """

    (
        array,
        was_vector,
        step,
        alpha,
        lower_terminal,
        normalization_value,
        normalization_description,
    ) = _prepare_inputs(
        samples,
        step,
        alpha,
        lower_terminal,
        normalization,
        normalization_name,
    )
    endpoint = _endpoint_values(array, step, alpha, normalization_value)
    if endpoint is not None:
        endpoint_name = (
            "alpha_zero_exact_extension_reference"
            if alpha == 0.0
            else "backward_difference_reference"
        )
        return _build_result(
            endpoint,
            was_vector=was_vector,
            n_times=array.shape[0],
            alpha=alpha,
            step=step,
            lower_terminal=lower_terminal,
            normalization_value=normalization_value,
            normalization_description=normalization_description,
            method=endpoint_name,
            backend="python_reference",
            kernel_rate=0.0 if alpha == 0.0 else None,
            complexity="O(n_times * dimension)",
        )

    kernel_rate, decay, scale = _kernel_parameters(
        step,
        alpha,
        normalization_value,
    )
    output = np.zeros_like(array)
    for index in range(1, array.shape[0]):
        accumulated = np.zeros(array.shape[1], dtype=np.float64)
        for interval in range(1, index + 1):
            lag = index - interval
            accumulated += (decay**lag) * (array[interval] - array[interval - 1])
        output[index] = scale * accumulated
    return _build_result(
        output,
        was_vector=was_vector,
        n_times=array.shape[0],
        alpha=alpha,
        step=step,
        lower_terminal=lower_terminal,
        normalization_value=normalization_value,
        normalization_description=normalization_description,
        method="cf_direct_interval_sum_piecewise_linear",
        backend="python_reference",
        kernel_rate=kernel_rate,
        complexity="O(n_times**2 * dimension); validation only",
    )


__all__ = [
    "CAPUTO_FABRIZIO_REFERENCES",
    "CaputoFabrizioDerivativeResult",
    "caputo_fabrizio_derivative",
    "caputo_fabrizio_derivative_reference",
]
