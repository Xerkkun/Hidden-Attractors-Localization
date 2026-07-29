"""Lyapunov diagnostics reconstructed from a scalar time series.

Stability: experimental
    The public result type and entry points are tested and reusable, while
    optional estimator parameters are outside the stable compatibility tier.

This module is deliberately separate from the variational Lyapunov routines.
It does not require a right-hand side, Jacobian, or numerical integrator.
Instead, it delegates scalar-delay reconstruction to the optional ``nolds``
backend:

* Rosenstein et al. for the largest Lyapunov exponent;
* Eckmann et al. for a finite-dimensional spectrum estimate.

Both outputs are finite-time, sampling-dependent diagnostics.  They do not
certify chaos, asymptotic exponents, or hiddenness.
"""

from __future__ import annotations

import warnings as warnings_module
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from threading import Lock
from typing import Any, Sequence

import numpy as np

from hidden_attractors._stability import EXPERIMENTAL, api_tier
from hidden_attractors.integrations.external_tools import require_external


ROSENSTEIN_METHOD = "nolds.lyap_r (Rosenstein et al. 1993)"
ECKMANN_METHOD = "nolds.lyap_e (Eckmann et al. 1986)"
EVIDENCE_STATUS = "finite_time_time_series_diagnostic"
SPECTRUM_STATUS = "finite_time_nolds_eckmann_scalar_reconstruction"
DEFAULT_MAX_PAIRWISE_MATRIX_BYTES = 256 * 1024 * 1024
_RANSAC_RANDOM_LOCK = Lock()


@api_tier(EXPERIMENTAL)
@dataclass(frozen=True)
class TimeSeriesLyapunovResult:
    """Structured scalar-time-series Lyapunov result.

    ``largest_exponent`` and every value in ``spectrum`` use inverse units of
    the supplied ``sample_interval``.  ``spectrum`` is sorted from largest to
    smallest before the Kaplan--Yorke dimension is evaluated.
    """

    largest_exponent: float
    spectrum: tuple[float, ...]
    kaplan_yorke_dimension: float
    kaplan_yorke_status: str
    spectrum_sum: float
    spectrum_status: str
    sample_interval: float
    sample_rate: float
    time_unit: str
    exponent_unit: str
    n_samples: int
    estimated_pairwise_matrix_bytes: int
    observable: str | None
    backend: str
    backend_version: str
    rosenstein_method: str
    eckmann_method: str
    rosenstein_parameters: dict[str, Any]
    eckmann_parameters: dict[str, Any]
    largest_sign_agrees_with_spectrum: bool
    rosenstein_slope_per_sample: float
    rosenstein_fit_r2: float | None
    rosenstein_divergence_index_unit: str
    rosenstein_divergence_trajectory: tuple[tuple[float, float], ...]
    rosenstein_divergence_time_trajectory: tuple[tuple[float, float], ...]
    evidence_status: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["spectrum"] = list(self.spectrum)
        payload["rosenstein_divergence_trajectory"] = [
            list(point) for point in self.rosenstein_divergence_trajectory
        ]
        payload["rosenstein_divergence_time_trajectory"] = [
            list(point) for point in self.rosenstein_divergence_time_trajectory
        ]
        payload["warnings"] = list(self.warnings)
        return payload


def _finite_signal(signal: Sequence[float]) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("signal must be a one-dimensional scalar time series")
    if values.size < 100:
        raise ValueError("signal must contain at least 100 samples")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "signal contains non-finite values; filtering would break uniform sampling"
        )
    if float(np.ptp(values)) == 0.0:
        raise ValueError("signal must not be constant")
    return values


def _positive_finite(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return parsed


def _resolve_sample_interval(
    sample_interval: float | None,
    sample_rate: float | None,
) -> float:
    if sample_interval is None and sample_rate is None:
        raise ValueError("provide sample_interval or sample_rate")
    interval = (
        None
        if sample_interval is None
        else _positive_finite(sample_interval, "sample_interval")
    )
    rate = (
        None
        if sample_rate is None
        else _positive_finite(sample_rate, "sample_rate")
    )
    if interval is None:
        assert rate is not None
        return 1.0 / rate
    if rate is not None and not np.isclose(
        interval * rate,
        1.0,
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError("sample_interval and sample_rate are not reciprocal")
    return interval


def _backend_version() -> str:
    try:
        return version("nolds")
    except PackageNotFoundError:
        return "unknown"


def _deduplicated_warning_text(
    captured: Sequence[warnings_module.WarningMessage],
    extra: Sequence[str] = (),
) -> tuple[str, ...]:
    fixed = (
        "Finite-time estimates reconstructed from one scalar observable.",
        "Results depend on sampling, delay embedding, neighborhood, and fit parameters.",
        "The nolds Eckmann spectrum and derived Kaplan-Yorke dimension are finite-data estimates.",
        "These diagnostics do not certify chaos, asymptotic exponents, or hiddenness.",
    )
    messages = [*fixed, *extra, *(str(item.message) for item in captured)]
    return tuple(dict.fromkeys(messages))


@api_tier(EXPERIMENTAL)
def kaplan_yorke_dimension(exponents: Sequence[float]) -> float:
    """Compute the Kaplan--Yorke dimension from a finite ordered spectrum.

    The input is sorted in descending order.  If the largest exponent is
    negative the returned dimension is zero.  If every cumulative sum is
    non-negative, the result is capped at the supplied spectrum dimension.
    """

    values = np.asarray(exponents, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("exponents must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("exponents must all be finite")

    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    nonnegative = np.flatnonzero(cumulative >= 0.0)
    if nonnegative.size == 0:
        return 0.0

    index = int(nonnegative[-1])
    if index == ordered.size - 1:
        return float(ordered.size)

    denominator = abs(float(ordered[index + 1]))
    if denominator == 0.0:
        raise ValueError(
            "Kaplan-Yorke dimension is undefined because the next exponent is zero"
        )
    dimension = float(index + 1) + float(cumulative[index]) / denominator
    return float(np.clip(dimension, 0.0, float(ordered.size)))


def _validate_estimator_parameters(
    *,
    rosenstein_emb_dim: int,
    rosenstein_lag: int | None,
    rosenstein_min_tsep: int | None,
    rosenstein_min_neighbors: int,
    rosenstein_trajectory_len: int,
    rosenstein_fit: str,
    rosenstein_fit_offset: int,
    eckmann_emb_dim: int,
    eckmann_matrix_dim: int,
    eckmann_min_neighbors: int | None,
    eckmann_min_tsep: int,
) -> None:
    if rosenstein_emb_dim < 2:
        raise ValueError("rosenstein_emb_dim must be at least 2")
    if rosenstein_lag is not None and rosenstein_lag < 1:
        raise ValueError("rosenstein_lag must be positive or None")
    if rosenstein_min_tsep is not None and rosenstein_min_tsep < 0:
        raise ValueError("rosenstein_min_tsep must be non-negative or None")
    if rosenstein_min_neighbors < 2:
        raise ValueError("rosenstein_min_neighbors must be at least 2")
    if rosenstein_trajectory_len < 2:
        raise ValueError("rosenstein_trajectory_len must be at least 2")
    if rosenstein_fit not in {"RANSAC", "poly"}:
        raise ValueError("rosenstein_fit must be 'RANSAC' or 'poly'")
    if not 0 <= rosenstein_fit_offset <= rosenstein_trajectory_len - 2:
        raise ValueError(
            "rosenstein_fit_offset must leave at least two fit points"
        )

    if eckmann_matrix_dim < 2:
        raise ValueError("eckmann_matrix_dim must be at least 2")
    if eckmann_emb_dim < eckmann_matrix_dim:
        raise ValueError("eckmann_emb_dim must be at least eckmann_matrix_dim")
    if (eckmann_emb_dim - 1) % (eckmann_matrix_dim - 1) != 0:
        raise ValueError(
            "(eckmann_emb_dim - 1) must be divisible by "
            "(eckmann_matrix_dim - 1)"
        )
    if eckmann_min_neighbors is not None and eckmann_min_neighbors < 2:
        raise ValueError("eckmann_min_neighbors must be at least 2 or None")
    if eckmann_min_tsep < 0:
        raise ValueError("eckmann_min_tsep must be non-negative")


def _preflight_sample_count(
    nolds: Any,
    *,
    n_samples: int,
    rosenstein_emb_dim: int,
    rosenstein_lag: int | None,
    rosenstein_min_tsep: int | None,
    rosenstein_trajectory_len: int,
    eckmann_emb_dim: int,
    eckmann_matrix_dim: int,
    eckmann_min_neighbors: int | None,
    eckmann_min_tsep: int,
) -> None:
    if rosenstein_lag is not None and rosenstein_min_tsep is not None:
        minimum = int(
            nolds.lyap_r_len(
                emb_dim=rosenstein_emb_dim,
                lag=rosenstein_lag,
                min_tsep=rosenstein_min_tsep,
                trajectory_len=rosenstein_trajectory_len,
            )
        )
        if n_samples < minimum:
            raise ValueError(
                f"Rosenstein configuration requires at least {minimum} samples"
            )

    min_neighbors = (
        min(2 * eckmann_matrix_dim, eckmann_matrix_dim + 4)
        if eckmann_min_neighbors is None
        else eckmann_min_neighbors
    )
    minimum = int(
        nolds.lyap_e_len(
            emb_dim=eckmann_emb_dim,
            matrix_dim=eckmann_matrix_dim,
            min_nb=min_neighbors,
            min_tsep=eckmann_min_tsep,
        )
    )
    if n_samples < minimum:
        raise ValueError(
            f"Eckmann configuration requires at least {minimum} samples"
        )


def _pairwise_matrix_bytes(
    n_samples: int,
    rosenstein_emb_dim: int,
    rosenstein_lag: int | None,
) -> int:
    lag = 1 if rosenstein_lag is None else rosenstein_lag
    orbit_vectors = max(0, n_samples - (rosenstein_emb_dim - 1) * lag)
    return int(orbit_vectors * orbit_vectors * np.dtype(np.float64).itemsize)


def _rosenstein_fit_diagnostics(
    debug_data: tuple[np.ndarray, np.ndarray, Sequence[float]],
    fit_offset: int,
) -> tuple[float | None, tuple[tuple[float, float], ...]]:
    ks, divergence, polynomial = debug_data
    x_values = np.asarray(ks, dtype=float)
    y_values = np.asarray(divergence, dtype=float)
    trajectory = tuple(
        (float(x_value), float(y_value))
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    fit_x = x_values[fit_offset:]
    fit_y = y_values[fit_offset:]
    if fit_x.size < 2:
        return None, trajectory
    predictions = np.polyval(np.asarray(polynomial, dtype=float), fit_x)
    residual = float(np.sum((fit_y - predictions) ** 2))
    centered = float(np.sum((fit_y - np.mean(fit_y)) ** 2))
    if centered == 0.0:
        return None, trajectory
    return float(1.0 - residual / centered), trajectory


@api_tier(EXPERIMENTAL)
def estimate_time_series_lyapunov(
    signal: Sequence[float],
    *,
    sample_interval: float | None = None,
    sample_rate: float | None = None,
    time_unit: str = "time_unit",
    observable: str | None = None,
    rosenstein_emb_dim: int = 10,
    rosenstein_lag: int | None = None,
    rosenstein_min_tsep: int | None = None,
    rosenstein_min_neighbors: int = 20,
    rosenstein_trajectory_len: int = 20,
    rosenstein_fit: str = "poly",
    rosenstein_fit_offset: int = 0,
    eckmann_emb_dim: int = 9,
    eckmann_matrix_dim: int = 3,
    eckmann_min_neighbors: int | None = None,
    eckmann_min_tsep: int = 0,
    random_seed: int = 0,
    max_pairwise_matrix_bytes: int = DEFAULT_MAX_PAIRWISE_MATRIX_BYTES,
) -> TimeSeriesLyapunovResult:
    """Estimate a largest exponent, spectrum, and Kaplan--Yorke dimension.

    Parameters are passed explicitly to ``nolds`` and recorded in the result.
    The default polynomial fit avoids an optional scikit-learn dependency.
    When RANSAC is requested, a module lock protects NumPy's process-global
    random state while ``random_seed`` makes the fit reproducible.
    """

    values = _finite_signal(signal)
    interval = _resolve_sample_interval(sample_interval, sample_rate)
    if not isinstance(time_unit, str) or not time_unit.strip():
        raise ValueError("time_unit must be a non-empty string")
    if (
        not isinstance(max_pairwise_matrix_bytes, int)
        or max_pairwise_matrix_bytes <= 0
    ):
        raise ValueError("max_pairwise_matrix_bytes must be a positive integer")
    _validate_estimator_parameters(
        rosenstein_emb_dim=rosenstein_emb_dim,
        rosenstein_lag=rosenstein_lag,
        rosenstein_min_tsep=rosenstein_min_tsep,
        rosenstein_min_neighbors=rosenstein_min_neighbors,
        rosenstein_trajectory_len=rosenstein_trajectory_len,
        rosenstein_fit=rosenstein_fit,
        rosenstein_fit_offset=rosenstein_fit_offset,
        eckmann_emb_dim=eckmann_emb_dim,
        eckmann_matrix_dim=eckmann_matrix_dim,
        eckmann_min_neighbors=eckmann_min_neighbors,
        eckmann_min_tsep=eckmann_min_tsep,
    )

    nolds = require_external("nolds")
    _preflight_sample_count(
        nolds,
        n_samples=int(values.size),
        rosenstein_emb_dim=rosenstein_emb_dim,
        rosenstein_lag=rosenstein_lag,
        rosenstein_min_tsep=rosenstein_min_tsep,
        rosenstein_trajectory_len=rosenstein_trajectory_len,
        eckmann_emb_dim=eckmann_emb_dim,
        eckmann_matrix_dim=eckmann_matrix_dim,
        eckmann_min_neighbors=eckmann_min_neighbors,
        eckmann_min_tsep=eckmann_min_tsep,
    )
    pairwise_bytes = _pairwise_matrix_bytes(
        int(values.size),
        rosenstein_emb_dim,
        rosenstein_lag,
    )
    if pairwise_bytes > max_pairwise_matrix_bytes:
        raise MemoryError(
            "Rosenstein pairwise-distance matrix would require approximately "
            f"{pairwise_bytes} bytes, above max_pairwise_matrix_bytes="
            f"{max_pairwise_matrix_bytes}; analyze a shorter deterministic window"
        )

    captured: list[warnings_module.WarningMessage]

    def run_estimators() -> tuple[
        float,
        tuple[np.ndarray, np.ndarray, Sequence[float]],
        np.ndarray,
        list[warnings_module.WarningMessage],
    ]:
        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            largest_result = nolds.lyap_r(
                values,
                emb_dim=rosenstein_emb_dim,
                lag=rosenstein_lag,
                min_tsep=rosenstein_min_tsep,
                tau=interval,
                min_neighbors=rosenstein_min_neighbors,
                trajectory_len=rosenstein_trajectory_len,
                fit=rosenstein_fit,
                debug_data=True,
                fit_offset=rosenstein_fit_offset,
            )
            largest = float(largest_result[0])
            debug_data = largest_result[1]
            spectrum_values = np.asarray(
                nolds.lyap_e(
                    values,
                    emb_dim=eckmann_emb_dim,
                    matrix_dim=eckmann_matrix_dim,
                    min_nb=eckmann_min_neighbors,
                    min_tsep=eckmann_min_tsep,
                    tau=interval,
                ),
                dtype=float,
            )
            return largest, debug_data, spectrum_values, list(caught)

    if rosenstein_fit == "RANSAC":
        with _RANSAC_RANDOM_LOCK:
            random_state = np.random.get_state()
            try:
                np.random.seed(int(random_seed))
                largest, debug_data, spectrum_values, captured = run_estimators()
            finally:
                np.random.set_state(random_state)
    else:
        largest, debug_data, spectrum_values, captured = run_estimators()

    if not np.isfinite(largest):
        raise RuntimeError("Rosenstein estimator returned a non-finite value")
    if (
        spectrum_values.ndim != 1
        or spectrum_values.size != eckmann_matrix_dim
        or not np.all(np.isfinite(spectrum_values))
    ):
        raise RuntimeError(
            "Eckmann estimator returned an invalid or non-finite spectrum"
        )

    ordered = tuple(float(item) for item in np.sort(spectrum_values)[::-1])
    dimension = kaplan_yorke_dimension(ordered)
    spectrum_sum = float(np.sum(spectrum_values))
    if spectrum_sum < 0.0:
        kaplan_yorke_status = "computed_from_finite_time_eckmann_spectrum"
        extra_warnings: tuple[str, ...] = ()
    else:
        kaplan_yorke_status = "spectrum_not_dissipatively_closed"
        extra_warnings = (
            "The estimated spectrum sum is non-negative; Kaplan-Yorke is capped "
            "at the reconstructed spectrum dimension and is not a closed "
            "dissipative estimate.",
        )
    fit_r2, divergence_trajectory = _rosenstein_fit_diagnostics(
        debug_data,
        rosenstein_fit_offset,
    )
    sign_agreement = bool((largest > 0.0) == (ordered[0] > 0.0))

    return TimeSeriesLyapunovResult(
        largest_exponent=largest,
        spectrum=ordered,
        kaplan_yorke_dimension=dimension,
        kaplan_yorke_status=kaplan_yorke_status,
        spectrum_sum=spectrum_sum,
        spectrum_status=SPECTRUM_STATUS,
        sample_interval=interval,
        sample_rate=1.0 / interval,
        time_unit=time_unit.strip(),
        exponent_unit=f"{time_unit.strip()}^-1",
        n_samples=int(values.size),
        estimated_pairwise_matrix_bytes=pairwise_bytes,
        observable=observable,
        backend="nolds",
        backend_version=_backend_version(),
        rosenstein_method=ROSENSTEIN_METHOD,
        eckmann_method=ECKMANN_METHOD,
        rosenstein_parameters={
            "emb_dim": rosenstein_emb_dim,
            "lag": rosenstein_lag,
            "min_tsep": rosenstein_min_tsep,
            "min_neighbors": rosenstein_min_neighbors,
            "trajectory_len": rosenstein_trajectory_len,
            "fit": rosenstein_fit,
            "fit_offset": rosenstein_fit_offset,
            "random_seed": int(random_seed),
        },
        eckmann_parameters={
            "emb_dim": eckmann_emb_dim,
            "matrix_dim": eckmann_matrix_dim,
            "min_neighbors": eckmann_min_neighbors,
            "min_tsep": eckmann_min_tsep,
        },
        largest_sign_agrees_with_spectrum=sign_agreement,
        rosenstein_slope_per_sample=largest * interval,
        rosenstein_fit_r2=fit_r2,
        rosenstein_divergence_index_unit="retained_sample_offset",
        rosenstein_divergence_trajectory=divergence_trajectory,
        rosenstein_divergence_time_trajectory=tuple(
            (offset * interval, log_distance)
            for offset, log_distance in divergence_trajectory
        ),
        evidence_status=EVIDENCE_STATUS,
        warnings=_deduplicated_warning_text(captured, extra_warnings),
    )


__all__ = [
    "ECKMANN_METHOD",
    "EVIDENCE_STATUS",
    "ROSENSTEIN_METHOD",
    "TimeSeriesLyapunovResult",
    "estimate_time_series_lyapunov",
    "kaplan_yorke_dimension",
]
