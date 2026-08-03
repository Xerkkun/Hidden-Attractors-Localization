"""Finite-grid basin entropy and final-state uncertainty diagnostics.

Stability: experimental

The functions in this module consume already classified basin samples.  They
are therefore shared by integer- and fractional-order simulations, but they do
not prove global basin structure, fractality, Wada boundaries, or hiddenness.
The grid, classifier, perturbation rule, and (for fractional systems) history
initialisation remain part of the numerical experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Literal

import numpy as np
from numba import njit


BASIN_ENTROPY_REFERENCE = "https://doi.org/10.1038/srep31416"
UNCERTAINTY_REFERENCE = "https://doi.org/10.1016/0375-9601(85)90193-8"


@dataclass(frozen=True, slots=True)
class BasinEntropyResult:
    """Entropy of a finite, regularly sampled two-dimensional basin grid."""

    basin_entropy: float
    boundary_basin_entropy: float
    boundary_box_fraction: float
    boundary_boxes: int
    evaluated_boxes: int
    total_boxes: int
    empty_boxes: int
    classified_samples: int
    ignored_samples: int
    dropped_samples: int
    observed_basins: int
    box_shape: tuple[int, int]
    partial_box_policy: str
    entropy_base: float
    log_two_threshold: float
    boundary_entropy_margin: float
    comparison_tolerance: float
    boundary_entropy_defined: bool
    log_two_criterion_applicable: bool
    log_two_criterion_reason: str
    boundary_entropy_above_log_two: bool
    status: str = "finite_grid_diagnostic"
    reference: str = BASIN_ENTROPY_REFERENCE


@dataclass(frozen=True, slots=True)
class UncertaintyFractionResult:
    """Final-state disagreement rate for one declared perturbation scale."""

    fraction: float
    uncertain_pairs: int
    evaluated_pairs: int
    ignored_pairs: int
    confidence: float
    confidence_interval: tuple[float, float]
    perturbation_scale: float | None
    scale_units: str | None
    perturbation_norm: str | None
    perturbation_direction: tuple[float, ...] | None
    interval_assumption: str = "independent_Bernoulli_pairs"
    status: str = "finite_sample_diagnostic"
    reference: str = UNCERTAINTY_REFERENCE


@dataclass(frozen=True, slots=True)
class UncertaintyExponentResult:
    """Log--log fit ``f(epsilon) = C epsilon**alpha`` over finite scales."""

    exponent: float
    intercept: float
    coefficient: float
    r_squared: float
    standard_error: float
    scales: np.ndarray
    fractions: np.ndarray
    sampling_space_dimension: float | None
    estimated_boundary_dimension: float | None
    dimension_estimate_admissible: bool | None
    status: str = "finite_scale_fit"
    reference: str = UNCERTAINTY_REFERENCE


@njit(cache=True, nogil=True)
def _basin_entropy_kernel(
    encoded: np.ndarray,
    n_labels: int,
    box_rows: int,
    box_columns: int,
    include_partial: bool,
) -> tuple[float, float, int, int, int]:
    """Accumulate natural-log box entropies on integer-encoded labels."""

    n_rows, n_columns = encoded.shape
    entropy_sum = 0.0
    boundary_entropy_sum = 0.0
    boundary_boxes = 0
    empty_boxes = 0
    classified_samples = 0
    row_limit = n_rows if include_partial else n_rows - n_rows % box_rows
    column_limit = n_columns if include_partial else n_columns - n_columns % box_columns
    for row_start in range(0, row_limit, box_rows):
        row_stop = min(row_start + box_rows, n_rows)
        for column_start in range(0, column_limit, box_columns):
            column_stop = min(column_start + box_columns, n_columns)
            counts = np.zeros(n_labels, dtype=np.int64)
            samples_in_box = 0
            for row in range(row_start, row_stop):
                for column in range(column_start, column_stop):
                    label = encoded[row, column]
                    if label >= 0:
                        counts[label] += 1
                        samples_in_box += 1
            if samples_in_box == 0:
                empty_boxes += 1
                continue
            classified_samples += samples_in_box
            occupied = 0
            box_entropy = 0.0
            for label in range(n_labels):
                count = counts[label]
                if count > 0:
                    occupied += 1
                    probability = count / samples_in_box
                    box_entropy -= probability * np.log(probability)
            entropy_sum += box_entropy
            if occupied > 1:
                boundary_boxes += 1
                boundary_entropy_sum += box_entropy
    return (
        entropy_sum,
        boundary_entropy_sum,
        boundary_boxes,
        empty_boxes,
        classified_samples,
    )


def _normalise_box_shape(box_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(box_size, (int, np.integer)):
        shape = (int(box_size), int(box_size))
    else:
        try:
            raw = tuple(box_size)
        except TypeError as exc:
            raise TypeError("box_size must be an integer or a pair of integers.") from exc
        if len(raw) != 2 or any(
            not isinstance(value, (int, np.integer)) for value in raw
        ):
            raise TypeError("box_size must be an integer or a pair of integers.")
        shape = (int(raw[0]), int(raw[1]))
    if shape[0] < 1 or shape[1] < 1:
        raise ValueError("box_size entries must be positive integers.")
    return shape


def basin_entropy(
    labels: np.ndarray,
    box_size: int | tuple[int, int],
    *,
    ignored_labels: Iterable[int] = (),
    entropy_base: float = np.e,
    partial_boxes: Literal["require_complete", "drop", "include_equal"] = (
        "require_complete"
    ),
) -> BasinEntropyResult:
    """Compute basin and boundary-basin entropy on a regular 2-D label grid.

    Boxes are non-overlapping and anchored at the array origin.  The canonical
    default requires equal complete boxes.  ``drop`` omits trailing partial
    boxes, while ``include_equal`` retains them with the same weight as a full
    box and is therefore an explicitly labelled finite-grid variant.  Ignored
    samples do not contribute to probabilities; a box containing only ignored
    samples is reported as empty and omitted from both entropy averages.
    """

    values = np.asarray(labels)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("labels must be a non-empty two-dimensional grid.")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("labels must contain integer basin class identifiers.")
    shape = _normalise_box_shape(box_size)
    base = float(entropy_base)
    if not np.isfinite(base) or base <= 1.0:
        raise ValueError("entropy_base must be finite and greater than one.")
    policy = str(partial_boxes).strip().lower()
    if policy not in {"require_complete", "drop", "include_equal"}:
        raise ValueError(
            "partial_boxes must be 'require_complete', 'drop', or 'include_equal'."
        )
    divisible = values.shape[0] % shape[0] == 0 and values.shape[1] % shape[1] == 0
    if policy == "require_complete" and not divisible:
        raise ValueError(
            "grid shape must be divisible by box_size when "
            "partial_boxes='require_complete'."
        )
    if policy == "drop" and (shape[0] > values.shape[0] or shape[1] > values.shape[1]):
        raise ValueError("partial_boxes='drop' would leave no complete boxes.")

    ignored = np.asarray(tuple(ignored_labels), dtype=np.int64).reshape(-1)
    values_i64 = np.ascontiguousarray(values, dtype=np.int64)
    valid = ~np.isin(values_i64, ignored) if ignored.size else np.ones(values.shape, dtype=bool)
    if not np.any(valid):
        raise ValueError("no classified samples remain after applying ignored_labels.")
    unique_labels = np.unique(values_i64[valid])
    encoded = np.full(values.shape, -1, dtype=np.int64)
    encoded[valid] = np.searchsorted(unique_labels, values_i64[valid])

    entropy_sum, boundary_sum, boundary_boxes, empty_boxes, classified = (
        _basin_entropy_kernel(
            np.ascontiguousarray(encoded),
            int(unique_labels.size),
            shape[0],
            shape[1],
            policy == "include_equal",
        )
    )
    if policy == "include_equal":
        row_boxes = (values.shape[0] + shape[0] - 1) // shape[0]
        column_boxes = (values.shape[1] + shape[1] - 1) // shape[1]
    else:
        row_boxes = values.shape[0] // shape[0]
        column_boxes = values.shape[1] // shape[1]
    total_boxes = int(row_boxes * column_boxes)
    evaluated_boxes = total_boxes - int(empty_boxes)
    log_base = float(np.log(base))
    entropy = float(entropy_sum / evaluated_boxes / log_base)
    boundary_entropy = (
        float(boundary_sum / boundary_boxes / log_base) if boundary_boxes else 0.0
    )
    threshold = float(np.log(2.0) / log_base)
    margin = boundary_entropy - threshold
    comparison_tolerance = float(
        32.0
        * np.finfo(float).eps
        * max(1.0, abs(boundary_entropy), abs(threshold))
    )
    ignored_count = int(np.count_nonzero(~valid))
    dropped_count = int(np.count_nonzero(valid) - classified)
    boundary_entropy_defined = bool(boundary_boxes > 0)
    if not boundary_entropy_defined:
        criterion_applicable = False
        criterion_reason = "no_boundary_boxes"
    elif unique_labels.size < 3:
        criterion_applicable = False
        criterion_reason = "fewer_than_three_observed_basins"
    elif policy == "include_equal":
        criterion_applicable = False
        criterion_reason = "unequal_partial_boxes_weighted_equally"
    elif ignored_count > 0:
        criterion_applicable = False
        criterion_reason = "ignored_samples_change_box_probabilities"
    else:
        criterion_applicable = True
        criterion_reason = "finite_grid_preconditions_only"
    return BasinEntropyResult(
        basin_entropy=entropy,
        boundary_basin_entropy=boundary_entropy,
        boundary_box_fraction=float(boundary_boxes / evaluated_boxes),
        boundary_boxes=int(boundary_boxes),
        evaluated_boxes=evaluated_boxes,
        total_boxes=total_boxes,
        empty_boxes=int(empty_boxes),
        classified_samples=int(classified),
        ignored_samples=ignored_count,
        dropped_samples=dropped_count,
        observed_basins=int(unique_labels.size),
        box_shape=shape,
        partial_box_policy=policy,
        entropy_base=base,
        log_two_threshold=threshold,
        boundary_entropy_margin=float(margin),
        comparison_tolerance=comparison_tolerance,
        boundary_entropy_defined=boundary_entropy_defined,
        log_two_criterion_applicable=criterion_applicable,
        log_two_criterion_reason=criterion_reason,
        boundary_entropy_above_log_two=bool(
            criterion_applicable and margin > comparison_tolerance
        ),
    )


def uncertainty_fraction(
    reference_labels: np.ndarray,
    perturbed_labels: np.ndarray,
    *,
    ignored_labels: Iterable[int] = (),
    confidence: float = 0.95,
    perturbation_scale: float | None = None,
    scale_units: str | None = None,
    perturbation_norm: str | None = None,
    perturbation_direction: Iterable[float] | None = None,
) -> UncertaintyFractionResult:
    """Return the fraction of valid pairs whose final basin labels disagree.

    The caller owns the perturbation construction.  When a scale is supplied,
    its units and norm are required and are retained in the result.  The
    interval is the Wilson binomial interval under independent Bernoulli pairs.
    For correlated grids it is descriptive only.  It does not include
    classifier or integration error.
    """

    reference = np.asarray(reference_labels)
    perturbed = np.asarray(perturbed_labels)
    if reference.shape != perturbed.shape or reference.size == 0:
        raise ValueError("reference_labels and perturbed_labels must have the same non-empty shape.")
    if not np.issubdtype(reference.dtype, np.integer) or not np.issubdtype(
        perturbed.dtype, np.integer
    ):
        raise TypeError("basin labels must be integer class identifiers.")
    level = float(confidence)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")

    metadata_supplied = any(
        value is not None
        for value in (scale_units, perturbation_norm, perturbation_direction)
    )
    if perturbation_scale is None:
        if metadata_supplied:
            raise ValueError(
                "perturbation_scale is required when perturbation metadata is supplied."
            )
        scale = None
        units = None
        norm = None
        direction = None
    else:
        scale = float(perturbation_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("perturbation_scale must be finite and positive.")
        units = str(scale_units).strip() if scale_units is not None else ""
        norm = str(perturbation_norm).strip() if perturbation_norm is not None else ""
        if not units or not norm:
            raise ValueError(
                "scale_units and perturbation_norm are required with perturbation_scale."
            )
        if perturbation_direction is None:
            direction = None
        else:
            try:
                direction_array = np.asarray(
                    tuple(perturbation_direction), dtype=float
                ).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "perturbation_direction must be an iterable of real values."
                ) from exc
            if direction_array.size == 0 or not np.all(np.isfinite(direction_array)):
                raise ValueError(
                    "perturbation_direction must contain finite real values."
                )
            if not np.any(direction_array != 0.0):
                raise ValueError("perturbation_direction must be nonzero.")
            direction = tuple(float(value) for value in direction_array)

    ignored = np.asarray(tuple(ignored_labels), dtype=np.int64).reshape(-1)
    if ignored.size:
        valid = ~np.isin(reference, ignored) & ~np.isin(perturbed, ignored)
    else:
        valid = np.ones(reference.shape, dtype=bool)
    evaluated = int(np.count_nonzero(valid))
    if evaluated == 0:
        raise ValueError("no valid label pairs remain after applying ignored_labels.")
    uncertain = int(np.count_nonzero(reference[valid] != perturbed[valid]))
    fraction = uncertain / evaluated

    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    z_squared = z * z
    denominator = 1.0 + z_squared / evaluated
    centre = (fraction + z_squared / (2.0 * evaluated)) / denominator
    half_width = (
        z
        * np.sqrt(
            fraction * (1.0 - fraction) / evaluated
            + z_squared / (4.0 * evaluated * evaluated)
        )
        / denominator
    )
    return UncertaintyFractionResult(
        fraction=float(fraction),
        uncertain_pairs=uncertain,
        evaluated_pairs=evaluated,
        ignored_pairs=int(reference.size - evaluated),
        confidence=level,
        confidence_interval=(float(centre - half_width), float(centre + half_width)),
        perturbation_scale=scale,
        scale_units=units,
        perturbation_norm=norm,
        perturbation_direction=direction,
    )


def estimate_uncertainty_exponent(
    scales: np.ndarray,
    fractions: np.ndarray,
    *,
    sampling_space_dimension: float | None = None,
) -> UncertaintyExponentResult:
    """Fit a finite-scale uncertainty exponent by ordinary log--log regression.

    All scale selection must be performed by the caller and remains visible in
    the returned arrays.  A high ``r_squared`` is not by itself evidence that
    the asymptotic scaling regime has been reached.
    """

    epsilon = np.asarray(scales, dtype=float).reshape(-1)
    uncertain = np.asarray(fractions, dtype=float).reshape(-1)
    if epsilon.size != uncertain.size or epsilon.size < 2:
        raise ValueError("scales and fractions must contain the same two or more values.")
    if (
        not np.all(np.isfinite(epsilon))
        or np.any(epsilon <= 0.0)
        or not np.all(np.isfinite(uncertain))
        or np.any(uncertain <= 0.0)
        or np.any(uncertain > 1.0)
    ):
        raise ValueError("scales must be positive and fractions must lie in (0, 1].")
    log_scale = np.log(epsilon)
    if np.ptp(log_scale) == 0.0:
        raise ValueError("at least two distinct scales are required.")
    log_fraction = np.log(uncertain)
    design = np.column_stack((log_scale, np.ones(log_scale.size)))
    coefficients, _, _, _ = np.linalg.lstsq(design, log_fraction, rcond=None)
    exponent = float(coefficients[0])
    intercept = float(coefficients[1])
    fitted = design @ coefficients
    residuals = log_fraction - fitted
    residual_sum = float(residuals @ residuals)
    centred = log_fraction - np.mean(log_fraction)
    total_sum = float(centred @ centred)
    constant_tolerance = np.finfo(float).eps * max(
        1.0, float(log_fraction @ log_fraction)
    )
    if total_sum <= constant_tolerance:
        r_squared = float("nan")
        fit_status = "degenerate_constant_response"
    else:
        r_squared = 1.0 - residual_sum / total_sum
        fit_status = "finite_scale_fit"
    if log_scale.size > 2:
        scale_sum = float(np.sum((log_scale - np.mean(log_scale)) ** 2))
        standard_error = float(np.sqrt((residual_sum / (log_scale.size - 2)) / scale_sum))
    else:
        standard_error = float("nan")

    dimension: float | None
    boundary_dimension: float | None
    admissible: bool | None
    if sampling_space_dimension is None:
        dimension = None
        boundary_dimension = None
        admissible = None
    else:
        dimension = float(sampling_space_dimension)
        if not np.isfinite(dimension) or dimension <= 0.0:
            raise ValueError("sampling_space_dimension must be finite and positive.")
        boundary_dimension = dimension - exponent
        admissible = bool(0.0 <= boundary_dimension <= dimension)
    return UncertaintyExponentResult(
        exponent=exponent,
        intercept=intercept,
        coefficient=float(np.exp(intercept)),
        r_squared=float(r_squared),
        standard_error=standard_error,
        scales=epsilon.copy(),
        fractions=uncertain.copy(),
        sampling_space_dimension=dimension,
        estimated_boundary_dimension=boundary_dimension,
        dimension_estimate_admissible=admissible,
        status=fit_status,
    )


__all__ = [
    "BASIN_ENTROPY_REFERENCE",
    "UNCERTAINTY_REFERENCE",
    "BasinEntropyResult",
    "UncertaintyExponentResult",
    "UncertaintyFractionResult",
    "basin_entropy",
    "estimate_uncertainty_exponent",
    "uncertainty_fraction",
]
