/*
 * Native Grassberger--Procaccia q=2 correlation-count kernel for HAFO.
 *
 * Storage and counting contract
 * -----------------------------
 * Points are finite float64 values in row-major (n_points, dimension) order.
 * Radii are finite, positive, and strictly increasing.  Only unordered pairs
 * i < j with j - i > theiler_window are eligible, and a pair is counted at a
 * radius only when distance(point_i, point_j) < radius (strict inequality).
 *
 * One binary search places each eligible pair into the first radius it enters.
 * Per-thread differential bins therefore need O(T * R) storage; a final prefix
 * sum produces cumulative uint64 counts for all radii without atomics in the
 * O(N^2) distance loop.
 *
 * Reference
 * ---------
 * P. Grassberger and I. Procaccia, "Measuring the strangeness of strange
 * attractors", Physica D 9 (1983), 189--208,
 * doi:10.1016/0167-2789(83)90298-1.
 *
 * This implementation is original HAFO code.  It contains no source copied
 * from pynamicalsys, DynamicalSystems.jl, or other third-party libraries.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "native_validation.h"

#if defined(_OPENMP)
  #include <omp.h>
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #define HAFO_CORR_EXPORT __declspec(dllexport)
#else
  #define HAFO_CORR_EXPORT __attribute__((visibility("default")))
#endif

enum {
    HAFO_CORR_OK = 0,
    HAFO_CORR_NULL_POINTER = -1,
    HAFO_CORR_INVALID_SHAPE = -2,
    HAFO_CORR_INVALID_RADIUS = -3,
    HAFO_CORR_INVALID_METRIC = -4,
    HAFO_CORR_ALLOCATION_FAILED = -5,
    HAFO_CORR_NONFINITE_INPUT = -6,
    HAFO_CORR_ALIASED_BUFFERS = -7,
    HAFO_CORR_SIZE_OVERFLOW = -8,
    HAFO_CORR_COUNT_OVERFLOW = -9
};

enum {
    HAFO_CORR_EUCLIDEAN = 0,
    HAFO_CORR_CHEBYSHEV = 1,
    HAFO_CORR_MANHATTAN = 2
};

HAFO_CORR_EXPORT int hafo_correlation_abi_version(void) {
    return 1;
}

HAFO_CORR_EXPORT const char *hafo_correlation_kernel_id(void) {
    return "hafo_correlation_sum_q2_v1";
}

HAFO_CORR_EXPORT int hafo_correlation_openmp_enabled(void) {
    #if defined(_OPENMP)
    return 1;
    #else
    return 0;
    #endif
}

HAFO_CORR_EXPORT const char *hafo_correlation_status(int status_code) {
    switch (status_code) {
        case HAFO_CORR_OK:
            return "ok";
        case HAFO_CORR_NULL_POINTER:
            return "null_pointer";
        case HAFO_CORR_INVALID_SHAPE:
            return "invalid_shape";
        case HAFO_CORR_INVALID_RADIUS:
            return "invalid_radius";
        case HAFO_CORR_INVALID_METRIC:
            return "invalid_metric";
        case HAFO_CORR_ALLOCATION_FAILED:
            return "allocation_failed";
        case HAFO_CORR_NONFINITE_INPUT:
            return "nonfinite_input";
        case HAFO_CORR_ALIASED_BUFFERS:
            return "aliased_buffers";
        case HAFO_CORR_SIZE_OVERFLOW:
            return "size_overflow";
        case HAFO_CORR_COUNT_OVERFLOW:
            return "count_overflow";
        default:
            return "unknown_status";
    }
}

HAFO_CORR_EXPORT const char *hafo_correlation_status_message(int status_code) {
    return hafo_correlation_status(status_code);
}

static int checked_size_product(size_t left, size_t right, size_t *product) {
    if (left != 0u && right > SIZE_MAX / left) {
        return 0;
    }
    *product = left * right;
    return 1;
}

static int ranges_overlap(
    const void *left,
    size_t left_bytes,
    const void *right,
    size_t right_bytes
) {
    uintptr_t left_address;
    uintptr_t right_address;

    if (left_bytes == 0u || right_bytes == 0u) {
        return 0;
    }
    left_address = (uintptr_t)left;
    right_address = (uintptr_t)right;
    if (left_address <= right_address) {
        return (right_address - left_address) < left_bytes;
    }
    return (left_address - right_address) < right_bytes;
}

static int eligible_pair_count(
    size_t n_points,
    size_t theiler_window,
    uint64_t *eligible_out
) {
    size_t separated_points;
    uint64_t left;
    uint64_t right;

    if (theiler_window >= n_points - 1u) {
        *eligible_out = 0u;
        return HAFO_CORR_OK;
    }
    separated_points = n_points - theiler_window - 1u;
    if ((uintmax_t)separated_points > (uintmax_t)UINT64_MAX) {
        return HAFO_CORR_COUNT_OVERFLOW;
    }
    left = (uint64_t)separated_points;
    if (left == UINT64_MAX) {
        return HAFO_CORR_COUNT_OVERFLOW;
    }
    right = left + 1u;
    if ((left & 1u) == 0u) {
        left /= 2u;
    } else {
        right /= 2u;
    }
    if (right != 0u && left > UINT64_MAX / right) {
        return HAFO_CORR_COUNT_OVERFLOW;
    }
    *eligible_out = left * right;
    return HAFO_CORR_OK;
}

static double pair_distance(
    const double *left,
    const double *right,
    size_t dimension,
    int metric
) {
    size_t component;

    if (metric == HAFO_CORR_EUCLIDEAN) {
        double distance = 0.0;
        for (component = 0u; component < dimension; ++component) {
            distance = hypot(distance, left[component] - right[component]);
        }
        return distance;
    }
    if (metric == HAFO_CORR_CHEBYSHEV) {
        double distance = 0.0;
        for (component = 0u; component < dimension; ++component) {
            const double difference = fabs(left[component] - right[component]);
            if (difference > distance) {
                distance = difference;
            }
        }
        return distance;
    }
    {
        double distance = 0.0;
        for (component = 0u; component < dimension; ++component) {
            distance += fabs(left[component] - right[component]);
        }
        return distance;
    }
}

static size_t first_strictly_larger_radius(
    const double *radii,
    size_t n_radii,
    double distance
) {
    size_t lower = 0u;
    size_t upper = n_radii;

    while (lower < upper) {
        const size_t middle = lower + (upper - lower) / 2u;
        if (distance < radii[middle]) {
            upper = middle;
        } else {
            lower = middle + 1u;
        }
    }
    return lower;
}

HAFO_CORR_EXPORT int hafo_correlation_sum_counts(
    const double *points,
    size_t n_points,
    size_t dimension,
    const double *radii,
    size_t n_radii,
    size_t theiler_window,
    int metric,
    uint64_t *counts_out,
    uint64_t *eligible_pairs_out
) {
    size_t point_values;
    size_t point_bytes;
    size_t radius_bytes;
    size_t count_bytes;
    size_t thread_count;
    size_t local_bin_count;
    size_t local_bin_bytes;
    size_t index;
    size_t radius_index;
    uint64_t eligible_pairs;
    uint64_t *local_bins;
    int status;

    if (points == NULL || radii == NULL || counts_out == NULL
            || eligible_pairs_out == NULL) {
        return HAFO_CORR_NULL_POINTER;
    }
    if (n_points < 2u || dimension == 0u || n_radii == 0u) {
        return HAFO_CORR_INVALID_SHAPE;
    }
    if (metric != HAFO_CORR_EUCLIDEAN
            && metric != HAFO_CORR_CHEBYSHEV
            && metric != HAFO_CORR_MANHATTAN) {
        return HAFO_CORR_INVALID_METRIC;
    }
    if (!checked_size_product(n_points, dimension, &point_values)
            || !checked_size_product(point_values, sizeof(double), &point_bytes)
            || !checked_size_product(n_radii, sizeof(double), &radius_bytes)
            || !checked_size_product(n_radii, sizeof(uint64_t), &count_bytes)) {
        return HAFO_CORR_SIZE_OVERFLOW;
    }
    if (ranges_overlap(points, point_bytes, counts_out, count_bytes)
            || ranges_overlap(radii, radius_bytes, counts_out, count_bytes)
            || ranges_overlap(
                points,
                point_bytes,
                eligible_pairs_out,
                sizeof(uint64_t)
            )
            || ranges_overlap(
                radii,
                radius_bytes,
                eligible_pairs_out,
                sizeof(uint64_t)
            )
            || ranges_overlap(
                counts_out,
                count_bytes,
                eligible_pairs_out,
                sizeof(uint64_t)
            )) {
        return HAFO_CORR_ALIASED_BUFFERS;
    }
    for (index = 0u; index < point_values; ++index) {
        if (!hafo_isfinite(points[index])) {
            return HAFO_CORR_NONFINITE_INPUT;
        }
    }
    for (radius_index = 0u; radius_index < n_radii; ++radius_index) {
        if (!hafo_isfinite(radii[radius_index])) {
            return HAFO_CORR_NONFINITE_INPUT;
        }
        if (!(radii[radius_index] > 0.0)
                || (radius_index > 0u
                    && !(radii[radius_index] > radii[radius_index - 1u]))) {
            return HAFO_CORR_INVALID_RADIUS;
        }
    }
    status = eligible_pair_count(n_points, theiler_window, &eligible_pairs);
    if (status != HAFO_CORR_OK) {
        return status;
    }

    #if defined(_OPENMP)
    thread_count = (size_t)omp_get_max_threads();
    #else
    thread_count = 1u;
    #endif
    if (thread_count == 0u
            || !checked_size_product(thread_count, n_radii, &local_bin_count)
            || !checked_size_product(
                local_bin_count,
                sizeof(uint64_t),
                &local_bin_bytes
            )) {
        return HAFO_CORR_SIZE_OVERFLOW;
    }
    local_bins = (uint64_t *)calloc(local_bin_count, sizeof(uint64_t));
    if (local_bins == NULL) {
        return HAFO_CORR_ALLOCATION_FAILED;
    }

    #if defined(_OPENMP)
    #pragma omp parallel
    #endif
    {
        size_t thread_index = 0u;
        uint64_t *thread_bins;
        size_t left_index;

        #if defined(_OPENMP)
        thread_index = (size_t)omp_get_thread_num();
        #endif
        thread_bins = local_bins + thread_index * n_radii;

        #if defined(_OPENMP)
        #pragma omp for schedule(guided, 4)
        #endif
        for (left_index = 0u; left_index < n_points; ++left_index) {
            size_t right_index;
            size_t first_right;

            if (theiler_window >= n_points - left_index - 1u) {
                continue;
            }
            first_right = left_index + theiler_window + 1u;
            for (right_index = first_right;
                    right_index < n_points;
                    ++right_index) {
                const double distance = pair_distance(
                    points + left_index * dimension,
                    points + right_index * dimension,
                    dimension,
                    metric
                );
                const size_t first_radius = first_strictly_larger_radius(
                    radii,
                    n_radii,
                    distance
                );
                if (first_radius < n_radii) {
                    thread_bins[first_radius] += 1u;
                }
            }
        }
    }

    {
        uint64_t cumulative = 0u;
        for (radius_index = 0u; radius_index < n_radii; ++radius_index) {
            uint64_t differential = 0u;
            size_t thread_index;
            for (thread_index = 0u; thread_index < thread_count; ++thread_index) {
                const uint64_t contribution =
                    local_bins[thread_index * n_radii + radius_index];
                if (UINT64_MAX - differential < contribution) {
                    free(local_bins);
                    return HAFO_CORR_COUNT_OVERFLOW;
                }
                differential += contribution;
            }
            if (UINT64_MAX - cumulative < differential) {
                free(local_bins);
                return HAFO_CORR_COUNT_OVERFLOW;
            }
            cumulative += differential;
            if (cumulative > eligible_pairs) {
                free(local_bins);
                return HAFO_CORR_COUNT_OVERFLOW;
            }
            counts_out[radius_index] = cumulative;
        }
    }

    *eligible_pairs_out = eligible_pairs;
    free(local_bins);
    return HAFO_CORR_OK;
}
