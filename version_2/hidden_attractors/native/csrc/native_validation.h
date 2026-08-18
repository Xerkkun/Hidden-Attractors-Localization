#ifndef HAFO_NATIVE_VALIDATION_H
#define HAFO_NATIVE_VALIDATION_H

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static inline int hafo_isfinite(double value) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_isfinite(value);
#else
    return isfinite(value);
#endif
}

static inline int hafo_checked_mul_size(size_t left, size_t right, size_t *result) {
    if (!result || (right != 0u && left > SIZE_MAX / right)) {
        return 0;
    }
    *result = left * right;
    return 1;
}

/*
 * Uniform-grid contract shared by the Python wrappers: the requested horizon
 * must be an integer multiple of h, up to a scale-aware floating-point
 * tolerance.  No partial or ceil-rounded final step is permitted.
 */
static inline int hafo_uniform_step_count(double t_final, double h, int *steps) {
    if (!steps || !hafo_isfinite(t_final) || !hafo_isfinite(h) ||
        t_final < 0.0 || !(h > 0.0)) {
        return 0;
    }
    const double ratio = t_final / h;
    if (!hafo_isfinite(ratio) || ratio > (double)(INT_MAX - 2)) {
        return 0;
    }
    const double nearest = nearbyint(ratio);
    const double reconstructed = nearest * h;
    const double scale = fmax(fabs(t_final), fmax(fabs(reconstructed), fabs(h)));
    const double horizon_ulp = fabs(nextafter(t_final, INFINITY) - t_final);
    const double reconstructed_ulp = fabs(
        nextafter(reconstructed, INFINITY) - reconstructed
    );
    const double tolerance = fmax(
        64.0 * DBL_EPSILON * scale,
        fmax(8.0 * horizon_ulp, 8.0 * reconstructed_ulp)
    );
    if (fabs(reconstructed - t_final) > tolerance || nearest < 0.0) {
        return 0;
    }
    *steps = (int)nearest;
    return 1;
}

static inline int hafo_positive_ratio_ceil(double numerator, double denominator, int *value) {
    if (!value || !hafo_isfinite(numerator) || !hafo_isfinite(denominator) ||
        !(numerator > 0.0) || !(denominator > 0.0)) {
        return 0;
    }
    const double ratio = ceil(numerator / denominator);
    if (!hafo_isfinite(ratio) || ratio > (double)(INT_MAX - 2) || ratio < 1.0) {
        return 0;
    }
    *value = (int)ratio;
    return 1;
}

static inline int hafo_values_are_finite(const double *values, size_t count) {
    if (!values && count != 0u) {
        return 0;
    }
    for (size_t index = 0u; index < count; ++index) {
        if (!hafo_isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}

#endif
