/*
 * Generic left-sided Grunwald-Letnikov history kernels for HAFO.
 *
 * Storage contract
 * ----------------
 * Samples and outputs are row-major arrays with shape (n_times, dimension).
 * An order is supplied for every component.  A history_window of zero means
 * full history; a positive value retains at most that many samples, including
 * the current one.  Input and output buffers must not alias.
 *
 * The weights are generated without Gamma-function evaluations:
 *
 *   w_0 = 1,
 *   w_k = w_(k-1) * (1 - (alpha + 1) / k)
 *       = (-1)^k binom(alpha, k).
 *
 * References
 * ----------
 * I. Podlubny, Fractional Differential Equations, Academic Press, 1999,
 * ISBN 978-0-12-558840-9 (Grunwald-Letnikov finite differences).
 * C. Lubich, "Discretized Fractional Calculus", SIAM J. Math. Anal.
 * 17(3), 704-719, 1986, doi:10.1137/0517050 (discrete convolutions).
 *
 * This implementation is original HAFO code and contains no source copied
 * from pynamicalsys or any other third-party numerical library.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "native_validation.h"

#if defined(_WIN32) || defined(__CYGWIN__)
  #define HAFO_GL_EXPORT __declspec(dllexport)
#else
  #define HAFO_GL_EXPORT __attribute__((visibility("default")))
#endif

enum {
    HAFO_GL_OK = 0,
    HAFO_GL_NULL_POINTER = -1,
    HAFO_GL_INVALID_SHAPE = -2,
    HAFO_GL_INVALID_ORDER = -3,
    HAFO_GL_INVALID_STEP = -4,
    HAFO_GL_ALLOCATION_FAILED = -5,
    HAFO_GL_NONFINITE_INPUT = -6,
    HAFO_GL_INVALID_MODE = -7,
    HAFO_GL_ALIASED_BUFFERS = -8,
    HAFO_GL_SIZE_OVERFLOW = -9
};

HAFO_GL_EXPORT int hafo_gl_abi_version(void) {
    return 1;
}

HAFO_GL_EXPORT const char *hafo_gl_kernel_id(void) {
    return "hafo_gl_direct_v1";
}

HAFO_GL_EXPORT int hafo_gl_openmp_enabled(void) {
    #if defined(_OPENMP)
    return 1;
    #else
    return 0;
    #endif
}

static int valid_order(double order) {
    return hafo_isfinite(order) && order > 0.0 && order <= 1.0;
}

HAFO_GL_EXPORT int hafo_gl_weights(
    double order,
    size_t count,
    double *weights_out
) {
    size_t k;

    if (count > 0u && weights_out == NULL) {
        return HAFO_GL_NULL_POINTER;
    }
    if (!valid_order(order)) {
        return HAFO_GL_INVALID_ORDER;
    }
    if (count == 0u) {
        return HAFO_GL_OK;
    }

    weights_out[0] = 1.0;
    for (k = 1u; k < count; ++k) {
        weights_out[k] = weights_out[k - 1u]
            * (1.0 - (order + 1.0) / (double)k);
    }
    return HAFO_GL_OK;
}

static int apply_gl_history(
    const double *restrict samples,
    size_t n_times,
    size_t dimension,
    double step,
    const double *restrict orders,
    int shift_initial,
    size_t history_window,
    int scale_as_derivative,
    double *restrict output
) {
    size_t component;
    size_t sample_index;
    size_t lag;
    size_t weight_count;
    double *weights;
    double *scales;

    if (samples == NULL || orders == NULL || output == NULL) {
        return HAFO_GL_NULL_POINTER;
    }
    if (samples == output) {
        return HAFO_GL_ALIASED_BUFFERS;
    }
    if (n_times == 0u || dimension == 0u) {
        return HAFO_GL_INVALID_SHAPE;
    }
    if (n_times > SIZE_MAX / dimension) {
        return HAFO_GL_SIZE_OVERFLOW;
    }
    if (shift_initial != 0 && shift_initial != 1) {
        return HAFO_GL_INVALID_MODE;
    }
    if (scale_as_derivative != 0 && scale_as_derivative != 1) {
        return HAFO_GL_INVALID_MODE;
    }
    if (scale_as_derivative && (!hafo_isfinite(step) || !(step > 0.0))) {
        return HAFO_GL_INVALID_STEP;
    }

    /* Validate once, outside the quadratic convolution.  Keeping isfinite()
     * out of the inner history loop is material for vectorization and branch
     * prediction on long trajectories. */
    for (component = 0u; component < dimension; ++component) {
        if (!valid_order(orders[component])) {
            return HAFO_GL_INVALID_ORDER;
        }
    }
    for (sample_index = 0u; sample_index < n_times * dimension; ++sample_index) {
        if (!hafo_isfinite(samples[sample_index])) {
            return HAFO_GL_NONFINITE_INPUT;
        }
    }

    weight_count = n_times;
    if (history_window > 0u && history_window < weight_count) {
        weight_count = history_window;
    }
    if (dimension > SIZE_MAX / weight_count
            || dimension * weight_count > SIZE_MAX / sizeof(double)
            || dimension > SIZE_MAX / sizeof(double)) {
        return HAFO_GL_SIZE_OVERFLOW;
    }
    weights = (double *)malloc(
        dimension * weight_count * sizeof(double)
    );
    scales = (double *)malloc(dimension * sizeof(double));
    if (weights == NULL || scales == NULL) {
        free(weights);
        free(scales);
        return HAFO_GL_ALLOCATION_FAILED;
    }

    /* Store weights lag-major so the component loop is contiguous and can be
     * vectorized.  Each output time is independent for this sampled-data
     * operator, so OpenMP distributes times rather than only a usually small
     * number of state components. */
    for (component = 0u; component < dimension; ++component) {
        const double order = orders[component];
        const double scale = scale_as_derivative
            ? pow(step, -order)
            : 1.0;

        if (!hafo_isfinite(scale)) {
            free(weights);
            free(scales);
            return HAFO_GL_NONFINITE_INPUT;
        }
        scales[component] = scale;
        weights[component] = 1.0;
    }
    for (lag = 1u; lag < weight_count; ++lag) {
        for (component = 0u; component < dimension; ++component) {
            weights[lag * dimension + component] =
                weights[(lag - 1u) * dimension + component]
                * (1.0 - (orders[component] + 1.0) / (double)lag);
        }
    }

    #if defined(_OPENMP)
    #pragma omp parallel for schedule(guided, 16) if(n_times >= 256u)
    #endif
    for (sample_index = 0u; sample_index < n_times; ++sample_index) {
        size_t available = sample_index + 1u;
        double *output_row = output + sample_index * dimension;

        if (available > weight_count) {
            available = weight_count;
        }
        for (component = 0u; component < dimension; ++component) {
            output_row[component] = 0.0;
        }
        for (lag = 0u; lag < available; ++lag) {
            const double *sample_row = samples
                + (sample_index - lag) * dimension;
            const double *weight_row = weights + lag * dimension;
            #if defined(_OPENMP)
            #pragma omp simd
            #endif
            for (component = 0u; component < dimension; ++component) {
                const double anchor = shift_initial ? samples[component] : 0.0;
                output_row[component] += weight_row[component]
                    * (sample_row[component] - anchor);
            }
        }
        #if defined(_OPENMP)
        #pragma omp simd
        #endif
        for (component = 0u; component < dimension; ++component) {
            output_row[component] *= scales[component];
        }
    }

    free(weights);
    free(scales);
    return HAFO_GL_OK;
}

HAFO_GL_EXPORT int hafo_gl_convolution(
    const double *samples,
    size_t n_times,
    size_t dimension,
    const double *orders,
    int shift_initial,
    size_t history_window,
    double *output
) {
    return apply_gl_history(
        samples,
        n_times,
        dimension,
        1.0,
        orders,
        shift_initial,
        history_window,
        0,
        output
    );
}

HAFO_GL_EXPORT int hafo_gl_derivative(
    const double *samples,
    size_t n_times,
    size_t dimension,
    double step,
    const double *orders,
    int shift_initial,
    size_t history_window,
    double *output
) {
    return apply_gl_history(
        samples,
        n_times,
        dimension,
        step,
        orders,
        shift_initial,
        history_window,
        1,
        output
    );
}
