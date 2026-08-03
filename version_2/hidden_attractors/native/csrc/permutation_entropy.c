/*
 * Native ordinal-pattern counting kernel for HAFO permutation entropy.
 *
 * Contract
 * --------
 * The input is a finite float64 signal.  A forward embedding window starting
 * at s contains signal[s + k * delay], k = 0, ..., m - 1.  Sorting those
 * values in ascending order yields a permutation of the original indices.
 * Its histogram bin is the zero-based lexicographic Lehmer rank.
 *
 * Exact ties are handled by one of three policies:
 *
 *   0 stable_index : equal values keep increasing original-index order;
 *   1 omit         : a window containing a tie is not counted;
 *   2 raise        : any tied window returns HAFO_PERM_TIED_WINDOW.
 *
 * The maximum embedding dimension is ten, so every local permutation and
 * factorial fits fixed stack storage and uint64_t.  OpenMP, when enabled by
 * the build, parallelizes independent windows.  A bounded per-thread
 * histogram avoids atomics when it fits in 64 MiB; larger histograms use
 * atomic updates instead of unbounded O(thread_count * m!) memory at m=10.
 *
 * References
 * ----------
 * C. Bandt and B. Pompe, "Permutation Entropy: A Natural Complexity Measure
 * for Time Series", Physical Review Letters 88 (2002), 174102,
 * doi:10.1103/PhysRevLett.88.174102.
 *
 * This implementation is original HAFO code.  It contains no source copied
 * from pynamicalsys, DynamicalSystems.jl, or other third-party libraries.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_OPENMP)
  #include <omp.h>
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #define HAFO_PERM_EXPORT __declspec(dllexport)
#else
  #define HAFO_PERM_EXPORT __attribute__((visibility("default")))
#endif

#define HAFO_PERM_MIN_EMBEDDING 2u
#define HAFO_PERM_MAX_EMBEDDING 10u
#define HAFO_PERM_OPENMP_MIN_WINDOWS 1024u
#define HAFO_PERM_LOCAL_HISTOGRAM_MAX_BYTES (64u * 1024u * 1024u)

enum {
    HAFO_PERM_OK = 0,
    HAFO_PERM_NULL_POINTER = -1,
    HAFO_PERM_INVALID_SHAPE = -2,
    HAFO_PERM_INVALID_EMBEDDING = -3,
    HAFO_PERM_INVALID_DELAY = -4,
    HAFO_PERM_INVALID_TIE_POLICY = -5,
    HAFO_PERM_INVALID_COUNTS_LENGTH = -6,
    HAFO_PERM_NONFINITE_INPUT = -7,
    HAFO_PERM_ALIASED_BUFFERS = -8,
    HAFO_PERM_SIZE_OVERFLOW = -9,
    HAFO_PERM_COUNT_OVERFLOW = -10,
    HAFO_PERM_TIED_WINDOW = -11
};

enum {
    HAFO_PERM_STABLE_INDEX = 0,
    HAFO_PERM_OMIT = 1,
    HAFO_PERM_RAISE = 2
};

static const uint64_t HAFO_PERM_FACTORIALS[11] = {
    UINT64_C(1),
    UINT64_C(1),
    UINT64_C(2),
    UINT64_C(6),
    UINT64_C(24),
    UINT64_C(120),
    UINT64_C(720),
    UINT64_C(5040),
    UINT64_C(40320),
    UINT64_C(362880),
    UINT64_C(3628800)
};

HAFO_PERM_EXPORT int hafo_permutation_abi_version(void) {
    return 1;
}

HAFO_PERM_EXPORT const char *hafo_permutation_kernel_id(void) {
    return "hafo_permutation_entropy_counts_v1";
}

HAFO_PERM_EXPORT int hafo_permutation_openmp_enabled(void) {
    #if defined(_OPENMP)
    return 1;
    #else
    return 0;
    #endif
}

HAFO_PERM_EXPORT size_t hafo_permutation_max_embedding_dimension(void) {
    return HAFO_PERM_MAX_EMBEDDING;
}

HAFO_PERM_EXPORT uint64_t hafo_permutation_pattern_count(
    size_t embedding_dimension
) {
    if (embedding_dimension > HAFO_PERM_MAX_EMBEDDING) {
        return UINT64_C(0);
    }
    return HAFO_PERM_FACTORIALS[embedding_dimension];
}

HAFO_PERM_EXPORT const char *hafo_permutation_status(int status_code) {
    switch (status_code) {
        case HAFO_PERM_OK:
            return "ok";
        case HAFO_PERM_NULL_POINTER:
            return "null_pointer";
        case HAFO_PERM_INVALID_SHAPE:
            return "invalid_shape";
        case HAFO_PERM_INVALID_EMBEDDING:
            return "invalid_embedding_dimension";
        case HAFO_PERM_INVALID_DELAY:
            return "invalid_delay";
        case HAFO_PERM_INVALID_TIE_POLICY:
            return "invalid_tie_policy";
        case HAFO_PERM_INVALID_COUNTS_LENGTH:
            return "invalid_counts_length";
        case HAFO_PERM_NONFINITE_INPUT:
            return "nonfinite_input";
        case HAFO_PERM_ALIASED_BUFFERS:
            return "aliased_buffers";
        case HAFO_PERM_SIZE_OVERFLOW:
            return "size_overflow";
        case HAFO_PERM_COUNT_OVERFLOW:
            return "count_overflow";
        case HAFO_PERM_TIED_WINDOW:
            return "tied_window";
        default:
            return "unknown_status";
    }
}

HAFO_PERM_EXPORT const char *hafo_permutation_status_message(
    int status_code
) {
    return hafo_permutation_status(status_code);
}

static int checked_size_product(
    size_t left,
    size_t right,
    size_t *product
) {
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

static int window_has_tie(
    const double *signal,
    size_t start,
    size_t embedding_dimension,
    size_t delay
) {
    size_t left;

    for (left = 0u; left < embedding_dimension; ++left) {
        size_t right;
        const double left_value = signal[start + left * delay];
        for (right = left + 1u; right < embedding_dimension; ++right) {
            if (left_value == signal[start + right * delay]) {
                return 1;
            }
        }
    }
    return 0;
}

static uint64_t ordinal_pattern_rank(
    const double *signal,
    size_t start,
    size_t embedding_dimension,
    size_t delay
) {
    size_t permutation[HAFO_PERM_MAX_EMBEDDING];
    size_t index;
    uint64_t rank = UINT64_C(0);

    for (index = 0u; index < embedding_dimension; ++index) {
        permutation[index] = index;
    }

    /* Stable insertion sort by (value, original_index). */
    for (index = 1u; index < embedding_dimension; ++index) {
        const size_t key = permutation[index];
        const double key_value = signal[start + key * delay];
        size_t position = index;

        while (position > 0u) {
            const size_t previous = permutation[position - 1u];
            const double previous_value = signal[start + previous * delay];
            const int key_precedes =
                (key_value < previous_value)
                || (key_value == previous_value && key < previous);
            if (!key_precedes) {
                break;
            }
            permutation[position] = previous;
            --position;
        }
        permutation[position] = key;
    }

    /* Zero-based lexicographic Lehmer rank of the sorted-index permutation. */
    for (index = 0u; index + 1u < embedding_dimension; ++index) {
        size_t later;
        uint64_t smaller_to_right = UINT64_C(0);
        for (later = index + 1u; later < embedding_dimension; ++later) {
            if (permutation[later] < permutation[index]) {
                ++smaller_to_right;
            }
        }
        rank += smaller_to_right
            * HAFO_PERM_FACTORIALS[embedding_dimension - index - 1u];
    }
    return rank;
}

HAFO_PERM_EXPORT int hafo_permutation_entropy_counts(
    const double *signal,
    size_t n_samples,
    size_t embedding_dimension,
    size_t delay,
    int tie_policy,
    uint64_t *counts_out,
    size_t counts_length,
    uint64_t *total_windows_out,
    uint64_t *valid_windows_out,
    uint64_t *tied_windows_out
) {
    size_t signal_bytes;
    size_t counts_bytes;
    size_t embedding_span;
    size_t total_windows_size;
    size_t index;
    uint64_t total_windows;
    uint64_t valid_windows = UINT64_C(0);
    uint64_t tied_windows = UINT64_C(0);
    const uint64_t expected_counts =
        embedding_dimension <= HAFO_PERM_MAX_EMBEDDING
        ? HAFO_PERM_FACTORIALS[embedding_dimension]
        : UINT64_C(0);

    if (signal == NULL || counts_out == NULL || total_windows_out == NULL
            || valid_windows_out == NULL || tied_windows_out == NULL) {
        return HAFO_PERM_NULL_POINTER;
    }
    if (embedding_dimension < HAFO_PERM_MIN_EMBEDDING
            || embedding_dimension > HAFO_PERM_MAX_EMBEDDING) {
        return HAFO_PERM_INVALID_EMBEDDING;
    }
    if (delay == 0u) {
        return HAFO_PERM_INVALID_DELAY;
    }
    if (tie_policy != HAFO_PERM_STABLE_INDEX
            && tie_policy != HAFO_PERM_OMIT
            && tie_policy != HAFO_PERM_RAISE) {
        return HAFO_PERM_INVALID_TIE_POLICY;
    }
    if ((uintmax_t)counts_length != (uintmax_t)expected_counts) {
        return HAFO_PERM_INVALID_COUNTS_LENGTH;
    }
    if (!checked_size_product(n_samples, sizeof(double), &signal_bytes)
            || !checked_size_product(
                counts_length,
                sizeof(uint64_t),
                &counts_bytes
            )
            || !checked_size_product(
                embedding_dimension - 1u,
                delay,
                &embedding_span
            )) {
        return HAFO_PERM_SIZE_OVERFLOW;
    }
    if (n_samples == 0u || embedding_span >= n_samples) {
        return HAFO_PERM_INVALID_SHAPE;
    }
    total_windows_size = n_samples - embedding_span;
    if ((uintmax_t)total_windows_size > (uintmax_t)UINT64_MAX) {
        return HAFO_PERM_COUNT_OVERFLOW;
    }
    total_windows = (uint64_t)total_windows_size;

    if (ranges_overlap(signal, signal_bytes, counts_out, counts_bytes)
            || ranges_overlap(
                signal,
                signal_bytes,
                total_windows_out,
                sizeof(uint64_t)
            )
            || ranges_overlap(
                signal,
                signal_bytes,
                valid_windows_out,
                sizeof(uint64_t)
            )
            || ranges_overlap(
                signal,
                signal_bytes,
                tied_windows_out,
                sizeof(uint64_t)
            )
            || ranges_overlap(
                counts_out,
                counts_bytes,
                total_windows_out,
                sizeof(uint64_t)
            )
            || ranges_overlap(
                counts_out,
                counts_bytes,
                valid_windows_out,
                sizeof(uint64_t)
            )
            || ranges_overlap(
                counts_out,
                counts_bytes,
                tied_windows_out,
                sizeof(uint64_t)
            )
            || ranges_overlap(
                total_windows_out,
                sizeof(uint64_t),
                valid_windows_out,
                sizeof(uint64_t)
            )
            || ranges_overlap(
                total_windows_out,
                sizeof(uint64_t),
                tied_windows_out,
                sizeof(uint64_t)
            )
            || ranges_overlap(
                valid_windows_out,
                sizeof(uint64_t),
                tied_windows_out,
                sizeof(uint64_t)
            )) {
        return HAFO_PERM_ALIASED_BUFFERS;
    }

    for (index = 0u; index < n_samples; ++index) {
        if (!isfinite(signal[index])) {
            return HAFO_PERM_NONFINITE_INPUT;
        }
    }
    for (index = 0u; index < counts_length; ++index) {
        counts_out[index] = UINT64_C(0);
    }
    *total_windows_out = total_windows;
    *valid_windows_out = UINT64_C(0);
    *tied_windows_out = UINT64_C(0);

    if (tie_policy == HAFO_PERM_RAISE) {
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static) reduction(+:tied_windows) \
            if(total_windows_size >= HAFO_PERM_OPENMP_MIN_WINDOWS)
        #endif
        for (index = 0u; index < total_windows_size; ++index) {
            if (window_has_tie(
                    signal,
                    index,
                    embedding_dimension,
                    delay
                )) {
                tied_windows += UINT64_C(1);
            }
        }
        if (tied_windows != UINT64_C(0)) {
            *valid_windows_out = total_windows - tied_windows;
            *tied_windows_out = tied_windows;
            return HAFO_PERM_TIED_WINDOW;
        }
    }

    tied_windows = UINT64_C(0);
    #if defined(_OPENMP)
    if (total_windows_size >= HAFO_PERM_OPENMP_MIN_WINDOWS) {
        const size_t thread_count = (size_t)omp_get_max_threads();
        size_t local_count_values;
        size_t local_count_bytes;
        uint64_t *local_counts = NULL;

        if (thread_count > 0u
                && checked_size_product(
                    thread_count,
                    counts_length,
                    &local_count_values
                )
                && checked_size_product(
                    local_count_values,
                    sizeof(uint64_t),
                    &local_count_bytes
                )
                && local_count_bytes <= HAFO_PERM_LOCAL_HISTOGRAM_MAX_BYTES) {
            local_counts = (uint64_t *)calloc(
                local_count_values,
                sizeof(uint64_t)
            );
        }
        if (local_counts != NULL) {
            #pragma omp parallel reduction(+:valid_windows,tied_windows)
            {
                const size_t thread_index = (size_t)omp_get_thread_num();
                uint64_t *thread_counts =
                    local_counts + thread_index * counts_length;
                size_t window_index;

                #pragma omp for schedule(static)
                for (window_index = 0u;
                        window_index < total_windows_size;
                        ++window_index) {
                    const int tied = window_has_tie(
                        signal,
                        window_index,
                        embedding_dimension,
                        delay
                    );
                    uint64_t rank;

                    if (tied) {
                        tied_windows += UINT64_C(1);
                        if (tie_policy == HAFO_PERM_OMIT) {
                            continue;
                        }
                    }
                    rank = ordinal_pattern_rank(
                        signal,
                        window_index,
                        embedding_dimension,
                        delay
                    );
                    if (rank < expected_counts) {
                        thread_counts[(size_t)rank] += UINT64_C(1);
                        valid_windows += UINT64_C(1);
                    }
                }
            }
            for (index = 0u; index < counts_length; ++index) {
                size_t thread_index;
                uint64_t combined = UINT64_C(0);
                for (thread_index = 0u;
                        thread_index < thread_count;
                        ++thread_index) {
                    const uint64_t contribution =
                        local_counts[thread_index * counts_length + index];
                    if (UINT64_MAX - combined < contribution) {
                        free(local_counts);
                        return HAFO_PERM_COUNT_OVERFLOW;
                    }
                    combined += contribution;
                }
                counts_out[index] = combined;
            }
            free(local_counts);
            if (valid_windows > total_windows || tied_windows > total_windows) {
                return HAFO_PERM_COUNT_OVERFLOW;
            }
            *valid_windows_out = valid_windows;
            *tied_windows_out = tied_windows;
            return HAFO_PERM_OK;
        }
    }
    #endif

    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static) \
        reduction(+:valid_windows,tied_windows) \
        if(total_windows_size >= HAFO_PERM_OPENMP_MIN_WINDOWS)
    #endif
    for (index = 0u; index < total_windows_size; ++index) {
        const int tied = window_has_tie(
            signal,
            index,
            embedding_dimension,
            delay
        );
        uint64_t rank;

        if (tied) {
            tied_windows += UINT64_C(1);
            if (tie_policy == HAFO_PERM_OMIT) {
                continue;
            }
        }
        rank = ordinal_pattern_rank(
            signal,
            index,
            embedding_dimension,
            delay
        );
        if (rank >= expected_counts) {
            /* Defensive invariant; unreachable for a valid permutation. */
            continue;
        }
        #if defined(_OPENMP)
        #pragma omp atomic update
        #endif
        counts_out[(size_t)rank] += UINT64_C(1);
        valid_windows += UINT64_C(1);
    }

    if (valid_windows > total_windows || tied_windows > total_windows) {
        return HAFO_PERM_COUNT_OVERFLOW;
    }
    *valid_windows_out = valid_windows;
    *tied_windows_out = tied_windows;
    return HAFO_PERM_OK;
}
