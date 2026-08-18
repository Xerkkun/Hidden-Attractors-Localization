#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include "fractional_integrators.h"
#include "native_validation.h"

API_EXPORT int fractional_integrators_abi_version(void) {
    return 3;
}

// Predefined RHS for Chua with saturation
API_EXPORT void chua_saturation_rhs_c(double t, const double *x, double *dx, int n, void *params) {
    (void)t;
    if (!x || !dx || !params || n < 3) return;
    ChuaSaturationParams *p = (ChuaSaturationParams *)params;
    double sigma = x[0];
    double sat_val = sigma;
    if (sat_val > 1.0) sat_val = 1.0;
    else if (sat_val < -1.0) sat_val = -1.0;
    double psi = (p->m0 - p->m1) * sat_val;
    
    dx[0] = -p->alpha * (p->m1 + 1.0) * x[0] + p->alpha * x[1] - p->alpha * psi;
    dx[1] = x[0] - x[1] + x[2];
    dx[2] = -p->beta * x[1] - p->gamma * x[2];
}

// Predefined RHS for Chua with arctan
API_EXPORT void chua_arctan_rhs_c(double t, const double *x, double *dx, int n, void *params) {
    (void)t;
    if (!x || !dx || !params || n < 3) return;
    ChuaArctanParams *p = (ChuaArctanParams *)params;
    double sigma = x[0];
    double phi = p->a1 * sigma + p->a2 * atan(p->rho * sigma);
    
    dx[0] = p->alpha * (x[1] - sigma - phi);
    dx[1] = x[0] - x[1] + x[2];
    dx[2] = -p->beta * x[1] - p->gamma * x[2];
}

// -----------------------------------------------------------------------------
// EFORK coefficients and memory component functions
// -----------------------------------------------------------------------------

typedef struct {
    double g1, g2, g3;
    double c2, c3;
    double a21, a31, a32;
    double w1, w2, w3;
    double inv_mem_factor;
} EFORK3;

static EFORK3 efork3_coeffs(double q, double h) {
    EFORK3 c;
    c.g1 = tgamma(1.0 + q);
    c.g2 = tgamma(1.0 + 2.0 * q);
    c.g3 = tgamma(1.0 + 3.0 * q);
    c.c2 = pow(1.0 / (2.0 * c.g1), 1.0 / q);
    c.c3 = pow(1.0 / (4.0 * c.g1), 1.0 / q);
    c.a21 = 1.0 / (2.0 * c.g1 * c.g1);
    c.a31 = ((c.g1 * c.g1) * c.g2 + 2.0 * (c.g2 * c.g2) - c.g3) /
            (4.0 * (c.g1 * c.g1) * (2.0 * (c.g2 * c.g2) - c.g3));
    c.a32 = -c.g2 / (4.0 * (2.0 * (c.g2 * c.g2) - c.g3));
    c.w1 = (8.0 * (c.g1 * c.g1 * c.g1) * (c.g2 * c.g2) -
            6.0 * (c.g1 * c.g1 * c.g1) * c.g3 + c.g2 * c.g3) /
           (c.g1 * c.g2 * c.g3);
    c.w2 = 2.0 * (c.g1 * c.g1) * (4.0 * (c.g2 * c.g2) - c.g3) / (c.g2 * c.g3);
    c.w3 = -8.0 * (c.g1 * c.g1) * (2.0 * (c.g2 * c.g2) - c.g3) / (c.g2 * c.g3);
    c.inv_mem_factor = 1.0 / (h * tgamma(2.0 - q));
    return c;
}

static void memory_component_precomputed(int k, const double *arr, int dim, const double *pow_expo, const EFORK3 *c, int memory_mode, int memory_window_length, double *out_mem) {
    for (int d = 0; d < dim; ++d) {
        out_mem[d] = 0.0;
    }
    int j_start = 0;
    if (memory_mode == 1) { // windowed
        j_start = k - memory_window_length;
        if (j_start < 0) j_start = 0;
    }
    for (int j = j_start; j < k; ++j) {
        int r = k - j;
        const double term = pow_expo[r] - pow_expo[r - 1];
        for (int d = 0; d < dim; ++d) {
            out_mem[d] += (arr[dim * (j + 1) + d] - arr[dim * j + d]) * term;
        }
    }
    for (int d = 0; d < dim; ++d) {
        out_mem[d] *= c->inv_mem_factor;
    }
}

// -----------------------------------------------------------------------------
// Unified General C Fractional Integrator
// -----------------------------------------------------------------------------

API_EXPORT int integrate_fractional_c(
    RhsCallback rhs,
    void *params,
    int dim,
    const double *x0,
    double q,
    double h,
    double t_final,
    int method,            // 0: ABM, 1: EFORK
    int memory_mode,       // 0: full, 1: window
    int memory_window_length,
    const double *history_times,
    const double *history_states,
    int history_len,
    size_t history_times_count,
    size_t history_states_count,
    double divergence_norm,
    double *out_times,
    double *out_states,
    size_t out_times_capacity,
    size_t out_states_capacity,
    int *out_steps,
    int *status_code,
    
    // Early stopping parameters
    int early_stop_enabled,
    int div_early_enabled,
    double div_early_norm,
    int div_consec_steps,
    double div_growth_factor,
    int eq_early_enabled,
    double eq_tol,
    double eq_deriv_tol,
    int eq_consec_steps,
    double eq_min_time,
    const double *equilibria_pts,
    int num_equilibria,
    size_t equilibria_values_count
) {
    size_t history_state_required = 0u;
    size_t equilibrium_values_required = 0u;
    size_t state_values_required = 0u;
    int nsteps = 0;

    if (!out_steps || !status_code) return -1;
    *out_steps = 0;
    *status_code = -1;

    // 1. Basic validation
    if (!rhs || !x0 || dim <= 0 || !(q > 0.0 && q < 1.0) ||
        !hafo_isfinite(q) || !hafo_isfinite(h) || !hafo_isfinite(t_final) ||
        !hafo_isfinite(divergence_norm) || !(divergence_norm > 0.0) ||
        (method != 0 && method != 1) ||
        (memory_mode != 0 && memory_mode != 1) ||
        (memory_mode == 0 && memory_window_length != 0) ||
        (memory_mode == 1 && memory_window_length < 1) ||
        history_len < 0 || num_equilibria < 0 ||
        (early_stop_enabled != 0 && early_stop_enabled != 1) ||
        (div_early_enabled != 0 && div_early_enabled != 1) ||
        (eq_early_enabled != 0 && eq_early_enabled != 1) ||
        !hafo_isfinite(div_early_norm) || !(div_early_norm > 0.0) ||
        div_consec_steps < 1 || !hafo_isfinite(div_growth_factor) ||
        !(div_growth_factor > 0.0) ||
        !hafo_isfinite(eq_tol) || !(eq_tol > 0.0) ||
        !hafo_isfinite(eq_deriv_tol) || !(eq_deriv_tol > 0.0) ||
        eq_consec_steps < 1 || !hafo_isfinite(eq_min_time) || eq_min_time < 0.0 ||
        !out_times || !out_states ||
        !hafo_uniform_step_count(t_final, h, &nsteps) ||
        !hafo_values_are_finite(x0, (size_t)dim)) {
        return -1;
    }

    if (history_len > 0) {
        if (!history_times || !history_states ||
            !hafo_checked_mul_size((size_t)history_len, (size_t)dim,
                                   &history_state_required) ||
            history_times_count < (size_t)history_len ||
            history_states_count < history_state_required ||
            !hafo_values_are_finite(history_times, (size_t)history_len) ||
            history_state_required > (size_t)INT_MAX ||
            !hafo_values_are_finite(history_states, history_state_required)) {
            return -1;
        }
        double time_scale = fmax(
            1.0, (double)(history_len > 1 ? history_len - 1 : 1) * fabs(h)
        );
        for (int index = 0; index < history_len; ++index) {
            time_scale = fmax(time_scale, fabs(history_times[index]));
        }
        const double time_tolerance = 64.0 * DBL_EPSILON * time_scale;
        if (fabs(history_times[history_len - 1]) > time_tolerance) return -1;
        for (int index = 1; index < history_len; ++index) {
            const double increment = history_times[index] - history_times[index - 1];
            if (!(increment > 0.0)) return -1;
        }
        for (int index = 0; index < history_len; ++index) {
            const double expected = (double)(index - (history_len - 1)) * h;
            if (fabs(history_times[index] - expected) > time_tolerance) return -1;
        }
        for (int d = 0; d < dim; ++d) {
            const double expected = x0[d];
            const double actual = history_states[(history_len - 1) * dim + d];
            const double tolerance = 64.0 * DBL_EPSILON *
                                     fmax(1.0, fmax(fabs(expected), fabs(actual)));
            if (fabs(expected - actual) > tolerance) return -1;
        }
    }

    if (num_equilibria > 0) {
        if (!equilibria_pts ||
            !hafo_checked_mul_size((size_t)num_equilibria, (size_t)dim,
                                   &equilibrium_values_required) ||
            equilibria_values_count < equilibrium_values_required ||
            equilibrium_values_required > (size_t)INT_MAX ||
            !hafo_values_are_finite(equilibria_pts, equilibrium_values_required)) {
            return -1;
        }
    }

    const int H = (history_len > 0) ? history_len : 1;
    if (nsteps > INT_MAX - H - 2) return -1;
    const int total_capacity = H + nsteps;
    if (!hafo_checked_mul_size((size_t)total_capacity, (size_t)dim,
                               &state_values_required) ||
        state_values_required > (size_t)INT_MAX) {
        return -1;
    }
    if (out_times_capacity < (size_t)total_capacity ||
        out_states_capacity < state_values_required) {
        return -4;
    }

    // 2. Allocate integration workspace
    double *t = (double *)calloc((size_t)total_capacity, sizeof(double));
    double *x = (double *)calloc(state_values_required, sizeof(double));
    if (!t || !x) {
        free(t); free(x);
        return -2;
    }

    // 3. Initialize with prehistory or standard x0
    if (history_len > 0) {
        for (int i = 0; i < H; ++i) {
            t[i] = history_times[i];
            for (int d = 0; d < dim; ++d) {
                x[i * dim + d] = history_states[i * dim + d];
            }
        }
        t[H - 1] = 0.0;
        for (int d = 0; d < dim; ++d) x[(H - 1) * dim + d] = x0[d];
    } else {
        t[0] = 0.0;
        for (int d = 0; d < dim; ++d) {
            x[d] = x0[d];
        }
    }

    *status_code = 0; // default ok
    int last_idx = H - 1;

    // Early Stop consecutive counter setups
    int div_consec_count = 0;
    int growth_consec_count = 0;
    double prev_norm = -1.0;
    
    int *eq_consec_counts = NULL;
    if (early_stop_enabled && eq_early_enabled && num_equilibria > 0 && equilibria_pts) {
        eq_consec_counts = (int *)calloc((size_t)num_equilibria, sizeof(int));
        if (!eq_consec_counts) {
            free(t);
            free(x);
            return -2;
        }
    }

    // -------------------------------------------------------------------------
    // Method 0: Adams-Bashforth-Moulton (ABM)
    // -------------------------------------------------------------------------
    if (method == 0) {
        double *fhist = (double *)calloc(state_values_required, sizeof(double));
        double *pow_q = (double *)malloc((size_t)(total_capacity + 2) * sizeof(double));
        double *pow_q1 = (double *)malloc((size_t)(total_capacity + 2) * sizeof(double));
        double *predictor = (double *)malloc((size_t)dim * sizeof(double));
        double *fp = (double *)malloc((size_t)dim * sizeof(double));
        double *corrected = (double *)malloc((size_t)dim * sizeof(double));

        if (!fhist || !pow_q || !pow_q1 || !predictor || !fp || !corrected) {
            free(t); free(x); free(fhist); free(pow_q); free(pow_q1); free(predictor); free(fp); free(corrected);
            if (eq_consec_counts) free(eq_consec_counts);
            return -3;
        }

        // Precompute q-powers
        for (int idx = 0; idx < total_capacity + 2; ++idx) {
            pow_q[idx] = pow((double)idx, q);
            pow_q1[idx] = pow((double)idx, q + 1.0);
        }

        // Evaluate historical derivatives
        for (int i = 0; i < H; ++i) {
            rhs(t[i], &x[i * dim], &fhist[i * dim], dim, params);
        }

        const double hq = pow(h, q);
        const double pred_scale = hq / tgamma(q + 1.0);
        const double corr_scale = hq / tgamma(q + 2.0);

        for (int i = H - 1; i < total_capacity - 1; ++i) {
            int s = 0;
            if (memory_mode == 1) { // windowed
                s = i - memory_window_length + 1;
                if (s < 0) s = 0;
            }
            int n_prime = i - s;

            // A. Predictor Step
            for (int d = 0; d < dim; ++d) {
                predictor[d] = x[s * dim + d];
            }
            for (int j = s; j <= i; ++j) {
                int r = i - j;
                double weight = pow_q[r + 1] - pow_q[r];
                for (int d = 0; d < dim; ++d) {
                    predictor[d] += pred_scale * weight * fhist[j * dim + d];
                }
            }

            const int local_step = i - (H - 1) + 1;
            double t_next = (local_step == nsteps) ? t_final : t[i] + h;
            rhs(t_next, predictor, fp, dim, params);

            // B. Corrector Step
            for (int d = 0; d < dim; ++d) {
                corrected[d] = x[s * dim + d];
            }

            if (n_prime == 0) {
                double a0 = q;
                for (int d = 0; d < dim; ++d) {
                    corrected[d] += corr_scale * (a0 * fhist[s * dim + d] + fp[d]);
                }
            } else {
                double a0 = pow_q1[n_prime] - ((double)n_prime - q) * pow_q[n_prime + 1];
                for (int d = 0; d < dim; ++d) {
                    corrected[d] += corr_scale * a0 * fhist[s * dim + d];
                }
                for (int j = s + 1; j <= i; ++j) {
                    int r = i - j;
                    double weight = pow_q1[r + 2] + pow_q1[r] - 2.0 * pow_q1[r + 1];
                    for (int d = 0; d < dim; ++d) {
                        corrected[d] += corr_scale * weight * fhist[j * dim + d];
                    }
                }
                for (int d = 0; d < dim; ++d) {
                    corrected[d] += corr_scale * fp[d];
                }
            }

            // C. Divergence Check & States Storage
            double norm = 0.0;
            for (int d = 0; d < dim; ++d) {
                norm += corrected[d] * corrected[d];
            }
            norm = sqrt(norm);

            t[i + 1] = t_next;
            for (int d = 0; d < dim; ++d) {
                x[(i + 1) * dim + d] = corrected[d];
            }
            rhs(t_next, corrected, &fhist[(i + 1) * dim], dim, params);

            last_idx = i + 1;

            // DIVERGENCIA EARLY STOP
            if (early_stop_enabled && div_early_enabled) {
                if (norm > div_early_norm) {
                    div_consec_count++;
                } else {
                    div_consec_count = 0;
                }
                if (prev_norm >= 0.0) {
                    if (norm > div_growth_factor * prev_norm) {
                        growth_consec_count++;
                    } else {
                        growth_consec_count = 0;
                    }
                }
                prev_norm = norm;
                if (div_consec_count >= div_consec_steps || growth_consec_count >= div_consec_steps) {
                    *status_code = 3; // diverged_early
                    break;
                }
            } else {
                prev_norm = norm;
            }

            // EQUILIBRIUM CONVERGENCE EARLY STOP
            if (early_stop_enabled && eq_early_enabled && eq_consec_counts && t_next >= eq_min_time) {
                int converged_idx = -1;
                double deriv_norm = 0.0;
                for (int d = 0; d < dim; ++d) {
                    const double derivative = fhist[(i + 1) * dim + d];
                    deriv_norm += derivative * derivative;
                }
                deriv_norm = sqrt(deriv_norm);
                for (int k = 0; k < num_equilibria; ++k) {
                    double diff_norm = 0.0;
                    for (int d = 0; d < dim; ++d) {
                        double diff = corrected[d] - equilibria_pts[k * dim + d];
                        diff_norm += diff * diff;
                    }
                    diff_norm = sqrt(diff_norm);

                    if (diff_norm < eq_tol && deriv_norm < eq_deriv_tol) {
                        eq_consec_counts[k]++;
                    } else {
                        eq_consec_counts[k] = 0;
                    }

                    if (eq_consec_counts[k] >= eq_consec_steps) {
                        converged_idx = k;
                        break;
                    }
                }
                if (converged_idx != -1) {
                    *status_code = 4; // converged_equilibrium_early
                    break;
                }
            }

            // Standard abort checks
            if (divergence_norm > 0.0 && norm > divergence_norm) {
                *status_code = 1; // diverged
                break;
            }
            if (!hafo_isfinite(norm)) {
                *status_code = 2; // nonfinite
                break;
            }
        }

        free(fhist); free(pow_q); free(pow_q1); free(predictor); free(fp); free(corrected);
    }
    // -------------------------------------------------------------------------
    // Method 1: Enhanced Fractional Order Runge-Kutta (EFORK)
    // -------------------------------------------------------------------------
    else if (method == 1) {
        double *k1 = (double *)malloc((size_t)dim * sizeof(double));
        double *k2 = (double *)malloc((size_t)dim * sizeof(double));
        double *k3 = (double *)malloc((size_t)dim * sizeof(double));
        double *tmp = (double *)malloc((size_t)dim * sizeof(double));
        double *f = (double *)malloc((size_t)dim * sizeof(double));
        double *mem_x = (double *)malloc((size_t)dim * sizeof(double));
        double *pow_expo_s1 = (double *)malloc((size_t)(total_capacity + 2) * sizeof(double));
        double *pow_expo_s2 = (double *)malloc((size_t)(total_capacity + 2) * sizeof(double));
        double *pow_expo_s3 = (double *)malloc((size_t)(total_capacity + 2) * sizeof(double));

        if (!k1 || !k2 || !k3 || !tmp || !f || !mem_x || !pow_expo_s1 || !pow_expo_s2 || !pow_expo_s3) {
            free(t); free(x); free(k1); free(k2); free(k3); free(tmp); free(f); free(mem_x);
            if (pow_expo_s1) free(pow_expo_s1);
            if (pow_expo_s2) free(pow_expo_s2);
            if (pow_expo_s3) free(pow_expo_s3);
            if (eq_consec_counts) free(eq_consec_counts);
            return -3;
        }

        const double hq = pow(h, q);
        const EFORK3 coeffs = efork3_coeffs(q, h);

        const double expo = 1.0 - q;
        for (int idx = 0; idx < total_capacity + 2; ++idx) {
            pow_expo_s1[idx] = pow((double)idx * h, expo);
            pow_expo_s2[idx] = pow(((double)idx + coeffs.c2) * h, expo);
            pow_expo_s3[idx] = pow(((double)idx + coeffs.c3) * h, expo);
        }

        for (int i = H - 1; i < total_capacity - 1; ++i) {
            // Stage 1
            memory_component_precomputed(i, x, dim, pow_expo_s1, &coeffs, memory_mode, memory_window_length, mem_x);
            rhs(t[i], &x[i * dim], f, dim, params);
            for (int d = 0; d < dim; ++d) {
                k1[d] = hq * (f[d] - mem_x[d]);
            }

            // Stage 2
            for (int d = 0; d < dim; ++d) {
                tmp[d] = x[i * dim + d] + coeffs.a21 * k1[d];
            }
            double t2 = t[i] + coeffs.c2 * h;
            memory_component_precomputed(i, x, dim, pow_expo_s2, &coeffs, memory_mode, memory_window_length, mem_x);
            rhs(t2, tmp, f, dim, params);
            for (int d = 0; d < dim; ++d) {
                k2[d] = hq * (f[d] - mem_x[d]);
            }

            // Stage 3
            for (int d = 0; d < dim; ++d) {
                tmp[d] = x[i * dim + d] + coeffs.a31 * k1[d] + coeffs.a32 * k2[d];
            }
            double t3 = t[i] + coeffs.c3 * h;
            memory_component_precomputed(i, x, dim, pow_expo_s3, &coeffs, memory_mode, memory_window_length, mem_x);
            rhs(t3, tmp, f, dim, params);
            for (int d = 0; d < dim; ++d) {
                k3[d] = hq * (f[d] - mem_x[d]);
            }

            // Prediction
            const int local_step = i - (H - 1) + 1;
            t[i + 1] = (local_step == nsteps) ? t_final : (double)local_step * h;
            double norm = 0.0;
            for (int d = 0; d < dim; ++d) {
                double val = x[i * dim + d] + coeffs.w1 * k1[d] + coeffs.w2 * k2[d] + coeffs.w3 * k3[d];
                x[(i + 1) * dim + d] = val;
                norm += val * val;
            }
            norm = sqrt(norm);

            last_idx = i + 1;

            double *state_ptr = &x[(i + 1) * dim];

            // DIVERGENCIA EARLY STOP
            if (early_stop_enabled && div_early_enabled) {
                if (norm > div_early_norm) {
                    div_consec_count++;
                } else {
                    div_consec_count = 0;
                }
                if (prev_norm >= 0.0) {
                    if (norm > div_growth_factor * prev_norm) {
                        growth_consec_count++;
                    } else {
                        growth_consec_count = 0;
                    }
                }
                prev_norm = norm;
                if (div_consec_count >= div_consec_steps || growth_consec_count >= div_consec_steps) {
                    *status_code = 3; // diverged_early
                    break;
                }
            } else {
                prev_norm = norm;
            }

            // EQUILIBRIUM CONVERGENCE EARLY STOP
            if (early_stop_enabled && eq_early_enabled && eq_consec_counts && t[i + 1] >= eq_min_time) {
                int converged_idx = -1;
                double deriv_norm = 0.0;
                rhs(t[i + 1], state_ptr, f, dim, params);
                for (int d = 0; d < dim; ++d) {
                    deriv_norm += f[d] * f[d];
                }
                deriv_norm = sqrt(deriv_norm);
                for (int k = 0; k < num_equilibria; ++k) {
                    double diff_norm = 0.0;
                    for (int d = 0; d < dim; ++d) {
                        double diff = state_ptr[d] - equilibria_pts[k * dim + d];
                        diff_norm += diff * diff;
                    }
                    diff_norm = sqrt(diff_norm);

                    if (diff_norm < eq_tol && deriv_norm < eq_deriv_tol) {
                        eq_consec_counts[k]++;
                    } else {
                        eq_consec_counts[k] = 0;
                    }

                    if (eq_consec_counts[k] >= eq_consec_steps) {
                        converged_idx = k;
                        break;
                    }
                }
                if (converged_idx != -1) {
                    *status_code = 4; // converged_equilibrium_early
                    break;
                }
            }

            // Standard abort checks
            if (divergence_norm > 0.0 && norm > divergence_norm) {
                *status_code = 1; // diverged
                break;
            }
            if (!hafo_isfinite(norm)) {
                *status_code = 2; // nonfinite
                break;
            }
        }

        free(k1); free(k2); free(k3); free(tmp); free(f); free(mem_x);
        free(pow_expo_s1); free(pow_expo_s2); free(pow_expo_s3);
    }

    // 4. Copy results back to Python pre-allocated buffers
    *out_steps = last_idx + 1;
    for (int i = 0; i <= last_idx; ++i) {
        out_times[i] = t[i];
        for (int d = 0; d < dim; ++d) {
            out_states[i * dim + d] = x[i * dim + d];
        }
    }

    if (eq_consec_counts) free(eq_consec_counts);
    free(t); free(x);
    return 0;
}

// -----------------------------------------------------------------------------
// Tempered-Caputo ABM in physical coordinates
// -----------------------------------------------------------------------------

static double robust_vector_norm(const double *values, int dim) {
    double norm = 0.0;
    for (int d = 0; d < dim; ++d) {
        norm = hypot(norm, values[d]);
    }
    return norm;
}

static int vector_is_finite(const double *values, int dim) {
    for (int d = 0; d < dim; ++d) {
        if (!hafo_isfinite(values[d])) return 0;
    }
    return 1;
}

API_EXPORT int integrate_tempered_caputo_abm_c(
    RhsCallback rhs,
    void *params,
    int dim,
    const double *x0,
    double q,
    double tempering,
    double h,
    int n_steps,
    int memory_mode,
    int memory_window_length,
    double divergence_norm,
    double *out_times,
    double *out_states,
    size_t out_times_capacity,
    size_t out_states_capacity,
    int *out_samples,
    int *status_code
) {
    if (!rhs || !x0 || dim <= 0 || !(q > 0.0 && q < 1.0) ||
        !hafo_isfinite(q) || !hafo_isfinite(tempering) || tempering < 0.0 ||
        !(h > 0.0) || !hafo_isfinite(h) || n_steps < 0 ||
        n_steps > INT_MAX - 3 ||
        (memory_mode != 0 && memory_mode != 1) ||
        (memory_mode == 1 && memory_window_length < 2) ||
        !(divergence_norm > 0.0) || !hafo_isfinite(divergence_norm) ||
        !hafo_values_are_finite(x0, (size_t)dim) ||
        !out_times || !out_states || !out_samples || !status_code) {
        return -1;
    }

    const int total_samples = n_steps + 1;
    const size_t total_samples_size = (size_t)total_samples;
    const size_t power_values = total_samples_size + 2u;
    size_t state_values = 0u;
    size_t times_bytes = 0u;
    size_t states_bytes = 0u;
    size_t powers_bytes = 0u;
    size_t vector_bytes = 0u;
    if (!hafo_checked_mul_size(total_samples_size, (size_t)dim, &state_values) ||
        state_values > (size_t)INT_MAX ||
        !hafo_checked_mul_size(total_samples_size, sizeof(double), &times_bytes) ||
        !hafo_checked_mul_size(state_values, sizeof(double), &states_bytes) ||
        !hafo_checked_mul_size(power_values, sizeof(double), &powers_bytes) ||
        !hafo_checked_mul_size((size_t)dim, sizeof(double), &vector_bytes)) {
        return -1;
    }
    if (out_times_capacity < total_samples_size ||
        out_states_capacity < state_values) {
        return -3;
    }
    double *times = (double *)calloc(1u, times_bytes);
    double *x = (double *)calloc(1u, states_bytes);
    double *fhist = (double *)calloc(1u, states_bytes);
    double *pow_q = (double *)malloc(powers_bytes);
    double *pow_q1 = (double *)malloc(powers_bytes);
    double *predictor = (double *)malloc(vector_bytes);
    double *fp = (double *)malloc(vector_bytes);
    double *corrected = (double *)malloc(vector_bytes);

    if (!times || !x || !fhist || !pow_q || !pow_q1 || !predictor || !fp || !corrected) {
        free(times); free(x); free(fhist); free(pow_q); free(pow_q1);
        free(predictor); free(fp); free(corrected);
        return -2;
    }

    for (int d = 0; d < dim; ++d) x[d] = x0[d];
    for (int idx = 0; idx < total_samples + 2; ++idx) {
        pow_q[idx] = pow((double)idx, q);
        pow_q1[idx] = pow((double)idx, q + 1.0);
    }

    *status_code = 0;
    int last_idx = 0;
    double initial_norm = robust_vector_norm(x, dim);
    if (!vector_is_finite(x, dim) || !hafo_isfinite(initial_norm)) {
        *status_code = 2;
        goto tempered_copy_results;
    }
    if (divergence_norm > 0.0 && initial_norm > divergence_norm) {
        *status_code = 1;
        goto tempered_copy_results;
    }

    rhs(0.0, x, fhist, dim, params);
    if (!vector_is_finite(fhist, dim)) {
        *status_code = 2;
        goto tempered_copy_results;
    }

    const double hq = pow(h, q);
    const double pred_scale = hq / tgamma(q + 1.0);
    const double corr_scale = hq / tgamma(q + 2.0);

    for (int i = 0; i < n_steps; ++i) {
        int s = 0;
        if (memory_mode == 1) {
            s = i - memory_window_length + 1;
            if (s < 0) s = 0;
        }
        const int n_prime = i - s;
        const double t_next = times[i] + h;
        const double anchor_damping = exp(
            -tempering * (((double)i + 1.0) - (double)s) * h
        );

        for (int d = 0; d < dim; ++d) {
            predictor[d] = anchor_damping * x[s * dim + d];
        }
        for (int j = s; j <= i; ++j) {
            const int r = i - j;
            const double weight = pow_q[r + 1] - pow_q[r];
            const double damping = exp(
                -tempering * (((double)i + 1.0) - (double)j) * h
            );
            for (int d = 0; d < dim; ++d) {
                predictor[d] += pred_scale * weight * damping * fhist[j * dim + d];
            }
        }
        if (!vector_is_finite(predictor, dim)) {
            *status_code = 2;
            break;
        }

        rhs(t_next, predictor, fp, dim, params);
        if (!vector_is_finite(fp, dim)) {
            *status_code = 2;
            break;
        }

        for (int d = 0; d < dim; ++d) {
            corrected[d] = anchor_damping * x[s * dim + d];
        }
        const double a0 = pow_q1[n_prime]
            - ((double)n_prime - q) * pow_q[n_prime + 1];
        const double first_damping = exp(
            -tempering * (((double)i + 1.0) - (double)s) * h
        );
        for (int d = 0; d < dim; ++d) {
            corrected[d] += corr_scale * a0 * first_damping * fhist[s * dim + d];
        }
        for (int j = s + 1; j <= i; ++j) {
            const int r = i - j;
            const double weight = pow_q1[r + 2] + pow_q1[r]
                - 2.0 * pow_q1[r + 1];
            const double damping = exp(
                -tempering * (((double)i + 1.0) - (double)j) * h
            );
            for (int d = 0; d < dim; ++d) {
                corrected[d] += corr_scale * weight * damping * fhist[j * dim + d];
            }
        }
        for (int d = 0; d < dim; ++d) {
            corrected[d] += corr_scale * fp[d];
            x[(i + 1) * dim + d] = corrected[d];
        }
        times[i + 1] = t_next;
        last_idx = i + 1;

        if (!vector_is_finite(corrected, dim)) {
            *status_code = 2;
            break;
        }
        const double norm = robust_vector_norm(corrected, dim);
        if (!hafo_isfinite(norm)) {
            *status_code = 2;
            break;
        }
        if (divergence_norm > 0.0 && norm > divergence_norm) {
            *status_code = 1;
            break;
        }

        rhs(t_next, corrected, &fhist[(i + 1) * dim], dim, params);
        if (!vector_is_finite(&fhist[(i + 1) * dim], dim)) {
            *status_code = 2;
            break;
        }
    }

tempered_copy_results:
    *out_samples = last_idx + 1;
    for (int i = 0; i <= last_idx; ++i) {
        out_times[i] = times[i];
        for (int d = 0; d < dim; ++d) {
            out_states[i * dim + d] = x[i * dim + d];
        }
    }

    free(times); free(x); free(fhist); free(pow_q); free(pow_q1);
    free(predictor); free(fp); free(corrected);
    return 0;
}
