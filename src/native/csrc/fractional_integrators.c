#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include "fractional_integrators.h"

// Predefined RHS for Chua with saturation
void chua_saturation_rhs_c(double t, const double *x, double *dx, int n, void *params) {
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
void chua_arctan_rhs_c(double t, const double *x, double *dx, int n, void *params) {
    ChuaArctanParams *p = (ChuaArctanParams *)params;
    double sigma = x[0];
    double psi = (p->n - p->m) * atan(sigma);
    
    dx[0] = -p->alpha * (1.0 + p->m) * x[0] + p->alpha * x[1] - p->alpha * psi;
    dx[1] = x[0] - x[1] + x[2];
    dx[2] = -p->beta * x[1] - p->gamma * x[2];
}

API_EXPORT void *get_chua_saturation_rhs(void) {
    return (void *)chua_saturation_rhs_c;
}

API_EXPORT void *get_chua_arctan_rhs(void) {
    return (void *)chua_arctan_rhs_c;
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

static void memory_component_general(int k, double t_eval, const double *t, const double *arr, int dim, double q, double h, const EFORK3 *c, int memory_mode, int memory_window_length, double *out_mem) {
    for (int d = 0; d < dim; ++d) {
        out_mem[d] = 0.0;
    }
    const double expo = 1.0 - q;
    int j_start = 0;
    if (memory_mode == 1) { // windowed
        j_start = k - memory_window_length;
        if (j_start < 0) j_start = 0;
    }
    for (int j = j_start; j < k; ++j) {
        const double v1 = pow(t_eval - t[j], expo);
        const double v2 = pow(t_eval - t[j + 1], expo);
        const double term = v1 - v2;
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
    double divergence_norm,
    double *out_times,
    double *out_states,
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
    int num_equilibria
) {
    // 1. Basic validation
    if (!rhs || !x0 || dim <= 0 || !(q > 0.0 && q <= 1.0) || !(h > 0.0) || t_final < 0.0 || !out_times || !out_states || !out_steps || !status_code) {
        return -1;
    }

    // EFORK is only valid for 0 < q < 1
    if (method == 1 && q >= 1.0) {
        *status_code = -1;
        return -1;
    }

    int H = (history_len > 0) ? history_len : 1;
    int nsteps = (int)ceil(t_final / h);
    if (nsteps < 0) nsteps = 0;
    int total_capacity = H + nsteps;

    // 2. Allocate integration workspace
    double *t = (double *)calloc((size_t)total_capacity, sizeof(double));
    double *x = (double *)calloc((size_t)total_capacity * (size_t)dim, sizeof(double));
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
    }

    // -------------------------------------------------------------------------
    // Method 0: Adams-Bashforth-Moulton (ABM)
    // -------------------------------------------------------------------------
    if (method == 0) {
        double *fhist = (double *)calloc((size_t)total_capacity * (size_t)dim, sizeof(double));
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

            double t_next = t[i] + h;
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
                for (int k = 0; k < num_equilibria; ++k) {
                    double diff_norm = 0.0;
                    for (int d = 0; d < dim; ++d) {
                        double diff = corrected[d] - equilibria_pts[k * dim + d];
                        diff_norm += diff * diff;
                    }
                    diff_norm = sqrt(diff_norm);

                    // Compute rhs vector norm at the corrected state to test derivative
                    double deriv_norm = 0.0;
                    double *dx_tmp = (double *)malloc((size_t)dim * sizeof(double));
                    if (dx_tmp) {
                        rhs(t_next, corrected, dx_tmp, dim, params);
                        for (int d = 0; d < dim; ++d) {
                            deriv_norm += dx_tmp[d] * dx_tmp[d];
                        }
                        deriv_norm = sqrt(deriv_norm);
                        free(dx_tmp);
                    } else {
                        deriv_norm = 9999.0;
                    }

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
            if (!isfinite(norm)) {
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

        if (!k1 || !k2 || !k3 || !tmp || !f || !mem_x) {
            free(t); free(x); free(k1); free(k2); free(k3); free(tmp); free(f); free(mem_x);
            if (eq_consec_counts) free(eq_consec_counts);
            return -3;
        }

        const double hq = pow(h, q);
        const EFORK3 coeffs = efork3_coeffs(q, h);

        for (int i = H - 1; i < total_capacity - 1; ++i) {
            // Stage 1
            memory_component_general(i, t[i], t, x, dim, q, h, &coeffs, memory_mode, memory_window_length, mem_x);
            rhs(t[i], &x[i * dim], f, dim, params);
            for (int d = 0; d < dim; ++d) {
                k1[d] = hq * (f[d] - mem_x[d]);
            }

            // Stage 2
            for (int d = 0; d < dim; ++d) {
                tmp[d] = x[i * dim + d] + coeffs.a21 * k1[d];
            }
            double t2 = t[i] + coeffs.c2 * h;
            memory_component_general(i, t2, t, x, dim, q, h, &coeffs, memory_mode, memory_window_length, mem_x);
            rhs(t2, tmp, f, dim, params);
            for (int d = 0; d < dim; ++d) {
                k2[d] = hq * (f[d] - mem_x[d]);
            }

            // Stage 3
            for (int d = 0; d < dim; ++d) {
                tmp[d] = x[i * dim + d] + coeffs.a31 * k1[d] + coeffs.a32 * k2[d];
            }
            double t3 = t[i] + coeffs.c3 * h;
            memory_component_general(i, t3, t, x, dim, q, h, &coeffs, memory_mode, memory_window_length, mem_x);
            rhs(t3, tmp, f, dim, params);
            for (int d = 0; d < dim; ++d) {
                k3[d] = hq * (f[d] - mem_x[d]);
            }

            // Prediction
            t[i + 1] = t[i] + h;
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
                for (int k = 0; k < num_equilibria; ++k) {
                    double diff_norm = 0.0;
                    for (int d = 0; d < dim; ++d) {
                        double diff = state_ptr[d] - equilibria_pts[k * dim + d];
                        diff_norm += diff * diff;
                    }
                    diff_norm = sqrt(diff_norm);

                    // Compute rhs vector norm at the predicted state to test derivative
                    double deriv_norm = 0.0;
                    double *dx_tmp = (double *)malloc((size_t)dim * sizeof(double));
                    if (dx_tmp) {
                        rhs(t[i + 1], state_ptr, dx_tmp, dim, params);
                        for (int d = 0; d < dim; ++d) {
                            deriv_norm += dx_tmp[d] * dx_tmp[d];
                        }
                        deriv_norm = sqrt(deriv_norm);
                        free(dx_tmp);
                    } else {
                        deriv_norm = 9999.0;
                    }

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
            if (!isfinite(norm)) {
                *status_code = 2; // nonfinite
                break;
            }
        }

        free(k1); free(k2); free(k3); free(tmp); free(f); free(mem_x);
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
