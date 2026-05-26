#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#if defined(_WIN32) || defined(__CYGWIN__)
  #define API_EXPORT __declspec(dllexport)
#else
  #define API_EXPORT __attribute__((visibility("default")))
#endif

// Signature for the Python callback
typedef void (*RhsCallback)(double t, const double *x, double *f);

static int ceil_steps(double t_final, double h) {
    if (!(h > 0.0) || t_final < 0.0) return -1;
    return (int)ceil(t_final / h);
}

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

static void memory_component_general(int k, double t_eval, const double *t, const double *arr, int dim, double q, double h, const EFORK3 *c, double *out_mem) {
    for (int d = 0; d < dim; ++d) {
        out_mem[d] = 0.0;
    }
    const double expo = 1.0 - q;
    for (int j = 0; j < k; ++j) {
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

API_EXPORT int integrate_general_efork_c(
    RhsCallback rhs,
    const double *x0,
    int dim,
    double q,
    double h,
    double t_final,
    double divergence_norm,
    double *traj_out
) {
    if (!rhs || !x0 || dim <= 0 || !(q > 0.0 && q < 1.0) || !(h > 0.0) || t_final < 0.0 || !traj_out) return -1;
    const int nsteps = ceil_steps(t_final, h);
    if (nsteps < 0) return -2;
    const int rows = nsteps + 1;

    double *t = (double*)calloc((size_t)rows, sizeof(double));
    double *x = (double*)calloc((size_t)rows * (size_t)dim, sizeof(double));
    double *k1 = (double*)malloc((size_t)dim * sizeof(double));
    double *k2 = (double*)malloc((size_t)dim * sizeof(double));
    double *k3 = (double*)malloc((size_t)dim * sizeof(double));
    double *tmp = (double*)malloc((size_t)dim * sizeof(double));
    double *f = (double*)malloc((size_t)dim * sizeof(double));
    double *mem = (double*)malloc((size_t)dim * sizeof(double));
    if (!t || !x || !k1 || !k2 || !k3 || !tmp || !f || !mem) {
        free(t); free(x); free(k1); free(k2); free(k3); free(tmp); free(f); free(mem);
        return -3;
    }

    t[0] = 0.0;
    for (int d = 0; d < dim; ++d) {
        x[d] = x0[d];
    }

    // Write initial state to trajectory output
    traj_out[0] = 0.0;
    for (int d = 0; d < dim; ++d) {
        traj_out[1 + d] = x0[d];
    }

    const double hq = pow(h, q);
    const EFORK3 c = efork3_coeffs(q, h);
    int last_idx = 0;
    int status = 0;

    for (int n = 0; n < nsteps; ++n) {
        // Stage 1
        double mem_x[100];
        if (dim <= 100) {
            if (n > 0) {
                memory_component_general(n, t[n], t, x, dim, q, h, &c, mem_x);
            } else {
                for (int d = 0; d < dim; ++d) mem_x[d] = 0.0;
            }
        }
        
        rhs(t[n], &x[dim * n], f);
        for (int d = 0; d < dim; ++d) {
            k1[d] = hq * (f[d] - mem_x[d]);
        }

        // Stage 2
        for (int d = 0; d < dim; ++d) {
            tmp[d] = x[dim * n + d] + c.a21 * k1[d];
        }
        const double t2 = t[n] + c.c2 * h;
        if (n > 0) {
            memory_component_general(n, t2, t, x, dim, q, h, &c, mem_x);
        } else {
            for (int d = 0; d < dim; ++d) mem_x[d] = 0.0;
        }
        rhs(t2, tmp, f);
        for (int d = 0; d < dim; ++d) {
            k2[d] = hq * (f[d] - mem_x[d]);
        }

        // Stage 3
        for (int d = 0; d < dim; ++d) {
            tmp[d] = x[dim * n + d] + c.a31 * k1[d] + c.a32 * k2[d];
        }
        const double t3 = t[n] + c.c3 * h;
        if (n > 0) {
            memory_component_general(n, t3, t, x, dim, q, h, &c, mem_x);
        } else {
            for (int d = 0; d < dim; ++d) mem_x[d] = 0.0;
        }
        rhs(t3, tmp, f);
        for (int d = 0; d < dim; ++d) {
            k3[d] = hq * (f[d] - mem_x[d]);
        }

        // Final prediction
        t[n + 1] = t[n] + h;
        double norm = 0.0;
        for (int d = 0; d < dim; ++d) {
            const double val = x[dim * n + d] + c.w1 * k1[d] + c.w2 * k2[d] + c.w3 * k3[d];
            x[dim * (n + 1) + d] = val;
            norm += val * val;
        }
        norm = sqrt(norm);

        // Save step to trajectory
        const int out_row = n + 1;
        traj_out[out_row * (dim + 1) + 0] = t[n + 1];
        for (int d = 0; d < dim; ++d) {
            traj_out[out_row * (dim + 1) + 1 + d] = x[dim * (n + 1) + d];
        }

        if (divergence_norm > 0.0 && norm > divergence_norm) {
            status = 1; // diverged
            last_idx = n + 1;
            break;
        }
        last_idx = n + 1;
    }

    free(t); free(x); free(k1); free(k2); free(k3); free(tmp); free(f); free(mem);
    return status == 1 ? (last_idx + 1) : rows;
}

API_EXPORT int integrate_general_abm_c(
    RhsCallback rhs,
    const double *x0,
    int dim,
    double q,
    double h,
    double t_final,
    double divergence_norm,
    double *traj_out
) {
    if (!rhs || !x0 || dim <= 0 || !(q > 0.0 && q < 1.0) || !(h > 0.0) || t_final < 0.0 || !traj_out) return -1;
    const int nsteps = ceil_steps(t_final, h);
    if (nsteps < 0) return -2;
    const int rows = nsteps + 1;

    double *state = (double*)calloc((size_t)rows * (size_t)dim, sizeof(double));
    double *fhist = (double*)calloc((size_t)rows * (size_t)dim, sizeof(double));
    double *pow_q = (double*)calloc((size_t)rows + 1u, sizeof(double));
    double *pow_q1 = (double*)calloc((size_t)rows + 1u, sizeof(double));
    double *predictor = (double*)malloc((size_t)dim * sizeof(double));
    double *fp = (double*)malloc((size_t)dim * sizeof(double));
    double *corrected = (double*)malloc((size_t)dim * sizeof(double));
    if (!state || !fhist || !pow_q || !pow_q1 || !predictor || !fp || !corrected) {
        free(state); free(fhist); free(pow_q); free(pow_q1); free(predictor); free(fp); free(corrected);
        return -3;
    }

    for (int i = 0; i <= rows; ++i) {
        pow_q[i] = pow((double)i, q);
        pow_q1[i] = pow((double)i, q + 1.0);
    }

    for (int d = 0; d < dim; ++d) {
        state[d] = x0[d];
    }
    rhs(0.0, state, fhist);

    traj_out[0] = 0.0;
    for (int d = 0; d < dim; ++d) {
        traj_out[1 + d] = x0[d];
    }

    const double hq = pow(h, q);
    const double pred_scale = hq / tgamma(q + 1.0);
    const double corr_scale = hq / tgamma(q + 2.0);
    int last_idx = 0;
    int status = 0;

    for (int i = 0; i < nsteps; ++i) {
        // Predictor step
        for (int d = 0; d < dim; ++d) {
            predictor[d] = state[d]; // anchor x0
        }
        for (int j = 0; j <= i; ++j) {
            const int r = i - j;
            const double weight = pow_q[r + 1] - pow_q[r];
            for (int d = 0; d < dim; ++d) {
                predictor[d] += pred_scale * weight * fhist[dim * j + d];
            }
        }

        const double t_next = (double)(i + 1) * h;
        rhs(t_next, predictor, fp);

        // Corrector step
        for (int d = 0; d < dim; ++d) {
            corrected[d] = state[d]; // anchor x0
        }

        if (i == 0) {
            for (int d = 0; d < dim; ++d) {
                corrected[d] += corr_scale * (q * fhist[d] + fp[d]);
            }
        } else {
            const double a0 = pow_q1[i] - ((double)i - q) * pow_q[i + 1];
            for (int d = 0; d < dim; ++d) {
                corrected[d] += corr_scale * a0 * fhist[d];
            }
            for (int j = 1; j <= i; ++j) {
                const int r = i - j + 1;
                const double weight = pow_q1[r + 1] + pow_q1[r - 1] - 2.0 * pow_q1[r];
                for (int d = 0; d < dim; ++d) {
                    corrected[d] += corr_scale * weight * fhist[dim * j + d];
                }
            }
            for (int d = 0; d < dim; ++d) {
                corrected[d] += corr_scale * fp[d];
            }
        }

        // Save step
        double norm = 0.0;
        for (int d = 0; d < dim; ++d) {
            state[dim * (i + 1) + d] = corrected[d];
            norm += corrected[d] * corrected[d];
        }
        norm = sqrt(norm);
        rhs(t_next, corrected, &fhist[dim * (i + 1)]);

        // Write trajectory
        const int out_row = i + 1;
        traj_out[out_row * (dim + 1) + 0] = t_next;
        for (int d = 0; d < dim; ++d) {
            traj_out[out_row * (dim + 1) + 1 + d] = corrected[d];
        }

        if (divergence_norm > 0.0 && norm > divergence_norm) {
            status = 1; // diverged
            last_idx = i + 1;
            break;
        }
        last_idx = i + 1;
    }

    free(state); free(fhist); free(pow_q); free(pow_q1); free(predictor); free(fp); free(corrected);
    return status == 1 ? (last_idx + 1) : rows;
}
