#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #define API_EXPORT __declspec(dllexport)
#else
  #define API_EXPORT __attribute__((visibility("default")))
#endif

typedef struct {
    double alpha_chua;
    double beta;
    double gamma_chua;
    double m0;
    double m1;
    double a1;
    double a2;
    double rho;
    int model; /* 0=piecewise, 1=arctan */
} ChuaParams;

typedef struct {
    double g1, g2, g3;
    double w1, w2, w3;
    double a21, a31, a32;
    double inv_mem_factor;
} EFORK3;

static ChuaParams G_PARAMS = {8.4562, 12.0732, 0.0052, -0.1768, -1.1468, 0.4, -1.5585, 1.0, 0};
static int G_WORKERS = 0;

static int ceil_steps(double t_final, double h) {
    if (!(h > 0.0) || t_final < 0.0) return -1;
    return (int)ceil(t_final / h);
}

static double sat_scalar(double x) {
    if (x < -1.0) return -1.0;
    if (x > 1.0) return 1.0;
    return x;
}

static double psi_value(double x, const ChuaParams *p) {
    if (p->model == 1) return p->a2 * atan(p->rho * x);
    return (p->m0 - p->m1) * sat_scalar(x);
}

static double base_slope(const ChuaParams *p) {
    return (p->model == 1) ? p->a1 : p->m1;
}

static void rhs_epsilon(const double x[3], const ChuaParams *p, double k, double eps, double f[3]) {
    const double bs = base_slope(p);
    const double sigma = x[0];
    const double psi = psi_value(sigma, p);
    const double p0x = -p->alpha_chua * (1.0 + bs + k) * x[0] + p->alpha_chua * x[1];
    f[0] = p0x - eps * p->alpha_chua * (psi - k * sigma);
    f[1] = x[0] - x[1] + x[2];
    f[2] = -p->beta * x[1] - p->gamma_chua * x[2];
}

static EFORK3 efork3_coeffs(double q, double h) {
    EFORK3 c;
    c.g1 = tgamma(1.0 + q);
    c.g2 = tgamma(1.0 + 2.0 * q);
    c.g3 = tgamma(1.0 + 3.0 * q);
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

static double memory_component(int k, const double *t, const double *arr, double q, double h, int nu, const EFORK3 *c) {
    int start = k - nu;
    if (start < 0) start = 0;
    double sum = 0.0;
    const double tn = t[k];
    const double expo = 1.0 - q;
    for (int j = start; j < k; ++j) {
        const double v1 = pow(tn - t[j], expo);
        const double v2 = pow(tn - t[j + 1], expo);
        sum += (arr[j + 1] - arr[j]) * (v1 - v2);
    }
    (void)h;
    return sum * c->inv_mem_factor;
}

static int integrate_internal(
    const ChuaParams *p,
    const double x0[3],
    double q,
    double h,
    double Lm,
    double t_final,
    double k,
    double eps,
    const double *history,
    int history_count,
    double **full_out,
    int *full_rows_out,
    int *segment_start_out
) {
    if (!(q > 0.0 && q <= 1.0) || !(h > 0.0) || !(Lm > 0.0) || t_final < 0.0) return -1;
    const int nsteps = ceil_steps(t_final, h);
    if (nsteps < 0) return -2;
    const int hist_count = (history && history_count > 0) ? history_count : 1;
    const int start_idx = hist_count - 1;
    const int total_rows = hist_count + nsteps;
    const int nu = (int)ceil(Lm / h) > 1 ? (int)ceil(Lm / h) : 1;
    const double hq = pow(h, q);
    const EFORK3 c = efork3_coeffs(q, h);

    double *t = (double*)calloc((size_t)total_rows, sizeof(double));
    double *x = (double*)calloc((size_t)total_rows, sizeof(double));
    double *y = (double*)calloc((size_t)total_rows, sizeof(double));
    double *z = (double*)calloc((size_t)total_rows, sizeof(double));
    double *full = (double*)malloc((size_t)total_rows * 4u * sizeof(double));
    if (!t || !x || !y || !z || !full) {
        free(t); free(x); free(y); free(z); free(full);
        return -3;
    }

    if (history && history_count > 0) {
        for (int i = 0; i < hist_count; ++i) {
            t[i] = history[4 * i + 0];
            x[i] = history[4 * i + 1];
            y[i] = history[4 * i + 2];
            z[i] = history[4 * i + 3];
        }
    } else {
        t[0] = 0.0;
        x[0] = x0[0];
        y[0] = x0[1];
        z[0] = x0[2];
    }

    double xn = x[start_idx], yn = y[start_idx], zn = z[start_idx];
    for (int n = start_idx; n < start_idx + nsteps; ++n) {
        double mem_x = 0.0, mem_y = 0.0, mem_z = 0.0;
        if (n > 0) {
            mem_x = memory_component(n, t, x, q, h, nu, &c);
            mem_y = memory_component(n, t, y, q, h, nu, &c);
            mem_z = memory_component(n, t, z, q, h, nu, &c);
        }

        double state[3], tmp[3], f[3];
        state[0] = xn; state[1] = yn; state[2] = zn;
        rhs_epsilon(state, p, k, eps, f);
        const double k1x = hq * (f[0] - mem_x);
        const double k1y = hq * (f[1] - mem_y);
        const double k1z = hq * (f[2] - mem_z);

        tmp[0] = xn + c.a21 * k1x;
        tmp[1] = yn + c.a21 * k1y;
        tmp[2] = zn + c.a21 * k1z;
        rhs_epsilon(tmp, p, k, eps, f);
        const double k2x = hq * f[0];
        const double k2y = hq * f[1];
        const double k2z = hq * f[2];

        tmp[0] = xn + c.a31 * k2x + c.a32 * k1x;
        tmp[1] = yn + c.a31 * k2y + c.a32 * k1y;
        tmp[2] = zn + c.a31 * k2z + c.a32 * k1z;
        rhs_epsilon(tmp, p, k, eps, f);
        const double k3x = hq * f[0];
        const double k3y = hq * f[1];
        const double k3z = hq * f[2];

        const double xn1 = xn + c.w1 * k1x + c.w2 * k2x + c.w3 * k3x;
        const double yn1 = yn + c.w1 * k1y + c.w2 * k2y + c.w3 * k3y;
        const double zn1 = zn + c.w1 * k1z + c.w2 * k2z + c.w3 * k3z;
        t[n + 1] = t[n] + h;
        x[n + 1] = xn1;
        y[n + 1] = yn1;
        z[n + 1] = zn1;
        xn = xn1; yn = yn1; zn = zn1;
    }

    for (int i = 0; i < total_rows; ++i) {
        full[4 * i + 0] = t[i];
        full[4 * i + 1] = x[i];
        full[4 * i + 2] = y[i];
        full[4 * i + 3] = z[i];
    }

    free(t); free(x); free(y); free(z);
    *full_out = full;
    *full_rows_out = total_rows;
    *segment_start_out = start_idx;
    return 0;
}

static int extract_memory_window(const double *full, int full_rows, double h, double Lm, double *history_out) {
    if (!full || full_rows <= 0 || !history_out) return 0;
    int nu = (int)ceil(Lm / h);
    if (nu < 1) nu = 1;
    int count = nu + 1;
    if (count > full_rows) count = full_rows;
    const int start = full_rows - count;
    const double final_t = full[4 * (full_rows - 1) + 0];
    for (int i = 0; i < count; ++i) {
        history_out[4 * i + 0] = full[4 * (start + i) + 0] - final_t;
        history_out[4 * i + 1] = full[4 * (start + i) + 1];
        history_out[4 * i + 2] = full[4 * (start + i) + 2];
        history_out[4 * i + 3] = full[4 * (start + i) + 3];
    }
    if (count > 0) history_out[4 * (count - 1)] = 0.0;
    return count;
}

static void write_segment(const double *full, int segment_start, int nsteps, double *out) {
    const double t0 = full[4 * segment_start + 0];
    for (int i = 0; i <= nsteps; ++i) {
        const int src = segment_start + i;
        out[4 * i + 0] = full[4 * src + 0] - t0;
        out[4 * i + 1] = full[4 * src + 1];
        out[4 * i + 2] = full[4 * src + 2];
        out[4 * i + 3] = full[4 * src + 3];
    }
}

static void copy_last_state(const double *full, int full_rows, double out3[3]) {
    const int idx = full_rows - 1;
    out3[0] = full[4 * idx + 1];
    out3[1] = full[4 * idx + 2];
    out3[2] = full[4 * idx + 3];
}

API_EXPORT void set_frac_chua_params(double alpha, double beta, double gamma, double m0, double m1) {
    G_PARAMS.alpha_chua = alpha;
    G_PARAMS.beta = beta;
    G_PARAMS.gamma_chua = gamma;
    G_PARAMS.m0 = m0;
    G_PARAMS.m1 = m1;
}

API_EXPORT void set_frac_chua_model(int model) {
    G_PARAMS.model = (model == 1) ? 1 : 0;
}

API_EXPORT void set_frac_chua_arctan_params(double a1, double a2, double rho) {
    G_PARAMS.a1 = a1;
    G_PARAMS.a2 = a2;
    G_PARAMS.rho = (rho > 0.0) ? rho : 1.0;
}

API_EXPORT void set_frac_backend_workers(int workers) {
    G_WORKERS = (workers > 0) ? workers : 0;
}

API_EXPORT int efork_rows(double t_final, double h) {
    const int nsteps = ceil_steps(t_final, h);
    return (nsteps < 0) ? -1 : nsteps + 1;
}

API_EXPORT int integrate_chua_efork3(
    double x0, double y0, double z0,
    double q, double h, double Lm, double t_final,
    double k, double eps,
    double *traj_out
) {
    if (!traj_out) return -10;
    const double seed[3] = {x0, y0, z0};
    double *full = NULL;
    int full_rows = 0, segment_start = 0;
    const int rc = integrate_internal(&G_PARAMS, seed, q, h, Lm, t_final, k, eps, NULL, 0, &full, &full_rows, &segment_start);
    if (rc != 0) return rc;
    const int nsteps = ceil_steps(t_final, h);
    write_segment(full, segment_start, nsteps, traj_out);
    free(full);
    return 0;
}

API_EXPORT int compute_continuation_efork3(
    const double *eps_values,
    int n_eps,
    const double *x_seed3,
    double q,
    double k,
    double h,
    double Lm,
    double t_transient,
    double t_keep,
    int memory_mode,
    int memory_update_source,
    double *x_in_out,
    double *x_transient_out,
    double *x_out_out,
    int *history_in_counts,
    int *history_out_counts,
    double *traj_out
) {
    if (!eps_values || !x_seed3 || !x_in_out || !x_transient_out || !x_out_out || !traj_out) return -10;
    if (n_eps <= 0) return -11;
    const int keep_steps = ceil_steps(t_keep, h);
    if (keep_steps < 0) return -12;
    int max_hist = (int)ceil(Lm / h) + 1;
    if (max_hist < 2) max_hist = 2;
    double *mem_current = (double*)calloc((size_t)max_hist * 4u, sizeof(double));
    double *mem_trans = (double*)calloc((size_t)max_hist * 4u, sizeof(double));
    double *mem_keep = (double*)calloc((size_t)max_hist * 4u, sizeof(double));
    double *mem_next = (double*)calloc((size_t)max_hist * 4u, sizeof(double));
    if (!mem_current || !mem_trans || !mem_keep || !mem_next) {
        free(mem_current); free(mem_trans); free(mem_keep); free(mem_next);
        return -13;
    }

    int mem_count = 0;
    double x_in[3] = {x_seed3[0], x_seed3[1], x_seed3[2]};
    for (int ie = 0; ie < n_eps; ++ie) {
        const double eps = eps_values[ie];
        x_in_out[3 * ie + 0] = x_in[0];
        x_in_out[3 * ie + 1] = x_in[1];
        x_in_out[3 * ie + 2] = x_in[2];
        if (history_in_counts) history_in_counts[ie] = (memory_mode == 1) ? mem_count : 0;

        double *xt_full = NULL;
        int xt_rows = 0, xt_seg = 0;
        const double *hist_ptr = (memory_mode == 1 && mem_count > 0) ? mem_current : NULL;
        const int hist_count = (memory_mode == 1 && mem_count > 0) ? mem_count : 0;
        int rc = integrate_internal(&G_PARAMS, x_in, q, h, Lm, t_transient, k, eps, hist_ptr, hist_count, &xt_full, &xt_rows, &xt_seg);
        if (rc != 0) {
            free(xt_full); free(mem_current); free(mem_trans); free(mem_keep); free(mem_next);
            return rc;
        }

        double x_trans[3];
        copy_last_state(xt_full, xt_rows, x_trans);
        x_transient_out[3 * ie + 0] = x_trans[0];
        x_transient_out[3 * ie + 1] = x_trans[1];
        x_transient_out[3 * ie + 2] = x_trans[2];

        int mem_trans_count = 0;
        if (memory_mode == 1) mem_trans_count = extract_memory_window(xt_full, xt_rows, h, Lm, mem_trans);

        double *xa_full = NULL;
        int xa_rows = 0, xa_seg = 0;
        rc = integrate_internal(
            &G_PARAMS, x_trans, q, h, Lm, t_keep, k, eps,
            (memory_mode == 1 && mem_trans_count > 0) ? mem_trans : NULL,
            (memory_mode == 1 && mem_trans_count > 0) ? mem_trans_count : 0,
            &xa_full, &xa_rows, &xa_seg
        );
        if (rc != 0) {
            free(xt_full); free(xa_full); free(mem_current); free(mem_trans); free(mem_keep); free(mem_next);
            return rc;
        }
        write_segment(xa_full, xa_seg, keep_steps, traj_out + (size_t)ie * (size_t)(keep_steps + 1) * 4u);

        double x_keep[3];
        copy_last_state(xa_full, xa_rows, x_keep);
        int next_count = 0;
        if (memory_mode == 1) {
            if (memory_update_source == 1) {
                next_count = extract_memory_window(xa_full, xa_rows, h, Lm, mem_keep);
                memcpy(mem_next, mem_keep, (size_t)next_count * 4u * sizeof(double));
            } else {
                next_count = mem_trans_count;
                memcpy(mem_next, mem_trans, (size_t)next_count * 4u * sizeof(double));
            }
        }

        if (memory_update_source == 1) {
            x_out_out[3 * ie + 0] = x_keep[0];
            x_out_out[3 * ie + 1] = x_keep[1];
            x_out_out[3 * ie + 2] = x_keep[2];
        } else {
            x_out_out[3 * ie + 0] = x_trans[0];
            x_out_out[3 * ie + 1] = x_trans[1];
            x_out_out[3 * ie + 2] = x_trans[2];
        }
        if (history_out_counts) history_out_counts[ie] = (memory_mode == 1) ? next_count : 0;

        x_in[0] = x_out_out[3 * ie + 0];
        x_in[1] = x_out_out[3 * ie + 1];
        x_in[2] = x_out_out[3 * ie + 2];
        if (memory_mode == 1) {
            mem_count = next_count;
            memcpy(mem_current, mem_next, (size_t)mem_count * 4u * sizeof(double));
        }

        free(xt_full);
        free(xa_full);
    }

    free(mem_current); free(mem_trans); free(mem_keep); free(mem_next);
    return 0;
}

static int extract_peaks(const double *traj, int rows, double t_burn, int max_peaks, double *peaks_out) {
    if (!traj || rows <= 0 || max_peaks <= 0 || !peaks_out) return 0;
    int start = 0;
    while (start < rows && traj[4 * start + 0] < t_burn) start++;
    if (start >= rows) return 0;

    int count = 0;
    for (int i = start + 1; i < rows - 1; ++i) {
        const double x0 = traj[4 * (i - 1) + 1];
        const double x1 = traj[4 * i + 1];
        const double x2 = traj[4 * (i + 1) + 1];
        if (x1 > x0 && x1 >= x2) count++;
    }
    if (count == 0) {
        double xmax = traj[4 * start + 1];
        for (int i = start + 1; i < rows; ++i) {
            const double x = traj[4 * i + 1];
            if (x > xmax) xmax = x;
        }
        peaks_out[0] = xmax;
        return 1;
    }
    const int skip = (count > max_peaks) ? (count - max_peaks) : 0;
    int seen = 0, written = 0;
    for (int i = start + 1; i < rows - 1; ++i) {
        const double x0 = traj[4 * (i - 1) + 1];
        const double x1 = traj[4 * i + 1];
        const double x2 = traj[4 * (i + 1) + 1];
        if (x1 > x0 && x1 >= x2) {
            if (seen >= skip && written < max_peaks) peaks_out[written++] = x1;
            seen++;
        }
    }
    return written;
}

static void params_for_value(ChuaParams *p, double *q, int param_type, double value) {
    if (param_type == 0) *q = value;
    else if (param_type == 1) p->alpha_chua = value;
    else if (param_type == 2) p->beta = value;
}

static int bifurcation_one_seed(
    const ChuaParams *p,
    const double seed[3],
    double q,
    double h,
    double Lm,
    double t_total,
    double t_burn,
    int max_peaks,
    double value,
    double *x_out,
    double *y_out,
    int *count_out,
    double final_state[3]
) {
    double *full = NULL;
    int full_rows = 0, seg = 0;
    const int rc = integrate_internal(p, seed, q, h, Lm, t_total, 0.0, 1.0, NULL, 0, &full, &full_rows, &seg);
    if (rc != 0) {
        free(full);
        return rc;
    }
    copy_last_state(full, full_rows, final_state);
    double *segment = full + 4 * seg;
    const int rows = full_rows - seg;
    const int count = extract_peaks(segment, rows, t_burn, max_peaks, y_out);
    for (int i = 0; i < count; ++i) x_out[i] = value;
    *count_out = count;
    free(full);
    return 0;
}

API_EXPORT int compute_bifurcation_sweep_efork3(
    int param_type,
    const double *values,
    int n_values,
    const double *seed_pos3,
    const double *seed_neg3,
    double base_q,
    double h,
    double Lm,
    double t_total,
    double t_burn,
    int max_peaks,
    int continue_seed,
    double *pos_x,
    double *pos_y,
    int *pos_count,
    double *neg_x,
    double *neg_y,
    int *neg_count
) {
    if (!values || !seed_pos3 || !seed_neg3 || !pos_x || !pos_y || !pos_count || !neg_x || !neg_y || !neg_count) return -10;
    if (n_values <= 0 || max_peaks <= 0) return -11;
    if (param_type < 0 || param_type > 2) return -12;

    ChuaParams base = G_PARAMS;
    if (continue_seed) {
        double cur_pos[3] = {seed_pos3[0], seed_pos3[1], seed_pos3[2]};
        double cur_neg[3] = {seed_neg3[0], seed_neg3[1], seed_neg3[2]};
        for (int i = 0; i < n_values; ++i) {
            ChuaParams p = base;
            double q = base_q;
            params_for_value(&p, &q, param_type, values[i]);
            double final_pos[3], final_neg[3];
            int rc = bifurcation_one_seed(&p, cur_pos, q, h, Lm, t_total, t_burn, max_peaks, values[i],
                                          pos_x + (size_t)i * max_peaks, pos_y + (size_t)i * max_peaks,
                                          pos_count + i, final_pos);
            if (rc != 0) return rc;
            rc = bifurcation_one_seed(&p, cur_neg, q, h, Lm, t_total, t_burn, max_peaks, values[i],
                                      neg_x + (size_t)i * max_peaks, neg_y + (size_t)i * max_peaks,
                                      neg_count + i, final_neg);
            if (rc != 0) return rc;
            cur_pos[0] = final_pos[0]; cur_pos[1] = final_pos[1]; cur_pos[2] = final_pos[2];
            cur_neg[0] = final_neg[0]; cur_neg[1] = final_neg[1]; cur_neg[2] = final_neg[2];
        }
        return 0;
    }

    int rc_global = 0;
    #ifdef _OPENMP
    const int threads = (G_WORKERS > 0) ? G_WORKERS : omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic) num_threads(threads)
    #endif
    for (int i = 0; i < n_values; ++i) {
        ChuaParams p = base;
        double q = base_q;
        params_for_value(&p, &q, param_type, values[i]);
        double final_pos[3], final_neg[3];
        int rc = bifurcation_one_seed(&p, seed_pos3, q, h, Lm, t_total, t_burn, max_peaks, values[i],
                                      pos_x + (size_t)i * max_peaks, pos_y + (size_t)i * max_peaks,
                                      pos_count + i, final_pos);
        if (rc == 0) {
            rc = bifurcation_one_seed(&p, seed_neg3, q, h, Lm, t_total, t_burn, max_peaks, values[i],
                                      neg_x + (size_t)i * max_peaks, neg_y + (size_t)i * max_peaks,
                                      neg_count + i, final_neg);
        }
        if (rc != 0) {
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            {
                if (rc_global == 0) rc_global = rc;
            }
        }
    }
    return rc_global;
}
