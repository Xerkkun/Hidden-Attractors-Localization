#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include "native_validation.h"

#if defined(_WIN32) || defined(__CYGWIN__)
  #define API_EXPORT __declspec(dllexport)
#else
  #define API_EXPORT __attribute__((visibility("default")))
#endif

API_EXPORT int chua_abm_abi_version(void) {
    return 3;
}

typedef struct {
    double alpha;
    double beta;
    double gamma;
    double m0;
    double m1;
} ChuaParams;

static ChuaParams G_PARAMS = {8.4562, 12.0732, 0.0052, -0.1768, -1.1468};

static int uniform_steps(double t_final, double h) {
    int steps = 0;
    return hafo_uniform_step_count(t_final, h, &steps) ? steps : -1;
}

static double chua_piecewise(double x, const ChuaParams *p) {
    return p->m1 * x + 0.5 * (p->m0 - p->m1) * (fabs(x + 1.0) - fabs(x - 1.0));
}

static void rhs(const double x[3], const ChuaParams *p, double f[3]) {
    f[0] = p->alpha * (x[1] - x[0] - chua_piecewise(x[0], p));
    f[1] = x[0] - x[1] + x[2];
    f[2] = -p->beta * x[1] - p->gamma * x[2];
}

/*
 * Lur'e deformation used for hidden-attractor seed continuation.
 *
 * The scalar nonlinearity is split as
 *   phi(sigma) = m1*sigma + psi(sigma),
 *   psi(sigma) = (m0-m1)*clip(sigma,-1,1),
 * and the internal continuation parameter eps satisfies
 *   eps=0: linear system with artificial feedback gain k,
 *   eps=1: original non-smooth Chua system.
 */
static void rhs_epsilon(const double x[3], const ChuaParams *p, double k, double eps, double f[3]) {
    double clipped = x[0];
    if (clipped > 1.0) clipped = 1.0;
    if (clipped < -1.0) clipped = -1.0;
    const double psi = (p->m0 - p->m1) * clipped;
    const double linear_x = -p->alpha * (1.0 + p->m1 + k) * x[0] + p->alpha * x[1];
    f[0] = linear_x - eps * p->alpha * (psi - k * x[0]);
    f[1] = x[0] - x[1] + x[2];
    f[2] = -p->beta * x[1] - p->gamma * x[2];
}

API_EXPORT void get_abm_chua_equilibria(double *out9) {
    const double slope = -G_PARAMS.beta / (G_PARAMS.gamma + G_PARAMS.beta);
    const double den = G_PARAMS.m1 - slope;
    const double xp = -(G_PARAMS.m0 - G_PARAMS.m1) / den;
    const double xm = (G_PARAMS.m0 - G_PARAMS.m1) / den;
    const double fp = chua_piecewise(xp, &G_PARAMS);
    const double fm = chua_piecewise(xm, &G_PARAMS);
    out9[0] = 0.0; out9[1] = 0.0; out9[2] = 0.0;
    out9[3] = xp; out9[4] = xp + fp; out9[5] = fp;
    out9[6] = xm; out9[7] = xm + fm; out9[8] = fm;
}

API_EXPORT void set_abm_chua_params(double alpha, double beta, double gamma, double m0, double m1) {
    G_PARAMS.alpha = alpha;
    G_PARAMS.beta = beta;
    G_PARAMS.gamma = gamma;
    G_PARAMS.m0 = m0;
    G_PARAMS.m1 = m1;
}

API_EXPORT int abm_rows(double t_final, double h) {
    const int nsteps = uniform_steps(t_final, h);
    return (nsteps < 0) ? -1 : nsteps + 1;
}

/*
 * Diethelm Adams-Bashforth-Moulton PECE method.
 *
 * With truncated_history=0 the history sums always begin at j=0, so each
 * output represents the Caputo initial-value problem on [0,t].  With
 * truncated_history=1 the sum begins at the first state retained by a sliding
 * window of length Lm and that retained state is the local integral anchor.
 * The latter is an explicitly labelled finite-memory restarted approximation;
 * it is not the full-history Caputo problem used for the Danca reference.
 */
static int integrate_chua_abm_impl(
    double x0,
    double y0,
    double z0,
    double q,
    double h,
    double Lm,
    double t_final,
    int truncated_history,
    double *out,
    size_t out_capacity
) {
    int nu = 1;
    size_t state_values = 0u;
    size_t output_values = 0u;
    const double initial[3] = {x0, y0, z0};
    if (!(q > 0.0 && q <= 1.0) || !hafo_isfinite(q) || !out ||
        (truncated_history != 0 && truncated_history != 1) ||
        !hafo_isfinite(G_PARAMS.alpha) || !hafo_isfinite(G_PARAMS.beta) ||
        !hafo_isfinite(G_PARAMS.gamma) || !hafo_isfinite(G_PARAMS.m0) ||
        !hafo_isfinite(G_PARAMS.m1) ||
        !hafo_values_are_finite(initial, 3u)) return -1;
    if (truncated_history && !hafo_positive_ratio_ceil(Lm, h, &nu)) return -1;
    const int nsteps = uniform_steps(t_final, h);
    if (nsteps < 0) return -2;
    const int rows = nsteps + 1;
    if (!truncated_history) nu = nsteps + 1;
    if (!hafo_checked_mul_size((size_t)rows, 3u, &state_values) ||
        state_values > (size_t)INT_MAX ||
        !hafo_checked_mul_size((size_t)rows, 4u, &output_values) ||
        output_values > (size_t)INT_MAX) return -2;
    if (out_capacity < output_values) return -4;

    double *state = (double*)calloc(state_values, sizeof(double));
    double *fhist = (double*)calloc(state_values, sizeof(double));
    double *pow_q = (double*)calloc((size_t)rows + 1u, sizeof(double));
    double *pow_q1 = (double*)calloc((size_t)rows + 1u, sizeof(double));
    if (!state || !fhist || !pow_q || !pow_q1) {
        free(state); free(fhist); free(pow_q); free(pow_q1);
        return -3;
    }

    for (int i = 0; i <= rows; ++i) {
        pow_q[i] = pow((double)i, q);
        pow_q1[i] = pow((double)i, q + 1.0);
    }

    state[0] = x0; state[1] = y0; state[2] = z0;
    rhs(state, &G_PARAMS, fhist);
    if (!hafo_values_are_finite(fhist, 3u)) {
        free(state); free(fhist); free(pow_q); free(pow_q1);
        return -7;
    }
    const double hq = pow(h, q);
    const double pred_scale = hq / tgamma(q + 1.0);
    const double corr_scale = hq / tgamma(q + 2.0);

    for (int i = 0; i < nsteps; ++i) {
        const int j0 = (truncated_history && i + 1 > nu) ? i + 1 - nu : 0;
        double predictor[3] = {
            state[3 * j0 + 0],
            state[3 * j0 + 1],
            state[3 * j0 + 2]
        };
        for (int j = j0; j <= i; ++j) {
            const int r = i - j;
            const double weight = pow_q[r + 1] - pow_q[r];
            predictor[0] += pred_scale * weight * fhist[3 * j + 0];
            predictor[1] += pred_scale * weight * fhist[3 * j + 1];
            predictor[2] += pred_scale * weight * fhist[3 * j + 2];
        }

        double fp[3];
        rhs(predictor, &G_PARAMS, fp);
        double corrected[3] = {
            state[3 * j0 + 0],
            state[3 * j0 + 1],
            state[3 * j0 + 2]
        };
        const int local_i = i - j0;
        if (local_i == 0) {
            corrected[0] += corr_scale * (q * fhist[3 * j0 + 0] + fp[0]);
            corrected[1] += corr_scale * (q * fhist[3 * j0 + 1] + fp[1]);
            corrected[2] += corr_scale * (q * fhist[3 * j0 + 2] + fp[2]);
        } else {
            const double a0 = pow_q1[local_i] - ((double)local_i - q) * pow_q[local_i + 1];
            corrected[0] += corr_scale * a0 * fhist[3 * j0 + 0];
            corrected[1] += corr_scale * a0 * fhist[3 * j0 + 1];
            corrected[2] += corr_scale * a0 * fhist[3 * j0 + 2];
            for (int j = j0 + 1; j <= i; ++j) {
                const int r = i - j + 1;
                const double weight = pow_q1[r + 1] + pow_q1[r - 1] - 2.0 * pow_q1[r];
                corrected[0] += corr_scale * weight * fhist[3 * j + 0];
                corrected[1] += corr_scale * weight * fhist[3 * j + 1];
                corrected[2] += corr_scale * weight * fhist[3 * j + 2];
            }
            corrected[0] += corr_scale * fp[0];
            corrected[1] += corr_scale * fp[1];
            corrected[2] += corr_scale * fp[2];
        }

        if (!hafo_values_are_finite(corrected, 3u)) {
            free(state); free(fhist); free(pow_q); free(pow_q1);
            return -7;
        }

        state[3 * (i + 1) + 0] = corrected[0];
        state[3 * (i + 1) + 1] = corrected[1];
        state[3 * (i + 1) + 2] = corrected[2];
        rhs(corrected, &G_PARAMS, &fhist[3 * (i + 1)]);
    }

    for (int i = 0; i < rows; ++i) {
        out[4 * i + 0] = (i == nsteps) ? t_final : (double)i * h;
        out[4 * i + 1] = state[3 * i + 0];
        out[4 * i + 2] = state[3 * i + 1];
        out[4 * i + 3] = state[3 * i + 2];
    }

    free(state); free(fhist); free(pow_q); free(pow_q1);
    return 0;
}

API_EXPORT int integrate_chua_abm_full_history(
    double x0,
    double y0,
    double z0,
    double q,
    double h,
    double t_final,
    double *out,
    size_t out_capacity
) {
    return integrate_chua_abm_impl(
        x0, y0, z0, q, h, 0.0, t_final, 0, out, out_capacity
    );
}

API_EXPORT int integrate_chua_abm_truncated_history(
    double x0,
    double y0,
    double z0,
    double q,
    double h,
    double Lm,
    double t_final,
    double *out,
    size_t out_capacity
) {
    return integrate_chua_abm_impl(
        x0, y0, z0, q, h, Lm, t_final, 1, out, out_capacity
    );
}

/*
 * Advance one PECE step for the Lur'e continuation problem.
 *
 * In full-history mode j0=0, hence the Caputo memory of all preceding eta
 * stages is retained. In truncated mode the retained state at j0 is the
 * restarted Volterra anchor for a window of duration Lm.
 */
static int advance_continuation_abm(
    double *state,
    double *fhist,
    int i,
    int nu,
    int truncated_history,
    double q,
    double pred_scale,
    double corr_scale,
    const double *pow_q,
    const double *pow_q1,
    double k,
    double eps
) {
    const int j0 = (truncated_history && i + 1 > nu) ? i + 1 - nu : 0;
    double predictor[3] = {
        state[3 * j0 + 0],
        state[3 * j0 + 1],
        state[3 * j0 + 2]
    };
    for (int j = j0; j <= i; ++j) {
        const int r = i - j;
        const double weight = pow_q[r + 1] - pow_q[r];
        predictor[0] += pred_scale * weight * fhist[3 * j + 0];
        predictor[1] += pred_scale * weight * fhist[3 * j + 1];
        predictor[2] += pred_scale * weight * fhist[3 * j + 2];
    }

    double fp[3];
    rhs_epsilon(predictor, &G_PARAMS, k, eps, fp);
    double corrected[3] = {
        state[3 * j0 + 0],
        state[3 * j0 + 1],
        state[3 * j0 + 2]
    };
    const int local_i = i - j0;
    if (local_i == 0) {
        corrected[0] += corr_scale * (q * fhist[3 * j0 + 0] + fp[0]);
        corrected[1] += corr_scale * (q * fhist[3 * j0 + 1] + fp[1]);
        corrected[2] += corr_scale * (q * fhist[3 * j0 + 2] + fp[2]);
    } else {
        const double a0 = pow_q1[local_i] - ((double)local_i - q) * pow_q[local_i + 1];
        corrected[0] += corr_scale * a0 * fhist[3 * j0 + 0];
        corrected[1] += corr_scale * a0 * fhist[3 * j0 + 1];
        corrected[2] += corr_scale * a0 * fhist[3 * j0 + 2];
        for (int j = j0 + 1; j <= i; ++j) {
            const int r = i - j + 1;
            const double weight = pow_q1[r + 1] + pow_q1[r - 1] - 2.0 * pow_q1[r];
            corrected[0] += corr_scale * weight * fhist[3 * j + 0];
            corrected[1] += corr_scale * weight * fhist[3 * j + 1];
            corrected[2] += corr_scale * weight * fhist[3 * j + 2];
        }
        corrected[0] += corr_scale * fp[0];
        corrected[1] += corr_scale * fp[1];
        corrected[2] += corr_scale * fp[2];
    }

    state[3 * (i + 1) + 0] = corrected[0];
    state[3 * (i + 1) + 1] = corrected[1];
    state[3 * (i + 1) + 2] = corrected[2];
    if (!hafo_values_are_finite(corrected, 3u)) return 0;
    rhs_epsilon(corrected, &G_PARAMS, k, eps, &fhist[3 * (i + 1)]);
    return hafo_values_are_finite(&fhist[3 * (i + 1)], 3u);
}

/*
 * ABM continuation through a public lambda=eps homotopy.
 *
 * The eta path is causal: the time index is not reset between parameter
 * stages. At an eta boundary, eps is interpreted right-continuously and the
 * endpoint derivative is evaluated using the new value before advancing.
 * This is a numerical continuation protocol, not a claim of an autonomous
 * periodic Caputo orbit.
 *
 * final_history_out returns rows [relative_time, x, y, z]. Full mode returns
 * the complete chronological state history; truncated mode returns precisely
 * the transported finite window.
 */
API_EXPORT int compute_continuation_abm(
    const double *lambda_values,
    int n_lambda,
    const double *seed3,
    double q,
    double k,
    double h,
    double Lm,
    double t_transient,
    double t_keep,
    int truncated_history,
    double *x_in_out,
    double *x_transient_out,
    double *x_out_out,
    int *history_in_counts,
    int *history_out_counts,
    double *traj_out,
    double *final_history_out,
    int final_history_capacity,
    int *final_history_count,
    size_t x_in_capacity,
    size_t x_transient_capacity,
    size_t x_out_capacity,
    size_t history_counts_capacity,
    size_t traj_capacity,
    size_t final_history_values_capacity
) {
    int nu_steps = 0;
    size_t stage_state_values = 0u;
    size_t state_values = 0u;
    size_t trajectory_rows = 0u;
    size_t trajectory_values = 0u;
    size_t final_history_values = 0u;
    if (!lambda_values || n_lambda < 1 || !seed3 || !x_in_out || !x_transient_out || !x_out_out ||
        !history_in_counts || !history_out_counts || !traj_out || !final_history_out || !final_history_count) return -1;
    if (!(q > 0.0 && q <= 1.0) || !hafo_isfinite(q) || !hafo_isfinite(k) ||
        (truncated_history != 0 && truncated_history != 1) ||
        !hafo_values_are_finite(lambda_values, (size_t)n_lambda) ||
        !hafo_values_are_finite(seed3, 3u) ||
        !hafo_isfinite(G_PARAMS.alpha) || !hafo_isfinite(G_PARAMS.beta) ||
        !hafo_isfinite(G_PARAMS.gamma) || !hafo_isfinite(G_PARAMS.m0) ||
        !hafo_isfinite(G_PARAMS.m1)) return -2;
    if (truncated_history && !hafo_positive_ratio_ceil(Lm, h, &nu_steps)) return -3;
    const int transient_steps = uniform_steps(t_transient, h);
    const int keep_steps = uniform_steps(t_keep, h);
    if (transient_steps < 0 || keep_steps < 0) return -4;
    if (transient_steps > INT_MAX - keep_steps) return -4;
    const int stage_steps = transient_steps + keep_steps;
    if (stage_steps > 0 && n_lambda > (INT_MAX - 2) / stage_steps) return -4;
    const int total_steps = n_lambda * stage_steps;
    const int rows = total_steps + 1;
    const int nu = truncated_history ? nu_steps : rows;

    if (!hafo_checked_mul_size((size_t)n_lambda, 3u, &stage_state_values) ||
        stage_state_values > (size_t)INT_MAX ||
        !hafo_checked_mul_size((size_t)rows, 3u, &state_values) ||
        state_values > (size_t)INT_MAX ||
        !hafo_checked_mul_size((size_t)n_lambda, (size_t)keep_steps + 1u,
                               &trajectory_rows) ||
        !hafo_checked_mul_size(trajectory_rows, 4u, &trajectory_values) ||
        final_history_capacity < 1 ||
        !hafo_checked_mul_size((size_t)final_history_capacity, 4u,
                               &final_history_values)) return -4;
    if (x_in_capacity < stage_state_values ||
        x_transient_capacity < stage_state_values ||
        x_out_capacity < stage_state_values ||
        history_counts_capacity < (size_t)n_lambda ||
        traj_capacity < trajectory_values ||
        final_history_values_capacity < final_history_values) return -6;

    double *state = (double*)calloc(state_values, sizeof(double));
    double *fhist = (double*)calloc(state_values, sizeof(double));
    double *pow_q = (double*)calloc((size_t)rows + 1u, sizeof(double));
    double *pow_q1 = (double*)calloc((size_t)rows + 1u, sizeof(double));
    if (!state || !fhist || !pow_q || !pow_q1) {
        free(state); free(fhist); free(pow_q); free(pow_q1);
        return -5;
    }

    for (int i = 0; i <= rows; ++i) {
        pow_q[i] = pow((double)i, q);
        pow_q1[i] = pow((double)i, q + 1.0);
    }
    state[0] = seed3[0]; state[1] = seed3[1]; state[2] = seed3[2];
    const double hq = pow(h, q);
    const double pred_scale = hq / tgamma(q + 1.0);
    const double corr_scale = hq / tgamma(q + 2.0);
    int cursor = 0;

    for (int stage = 0; stage < n_lambda; ++stage) {
        const double eps = lambda_values[stage];
        rhs_epsilon(&state[3 * cursor], &G_PARAMS, k, eps, &fhist[3 * cursor]);
        if (!hafo_values_are_finite(&fhist[3 * cursor], 3u)) {
            free(state); free(fhist); free(pow_q); free(pow_q1);
            return -7;
        }
        x_in_out[3 * stage + 0] = state[3 * cursor + 0];
        x_in_out[3 * stage + 1] = state[3 * cursor + 1];
        x_in_out[3 * stage + 2] = state[3 * cursor + 2];
        if (stage == 0) {
            history_in_counts[stage] = 0;
        } else if (truncated_history && cursor + 1 > nu + 1) {
            history_in_counts[stage] = nu + 1;
        } else {
            history_in_counts[stage] = cursor + 1;
        }

        for (int step = 0; step < transient_steps; ++step) {
            if (!advance_continuation_abm(state, fhist, cursor, nu,
                                          truncated_history, q, pred_scale,
                                          corr_scale, pow_q, pow_q1, k, eps)) {
                free(state); free(fhist); free(pow_q); free(pow_q1);
                return -7;
            }
            cursor += 1;
        }
        x_transient_out[3 * stage + 0] = state[3 * cursor + 0];
        x_transient_out[3 * stage + 1] = state[3 * cursor + 1];
        x_transient_out[3 * stage + 2] = state[3 * cursor + 2];
        const size_t traj_offset = (size_t)stage * ((size_t)keep_steps + 1u) * 4u;
        traj_out[traj_offset + 0] = 0.0;
        traj_out[traj_offset + 1] = state[3 * cursor + 0];
        traj_out[traj_offset + 2] = state[3 * cursor + 1];
        traj_out[traj_offset + 3] = state[3 * cursor + 2];
        for (int step = 0; step < keep_steps; ++step) {
            if (!advance_continuation_abm(state, fhist, cursor, nu,
                                          truncated_history, q, pred_scale,
                                          corr_scale, pow_q, pow_q1, k, eps)) {
                free(state); free(fhist); free(pow_q); free(pow_q1);
                return -7;
            }
            cursor += 1;
            const size_t offset = traj_offset + ((size_t)step + 1u) * 4u;
            traj_out[offset + 0] = (step + 1 == keep_steps)
                ? t_keep : (double)(step + 1) * h;
            traj_out[offset + 1] = state[3 * cursor + 0];
            traj_out[offset + 2] = state[3 * cursor + 1];
            traj_out[offset + 3] = state[3 * cursor + 2];
        }
        x_out_out[3 * stage + 0] = state[3 * cursor + 0];
        x_out_out[3 * stage + 1] = state[3 * cursor + 1];
        x_out_out[3 * stage + 2] = state[3 * cursor + 2];
        if (truncated_history && cursor + 1 > nu + 1) {
            history_out_counts[stage] = nu + 1;
        } else {
            history_out_counts[stage] = cursor + 1;
        }
    }

    const int count = truncated_history && cursor + 1 > nu + 1 ? nu + 1 : cursor + 1;
    *final_history_count = count;
    if (final_history_capacity < count) {
        free(state); free(fhist); free(pow_q); free(pow_q1);
        return -6;
    }
    const int start = cursor + 1 - count;
    for (int i = 0; i < count; ++i) {
        final_history_out[4 * i + 0] = (double)(start + i - cursor) * h;
        final_history_out[4 * i + 1] = state[3 * (start + i) + 0];
        final_history_out[4 * i + 2] = state[3 * (start + i) + 1];
        final_history_out[4 * i + 3] = state[3 * (start + i) + 2];
    }

    free(state); free(fhist); free(pow_q); free(pow_q1);
    return 0;
}
