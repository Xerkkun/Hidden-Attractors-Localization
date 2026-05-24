#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#if defined(_WIN32) || defined(__CYGWIN__)
  #define API_EXPORT __declspec(dllexport)
#else
  #define API_EXPORT __attribute__((visibility("default")))
#endif

typedef struct {
    double alpha;
    double beta;
    double gamma;
    double m0;
    double m1;
} ChuaParams;

static ChuaParams G_PARAMS = {8.4562, 12.0732, 0.0052, -0.1768, -1.1468};

static int ceil_steps(double t_final, double h) {
    if (!(h > 0.0) || t_final < 0.0) return -1;
    return (int)ceil(t_final / h);
}

static double chua_piecewise(double x, const ChuaParams *p) {
    return p->m1 * x + 0.5 * (p->m0 - p->m1) * (fabs(x + 1.0) - fabs(x - 1.0));
}

static void rhs(const double x[3], const ChuaParams *p, double f[3]) {
    f[0] = p->alpha * (x[1] - x[0] - chua_piecewise(x[0], p));
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
    const int nsteps = ceil_steps(t_final, h);
    return (nsteps < 0) ? -1 : nsteps + 1;
}

/*
 * Full-history Diethelm Adams-Bashforth-Moulton PECE method.
 *
 * The history sums always begin at j=0. There is no finite-memory parameter:
 * each output therefore represents the Caputo initial-value problem on [0,t].
 * The formulas mirror the existing Python ABM reference used for Danca runs.
 */
API_EXPORT int integrate_chua_abm_full_history(
    double x0,
    double y0,
    double z0,
    double q,
    double h,
    double t_final,
    double *out
) {
    if (!(q > 0.0 && q <= 1.0) || !(h > 0.0) || t_final < 0.0 || !out) return -1;
    const int nsteps = ceil_steps(t_final, h);
    if (nsteps < 0) return -2;
    const int rows = nsteps + 1;

    double *state = (double*)calloc((size_t)rows * 3u, sizeof(double));
    double *fhist = (double*)calloc((size_t)rows * 3u, sizeof(double));
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
    const double hq = pow(h, q);
    const double pred_scale = hq / tgamma(q + 1.0);
    const double corr_scale = hq / tgamma(q + 2.0);

    for (int i = 0; i < nsteps; ++i) {
        double predictor[3] = {x0, y0, z0};
        for (int j = 0; j <= i; ++j) {
            const int r = i - j;
            const double weight = pow_q[r + 1] - pow_q[r];
            predictor[0] += pred_scale * weight * fhist[3 * j + 0];
            predictor[1] += pred_scale * weight * fhist[3 * j + 1];
            predictor[2] += pred_scale * weight * fhist[3 * j + 2];
        }

        double fp[3];
        rhs(predictor, &G_PARAMS, fp);
        double corrected[3] = {x0, y0, z0};
        if (i == 0) {
            corrected[0] += corr_scale * (q * fhist[0] + fp[0]);
            corrected[1] += corr_scale * (q * fhist[1] + fp[1]);
            corrected[2] += corr_scale * (q * fhist[2] + fp[2]);
        } else {
            const double a0 = pow_q1[i] - ((double)i - q) * pow_q[i + 1];
            corrected[0] += corr_scale * a0 * fhist[0];
            corrected[1] += corr_scale * a0 * fhist[1];
            corrected[2] += corr_scale * a0 * fhist[2];
            for (int j = 1; j <= i; ++j) {
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
        rhs(corrected, &G_PARAMS, &fhist[3 * (i + 1)]);
    }

    for (int i = 0; i < rows; ++i) {
        out[4 * i + 0] = (double)i * h;
        out[4 * i + 1] = state[3 * i + 0];
        out[4 * i + 2] = state[3 * i + 1];
        out[4 * i + 3] = state[3 * i + 2];
    }

    free(state); free(fhist); free(pow_q); free(pow_q1);
    return 0;
}
