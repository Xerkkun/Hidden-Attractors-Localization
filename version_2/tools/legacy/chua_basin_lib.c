#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #define API_EXPORT __declspec(dllexport)
#else
  #define API_EXPORT __attribute__((visibility("default")))
#endif

// ============================================================
// Parámetros globales del sistema de Chua fraccionario
// ============================================================
// Python sobrescribe estos valores con set_chua_params(...), set_chua_model(...)
// y pasa q por llamada a compute_basin_*; estos defaults son el caso piecewise.
static double G_ALPHA = 8.4562;
static double G_BETA  = 12.0732;
static double G_GAMMA = 0.0052;
static double G_M0    = -0.1768;
static double G_M1    = -1.1468;
static int    G_MODEL = 0; // 0=piecewise, 1=arctan
static double G_A1    = 0.4;
static double G_A2    = -1.5585;
static double G_RHO   = 1.0;
static int    G_WORKERS = 0;

// Clases
#define CLS_EQ         0
#define CLS_HIDDEN_POS 1
#define CLS_HIDDEN_NEG 2
#define CLS_DIV        3
#define CLS_UNKNOWN    4

// ============================================================
// Utilidades
// ============================================================
static inline double f_chua(double x) {
    if (G_MODEL == 1) {
        return G_A1 * x + G_A2 * atan(G_RHO * x);
    }
    // f(x)=m1*x+psi(x), psi(x)=(m0-m1)*sat(x).
    return G_M1*x + 0.5*(G_M0 - G_M1)*(fabs(x + 1.0) - fabs(x - 1.0));
}

static inline double nrm2(double x, double y, double z) {
    return x*x + y*y + z*z;
}

static inline void rhs(double x, double y, double z, double *fx, double *fy, double *fz) {
    *fx = G_ALPHA * (y - x - f_chua(x));
    *fy = x - y + z;
    *fz = -G_BETA * y - G_GAMMA * z;
}

static void chua_equilibria_internal(double *eq_out9) {
    // eq_out9 = [xE0,yE0,zE0, xEp,yEp,zEp, xEm,yEm,zEm]
    eq_out9[0] = 0.0; eq_out9[1] = 0.0; eq_out9[2] = 0.0;

    if (G_MODEL == 1) {
        const double coeff = 1.0 + G_A1 - G_GAMMA / (G_BETA + G_GAMMA);
        double prevx = 1e-8;
        double prev = coeff * prevx + G_A2 * atan(G_RHO * prevx);
        double xp = NAN;
        for (int i = 1; i <= 20000; ++i) {
            double x = 100.0 * (double)i / 20000.0;
            double cur = coeff * x + G_A2 * atan(G_RHO * x);
            if (prev * cur < 0.0) {
                double lo = prevx, hi = x, flo = prev;
                for (int it = 0; it < 80; ++it) {
                    double mid = 0.5 * (lo + hi);
                    double fm = coeff * mid + G_A2 * atan(G_RHO * mid);
                    if (fabs(fm) < 1e-14) { lo = hi = mid; break; }
                    if (flo * fm <= 0.0) hi = mid;
                    else { lo = mid; flo = fm; }
                }
                xp = 0.5 * (lo + hi);
                break;
            }
            prevx = x;
            prev = cur;
        }
        if (!isfinite(xp)) {
            eq_out9[3] = eq_out9[4] = eq_out9[5] = NAN;
            eq_out9[6] = eq_out9[7] = eq_out9[8] = NAN;
            return;
        }
        const double yp = G_GAMMA / (G_BETA + G_GAMMA) * xp;
        const double zp = -G_BETA / (G_BETA + G_GAMMA) * xp;
        eq_out9[3] = xp;  eq_out9[4] = yp;  eq_out9[5] = zp;
        eq_out9[6] = -xp; eq_out9[7] = -yp; eq_out9[8] = -zp;
        return;
    }

    const double s = -G_BETA / (G_GAMMA + G_BETA);
    const double den = (G_M1 - s);

    if (fabs(den) < 1e-15) {
        eq_out9[3] = eq_out9[4] = eq_out9[5] = NAN;
        eq_out9[6] = eq_out9[7] = eq_out9[8] = NAN;
        return;
    }

    const double xplus  = -(G_M0 - G_M1) / den;
    const double xminus =  (G_M0 - G_M1) / den;
    const double fxp = f_chua(xplus);
    const double fxm = f_chua(xminus);

    eq_out9[3] = xplus;  eq_out9[4] = xplus  + fxp;  eq_out9[5] = fxp;
    eq_out9[6] = xminus; eq_out9[7] = xminus + fxm;  eq_out9[8] = fxm;
}

// ============================================================
// EFORK
// ============================================================
typedef struct {
    double alpha_frac;
    double g1, g2, g3;
    double w1, w2, w3;
    double a21, a31, a32;
} EFORKCoeffs;

static EFORKCoeffs efork_coeffs(double alpha_frac) {
    EFORKCoeffs c;
    c.alpha_frac = alpha_frac;
    c.g1 = tgamma(1.0 + alpha_frac);
    c.g2 = tgamma(1.0 + 2.0*alpha_frac);
    c.g3 = tgamma(1.0 + 3.0*alpha_frac);

    c.a21 = 1.0 / (2.0 * c.g1 * c.g1);
    c.a31 = ((c.g1*c.g1)*c.g2 + 2.0*(c.g2*c.g2) - c.g3)
          / (4.0*(c.g1*c.g1)*(2.0*(c.g2*c.g2) - c.g3));
    c.a32 = -c.g2 / (4.0 * (2.0*(c.g2*c.g2) - c.g3));

    c.w1 = (8.0*(c.g1*c.g1*c.g1)*(c.g2*c.g2) - 6.0*(c.g1*c.g1*c.g1)*c.g3 + c.g2*c.g3)
         / (c.g1*c.g2*c.g3);
    c.w2 = 2.0*(c.g1*c.g1)*(4.0*(c.g2*c.g2) - c.g3) / (c.g2*c.g3);
    c.w3 = -8.0*(c.g1*c.g1)*(2.0*(c.g2*c.g2) - c.g3) / (c.g2*c.g3);

    return c;
}

static double memory_fractional_scalar(
    int k,
    double t,
    const double *arr,
    const double *vtn,
    double h,
    double alpha_frac,
    int nu
) {
    const int start = (k > nu) ? (k - nu) : 0;
    const double gamma_term = tgamma(2.0 - alpha_frac);
    double s = 0.0;

    for (int j = start; j < k; ++j) {
        const double t0 = vtn[j];
        const double t1 = vtn[j + 1];
        const double v1 = pow(t - t0, 1.0 - alpha_frac);
        const double v2 = pow(t - t1, 1.0 - alpha_frac);
        s += (arr[j + 1] - arr[j]) * (v1 - v2);
    }
    return s / (h * gamma_term);
}

static int classify_point(
    double x0,
    double y0,
    double z0,
    double q,
    double h,
    double Lm,
    double TMAX,
    double TBURN,
    double R_DIV,
    double R_BOUND,
    double EPS_EQ,
    int CAP_WIN,
    double MEAN_X_GAP
) {
    const int N = (int)ceil(TMAX / h);
    const int burn = (int)floor(TBURN / h);
    const int nu = ((int)floor(Lm / h) > 1) ? (int)floor(Lm / h) : 1;
    const double ha = pow(h, q);
    const double Rdiv2 = R_DIV * R_DIV;
    const double Rbnd2 = R_BOUND * R_BOUND;
    const double eps2 = EPS_EQ * EPS_EQ;

    double eq[9];
    chua_equilibria_internal(eq);

    double *t = (double*)malloc((size_t)(N + 1) * sizeof(double));
    double *x = (double*)malloc((size_t)(N + 1) * sizeof(double));
    double *y = (double*)malloc((size_t)(N + 1) * sizeof(double));
    double *z = (double*)malloc((size_t)(N + 1) * sizeof(double));
    if (!t || !x || !y || !z) {
        free(t); free(x); free(y); free(z);
        return CLS_UNKNOWN;
    }

    EFORKCoeffs c = efork_coeffs(q);

    t[0] = 0.0;
    x[0] = x0;
    y[0] = y0;
    z[0] = z0;

    // Primer paso sin memoria
    double dx, dy, dz;
    rhs(x0, y0, z0, &dx, &dy, &dz);
    double K1x = ha * dx, K1y = ha * dy, K1z = ha * dz;

    double x2 = x0 + c.a21*K1x;
    double y2 = y0 + c.a21*K1y;
    double z2 = z0 + c.a21*K1z;
    rhs(x2, y2, z2, &dx, &dy, &dz);
    double K2x = ha * dx, K2y = ha * dy, K2z = ha * dz;

    double x3 = x0 + c.a31*K1x + c.a32*K2x;
    double y3 = y0 + c.a31*K1y + c.a32*K2y;
    double z3 = z0 + c.a31*K1z + c.a32*K2z;
    rhs(x3, y3, z3, &dx, &dy, &dz);
    double K3x = ha * dx, K3y = ha * dy, K3z = ha * dz;

    x[1] = x0 + c.w1*K1x + c.w2*K2x + c.w3*K3x;
    y[1] = y0 + c.w1*K1y + c.w2*K2y + c.w3*K3y;
    z[1] = z0 + c.w1*K1z + c.w2*K2z + c.w3*K3z;
    t[1] = h;

    int hit0 = 0, hitp = 0, hitm = 0;
    double mean_x = 0.0;
    double mean_r2 = 0.0;
    int cnt_tail = 0;

    if (nrm2(x[1], y[1], z[1]) > Rdiv2) {
        free(t); free(x); free(y); free(z);
        return CLS_DIV;
    }

    for (int n = 1; n < N; ++n) {
        const double tn = n * h;

        const double mem_x = memory_fractional_scalar(n, tn, x, t, h, q, nu);
        const double mem_y = memory_fractional_scalar(n, tn, y, t, h, q, nu);
        const double mem_z = memory_fractional_scalar(n, tn, z, t, h, q, nu);

        rhs(x[n], y[n], z[n], &dx, &dy, &dz);
        K1x = ha * (dx - mem_x);
        K1y = ha * (dy - mem_y);
        K1z = ha * (dz - mem_z);

        x2 = x[n] + c.a21*K1x;
        y2 = y[n] + c.a21*K1y;
        z2 = z[n] + c.a21*K1z;
        rhs(x2, y2, z2, &dx, &dy, &dz);
        K2x = ha * dx; K2y = ha * dy; K2z = ha * dz;

        x3 = x[n] + c.a31*K1x + c.a32*K2x;
        y3 = y[n] + c.a31*K1y + c.a32*K2y;
        z3 = z[n] + c.a31*K1z + c.a32*K2z;
        rhs(x3, y3, z3, &dx, &dy, &dz);
        K3x = ha * dx; K3y = ha * dy; K3z = ha * dz;

        x[n + 1] = x[n] + c.w1*K1x + c.w2*K2x + c.w3*K3x;
        y[n + 1] = y[n] + c.w1*K1y + c.w2*K2y + c.w3*K3y;
        z[n + 1] = z[n] + c.w1*K1z + c.w2*K2z + c.w3*K3z;
        t[n + 1] = (n + 1) * h;

        if (nrm2(x[n + 1], y[n + 1], z[n + 1]) > Rdiv2) {
            free(t); free(x); free(y); free(z);
            return CLS_DIV;
        }

        if (n >= burn) {
            cnt_tail++;
            const double r2 = nrm2(x[n + 1], y[n + 1], z[n + 1]);
            mean_x += (x[n + 1] - mean_x) / (double)cnt_tail;
            mean_r2 += (r2 - mean_r2) / (double)cnt_tail;

            const double d0 = (x[n + 1] - eq[0])*(x[n + 1] - eq[0])
                            + (y[n + 1] - eq[1])*(y[n + 1] - eq[1])
                            + (z[n + 1] - eq[2])*(z[n + 1] - eq[2]);
            const double dp = (x[n + 1] - eq[3])*(x[n + 1] - eq[3])
                            + (y[n + 1] - eq[4])*(y[n + 1] - eq[4])
                            + (z[n + 1] - eq[5])*(z[n + 1] - eq[5]);
            const double dm = (x[n + 1] - eq[6])*(x[n + 1] - eq[6])
                            + (y[n + 1] - eq[7])*(y[n + 1] - eq[7])
                            + (z[n + 1] - eq[8])*(z[n + 1] - eq[8]);

            hit0 = (d0 <= eps2) ? (hit0 + 1) : 0;
            hitp = (dp <= eps2) ? (hitp + 1) : 0;
            hitm = (dm <= eps2) ? (hitm + 1) : 0;

            if (hit0 >= CAP_WIN || hitp >= CAP_WIN || hitm >= CAP_WIN) {
                free(t); free(x); free(y); free(z);
                return CLS_EQ;
            }
        }
    }

    free(t); free(x); free(y); free(z);

    if (cnt_tail <= 0) return CLS_UNKNOWN;
    if (mean_r2 < Rbnd2) {
        if (mean_x >  MEAN_X_GAP) return CLS_HIDDEN_POS;
        if (mean_x < -MEAN_X_GAP) return CLS_HIDDEN_NEG;
        return CLS_UNKNOWN;
    }
    return CLS_DIV;
}

// ============================================================
// API pública
// ============================================================
API_EXPORT void set_chua_params(double alpha, double beta, double gamma, double m0, double m1) {
    G_ALPHA = alpha;
    G_BETA  = beta;
    G_GAMMA = gamma;
    G_M0    = m0;
    G_M1    = m1;
}

API_EXPORT void set_chua_model(int model) {
    G_MODEL = (model == 1) ? 1 : 0;
}

API_EXPORT void set_chua_arctan_params(double a1, double a2, double rho) {
    G_A1 = a1;
    G_A2 = a2;
    G_RHO = (rho > 0.0) ? rho : 1.0;
}

API_EXPORT void set_basin_workers(int workers) {
    G_WORKERS = (workers > 0) ? workers : 0;
}

API_EXPORT void get_chua_params(double *out5) {
    if (!out5) return;
    out5[0] = G_ALPHA;
    out5[1] = G_BETA;
    out5[2] = G_GAMMA;
    out5[3] = G_M0;
    out5[4] = G_M1;
}

API_EXPORT void get_equilibria(double *eq_out9) {
    if (!eq_out9) return;
    chua_equilibria_internal(eq_out9);
}

API_EXPORT int classify_basin_point(
    double x0,
    double y0,
    double z0,
    double q,
    double h,
    double Lm,
    double TMAX,
    double TBURN,
    double R_DIV,
    double R_BOUND,
    double EPS_EQ,
    int CAP_WIN,
    double MEAN_X_GAP
) {
    if (!(q > 0.0 && q <= 1.0)) return -3;
    if (!(h > 0.0 && Lm > 0.0 && TMAX > 0.0 && TBURN >= 0.0)) return -4;
    return classify_point(
        x0, y0, z0, q, h, Lm, TMAX, TBURN,
        R_DIV, R_BOUND, EPS_EQ, CAP_WIN, MEAN_X_GAP
    );
}

API_EXPORT int compute_basin_xy(
    int nx,
    int ny,
    double xmin,
    double xmax,
    double ymin,
    double ymax,
    double z0,
    double q,
    double h,
    double Lm,
    double TMAX,
    double TBURN,
    double R_DIV,
    double R_BOUND,
    double EPS_EQ,
    int CAP_WIN,
    double MEAN_X_GAP,
    int *out_classes
) {
    if (nx <= 1 || ny <= 1 || !out_classes) return -1;
    if (!(xmax > xmin) || !(ymax > ymin)) return -2;
    if (!(q > 0.0 && q <= 1.0)) return -3;
    if (!(h > 0.0 && Lm > 0.0 && TMAX > 0.0 && TBURN >= 0.0)) return -4;

    const double dx = (xmax - xmin) / (double)(nx - 1);
    const double dy = (ymax - ymin) / (double)(ny - 1);

    int rows_done = 0;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(G_WORKERS > 0 ? G_WORKERS : omp_get_max_threads())
    #endif
    for (int i = 0; i < ny; ++i) {
        const double y = ymin + (double)i * dy;
        for (int j = 0; j < nx; ++j) {
            const double x = xmin + (double)j * dx;
            out_classes[(size_t)i * (size_t)nx + (size_t)j] = classify_point(
                x, y, z0, q, h, Lm, TMAX, TBURN,
                R_DIV, R_BOUND, EPS_EQ, CAP_WIN, MEAN_X_GAP
            );
        }

        #ifdef _OPENMP
        #pragma omp atomic
        #endif
        rows_done++;

        if ((rows_done % 10) == 0 || rows_done == ny) {
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            {
                fprintf(stderr, "[basin C xy %4d/%4d] %5.1f%%\n", rows_done, ny,
                        100.0 * (double)rows_done / (double)ny);
                fflush(stderr);
            }
        }
    }
    return 0;
}

API_EXPORT int compute_basin_plane(
    int nx,
    int ny,
    double umin,
    double umax,
    double vmin,
    double vmax,
    double fixed,
    int plane,
    double q,
    double h,
    double Lm,
    double TMAX,
    double TBURN,
    double R_DIV,
    double R_BOUND,
    double EPS_EQ,
    int CAP_WIN,
    double MEAN_X_GAP,
    int *out_classes
) {
    if (nx <= 1 || ny <= 1 || !out_classes) return -1;
    if (!(umax > umin) || !(vmax > vmin)) return -2;
    if (!(q > 0.0 && q <= 1.0)) return -3;
    if (!(h > 0.0 && Lm > 0.0 && TMAX > 0.0 && TBURN >= 0.0)) return -4;
    if (plane < 0 || plane > 2) return -5;

    const double du = (umax - umin) / (double)(nx - 1);
    const double dv = (vmax - vmin) / (double)(ny - 1);

    int rows_done = 0;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(G_WORKERS > 0 ? G_WORKERS : omp_get_max_threads())
    #endif
    for (int i = 0; i < ny; ++i) {
        const double v = vmin + (double)i * dv;
        for (int j = 0; j < nx; ++j) {
            const double u = umin + (double)j * du;
            double x0, y0, z0;
            if (plane == 0) {
                x0 = u; y0 = v; z0 = fixed;
            } else if (plane == 1) {
                x0 = u; y0 = fixed; z0 = v;
            } else {
                x0 = fixed; y0 = u; z0 = v;
            }
            out_classes[(size_t)i * (size_t)nx + (size_t)j] = classify_point(
                x0, y0, z0, q, h, Lm, TMAX, TBURN,
                R_DIV, R_BOUND, EPS_EQ, CAP_WIN, MEAN_X_GAP
            );
        }

        #ifdef _OPENMP
        #pragma omp atomic
        #endif
        rows_done++;

        if ((rows_done % 10) == 0 || rows_done == ny) {
            #ifdef _OPENMP
            #pragma omp critical
            #endif
            {
                const char *names[3] = {"xy", "xz", "yz"};
                fprintf(stderr, "[basin C %s %4d/%4d] %5.1f%%\n", names[plane], rows_done, ny,
                        100.0 * (double)rows_done / (double)ny);
                fflush(stderr);
            }
        }
    }
    return 0;
}
