#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*
chua_frac_lyapunov_efork_benettin.c

Cálculo operacional del espectro de Lyapunov para el sistema de Chua fraccionario
(con derivada de Caputo y memoria truncada) usando:

1) Integración del sistema base + sistema variacional linealizado
2) Esquema EFORK-3 explícito con memoria truncada
3) Reortogonalización tipo Benettin / Gram-Schmidt por bloques
4) Acumulación estándar por logaritmos de los factores de estiramiento

Referencias conceptuales:
- Benettin et al. (1980), Meccanica 15: método estándar de espectro LE para ODEs.
- Skokos (2010): revisión del método estándar, variacionales + Gram-Schmidt.

IMPORTANTE
----------
Este archivo implementa una EXTENSIÓN OPERACIONAL al caso fraccionario con memoria truncada.
No pretende ser una prueba teórica completa para el caso Caputo. La parte estándar es Benettin.
La parte novedosa/práctica aquí es:

(A) integrar el sistema aumentado (estado + 3 vectores tangentes) con EFORK-3,
(B) aplicar la misma transformación lineal M = R^{-1} no solo al instante actual, sino
    a toda la ventana de memoria truncada de los vectores tangentes, de modo que la
    renormalización sea consistente con la no localidad del integrador de memoria corta.

Esto permite enchufar el cálculo LE al pipeline ya existente de Chua fraccionario.
*/

typedef struct {
    double alpha_chua;
    double beta;
    double gamma_chua;
    double m0;
    double m1;
    double q;
    double h;
    double Lm;
    double t_burn;
    int    n_blocks;
    double t_block;
} Params;

typedef struct {
    double g1, g2, g3;
    double w1, w2, w3;
    double a21, a31, a32;
} EFORK3;

static double sat_scalar(double x) {
    if (x < -1.0) return -1.0;
    if (x >  1.0) return  1.0;
    return x;
}

static void chua_rhs(const double *x, double *f, const Params *p) {
    const double x1 = x[0], x2 = x[1], x3 = x[2];
    /* f_chua(x)=m1*x+psi(x), psi(x)=(m0-m1)*sat(x); p->q es frac_order. */
    const double phi = p->m1 * x1 + (p->m0 - p->m1) * sat_scalar(x1);
    f[0] = p->alpha_chua * (x2 - x1 - phi);
    f[1] = x1 - x2 + x3;
    f[2] = -(p->beta * x2 + p->gamma_chua * x3);
}

static double chua_phi_prime(double x, const Params *p) {
    /* Derivada por tramos; en |x|=1 se toma la rama exterior por conveniencia numérica. */
    if (x > -1.0 && x < 1.0) return p->m0;
    return p->m1;
}

static void jacobian_apply(const double *x, const double *v, double *Jv, const Params *p) {
    const double dphi = chua_phi_prime(x[0], p);
    Jv[0] = p->alpha_chua * (v[1] - v[0] - dphi * v[0]);
    Jv[1] = v[0] - v[1] + v[2];
    Jv[2] = -(p->beta * v[1] + p->gamma_chua * v[2]);
}

static void rhs_augmented(const double *u, double *f, const Params *p) {
    /* u = [x(3), d1(3), d2(3), d3(3)] */
    chua_rhs(u, f, p);
    jacobian_apply(u, u + 3, f + 3, p);
    jacobian_apply(u, u + 6, f + 6, p);
    jacobian_apply(u, u + 9, f + 9, p);
}

static EFORK3 efork3_coeffs(double q) {
    EFORK3 c;
    c.g1 = tgamma(1.0 + q);
    c.g2 = tgamma(1.0 + 2.0 * q);
    c.g3 = tgamma(1.0 + 3.0 * q);

    c.a21 = 1.0 / (2.0 * c.g1 * c.g1);
    c.a31 = ((c.g1*c.g1) * c.g2 + 2.0 * (c.g2*c.g2) - c.g3) /
            (4.0 * (c.g1*c.g1) * (2.0 * (c.g2*c.g2) - c.g3));
    c.a32 = -c.g2 / (4.0 * (2.0 * (c.g2*c.g2) - c.g3));

    c.w1 = (8.0 * (c.g1*c.g1*c.g1) * (c.g2*c.g2) - 6.0 * (c.g1*c.g1*c.g1) * c.g3 + c.g2*c.g3) /
           (c.g1 * c.g2 * c.g3);
    c.w2 = 2.0 * (c.g1*c.g1) * (4.0 * (c.g2*c.g2) - c.g3) / (c.g2 * c.g3);
    c.w3 = -8.0 * (c.g1*c.g1) * (2.0 * (c.g2*c.g2) - c.g3) / (c.g2 * c.g3);
    return c;
}

static double memory_component(const double *arr, const double *tgrid,
                               int k, double t, double h, double q, int nu) {
    int start = k - nu;
    if (start < 0) start = 0;
    double s = 0.0;
    const double gamma_term = tgamma(2.0 - q);
    for (int j = start; j < k; ++j) {
        double t0 = tgrid[j];
        double t1 = tgrid[j + 1];
        double v1 = pow(t - t0, 1.0 - q);
        double v2 = pow(t - t1, 1.0 - q);
        s += (arr[j + 1] - arr[j]) * (v1 - v2);
    }
    return s / (h * gamma_term);
}

static void copy_vector(double *dst, const double *src, int n) {
    for (int i = 0; i < n; ++i) dst[i] = src[i];
}

static void qr_gram_schmidt_3x3(const double Yin[3][3], double Q[3][3], double R[3][3]) {
    memset(R, 0, 9 * sizeof(double));
    double v[3];

    /* col 0 */
    for (int i = 0; i < 3; ++i) v[i] = Yin[i][0];
    R[0][0] = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (R[0][0] < 1e-300) R[0][0] = 1e-300;
    for (int i = 0; i < 3; ++i) Q[i][0] = v[i] / R[0][0];

    /* col 1 */
    for (int i = 0; i < 3; ++i) v[i] = Yin[i][1];
    R[0][1] = Q[0][0]*v[0] + Q[1][0]*v[1] + Q[2][0]*v[2];
    for (int i = 0; i < 3; ++i) v[i] -= R[0][1] * Q[i][0];
    R[1][1] = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (R[1][1] < 1e-300) R[1][1] = 1e-300;
    for (int i = 0; i < 3; ++i) Q[i][1] = v[i] / R[1][1];

    /* col 2 */
    for (int i = 0; i < 3; ++i) v[i] = Yin[i][2];
    R[0][2] = Q[0][0]*v[0] + Q[1][0]*v[1] + Q[2][0]*v[2];
    for (int i = 0; i < 3; ++i) v[i] -= R[0][2] * Q[i][0];
    R[1][2] = Q[0][1]*v[0] + Q[1][1]*v[1] + Q[2][1]*v[2];
    for (int i = 0; i < 3; ++i) v[i] -= R[1][2] * Q[i][1];
    R[2][2] = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (R[2][2] < 1e-300) R[2][2] = 1e-300;
    for (int i = 0; i < 3; ++i) Q[i][2] = v[i] / R[2][2];
}

static void invert_upper_triangular_3x3(const double R[3][3], double M[3][3]) {
    memset(M, 0, 9 * sizeof(double));
    M[2][2] = 1.0 / R[2][2];
    M[1][1] = 1.0 / R[1][1];
    M[0][0] = 1.0 / R[0][0];

    M[1][2] = -R[1][2] / (R[1][1] * R[2][2]);
    M[0][1] = -R[0][1] / (R[0][0] * R[1][1]);
    M[0][2] = (R[0][1]*R[1][2] - R[0][2]*R[1][1]) / (R[0][0] * R[1][1] * R[2][2]);
}

static void right_multiply_history_window(double **hist, int start_idx, int end_idx, const double M[3][3]) {
    /* hist[c][n], c=0..11. Solo transforma columnas tangentes (3..11) para n in [start_idx, end_idx]. */
    for (int n = start_idx; n <= end_idx; ++n) {
        double Y[3][3], Ynew[3][3];
        for (int i = 0; i < 3; ++i) {
            Y[i][0] = hist[3 + i][n];
            Y[i][1] = hist[6 + i][n];
            Y[i][2] = hist[9 + i][n];
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Ynew[i][j] = Y[i][0]*M[0][j] + Y[i][1]*M[1][j] + Y[i][2]*M[2][j];
            }
        }
        for (int i = 0; i < 3; ++i) {
            hist[3 + i][n] = Ynew[i][0];
            hist[6 + i][n] = Ynew[i][1];
            hist[9 + i][n] = Ynew[i][2];
        }
    }
}

static void run_fractional_le(const double x_init[3], const Params *p, const char *csv_path) {
    const int D = 12;
    const int burn_steps = (int) llround(p->t_burn / p->h);
    const int block_steps = (int) llround(p->t_block / p->h);
    const int total_steps = burn_steps + p->n_blocks * block_steps;
    const int nu = (int) ceil(p->Lm / p->h);
    const double hq = pow(p->h, p->q);
    const EFORK3 c = efork3_coeffs(p->q);

    double *tgrid = (double*) calloc((size_t)total_steps + 1, sizeof(double));
    double **hist = (double**) calloc(D, sizeof(double*));
    for (int i = 0; i < D; ++i) hist[i] = (double*) calloc((size_t)total_steps + 1, sizeof(double));
    if (!tgrid || !hist) {
        fprintf(stderr, "No se pudo asignar memoria.\n");
        exit(1);
    }

    tgrid[0] = 0.0;
    hist[0][0] = x_init[0];
    hist[1][0] = x_init[1];
    hist[2][0] = x_init[2];
    /* Base tangente identidad */
    hist[3][0] = 1.0; hist[4][0] = 0.0; hist[5][0] = 0.0;
    hist[6][0] = 0.0; hist[7][0] = 1.0; hist[8][0] = 0.0;
    hist[9][0] = 0.0; hist[10][0] = 0.0; hist[11][0] = 1.0;

    double u[D], rhs1[D], rhs2[D], rhs3[D];
    double mem[D], tmp[D];
    double sum_logs[3] = {0.0, 0.0, 0.0};

    FILE *fcsv = NULL;
    if (csv_path && strlen(csv_path) > 0) {
        fcsv = fopen(csv_path, "w");
        if (!fcsv) {
            fprintf(stderr, "No se pudo abrir CSV de convergencia: %s\n", csv_path);
            exit(1);
        }
        fprintf(fcsv, "block,time,lambda1,lambda2,lambda3,stretch1,stretch2,stretch3\n");
    }

    for (int n = 0; n < total_steps; ++n) {
        double t = n * p->h;
        tgrid[n+1] = (n + 1) * p->h;

        for (int i = 0; i < D; ++i) u[i] = hist[i][n];

        /* memoria para K1 */
        for (int i = 0; i < D; ++i) mem[i] = memory_component(hist[i], tgrid, n, t, p->h, p->q, nu);

        rhs_augmented(u, rhs1, p);
        for (int i = 0; i < D; ++i) rhs1[i] -= mem[i];
        for (int i = 0; i < D; ++i) tmp[i] = u[i] + c.a21 * hq * rhs1[i];

        rhs_augmented(tmp, rhs2, p);
        for (int i = 0; i < D; ++i) tmp[i] = u[i] + c.a31 * hq * rhs2[i] + c.a32 * hq * rhs1[i];

        rhs_augmented(tmp, rhs3, p);

        for (int i = 0; i < D; ++i) {
            hist[i][n+1] = u[i] + hq * (c.w1 * rhs1[i] + c.w2 * rhs2[i] + c.w3 * rhs3[i]);
        }

        /* Ortonormalización cada bloque, después del burn-in */
        if ((n + 1) > burn_steps && ((n + 1 - burn_steps) % block_steps == 0)) {
            int block_id = (n + 1 - burn_steps) / block_steps;
            double Y[3][3], Q[3][3], R[3][3], M[3][3];

            for (int i = 0; i < 3; ++i) {
                Y[i][0] = hist[3 + i][n+1];
                Y[i][1] = hist[6 + i][n+1];
                Y[i][2] = hist[9 + i][n+1];
            }

            qr_gram_schmidt_3x3(Y, Q, R);
            invert_upper_triangular_3x3(R, M);

            /* Benettin estándar: acumular log de los factores de estiramiento */
            sum_logs[0] += log(fabs(R[0][0]));
            sum_logs[1] += log(fabs(R[1][1]));
            sum_logs[2] += log(fabs(R[2][2]));

            /* Transformar toda la ventana de memoria de los vectores tangentes */
            int start_idx = (n + 1) - nu;
            if (start_idx < 0) start_idx = 0;
            right_multiply_history_window(hist, start_idx, n + 1, M);

            /* Asegurar que en el instante actual quede exactamente Q */
            for (int i = 0; i < 3; ++i) {
                hist[3 + i][n+1] = Q[i][0];
                hist[6 + i][n+1] = Q[i][1];
                hist[9 + i][n+1] = Q[i][2];
            }

            double Tacc = block_id * p->t_block;
            double l1 = sum_logs[0] / Tacc;
            double l2 = sum_logs[1] / Tacc;
            double l3 = sum_logs[2] / Tacc;
            if (fcsv) {
                fprintf(fcsv, "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                        block_id, Tacc, l1, l2, l3, fabs(R[0][0]), fabs(R[1][1]), fabs(R[2][2]));
            }
        }
    }

    double Tfinal = p->n_blocks * p->t_block;
    printf("# LE_frac_standard %.17g %.17g %.17g\n", sum_logs[0]/Tfinal, sum_logs[1]/Tfinal, sum_logs[2]/Tfinal);
    printf("# final_state %.17g %.17g %.17g\n", hist[0][total_steps], hist[1][total_steps], hist[2][total_steps]);

    if (fcsv) fclose(fcsv);
    for (int i = 0; i < D; ++i) free(hist[i]);
    free(hist);
    free(tgrid);
}

int main(int argc, char *argv[]) {
    if (argc != 15) {
        fprintf(stderr,
                "Uso: %s x0 y0 z0 alpha_chua beta gamma_chua m0 m1 q h Lm t_burn n_blocks t_block [NO: csv va fijo por env CHUA_LE_CSV]\n",
                argv[0]);
        fprintf(stderr,
                "Ejemplo: %s 5.85176778548633 0.370408600306164 -8.36097293442065 8.4562 12.0732 0.0052 -0.1768 -1.1468 0.9998 0.01 20 100 500 0.5\n",
                argv[0]);
        return 1;
    }

    Params p;
    double x0[3];
    x0[0] = strtod(argv[1], NULL);
    x0[1] = strtod(argv[2], NULL);
    x0[2] = strtod(argv[3], NULL);
    p.alpha_chua = strtod(argv[4], NULL);
    p.beta = strtod(argv[5], NULL);
    p.gamma_chua = strtod(argv[6], NULL);
    p.m0 = strtod(argv[7], NULL);
    p.m1 = strtod(argv[8], NULL);
    p.q = strtod(argv[9], NULL);
    if (!(p.q > 0.0 && p.q <= 1.0)) {
        fprintf(stderr, "El orden fraccionario q debe cumplir 0 < q <= 1.\n");
        return 2;
    }
    p.h = strtod(argv[10], NULL);
    p.Lm = strtod(argv[11], NULL);
    p.t_burn = strtod(argv[12], NULL);
    p.n_blocks = (int) strtol(argv[13], NULL, 10);
    p.t_block = strtod(argv[14], NULL);

    const char *csv_path = getenv("CHUA_LE_CSV");
    if (!csv_path) csv_path = "chua_frac_le_convergence.csv";

    run_fractional_le(x0, &p, csv_path);
    return 0;
}
