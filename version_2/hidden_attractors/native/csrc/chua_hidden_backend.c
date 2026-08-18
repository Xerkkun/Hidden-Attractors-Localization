#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include "native_validation.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define CLS_EQ 0
#define CLS_DIV 1
#define CLS_TARGET 2
#define CLS_OTHER 3
#define CLS_UNKNOWN 4
#define CLS_NONE (-1)

static const char *CLASS_NAMES[] = {"EQ", "DIV", "TARGET", "OTHER", "UNKNOWN"};

typedef struct { double alpha_chua, beta, gamma_chua, m0, m1, a1, a2, rho; int model; } ChuaParams;
typedef struct {
    double alpha_frac, g1, g2, g3, w1, w2, w3, a21, a31, a32, inv_mem_factor;
} EFORKCoeffs;
typedef struct { double h, Lm, TMAX_REF, TMAX_TEST, TBURN_REF, TBURN_TEST; } IntegrationCfg;
typedef struct {
    double R_DIV, EPS_EQ, SEC_TOL, HIT_FRAC_REQ;
    int CAP_WIN, MIN_SEC_MATCH, TEST_MAX_SEC;
} ThresholdCfg;
typedef struct { int n_radii, nsamples; double *radii; uint64_t random_seed; char eq_filter[64]; } SamplingCfg;
typedef struct { char csv_out[512], ref_out[512], summary_csv_out[512]; } FileCfg;
typedef struct { int n_eq; char names[3][3]; double val[3][3]; } EqSet;
typedef struct { double x0, y0, z0; int cls, sec_total, sec_hits; double hit_frac; } SampleRow;
typedef struct { int ref_points, total_target_hits; } RunStats;

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

static void die_errno(const char *msg) {
    fprintf(stderr, "%s: %s\n", msg, strerror(errno));
    exit(EXIT_FAILURE);
}

static const char *find_arg(int argc, char **argv, const char *key) {
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return NULL;
}

static int has_arg(int argc, char **argv, const char *key) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], key) == 0) return 1;
    }
    return 0;
}

static double parse_finite_text(const char *text, const char *key) {
    char *end = NULL;
    double value;
    if (!text || !*text) {
        fprintf(stderr, "Valor invalido para %s\n", key);
        exit(EXIT_FAILURE);
    }
    errno = 0;
    value = strtod(text, &end);
    if (errno == ERANGE || end == text || !end || *end != '\0' ||
        !hafo_isfinite(value)) {
        fprintf(stderr, "Valor invalido para %s: %s\n", key, text);
        exit(EXIT_FAILURE);
    }
    return value;
}

static double parse_double_arg(int argc, char **argv, const char *key) {
    const char *text = find_arg(argc, argv, key);
    if (!text) {
        fprintf(stderr, "Falta argumento %s\n", key);
        exit(EXIT_FAILURE);
    }
    return parse_finite_text(text, key);
}

static int parse_int_arg(int argc, char **argv, const char *key) {
    const char *text = find_arg(argc, argv, key);
    char *end = NULL;
    long value;
    if (!text) {
        fprintf(stderr, "Falta argumento %s\n", key);
        exit(EXIT_FAILURE);
    }
    errno = 0;
    value = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || !end || *end != '\0' ||
        value < INT_MIN || value > INT_MAX) {
        fprintf(stderr, "Valor invalido para %s: %s\n", key, text);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

static uint64_t parse_u64_arg(int argc, char **argv, const char *key) {
    const char *text = find_arg(argc, argv, key);
    char *end = NULL;
    unsigned long long value;
    if (!text) {
        fprintf(stderr, "Falta argumento %s\n", key);
        exit(EXIT_FAILURE);
    }
    if (*text == '-') {
        fprintf(stderr, "Valor invalido para %s: %s\n", key, text);
        exit(EXIT_FAILURE);
    }
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || !end || *end != '\0' ||
        value > (unsigned long long)UINT64_MAX) {
        fprintf(stderr, "Valor invalido para %s: %s\n", key, text);
        exit(EXIT_FAILURE);
    }
    return (uint64_t)value;
}

static void parse_vec3_arg(
    int argc, char **argv, const char *key, double out[3]
) {
    const char *text = find_arg(argc, argv, key);
    char *buffer;
    char *save = NULL;
    char *token;
    if (!text) {
        fprintf(stderr, "Falta argumento %s\n", key);
        exit(EXIT_FAILURE);
    }
    buffer = strdup(text);
    if (!buffer) die_errno("No se pudo reservar memoria para vec3");
    token = strtok_r(buffer, ",", &save);
    for (int i = 0; i < 3; ++i) {
        if (!token) {
            free(buffer);
            fprintf(stderr, "Se esperaban exactamente 3 componentes en %s\n", key);
            exit(EXIT_FAILURE);
        }
        out[i] = parse_finite_text(token, key);
        token = strtok_r(NULL, ",", &save);
    }
    if (token) {
        free(buffer);
        fprintf(stderr, "Se esperaban exactamente 3 componentes en %s\n", key);
        exit(EXIT_FAILURE);
    }
    free(buffer);
}

static void parse_radii_arg(
    int argc, char **argv, const char *key, SamplingCfg *sampling
) {
    const char *text = find_arg(argc, argv, key);
    char *buffer;
    char *save = NULL;
    char *token;
    size_t count = 1u;
    size_t bytes = 0u;
    int index = 0;
    if (!text || !*text) die("Falta argumento --radii");
    buffer = strdup(text);
    if (!buffer) die_errno("No se pudo reservar memoria para radii");
    for (const char *cursor = text; *cursor; ++cursor) {
        if (*cursor == ',') {
            if (count >= (size_t)INT_MAX) {
                free(buffer);
                die("Demasiados radios");
            }
            ++count;
        }
    }
    if (!hafo_checked_mul_size(count, sizeof(double), &bytes)) {
        free(buffer);
        die("Tamano de radii fuera de rango");
    }
    sampling->radii = (double *)malloc(bytes);
    if (!sampling->radii) {
        free(buffer);
        die_errno("No se pudo reservar memoria para radii[]");
    }
    token = strtok_r(buffer, ",", &save);
    while (token) {
        const double radius = parse_finite_text(token, key);
        if (!(radius > 0.0)) {
            free(buffer);
            free(sampling->radii);
            sampling->radii = NULL;
            die("Todos los radios deben ser positivos");
        }
        sampling->radii[index++] = radius;
        token = strtok_r(NULL, ",", &save);
    }
    if ((size_t)index != count) {
        free(buffer);
        free(sampling->radii);
        sampling->radii = NULL;
        die("La lista de radios contiene componentes vacios");
    }
    sampling->n_radii = index;
    free(buffer);
}

static inline int parse_model_name(const char *name) {
    if (!name || strcmp(name, "piecewise") == 0 ||
        strcmp(name, "nonsmooth") == 0) {
        return 0;
    }
    if (strcmp(name, "arctan") == 0 || strcmp(name, "atan") == 0 ||
        strcmp(name, "smooth") == 0) {
        return 1;
    }
    fprintf(stderr, "Modelo invalido: %s\n", name);
    exit(EXIT_FAILURE);
}

static inline double optional_double_arg(
    int argc, char **argv, const char *key, double default_value
) {
    const char *text = find_arg(argc, argv, key);
    return text ? parse_finite_text(text, key) : default_value;
}
/* piecewise: f(x)=m1*x+psi(x), psi(x)=(m0-m1)*sat(x). */
static inline double f_chua_value(double x, const ChuaParams *p) {
    if (p->model == 1) return p->a1 * x + p->a2 * atan(p->rho * x);
    return p->m1 * x + 0.5 * (p->m0 - p->m1) *
           (fabs(x + 1.0) - fabs(x - 1.0));
}

static inline void chua_rhs_xyz(
    double x, double y, double z, const ChuaParams *p,
    double *dx, double *dy, double *dz
) {
    *dx = p->alpha_chua * (y - x - f_chua_value(x, p));
    *dy = x - y + z;
    *dz = -p->beta * y - p->gamma_chua * z;
}

static inline double xdot_func(
    double x, double y, const ChuaParams *p
) {
    return p->alpha_chua * (y - x - f_chua_value(x, p));
}

static void chua_equilibria(const ChuaParams *p, EqSet *eqs){
    eqs->n_eq=1;
    strcpy(eqs->names[0],"E0");
    eqs->val[0][0]=eqs->val[0][1]=eqs->val[0][2]=0.0;
    if(p->model==1){
        double coeff=1.0+p->a1-p->gamma_chua/(p->beta+p->gamma_chua);
        double prevx=1e-8, prev=coeff*prevx + p->a2*atan(p->rho*prevx);
        double xp=NAN;
        for(int i=1;i<=20000;++i){
            double x=100.0*(double)i/20000.0;
            double cur=coeff*x + p->a2*atan(p->rho*x);
            if(prev*cur<0.0){
                double lo=prevx, hi=x, flo=prev;
                for(int it=0;it<80;++it){
                    double mid=0.5*(lo+hi);
                    double fm=coeff*mid + p->a2*atan(p->rho*mid);
                    if(fabs(fm)<1e-14){ lo=hi=mid; break; }
                    if(flo*fm<=0.0) hi=mid; else { lo=mid; flo=fm; }
                }
                xp=0.5*(lo+hi);
                break;
            }
            prevx=x; prev=cur;
        }
        if(hafo_isfinite(xp)){
            double yp=p->gamma_chua/(p->beta+p->gamma_chua)*xp;
            double zp=-p->beta/(p->beta+p->gamma_chua)*xp;
            strcpy(eqs->names[1],"E+");
            strcpy(eqs->names[2],"E-");
            eqs->val[1][0]= xp; eqs->val[1][1]= yp; eqs->val[1][2]= zp;
            eqs->val[2][0]=-xp; eqs->val[2][1]=-yp; eqs->val[2][2]=-zp;
            eqs->n_eq=3;
        }
        return;
    }
    double A=p->m0-p->m1;
    double den=(p->beta+p->gamma_chua)*p->m1 + p->beta;
    if(fabs(den)<1e-14) return;
    double xp=-(p->beta+p->gamma_chua)*A/den;
    if(fabs(xp)>1.0){
        double fp=p->m1*xp + A;
        strcpy(eqs->names[1],"E+");
        strcpy(eqs->names[2],"E-");
        eqs->val[1][0]= xp;  eqs->val[1][1]= xp+fp;  eqs->val[1][2]= fp;
        eqs->val[2][0]=-xp;  eqs->val[2][1]=-(xp+fp); eqs->val[2][2]=-fp;
        eqs->n_eq=3;
    }
}

static EFORKCoeffs efork_coeffs(double alpha_frac, double h){
    EFORKCoeffs c;
    c.alpha_frac=alpha_frac;
    c.g1=tgamma(1.0+alpha_frac);
    c.g2=tgamma(1.0+2.0*alpha_frac);
    c.g3=tgamma(1.0+3.0*alpha_frac);
    c.a21=1.0/(2.0*c.g1*c.g1);
    c.a31=((c.g1*c.g1)*c.g2 + 2.0*(c.g2*c.g2) - c.g3)/(4.0*(c.g1*c.g1)*(2.0*(c.g2*c.g2)-c.g3));
    c.a32=-c.g2/(4.0*(2.0*(c.g2*c.g2)-c.g3));
    c.w1=(8.0*(c.g1*c.g1*c.g1)*(c.g2*c.g2)-6.0*(c.g1*c.g1*c.g1)*c.g3 + c.g2*c.g3)/(c.g1*c.g2*c.g3);
    c.w2=2.0*(c.g1*c.g1)*(4.0*(c.g2*c.g2)-c.g3)/(c.g2*c.g3);
    c.w3=-8.0*(c.g1*c.g1)*(2.0*(c.g2*c.g2)-c.g3)/(c.g2*c.g3);
    c.inv_mem_factor=1.0/(h*tgamma(2.0-alpha_frac));
    return c;
}
static inline double memory_fractional_scalar(
    int k,
    double time,
    const double *values,
    const double *times,
    int memory_steps,
    const EFORKCoeffs *coefficients
) {
    int start = k - memory_steps;
    double sum = 0.0;
    const double exponent = 1.0 - coefficients->alpha_frac;
    if (start < 0) start = 0;
    for (int index = start; index < k; ++index) {
        const double upper = pow(time - times[index], exponent);
        const double lower = pow(time - times[index + 1], exponent);
        sum += (values[index + 1] - values[index]) * (upper - lower);
    }
    return sum * coefficients->inv_mem_factor;
}

static int efork_chua_caputo_steps(
    const ChuaParams *p,
    const double x0[3],
    double alpha_frac,
    double h,
    int steps,
    int memory_steps,
    double **t_out,
    double **X_out,
    int *N_out
) {
    const EFORKCoeffs coef = efork_coeffs(alpha_frac, h);
    const size_t point_count = (size_t)steps + 1u;
    size_t time_bytes = 0u;
    size_t state_values = 0u;
    size_t state_bytes = 0u;
    double *t = NULL;
    double *x = NULL;
    double *y = NULL;
    double *z = NULL;
    double *states = NULL;
    const double ha = pow(h, alpha_frac);
    double xn = x0[0];
    double yn = x0[1];
    double zn = x0[2];
    double dx;
    double dy;
    double dz;
    double K1x;
    double K1y;
    double K1z;
    double K2x;
    double K2y;
    double K2z;
    double K3x;
    double K3y;
    double K3z;
    double xn1;
    double yn1;
    double zn1;

    if (!hafo_checked_mul_size(point_count, sizeof(double), &time_bytes) ||
        !hafo_checked_mul_size(point_count, 3u, &state_values) ||
        !hafo_checked_mul_size(state_values, sizeof(double), &state_bytes)) {
        return 0;
    }
    (void)time_bytes;
    t = (double *)calloc(point_count, sizeof(double));
    x = (double *)calloc(point_count, sizeof(double));
    y = (double *)calloc(point_count, sizeof(double));
    z = (double *)calloc(point_count, sizeof(double));
    if (!t || !x || !y || !z) {
        free(t);
        free(x);
        free(y);
        free(z);
        return 0;
    }

    t[0] = 0.0;
    x[0] = xn;
    y[0] = yn;
    z[0] = zn;
    chua_rhs_xyz(xn, yn, zn, p, &dx, &dy, &dz);
    K1x = ha * dx;
    K1y = ha * dy;
    K1z = ha * dz;
    chua_rhs_xyz(
        xn + coef.a21 * K1x,
        yn + coef.a21 * K1y,
        zn + coef.a21 * K1z,
        p,
        &dx,
        &dy,
        &dz
    );
    K2x = ha * dx;
    K2y = ha * dy;
    K2z = ha * dz;
    chua_rhs_xyz(
        xn + coef.a31 * K1x + coef.a32 * K2x,
        yn + coef.a31 * K1y + coef.a32 * K2y,
        zn + coef.a31 * K1z + coef.a32 * K2z,
        p,
        &dx,
        &dy,
        &dz
    );
    K3x = ha * dx;
    K3y = ha * dy;
    K3z = ha * dz;
    xn1 = xn + coef.w1 * K1x + coef.w2 * K2x + coef.w3 * K3x;
    yn1 = yn + coef.w1 * K1y + coef.w2 * K2y + coef.w3 * K3y;
    zn1 = zn + coef.w1 * K1z + coef.w2 * K2z + coef.w3 * K3z;
    t[1] = h;
    x[1] = xn1;
    y[1] = yn1;
    z[1] = zn1;
    xn = xn1;
    yn = yn1;
    zn = zn1;

    for (int n = 1; n < steps; ++n) {
        const double tn = (double)n * h;
        const double mem_x = memory_fractional_scalar(
            n, tn, x, t, memory_steps, &coef
        );
        const double mem_y = memory_fractional_scalar(
            n, tn, y, t, memory_steps, &coef
        );
        const double mem_z = memory_fractional_scalar(
            n, tn, z, t, memory_steps, &coef
        );
        chua_rhs_xyz(xn, yn, zn, p, &dx, &dy, &dz);
        K1x = ha * (dx - mem_x);
        K1y = ha * (dy - mem_y);
        K1z = ha * (dz - mem_z);
        chua_rhs_xyz(
            xn + coef.a21 * K1x,
            yn + coef.a21 * K1y,
            zn + coef.a21 * K1z,
            p,
            &dx,
            &dy,
            &dz
        );
        K2x = ha * dx;
        K2y = ha * dy;
        K2z = ha * dz;
        chua_rhs_xyz(
            xn + coef.a31 * K1x + coef.a32 * K2x,
            yn + coef.a31 * K1y + coef.a32 * K2y,
            zn + coef.a31 * K1z + coef.a32 * K2z,
            p,
            &dx,
            &dy,
            &dz
        );
        K3x = ha * dx;
        K3y = ha * dy;
        K3z = ha * dz;
        xn1 = xn + coef.w1 * K1x + coef.w2 * K2x + coef.w3 * K3x;
        yn1 = yn + coef.w1 * K1y + coef.w2 * K2y + coef.w3 * K3y;
        zn1 = zn + coef.w1 * K1z + coef.w2 * K2z + coef.w3 * K3z;
        t[n + 1] = (double)(n + 1) * h;
        x[n + 1] = xn1;
        y[n + 1] = yn1;
        z[n + 1] = zn1;
        xn = xn1;
        yn = yn1;
        zn = zn1;
    }

    states = (double *)malloc(state_bytes);
    if (!states) {
        free(t);
        free(x);
        free(y);
        free(z);
        return 0;
    }
    for (int i = 0; i <= steps; ++i) {
        states[3 * i + 0] = x[i];
        states[3 * i + 1] = y[i];
        states[3 * i + 2] = z[i];
    }
    free(x);
    free(y);
    free(z);
    *t_out = t;
    *X_out = states;
    *N_out = steps + 1;
    return 1;
}

static int efork_chua_caputo(
    const ChuaParams *p, double x0[3], double alpha_frac, double h,
    double t_final, double Lm, double **t_out, double **X_out, int *N_out
) {
    int steps = 0;
    int memory_steps = 0;
    size_t state_values = 0u;
    const double parameters[8] = {
        p ? p->alpha_chua : 0.0, p ? p->beta : 0.0,
        p ? p->gamma_chua : 0.0, p ? p->m0 : 0.0,
        p ? p->m1 : 0.0, p ? p->a1 : 0.0,
        p ? p->a2 : 0.0, p ? p->rho : 0.0
    };
    if (!p || !x0 || !t_out || !X_out || !N_out ||
        (p->model != 0 && p->model != 1) ||
        !hafo_values_are_finite(parameters, 8u) ||
        !hafo_values_are_finite(x0, 3u) ||
        !(alpha_frac > 0.0 && alpha_frac <= 1.0) ||
        !hafo_isfinite(alpha_frac) ||
        !hafo_uniform_step_count(t_final, h, &steps) || steps < 1 ||
        !hafo_positive_ratio_ceil(Lm, h, &memory_steps) ||
        steps > (INT_MAX - 3) / 3 ||
        !hafo_checked_mul_size((size_t)steps + 1u, 3u, &state_values)) {
        return 0;
    }
    if (!efork_chua_caputo_steps(
            p, x0, alpha_frac, h, steps, memory_steps,
            t_out, X_out, N_out)) {
        return 0;
    }
    if (*N_out != steps + 1 ||
        !hafo_values_are_finite(*t_out, (size_t)*N_out) ||
        !hafo_values_are_finite(*X_out, state_values)) {
        free(*t_out);
        free(*X_out);
        *t_out = NULL;
        *X_out = NULL;
        *N_out = 0;
        return 0;
    }
    (*t_out)[steps] = t_final;
    return 1;
}

static int classify_equilibrium_or_divergence(
    const double *states,
    int sample_count,
    const EqSet *equilibria,
    const ThresholdCfg *thresholds
) {
    const double divergence_squared = thresholds->R_DIV * thresholds->R_DIV;
    const double equilibrium_squared = thresholds->EPS_EQ * thresholds->EPS_EQ;
    int hits[3] = {0, 0, 0};
    for (int index = 0; index < sample_count; ++index) {
        const double x = states[3 * index + 0];
        const double y = states[3 * index + 1];
        const double z = states[3 * index + 2];
        const double radius_squared = x * x + y * y + z * z;
        if (!hafo_isfinite(x) || !hafo_isfinite(y) || !hafo_isfinite(z) ||
            !hafo_isfinite(radius_squared) ||
            radius_squared > divergence_squared) {
            return CLS_DIV;
        }
        for (int equilibrium = 0; equilibrium < equilibria->n_eq; ++equilibrium) {
            const double dx = x - equilibria->val[equilibrium][0];
            const double dy = y - equilibria->val[equilibrium][1];
            const double dz = z - equilibria->val[equilibrium][2];
            const double distance_squared = dx * dx + dy * dy + dz * dz;
            if (distance_squared <= equilibrium_squared) {
                ++hits[equilibrium];
                if (hits[equilibrium] >= thresholds->CAP_WIN) return CLS_EQ;
            } else {
                hits[equilibrium] = 0;
            }
        }
    }
    return CLS_NONE;
}

static int section_points(
    const double *times,
    const double *states,
    int sample_count,
    const ChuaParams *parameters,
    double burn_time,
    int max_points,
    double *points
) {
    int start = 1;
    int count = 0;
    while (start < sample_count && times[start] < burn_time) ++start;
    for (int index = start; index < sample_count; ++index) {
        const double previous_x = states[3 * (index - 1) + 0];
        const double x = states[3 * index + 0];
        if (previous_x < 0.0 && x >= 0.0 &&
            xdot_func(states[3 * index], states[3 * index + 1], parameters) > 0.0) {
            const double fraction = -previous_x / (x - previous_x + 1.0e-30);
            points[2 * count + 0] = states[3 * (index - 1) + 1] +
                fraction * (states[3 * index + 1] - states[3 * (index - 1) + 1]);
            points[2 * count + 1] = states[3 * (index - 1) + 2] +
                fraction * (states[3 * index + 2] - states[3 * (index - 1) + 2]);
            ++count;
            if (count >= max_points) break;
        }
    }
    return count;
}

static inline double min_dist_to_ref(
    const double *reference, int reference_count, double y, double z
) {
    double best = INFINITY;
    if (reference_count <= 0) return best;
    for (int index = 0; index < reference_count; ++index) {
        const double dy = reference[2 * index + 0] - y;
        const double dz = reference[2 * index + 1] - z;
        const double squared = dy * dy + dz * dz;
        if (squared < best) best = squared;
    }
    return sqrt(best);
}

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t value = (*state += UINT64_C(0x9E3779B97F4A7C15));
    value = (value ^ (value >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}

static double rng_uniform01(uint64_t *state) {
    const uint64_t value = splitmix64_next(state);
    return (double)(value >> 11) * (1.0 / 9007199254740992.0);
}

static void sample_in_ball(
    const double center[3], double radius, uint64_t *state, double out[3]
) {
    for (;;) {
        const double ux = 2.0 * rng_uniform01(state) - 1.0;
        const double uy = 2.0 * rng_uniform01(state) - 1.0;
        const double uz = 2.0 * rng_uniform01(state) - 1.0;
        if (ux * ux + uy * uy + uz * uz <= 1.0) {
            out[0] = center[0] + radius * ux;
            out[1] = center[1] + radius * uy;
            out[2] = center[2] + radius * uz;
            return;
        }
    }
}

static uint64_t make_sample_seed(
    uint64_t base, int equilibrium, int radius, int sample
) {
    uint64_t state = base;
    state ^= UINT64_C(0xA24BAED4963EE407) * (uint64_t)(equilibrium + 1);
    state ^= UINT64_C(0x9FB21C651E98DF25) * (uint64_t)(radius + 1);
    state ^= UINT64_C(0xD6E8FEB86659FD93) * (uint64_t)(sample + 1);
    return state;
}

static int eq_filter_allows(const SamplingCfg *s,const char *name){
    const char *filter=s->eq_filter[0] ? s->eq_filter : getenv("HIDDEN_VERIFY_EQ_FILTER");
    if(!filter || !filter[0] || strcmp(filter,"all")==0 || strcmp(filter,"ALL")==0) return 1;
    char buf[64];
    const int written=snprintf(buf,sizeof(buf),"%s",filter);
    if(written<0 || (size_t)written>=sizeof(buf)) die("Filtro de equilibrios demasiado largo");
    char *save=NULL;
    char *tok=strtok_r(buf,",",&save);
    while(tok){
        while(*tok==' '||*tok=='\t') tok++;
        char *end=tok+strlen(tok);
        while(end>tok && (end[-1]==' '||end[-1]=='\t'||end[-1]=='\r'||end[-1]=='\n')) *--end='\0';
        if(strcmp(tok,name)==0) return 1;
        tok=strtok_r(NULL,",",&save);
    }
    return 0;
}

static int build_reference_section(
    const ChuaParams *parameters,
    const EqSet *equilibria,
    const IntegrationCfg *integration,
    const ThresholdCfg *thresholds,
    double fractional_order,
    const double target_seed[3],
    double **reference_out,
    int *reference_count_out
) {
    double *times = NULL;
    double *states = NULL;
    double *reference;
    int sample_count = 0;
    int classification;
    int reference_count;
    const int max_points = thresholds->TEST_MAX_SEC * 4;
    double seed[3] = {target_seed[0], target_seed[1], target_seed[2]};

    if (!efork_chua_caputo(
            parameters, seed, fractional_order, integration->h,
            integration->TMAX_REF, integration->Lm,
            &times, &states, &sample_count)) {
        die("No se pudo integrar la semilla de referencia");
    }
    classification = classify_equilibrium_or_divergence(
        states, sample_count, equilibria, thresholds
    );
    if (classification == CLS_EQ || classification == CLS_DIV) {
        free(times);
        free(states);
        return classification;
    }
    reference = (double *)malloc((size_t)max_points * 2u * sizeof(double));
    if (!reference) die_errno("No se pudo reservar memoria para la referencia");
    reference_count = section_points(
        times, states, sample_count, parameters,
        integration->TBURN_REF, max_points, reference
    );
    free(times);
    free(states);
    if (reference_count < thresholds->MIN_SEC_MATCH) {
        free(reference);
        return CLS_UNKNOWN;
    }
    *reference_out = reference;
    *reference_count_out = reference_count;
    return CLS_TARGET;
}

static int classify_to_target(
    const ChuaParams *parameters,
    const EqSet *equilibria,
    const IntegrationCfg *integration,
    const ThresholdCfg *thresholds,
    double fractional_order,
    const double *reference,
    int reference_count,
    const double x0[3],
    int *section_total,
    int *section_hits,
    double *hit_fraction
) {
    double *times = NULL;
    double *states = NULL;
    double *section;
    int sample_count = 0;
    int classification;
    int count;
    int hits = 0;
    double seed[3] = {x0[0], x0[1], x0[2]};

    if (!efork_chua_caputo(
            parameters, seed, fractional_order, integration->h,
            integration->TMAX_TEST, integration->Lm,
            &times, &states, &sample_count)) {
        die("No se pudo integrar una trayectoria de prueba");
    }
    classification = classify_equilibrium_or_divergence(
        states, sample_count, equilibria, thresholds
    );
    if (classification != CLS_NONE) {
        *section_total = 0;
        *section_hits = 0;
        *hit_fraction = 0.0;
        free(times);
        free(states);
        return classification;
    }
    section = (double *)malloc(
        (size_t)thresholds->TEST_MAX_SEC * 2u * sizeof(double)
    );
    if (!section) die_errno("No se pudo reservar memoria para la seccion");
    count = section_points(
        times, states, sample_count, parameters, integration->TBURN_TEST,
        thresholds->TEST_MAX_SEC, section
    );
    free(times);
    free(states);
    if (count < thresholds->MIN_SEC_MATCH) {
        *section_total = count;
        *section_hits = 0;
        *hit_fraction = 0.0;
        free(section);
        return CLS_UNKNOWN;
    }
    for (int index = 0; index < count; ++index) {
        const double distance = min_dist_to_ref(
            reference, reference_count,
            section[2 * index], section[2 * index + 1]
        );
        if (distance <= thresholds->SEC_TOL) ++hits;
    }
    *section_total = count;
    *section_hits = hits;
    *hit_fraction = (double)hits / (double)count;
    free(section);
    return *hit_fraction >= thresholds->HIT_FRAC_REQ ? CLS_TARGET : CLS_OTHER;
}

static void write_reference_csv(
    const char *path, const double *reference, int reference_count
) {
    FILE *file = fopen(path, "w");
    if (!file) die_errno("No se pudo abrir reference_section.csv");
    fprintf(file, "y,z\n");
    for (int index = 0; index < reference_count; ++index) {
        fprintf(
            file, "%.17g,%.17g\n",
            reference[2 * index], reference[2 * index + 1]
        );
    }
    fclose(file);
}

static void run_backend(
    const ChuaParams *parameters,
    double fractional_order,
    const double target_seed[3],
    const IntegrationCfg *integration,
    const ThresholdCfg *thresholds,
    const SamplingCfg *sampling,
    const FileCfg *files,
    RunStats *stats
) {
    EqSet equilibria;
    double *reference = NULL;
    int reference_count = 0;
    int total_target_hits = 0;
    int reference_class;
    FILE *samples_file;
    FILE *summary_file;
    const char *active_filter = sampling->eq_filter[0]
        ? sampling->eq_filter
        : getenv("HIDDEN_VERIFY_EQ_FILTER");

    chua_equilibria(parameters, &equilibria);
    printf("Equilibrios:\n");
    for (int index = 0; index < equilibria.n_eq; ++index) {
        printf(
            "%s = (%.10f, %.10f, %.10f)\n",
            equilibria.names[index], equilibria.val[index][0],
            equilibria.val[index][1], equilibria.val[index][2]
        );
    }
    printf(
        "\nSemilla objetivo = (%.10f, %.10f, %.10f)\n\n",
        target_seed[0], target_seed[1], target_seed[2]
    );
    if (active_filter && active_filter[0]) {
        printf("Filtro de equilibrios: %s\n\n", active_filter);
    }

    reference_class = build_reference_section(
        parameters, &equilibria, integration, thresholds,
        fractional_order, target_seed, &reference, &reference_count
    );
    if (reference_class != CLS_TARGET) {
        die(
            "No se pudo construir una referencia robusta del atractor "
            "objetivo. Ajusta tiempos, h, Lm o la semilla."
        );
    }
    stats->ref_points = reference_count;
    write_reference_csv(files->ref_out, reference, reference_count);
    printf(
        "Referencia construida con %d puntos de seccion.\n\n",
        reference_count
    );

    samples_file = fopen(files->csv_out, "w");
    if (!samples_file) die_errno("No se pudo abrir csv_out");
    fprintf(
        samples_file,
        "equilibrium,radius,sample_id,x0,y0,z0,class,"
        "sec_total,sec_hits,hit_frac\n"
    );
    summary_file = fopen(files->summary_csv_out, "w");
    if (!summary_file) die_errno("No se pudo abrir summary_csv_out");
    fprintf(
        summary_file, "equilibrium,radius,EQ,DIV,TARGET,OTHER,UNKNOWN\n"
    );

    for (int equilibrium = 0; equilibrium < equilibria.n_eq; ++equilibrium) {
        if (!eq_filter_allows(sampling, equilibria.names[equilibrium])) continue;
        printf(
            "=== Muestreo alrededor de %s ===\n",
            equilibria.names[equilibrium]
        );
        for (int radius_index = 0;
             radius_index < sampling->n_radii;
             ++radius_index) {
            const double radius = sampling->radii[radius_index];
            const int sample_count = sampling->nsamples;
            int counts[5] = {0, 0, 0, 0, 0};
            SampleRow *rows = (SampleRow *)calloc(
                (size_t)sample_count, sizeof(SampleRow)
            );
            if (!rows) die_errno("No se pudo reservar memoria para rows");
#ifdef _OPENMP
#pragma omp parallel for if(sample_count > 1)
#endif
            for (int sample = 0; sample < sample_count; ++sample) {
                uint64_t random_state = make_sample_seed(
                    sampling->random_seed, equilibrium, radius_index, sample
                );
                double x0[3];
                sample_in_ball(
                    equilibria.val[equilibrium], radius, &random_state, x0
                );
                rows[sample].x0 = x0[0];
                rows[sample].y0 = x0[1];
                rows[sample].z0 = x0[2];
                rows[sample].cls = classify_to_target(
                    parameters, &equilibria, integration, thresholds,
                    fractional_order, reference, reference_count, x0,
                    &rows[sample].sec_total, &rows[sample].sec_hits,
                    &rows[sample].hit_frac
                );
            }
            for (int sample = 0; sample < sample_count; ++sample) {
                int classification = rows[sample].cls;
                if (classification < 0 || classification > 4) {
                    classification = CLS_UNKNOWN;
                }
                ++counts[classification];
                fprintf(
                    samples_file,
                    "%s,%.17g,%d,%.17g,%.17g,%.17g,%s,%d,%d,%.17g\n",
                    equilibria.names[equilibrium], radius, sample,
                    rows[sample].x0, rows[sample].y0, rows[sample].z0,
                    CLASS_NAMES[classification], rows[sample].sec_total,
                    rows[sample].sec_hits, rows[sample].hit_frac
                );
            }
            fprintf(
                summary_file, "%s,%.17g,%d,%d,%d,%d,%d\n",
                equilibria.names[equilibrium], radius, counts[CLS_EQ],
                counts[CLS_DIV], counts[CLS_TARGET], counts[CLS_OTHER],
                counts[CLS_UNKNOWN]
            );
            total_target_hits += counts[CLS_TARGET];
            printf(
                "r=%-8.1e  EQ=%3d  DIV=%3d  TARGET=%3d  OTHER=%3d  "
                "UNKNOWN=%3d\n",
                radius, counts[CLS_EQ], counts[CLS_DIV], counts[CLS_TARGET],
                counts[CLS_OTHER], counts[CLS_UNKNOWN]
            );
            fflush(samples_file);
            fflush(summary_file);
            free(rows);
        }
        printf("\n");
    }
    fclose(samples_file);
    fclose(summary_file);
    free(reference);
    stats->total_target_hits = total_target_hits;
}

static void checked_path_copy(
    char *destination, size_t capacity, const char *source
) {
    int written;
    if (!destination || capacity == 0u || !source || !*source) {
        die("Las rutas de salida no pueden estar vacias");
    }
    written = snprintf(destination, capacity, "%s", source);
    if (written < 0 || (size_t)written >= capacity) {
        die("Ruta de salida demasiado larga");
    }
}

static void validated_run_backend(
    const ChuaParams *p,
    double frac_order,
    const double target_seed[3],
    const IntegrationCfg *integ,
    const ThresholdCfg *thr,
    const SamplingCfg *sampling,
    const FileCfg *files,
    RunStats *stats
) {
    int ref_steps = 0;
    int test_steps = 0;
    int burn_ref_steps = 0;
    int burn_test_steps = 0;
    int memory_steps = 0;
    size_t row_bytes = 0u;

    if (!p || !integ || !thr || !sampling || !files || !stats ||
        !hafo_values_are_finite(target_seed, 3u) ||
        !(p->alpha_chua > 0.0) || !(p->beta > 0.0) ||
        (p->model != 0 && p->model != 1) ||
        (p->model == 1 && !(p->rho > 0.0))) {
        die("Parametros de Chua invalidos");
    }
    if (!hafo_uniform_step_count(integ->TMAX_REF, integ->h, &ref_steps) ||
        ref_steps < 1 ||
        !hafo_uniform_step_count(integ->TMAX_TEST, integ->h, &test_steps) ||
        test_steps < 1 ||
        !hafo_uniform_step_count(integ->TBURN_REF, integ->h, &burn_ref_steps) ||
        !hafo_uniform_step_count(integ->TBURN_TEST, integ->h, &burn_test_steps) ||
        burn_ref_steps > ref_steps || burn_test_steps > test_steps ||
        !hafo_positive_ratio_ceil(integ->Lm, integ->h, &memory_steps)) {
        die("h, Lm y los horizontes deben definir mallas uniformes validas");
    }
    if (!(frac_order > 0.0 && frac_order <= 1.0) ||
        !(thr->R_DIV > 0.0) || !(thr->EPS_EQ > 0.0) ||
        !(thr->SEC_TOL > 0.0) ||
        !(thr->HIT_FRAC_REQ >= 0.0 && thr->HIT_FRAC_REQ <= 1.0) ||
        thr->CAP_WIN < 1 || thr->MIN_SEC_MATCH < 1 ||
        thr->TEST_MAX_SEC < 1 || thr->MIN_SEC_MATCH > thr->TEST_MAX_SEC ||
        thr->TEST_MAX_SEC > INT_MAX / 8 ||
        sampling->n_radii < 1 || sampling->nsamples < 1) {
        die("Umbrales o conteos invalidos");
    }
    if (!hafo_checked_mul_size(
            (size_t)sampling->nsamples, sizeof(SampleRow), &row_bytes)) {
        die("Tamano de muestreo fuera de rango");
    }
    (void)memory_steps;
    (void)row_bytes;
    run_backend(p, frac_order, target_seed, integ, thr, sampling, files, stats);
}

static void usage(const char *program) {
    fprintf(
        stderr,
        "Uso: %s\n"
        "  --alpha_chua A --beta B --gamma_chua G --m0 M0 --m1 M1\n"
        "  --frac_order q --target_seed x,y,z\n"
        "  --h h --Lm Lm --TMAX_REF tr --TMAX_TEST tt\n"
        "  --TBURN_REF br --TBURN_TEST bt\n"
        "  --R_DIV rdiv --EPS_EQ epseq --CAP_WIN cap --SEC_TOL stol\n"
        "  --MIN_SEC_MATCH msec --TEST_MAX_SEC tsec --HIT_FRAC_REQ hreq\n"
        "  --radii r1,r2,... --nsamples N --random_seed S\n"
        "  --csv_out out.csv --ref_out ref.csv "
        "--summary_csv_out summary.csv\n",
        program
    );
}

int main(int argc, char **argv) {
    ChuaParams parameters;
    IntegrationCfg integration;
    ThresholdCfg thresholds;
    SamplingCfg sampling = {0};
    FileCfg files;
    RunStats stats = {0, 0};
    double fractional_order;
    double target_seed[3];
    const char *csv_out;
    const char *reference_out;
    const char *summary_out;

    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc == 1 || has_arg(argc, argv, "--help")) {
        usage(argv[0]);
        return argc == 1 ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    parameters.alpha_chua = parse_double_arg(argc, argv, "--alpha_chua");
    parameters.beta = parse_double_arg(argc, argv, "--beta");
    parameters.gamma_chua = parse_double_arg(argc, argv, "--gamma_chua");
    parameters.m0 = parse_double_arg(argc, argv, "--m0");
    parameters.m1 = parse_double_arg(argc, argv, "--m1");
    parameters.model = parse_model_name(find_arg(argc, argv, "--model"));
    parameters.a1 = optional_double_arg(argc, argv, "--a1", 0.4);
    parameters.a2 = optional_double_arg(argc, argv, "--a2", -1.5585);
    parameters.rho = optional_double_arg(argc, argv, "--rho", 1.0);
    if (!(parameters.rho > 0.0)) die("--rho debe ser positivo");

    fractional_order = parse_double_arg(argc, argv, "--frac_order");
    if (!(fractional_order > 0.0 && fractional_order <= 1.0)) {
        die("El orden fraccionario q debe cumplir 0 < q <= 1.");
    }
    parse_vec3_arg(argc, argv, "--target_seed", target_seed);

    integration.h = parse_double_arg(argc, argv, "--h");
    integration.Lm = parse_double_arg(argc, argv, "--Lm");
    integration.TMAX_REF = parse_double_arg(argc, argv, "--TMAX_REF");
    integration.TMAX_TEST = parse_double_arg(argc, argv, "--TMAX_TEST");
    integration.TBURN_REF = parse_double_arg(argc, argv, "--TBURN_REF");
    integration.TBURN_TEST = parse_double_arg(argc, argv, "--TBURN_TEST");

    thresholds.R_DIV = parse_double_arg(argc, argv, "--R_DIV");
    thresholds.EPS_EQ = parse_double_arg(argc, argv, "--EPS_EQ");
    thresholds.CAP_WIN = parse_int_arg(argc, argv, "--CAP_WIN");
    thresholds.SEC_TOL = parse_double_arg(argc, argv, "--SEC_TOL");
    thresholds.MIN_SEC_MATCH = parse_int_arg(argc, argv, "--MIN_SEC_MATCH");
    thresholds.TEST_MAX_SEC = parse_int_arg(argc, argv, "--TEST_MAX_SEC");
    thresholds.HIT_FRAC_REQ = parse_double_arg(argc, argv, "--HIT_FRAC_REQ");

    parse_radii_arg(argc, argv, "--radii", &sampling);
    sampling.nsamples = parse_int_arg(argc, argv, "--nsamples");
    sampling.random_seed = parse_u64_arg(argc, argv, "--random_seed");

    csv_out = find_arg(argc, argv, "--csv_out");
    reference_out = find_arg(argc, argv, "--ref_out");
    summary_out = find_arg(argc, argv, "--summary_csv_out");
    if (!csv_out || !reference_out || !summary_out) {
        free(sampling.radii);
        die("Faltan rutas de salida: --csv_out, --ref_out o --summary_csv_out");
    }
    checked_path_copy(files.csv_out, sizeof(files.csv_out), csv_out);
    checked_path_copy(files.ref_out, sizeof(files.ref_out), reference_out);
    checked_path_copy(
        files.summary_csv_out, sizeof(files.summary_csv_out), summary_out
    );

    validated_run_backend(
        &parameters,
        fractional_order,
        target_seed,
        &integration,
        &thresholds,
        &sampling,
        &files,
        &stats
    );

    printf("Resumen final:\n");
    if (stats.total_target_hits == 0) {
        printf(
            "No se detectaron trayectorias que cayeran en el atractor objetivo "
            "desde las vecindades muestreadas de los equilibrios.\n"
        );
    } else {
        printf(
            "Se detectaron %d trayectorias clasificadas como TARGET.\n",
            stats.total_target_hits
        );
    }
    printf(
        "\nArchivos numericos generados:\n- %s\n- %s\n- %s\n",
        files.csv_out,
        files.ref_out,
        files.summary_csv_out
    );
    free(sampling.radii);
    return EXIT_SUCCESS;
}

