#ifndef FRACTIONAL_INTEGRATORS_H
#define FRACTIONAL_INTEGRATORS_H

#if defined(_WIN32) || defined(__CYGWIN__)
  #define API_EXPORT __declspec(dllexport)
#else
  #define API_EXPORT __attribute__((visibility("default")))
#endif

// Signature for the general RHS callback
typedef void (*RhsCallback)(double t, const double *x, double *dx, int n, void *params);

// Structure for Chua saturation parameters
typedef struct {
    double alpha;
    double beta;
    double gamma;
    double m0;
    double m1;
} ChuaSaturationParams;

// Structure for Chua arctan parameters
typedef struct {
    double alpha;
    double beta;
    double gamma;
    double m;
    double n;
} ChuaArctanParams;

// Predefined RHS function pointers getter
API_EXPORT void *get_chua_saturation_rhs(void);
API_EXPORT void *get_chua_arctan_rhs(void);

// Main general fractional integrator function
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
    int *status_code
);

#endif // FRACTIONAL_INTEGRATORS_H
