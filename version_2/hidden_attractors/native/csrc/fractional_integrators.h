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
    double a1;
    double a2;
    double rho;
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
);

#endif // FRACTIONAL_INTEGRATORS_H
