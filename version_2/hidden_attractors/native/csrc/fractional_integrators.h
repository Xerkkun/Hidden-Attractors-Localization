#ifndef FRACTIONAL_INTEGRATORS_H
#define FRACTIONAL_INTEGRATORS_H

#include <stddef.h>

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

// Predefined RHS entry points.  ctypes obtains their typed addresses directly;
// ISO C does not permit round-tripping function pointers through void pointers.
API_EXPORT void chua_saturation_rhs_c(
    double t, const double *x, double *dx, int n, void *params
);
API_EXPORT void chua_arctan_rhs_c(
    double t, const double *x, double *dx, int n, void *params
);

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
    size_t history_times_count,
    size_t history_states_count,
    double divergence_norm,
    double *out_times,
    double *out_states,
    size_t out_times_capacity,
    size_t out_states_capacity,
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
    int num_equilibria,
    size_t equilibria_values_count
);

// Tempered-Caputo ABM written directly in physical coordinates.  The
// exponential conjugation is applied to the history weights, so the stored
// state and divergence norm remain physical and cannot overflow merely because
// exp(lambda*t) is large.
API_EXPORT int integrate_tempered_caputo_abm_c(
    RhsCallback rhs,
    void *params,
    int dim,
    const double *x0,
    double q,
    double tempering,
    double h,
    int n_steps,
    int memory_mode,       // 0: full, 1: sliding restart/window
    int memory_window_length,
    double divergence_norm,
    double *out_times,
    double *out_states,
    size_t out_times_capacity,
    size_t out_states_capacity,
    int *out_samples,
    int *status_code
);

#endif // FRACTIONAL_INTEGRATORS_H
