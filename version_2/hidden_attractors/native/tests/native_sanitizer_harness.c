#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>

#include "../csrc/fractional_integrators.h"

int integrate_general_efork_c(
    void (*rhs)(double, const double *, double *),
    const double *x0, int dim, double q, double h, double t_final,
    double divergence_norm, double *trajectory, size_t capacity
);
int integrate_general_abm_c(
    void (*rhs)(double, const double *, double *),
    const double *x0, int dim, double q, double h, double t_final,
    double divergence_norm, double *trajectory, size_t capacity
);

static void fractional_decay(
    double time, const double *state, double *derivative,
    int dimension, void *params
) {
    (void)time;
    (void)params;
    for (int index = 0; index < dimension; ++index) {
        derivative[index] = -state[index];
    }
}

static void general_decay(
    double time, const double *state, double *derivative
) {
    (void)time;
    derivative[0] = -state[0];
}

static int finite_double(double value) {
    return value == value && value != INFINITY && value != -INFINITY;
}

static int call_fractional(double horizon, size_t time_capacity) {
    const double x0[1] = {1.0};
    double times[4] = {0.0};
    double states[4] = {0.0};
    int samples = 0;
    int status = -1;
    const int rc = integrate_fractional_c(
        fractional_decay, NULL, 1, x0, 0.9, 0.1, horizon,
        0, 0, 0, NULL, NULL, 0, 0u, 0u, 120.0,
        times, states, time_capacity, 4u, &samples, &status,
        0, 0, 80.0, 5, 1.25, 0, 1.0e-3, 1.0e-4, 200, 5.0,
        NULL, 0, 0u
    );
    if (rc == 0 &&
        (samples != 4 || status != 0 || times[3] != 0.3 ||
         !finite_double(states[3]))) {
        return -99;
    }
    return rc;
}

static int call_fractional_contract(
    int method,
    int memory_mode,
    int memory_window,
    const double *history_times,
    const double *history_states,
    int history_len,
    size_t history_time_count,
    size_t history_state_count,
    const double *equilibria,
    int equilibrium_count,
    size_t equilibrium_value_count
) {
    const double x0[1] = {1.0};
    double times[4] = {0.0};
    double states[4] = {0.0};
    int samples = 0;
    int status = -1;
    return integrate_fractional_c(
        fractional_decay, NULL, 1, x0, 0.9, 0.1, 0.3,
        method, memory_mode, memory_window,
        history_times, history_states, history_len,
        history_time_count, history_state_count, 120.0,
        times, states, 4u, 4u, &samples, &status,
        0, 0, 80.0, 5, 1.25, 0, 1.0e-3, 1.0e-4, 200, 5.0,
        equilibria, equilibrium_count, equilibrium_value_count
    );
}

static int run_fractional_contract_checks(void) {
    const double history_times[1] = {0.0};
    const double history_states[1] = {1.0};
    const double equilibria[1] = {0.0};
    if (call_fractional_contract(
            99, 0, 0, NULL, NULL, 0, 0u, 0u, NULL, 0, 0u) != -1) {
        return 30;
    }
    if (call_fractional_contract(
            0, 99, 0, NULL, NULL, 0, 0u, 0u, NULL, 0, 0u) != -1) {
        return 31;
    }
    if (call_fractional_contract(
            0, 1, 0, NULL, NULL, 0, 0u, 0u, NULL, 0, 0u) != -1) {
        return 32;
    }
    if (call_fractional_contract(
            0, 0, 0, history_times, NULL, 1, 1u, 0u,
            NULL, 0, 0u) != -1) {
        return 33;
    }
    if (call_fractional_contract(
            0, 0, 0, NULL, NULL, 1, 0u, 0u,
            NULL, 0, 0u) != -1) {
        return 36;
    }
    if (call_fractional_contract(
            0, 0, 0, NULL, history_states, 1, 0u, 1u,
            NULL, 0, 0u) != -1) {
        return 37;
    }
    if (call_fractional_contract(
            0, 0, 0, history_times, history_states, 1, 1u, 0u,
            NULL, 0, 0u) != -1) {
        return 34;
    }
    if (call_fractional_contract(
            0, 0, 0, NULL, NULL, 0, 0u, 0u,
            equilibria, 1, 0u) != -1) {
        return 35;
    }
    return 0;
}

static int run_general_checks(void) {
    const double x0[1] = {1.0};
    double trajectory[8] = {0.0};
    int rc = integrate_general_efork_c(
        general_decay, x0, 1, 0.9, 0.1, 0.3, 120.0, trajectory, 8u
    );
    if (rc != 4 || trajectory[6] != 0.3 || !finite_double(trajectory[7])) {
        return 20;
    }
    rc = integrate_general_abm_c(
        general_decay, x0, 1, 0.9, 0.1, 0.3, 120.0, trajectory, 8u
    );
    if (rc != 4 || trajectory[6] != 0.3 || !finite_double(trajectory[7])) {
        return 21;
    }
    if (integrate_general_efork_c(
            general_decay, x0, 1, 0.9, 0.1, 0.25, 120.0,
            trajectory, 8u) != -1) {
        return 22;
    }
    if (integrate_general_abm_c(
            general_decay, x0, 1, 0.9, 0.1, 0.3, 120.0,
            trajectory, 7u) != -4) {
        return 23;
    }
    return 0;
}

static int run_tempered_checks(void) {
    const double x0[1] = {1.0};
    double times[4] = {0.0};
    double states[4] = {0.0};
    int samples = 0;
    int status = -1;
    int rc = integrate_tempered_caputo_abm_c(
        fractional_decay, NULL, 1, x0, 0.9, 0.1, 0.1, 3,
        0, 0, 120.0, times, states, 4u, 4u, &samples, &status
    );
    if (rc != 0 || samples != 4 || status != 0 || fabs(times[3] - 0.3) > 1e-12 ||
        !finite_double(states[3])) {
        return 40;
    }
    if (integrate_tempered_caputo_abm_c(
            fractional_decay, NULL, 1, x0, 0.9, 0.1, 0.1, 3,
            0, 0, 120.0, times, states, 4u, 3u, &samples, &status) != -3) {
        return 41;
    }
    if (integrate_tempered_caputo_abm_c(
            fractional_decay, NULL, 1, x0, 0.9, 0.1, 0.1, INT_MAX,
            0, 0, 120.0, times, states, 4u, 4u, &samples, &status) != -1) {
        return 42;
    }
    return 0;
}

int main(void) {
    const int fractional_valid = call_fractional(0.3, 4u);
    const int fractional_grid = call_fractional(0.25, 4u);
    const int fractional_capacity = call_fractional(0.3, 3u);
    const int fractional_contract = run_fractional_contract_checks();
    const int general_status = run_general_checks();
    const int tempered_status = run_tempered_checks();
    if (fractional_valid != 0 || fractional_grid != -1 ||
        fractional_capacity != -4 || fractional_contract != 0 ||
        general_status != 0 || tempered_status != 0) {
        fprintf(
            stderr,
            "harness failed: frac=%d grid=%d cap=%d contract=%d general=%d "
            "tempered=%d\n",
            fractional_valid, fractional_grid, fractional_capacity,
            fractional_contract, general_status, tempered_status
        );
        return 1;
    }
    puts("native sanitizer harness: ok");
    return 0;
}
