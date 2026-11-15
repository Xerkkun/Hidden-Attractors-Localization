#ifndef HAFO_CHUA_EQUILIBRIUM_CLASSIFIER_H
#define HAFO_CHUA_EQUILIBRIUM_CLASSIFIER_H

/* These are finite observation policies, not asymptotic convergence tests. */
#define HAFO_EQ_LEGACY_V1 0
#define HAFO_EQ_TERMINAL_V2 1

typedef struct {
    int streak[3];
    int visited;
    int terminal_equilibrium;
    int first_threshold_index;
    int first_nonfinite_index;
} HafoEquilibriumObservation;

static HafoEquilibriumObservation hafo_eq_observation_init(void) {
    HafoEquilibriumObservation result = {{0, 0, 0}, 0, -1, -1, -1};
    return result;
}

/* All samples are observed for failure/escape. Only eligible samples count
   toward the terminal window. A qualified earlier visit remains a diagnostic. */
static void hafo_eq_observe(
    HafoEquilibriumObservation *observation,
    const double state[3], int index, const double *equilibria, int n_eq,
    double epsilon_squared, int cap_win, double divergence_squared,
    int eligible
) {
    double radius_squared = state[0]*state[0] + state[1]*state[1] + state[2]*state[2];
    observation->terminal_equilibrium = -1;
    if (!hafo_isfinite(state[0]) || !hafo_isfinite(state[1]) ||
        !hafo_isfinite(state[2]) || !hafo_isfinite(radius_squared)) {
        if (observation->first_nonfinite_index < 0)
            observation->first_nonfinite_index = index;
    } else if (radius_squared > divergence_squared &&
               observation->first_threshold_index < 0) {
        observation->first_threshold_index = index;
    }
    for (int equilibrium = 0; equilibrium < n_eq; ++equilibrium) {
        double dx = state[0] - equilibria[3*equilibrium];
        double dy = state[1] - equilibria[3*equilibrium + 1];
        double dz = state[2] - equilibria[3*equilibrium + 2];
        double distance_squared = dx*dx + dy*dy + dz*dz;
        if (eligible && hafo_isfinite(distance_squared) &&
            distance_squared <= epsilon_squared) {
            if (observation->streak[equilibrium] < cap_win)
                ++observation->streak[equilibrium];
            if (observation->streak[equilibrium] >= cap_win) {
                observation->visited = 1;
                observation->terminal_equilibrium = equilibrium;
            }
        } else {
            observation->streak[equilibrium] = 0;
        }
    }
}

static const char *hafo_eq_classifier_version(int policy) {
    return policy == HAFO_EQ_TERMINAL_V2 ? "chua_terminal_v2" : "chua_early_capture_v1";
}

#endif
