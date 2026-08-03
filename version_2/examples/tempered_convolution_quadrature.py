"""Manufactured nonlinear system for tempered BDF convolution quadrature.

The example is a finite sampled-operator check, not an FDE solver and not
evidence of chaos, attraction, or hiddenness.  For each component it fixes

    x_i(t) = exp(-lambda_i*tau) * (x0_i + a_i*tau**beta_i)

and constructs the nonlinear equation

    CaputoTempered(q_i, lambda_i) x_i = x_i**2 + g_i(t),

where ``g_i`` is chosen from the analytic tempered-Caputo derivative.  HAFO
then evaluates the left-hand side with uncorrected-start BDF2 CQ.
"""

from __future__ import annotations

import json
from math import gamma

import numpy as np

from hidden_attractors.fractional import (
    TEMPERED_CAPUTO_INITIAL_CONDITION,
    tempered_convolution_quadrature,
)


def run_example(
    *,
    intervals: int = 512,
    backend: str = "fft",
) -> dict[str, object]:
    """Return an auditable endpoint and post-startup residual summary."""

    if intervals < 32:
        raise ValueError("intervals must be at least 32 for the residual summary.")
    lower_terminal = 0.7
    duration = 1.0
    elapsed = np.linspace(0.0, duration, intervals + 1)
    times = lower_terminal + elapsed
    orders = np.array([0.58, 0.84])
    tempering = np.array([0.35, 1.10])
    initial = np.array([0.8, -0.3])
    amplitudes = np.array([1.2, 0.7])
    powers = np.array([3.0, 4.0])

    states = np.exp(-elapsed[:, None] * tempering[None, :]) * (
        initial[None, :] + amplitudes[None, :] * elapsed[:, None] ** powers
    )
    coefficients = np.array(
        [
            gamma(float(power) + 1.0)
            / gamma(float(power) + 1.0 - float(order))
            for power, order in zip(powers, orders, strict=True)
        ]
    )
    exact_operator = np.exp(
        -elapsed[:, None] * tempering[None, :]
    ) * (
        amplitudes[None, :]
        * coefficients[None, :]
        * elapsed[:, None] ** (powers - orders)
    )

    # The manufactured forcing is explicit: exact_operator = states**2 + g(t).
    forcing = exact_operator - states**2
    nonlinear_rhs = states**2 + forcing
    result = tempered_convolution_quadrature(
        states,
        orders,
        tempering=tempering,
        bdf_order=2,
        definition="tempered_caputo",
        times=times,
        lower_terminal=lower_terminal,
        initial_condition_semantics=TEMPERED_CAPUTO_INITIAL_CONDITION,
        backend=backend,
    )
    residual = np.asarray(result.values) - nonlinear_rhs
    startup_cut = max(4, intervals // 16)

    return {
        "definition": result.definition,
        "tempering_convention": result.tempering_convention,
        "bdf_order": result.bdf_order,
        "backend": result.backend,
        "intervals": intervals,
        "orders": orders.tolist(),
        "tempering": tempering.tolist(),
        "endpoint_abs_error": np.abs(residual[-1]).tolist(),
        "post_startup_max_abs_error": np.max(
            np.abs(residual[startup_cut:]), axis=0
        ).tolist(),
        "startup_cut_index": startup_cut,
        "positive_exponential_materialized": (
            result.positive_exponential_materialized
        ),
        "starting_corrections": result.starting_corrections,
        "scope": result.scope,
        "evidence_boundary": (
            "manufactured finite-grid consistency only; not a nonlinear FDE "
            "solver validation or evidence of chaos, attraction, or hiddenness"
        ),
    }


def main() -> None:
    """Print the portable JSON summary."""

    print(json.dumps(run_example(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
