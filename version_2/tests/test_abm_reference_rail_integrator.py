"""The reference convolution preserves the complete single-block PECE history."""
import numpy as np
import pytest

from hidden_attractors.integrations.abm_fractional import integrate_fractional_abm
from hidden_attractors.integrations.abm_reference import integrate_reference_abm


@pytest.mark.parametrize("orders", [[1.0, 1.0], [.9, .9], [.5, .9], [.7, 1.0]])
@pytest.mark.parametrize("block_size", [1, 7, 256])
def test_fft_rail_matches_direct_full_history(orders, block_size):
    def rhs(t, state):
        return np.array([-.4 * state[0] + .1 * state[1] + .01 * t,
                         -.2 * state[1] + .05 * state[0]])
    args = (rhs, np.array([.1, -.2]), orders, .01)
    _, direct, direct_status = integrate_fractional_abm(*args, 43)
    _, rail, rail_status = integrate_reference_abm(*args, .43, convolution_block_size=block_size)
    assert rail_status == direct_status == "ok"
    np.testing.assert_allclose(rail, direct, rtol=3e-13, atol=3e-14)


def test_constant_field_matches_caputo_closed_form_across_blocks():
    from scipy.special import gamma
    orders = np.array([.5, .9, 1.0])
    initial = np.array([.1, -.2, .3])
    forcing = np.array([.7, .4, -.3])
    times, states, status = integrate_reference_abm(
        lambda t, x: forcing, initial, orders, .01, 1.7, convolution_block_size=13)
    exact = initial + times[:, None]**orders * forcing / gamma(orders + 1)
    assert status == "ok"
    np.testing.assert_allclose(states, exact, rtol=1e-11, atol=2e-12)


def test_reference_does_not_restart_caputo_anchor_at_fft_boundaries():
    rhs = lambda t, x: np.ones(1)
    _, rail, status = integrate_reference_abm(rhs, np.zeros(1), [.5], .01, .2, convolution_block_size=10)
    _, first, _ = integrate_fractional_abm(rhs, np.zeros(1), [.5], .01, 10)
    _, reset, _ = integrate_fractional_abm(rhs, first[-1], [.5], .01, 10)
    assert status == "ok"
    assert abs(rail[-1, 0] - reset[-1, 0]) > .1


def test_failure_does_not_return_uncomputed_trailing_states():
    times, states, status = integrate_reference_abm(
        lambda t, x: np.ones(1), np.zeros(1), [.5], .01, .2,
        convolution_block_size=3, divergence_norm=.2)
    assert status == "diverged"
    assert len(times) == len(states) < 21
    assert states[-1, 0] > .2


def test_half_order_linear_exact_solution_refines_across_fft_blocks():
    from scipy.special import erfcx
    errors = []
    # At h=.04 accidental cancellation in the nonsmooth startup error can
    # beat h=.02. Check the refinement range, not an unsupported all-h law.
    for h in [.02, .01, .005, .0025, .00125, .000625]:
        times, states, status = integrate_reference_abm(
            lambda t, x: -x, np.ones(1), [.5], h, 2.0, convolution_block_size=13)
        assert status == "ok"
        errors.append(float(np.max(abs(states[:, 0] - erfcx(np.sqrt(times))))))
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    assert errors[-1] < .0001
