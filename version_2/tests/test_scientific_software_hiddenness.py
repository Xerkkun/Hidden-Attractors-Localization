"""Lightweight operational hiddenness-classification tests."""

from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.verification.classifiers import classify_hiddenness_verdict
from hidden_attractors.verification.hiddenness import evaluate_target_match


def test_nn_percentile_target_match_accepts_close_clouds_and_rejects_separated_clouds() -> None:
    rng = np.random.default_rng(1234)
    reference = rng.normal(size=(300, 3))
    close = reference + np.array([1.0e-3, -2.0e-3, 1.0e-3])
    separated = reference + np.array([10.0, 0.0, 0.0])

    assert evaluate_target_match(close, reference, metric="nn_percentile", tolerance=1.0e-2)
    assert not evaluate_target_match(separated, reference, metric="nn_percentile", tolerance=1.0e-2)


@pytest.mark.parametrize(
    ("target_hits", "seed_reached", "numerical_failures", "expected"),
    [
        (1, True, 0, "self_excited_contact_detected"),
        (0, True, 0, "compatible_with_hiddenness_under_sampled_radii"),
        (0, False, 0, "not_supported"),
        (0, True, 1, "numerical_failure"),
    ],
)
def test_hiddenness_verdict_uses_operational_sampled_radii_states(
    target_hits: int, seed_reached: bool, numerical_failures: int, expected: str
) -> None:
    assert (
        classify_hiddenness_verdict(
            target_hits_from_equilibria=target_hits,
            equilibria_count=3,
            unstable_equilibria_count=2,
            seed_reached_attractor=seed_reached,
            numerical_failures=numerical_failures,
        )
        == expected
    )

