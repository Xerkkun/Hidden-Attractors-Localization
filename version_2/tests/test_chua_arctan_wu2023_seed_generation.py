"""Seed and target-dynamics filter checks for Wu2023 arctan Chua."""

from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.diagnostics.periodicity import classify_post_transient_periodicity
from hidden_attractors.models import chua_arctan_wu2023_parameters
from hidden_attractors.seed_generation.chua import chua_matrices, describing_function
from hidden_attractors.seed_generation.chua_arctan_wu2023 import (
    find_centered_arctan_wu2023_branches,
    transfer_function_arctan_wu2023,
)
from hidden_attractors.seed_generation.core import fractional_iomega_power


def test_fractional_transfer_uses_lambda_equal_jomega_to_q() -> None:
    p = chua_arctan_wu2023_parameters()
    omega = 2.0991692817
    q = 0.99
    pmat, bvec, rvec = chua_matrices(p)
    lam = fractional_iomega_power(omega, q)
    expected = rvec @ np.linalg.solve(lam * np.eye(3) - pmat, bvec)
    integer_wrong = rvec @ np.linalg.solve(1j * omega * np.eye(3) - pmat, bvec)

    assert transfer_function_arctan_wu2023(omega, q, p) == expected
    assert abs(expected - integer_wrong) > 1.0e-3


def test_arctan_describing_function_sign_and_centered_branch_closure() -> None:
    p = chua_arctan_wu2023_parameters()
    branches = find_centered_arctan_wu2023_branches(q=0.99, params=p, nscan=20000)

    assert len(branches) == 2
    for branch in branches:
        n_value = describing_function(branch.amplitude, p)
        response = transfer_function_arctan_wu2023(branch.omega, 0.99, p)
        assert n_value < 0.0
        assert branch.gain == pytest.approx(n_value, abs=1.0e-14)
        assert abs(1.0 - response * n_value) < 1.0e-9
        assert abs(1.0 + response * n_value) > 1.9


def test_periodic_target_trajectory_is_rejected_only_after_integration() -> None:
    h = 0.01
    time = np.arange(0.0, 40.0, h)
    trajectory = np.column_stack([time, np.sin(2 * np.pi * time), np.cos(2 * np.pi * time), 0.4 * np.sin(2 * np.pi * time)])
    result = classify_post_transient_periodicity(
        trajectory,
        h=h,
        config={"t_transient": 10.0, "require_two_components": True},
    )

    assert result["candidate_label"] == "regular_periodic_rejected"
    assert result["periodic_post_transient"] is True
    assert result["n_periodic_components"] >= 2
    assert result["hidden_candidate_allowed"] is False
