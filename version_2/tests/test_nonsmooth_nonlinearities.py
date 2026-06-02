"""Unit tests for NonSmoothNonlinearityValidator."""

from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.systems import get_system
from hidden_attractors.validation.nonsmooth import NonSmoothNonlinearityValidator


def test_nonsmooth_properties():
    """Verify that chua-nonsmooth registers as sat, continuous, Lipschitz, with boundaries at +-1."""
    system = get_system("chua-nonsmooth")
    props = NonSmoothNonlinearityValidator.analyze_nonlinearity(system)
    assert props["type"] == "sat"
    assert props["continuous"] is True
    assert props["lipschitz"] is True
    assert props["switching_surfaces"] == [-1.0, 1.0]


def test_regional_jacobian():
    """Verify regional Jacobian calculation on different sides of switching boundaries."""
    system = get_system("chua-nonsmooth")
    
    # Inside region: |x| < 1
    J_inside = NonSmoothNonlinearityValidator.jacobian_region(system, np.array([0.5, 0.0, 0.0]))
    # Outside region: |x| > 1
    J_outside = NonSmoothNonlinearityValidator.jacobian_region(system, np.array([1.5, 0.0, 0.0]))
    
    # They should differ because of the slope (m0 vs m1)
    assert not np.array_equal(J_inside, J_outside)


def test_crossing_detection():
    """Verify that crossing a boundary triggers an alert and crossing report."""
    system = get_system("chua-nonsmooth")
    
    # Trajectory crossing x = 1.0 at t=1.0
    trajectory = np.array([
        [0.0, 0.8, 0.0, 0.0],
        [1.0, 1.2, 0.0, 0.0]
    ])
    
    res = NonSmoothNonlinearityValidator.detect_switching_crossings(trajectory, system)
    assert res["crossings_detected"] == 1
    assert len(res["crossings"]) == 1
    assert res["crossings"][0]["surface"] == 1.0
    assert len(res["warnings"]) > 0
    assert "cruza superficie" in res["warnings"][0]


def test_matignon_indeterminate_boundary():
    """Verify that equilibria exactly on switching boundaries return nonsmooth_indeterminate."""
    system = get_system("chua-nonsmooth")
    
    # Point on boundary: x = 1.0
    eq = np.array([1.0, 0.0, 0.0])
    res = NonSmoothNonlinearityValidator.validate_equilibrium_stability(system, eq, 0.9998)
    assert res == "nonsmooth_indeterminate"
