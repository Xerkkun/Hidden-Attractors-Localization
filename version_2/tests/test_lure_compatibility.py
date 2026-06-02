"""Unit tests for LureCompatibilityValidator."""

from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.lure.compatibility import LureCompatibilityValidator
from hidden_attractors.systems import get_system
from hidden_attractors.systems.base import ChaoticSystem


def test_lure_direct_compatibility():
    """Verify that chua-nonsmooth registers as LURE_DIRECT."""
    system = get_system("chua-nonsmooth")
    report = LureCompatibilityValidator.validate(system)
    assert report["class"] == "LURE_DIRECT"
    assert report["residual_norm"] < 1e-10
    assert "classic" in report["allowed_methods"]
    assert "machado" in report["allowed_methods"]


def test_lure_incompatibility():
    """Verify that a system without a Lur'e split registers as NOT_COMPATIBLE."""
    mock_system = ChaoticSystem(
        name="incompatible-mock",
        dimension=3,
        rhs=lambda s, p: np.zeros(3),
    )
    report = LureCompatibilityValidator.validate(mock_system)
    assert report["class"] == "NOT_COMPATIBLE"
    assert report["residual_norm"] == float("inf")
    assert not report["allowed_methods"]


def test_force_heuristic_describing_function():
    """Verify that if force_heuristic_describing_function is enabled, allowed_methods has classic."""
    mock_system = ChaoticSystem(
        name="incompatible-mock",
        dimension=3,
        rhs=lambda s, p: np.zeros(3),
    )
    config = {"force_heuristic_describing_function": True}
    report = LureCompatibilityValidator.validate(mock_system, config=config)
    assert report["class"] == "NOT_COMPATIBLE"
    assert "classic" in report["allowed_methods"]
