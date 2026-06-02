"""Unit tests for SymmetryValidator."""

from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.systems import get_system
from hidden_attractors.validation.symmetry import SymmetryValidator


def test_symmetry_detection():
    """Verify that inversion symmetry is successfully detected in chua-nonsmooth."""
    system = get_system("chua-nonsmooth")
    symmetries = SymmetryValidator.detect_symmetries(system)
    assert "inversion" in symmetries


def test_symmetric_seeds_generation_and_deduplication():
    """Verify that symmetric seeds are generated and deduplicated correctly."""
    system = get_system("chua-nonsmooth")
    
    seeds = [
        {
            "candidate_id": "seed_0",
            "seed_id": "seed_0",
            "x0": [2.0, 0.5, -3.0],
            "seed": [2.0, 0.5, -3.0],
        }
    ]
    
    # Generate symmetric seeds
    all_seeds = SymmetryValidator.generate_symmetric_seeds(system, seeds)
    
    # We should get 2 seeds (parent + inversion)
    assert len(all_seeds) == 2
    assert all_seeds[1]["seed_id"] == "seed_0_sym_inversion"
    assert all_seeds[1]["is_symmetric_generated"] is True
    assert all_seeds[1]["x0"] == [-2.0, -0.5, 3.0]


def test_symmetric_seeds_deduplicate_invariant():
    """Verify that an invariant seed (e.g. at origin [0,0,0]) is not duplicated."""
    system = get_system("chua-nonsmooth")
    
    # The origin is invariant under inversion: T(0,0,0) = (0,0,0)
    seeds = [
        {
            "candidate_id": "origin",
            "seed_id": "origin",
            "x0": [0.0, 0.0, 0.0],
            "seed": [0.0, 0.0, 0.0],
        }
    ]
    
    all_seeds = SymmetryValidator.generate_symmetric_seeds(system, seeds)
    # Origin is invariant, so it should not generate a new seed
    assert len(all_seeds) == 1
