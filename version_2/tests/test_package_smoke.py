"""Smoke tests for the package-level API."""

from __future__ import annotations

import numpy as np

from hidden_attractors import chua_piecewise_parameters, load_final_candidate_records
from hidden_attractors.models import equilibria_piecewise, rhs_piecewise
from hidden_attractors.native.backends import C_SOURCE_ROOT


def test_chua_equilibria_are_vector_field_zeros() -> None:
    params = chua_piecewise_parameters()
    for eq in equilibria_piecewise(params).values():
        assert np.linalg.norm(rhs_piecewise(eq, params)) < 1.0e-10


def test_final_candidate_loader_returns_three_records() -> None:
    records = load_final_candidate_records()
    assert len(records) == 3
    assert all(record.q > 0.0 for record in records)


def test_native_c_sources_are_packaged() -> None:
    assert (C_SOURCE_ROOT / "chua_frac_backend_lib.c").exists()
    assert (C_SOURCE_ROOT / "chua_basin_lib.c").exists()
