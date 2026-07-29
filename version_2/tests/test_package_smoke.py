"""Smoke tests for portable package-level APIs."""

from __future__ import annotations

import inspect

import numpy as np

import hidden_attractors as ha
from hidden_attractors import chua_nonsmooth_parameters
from hidden_attractors.models import equilibria_nonsmooth, rhs_nonsmooth
from hidden_attractors.native.backends import C_SOURCE_ROOT


def test_chua_equilibria_are_vector_field_zeros() -> None:
    params = chua_nonsmooth_parameters()
    for equilibrium in equilibria_nonsmooth(params).values():
        assert np.linalg.norm(rhs_nonsmooth(equilibrium, params)) < 1.0e-10


def test_native_c_sources_are_packaged() -> None:
    assert (C_SOURCE_ROOT / "chua_frac_backend_lib.c").exists()
    assert (C_SOURCE_ROOT / "chua_basin_lib.c").exists()


def test_candidate_loader_requires_an_explicit_portable_source() -> None:
    from hidden_attractors.candidates import load_final_candidate_records

    signature = inspect.signature(load_final_candidate_records)
    parameter = signature.parameters["source_dir"]
    assert parameter.default is inspect.Parameter.empty
    assert "load_final_candidate_records" not in ha.PUBLIC_API_STABLE
    assert not hasattr(ha, "load_final_candidate_records")
