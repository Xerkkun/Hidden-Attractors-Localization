"""Smoke tests for the package-level API.

These tests verify that the public API is importable and functional in a clean
checkout without any generated output files.  Tests that require runtime output
files (e.g. ``load_final_candidate_records``) are skipped when those files are
absent so that CI on a fresh clone always passes.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from hidden_attractors import chua_piecewise_parameters
from hidden_attractors.models import equilibria_piecewise, rhs_piecewise
from hidden_attractors.native.backends import C_SOURCE_ROOT


# ── Helper evaluated at collection time ──────────────────────────────────────

def _candidate_data_available() -> bool:
    """Return True if the runtime candidate output files exist in this checkout."""
    from hidden_attractors.paths import OUTPUTS
    targeted = (
        OUTPUTS
        / "extended_search"
        / "machado_targeted_verification_lm10_20260515_182252"
        / "machado_targeted_summary.json"
    )
    return targeted.exists()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_chua_equilibria_are_vector_field_zeros() -> None:
    params = chua_piecewise_parameters()
    for eq in equilibria_piecewise(params).values():
        assert np.linalg.norm(rhs_piecewise(eq, params)) < 1.0e-10


def test_native_c_sources_are_packaged() -> None:
    assert (C_SOURCE_ROOT / "chua_frac_backend_lib.c").exists()
    assert (C_SOURCE_ROOT / "chua_basin_lib.c").exists()


def test_final_candidate_loader_api_is_callable() -> None:
    """``load_final_candidate_records`` must be importable and require no
    positional arguments — verifiable without any output files."""
    from hidden_attractors import load_final_candidate_records

    sig = inspect.signature(load_final_candidate_records)
    assert all(
        p.default is not inspect.Parameter.empty
        for p in sig.parameters.values()
    ), "load_final_candidate_records must have only optional parameters"


@pytest.mark.skipif(
    not _candidate_data_available(),
    reason=(
        "Runtime output files (outputs/extended_search/...) are not present. "
        "Run the full pipeline locally to generate them, then re-run this test."
    ),
)
def test_final_candidate_loader_returns_three_records() -> None:
    """Full integration test — only runs when output files are present."""
    from hidden_attractors import load_final_candidate_records

    records = load_final_candidate_records()
    assert len(records) == 3
    assert all(record.q > 0.0 for record in records)
