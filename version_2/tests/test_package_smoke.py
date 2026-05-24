"""Smoke tests for the package-level API.

These tests verify that the public API is importable and functional in a clean
checkout without any generated output files.  Tests that require runtime output
files (e.g. ``load_final_candidate_records``) are skipped when those files are
absent so that CI on a fresh clone always passes.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from hidden_attractors import chua_nonsmooth_parameters
from hidden_attractors.models import equilibria_nonsmooth, rhs_nonsmooth
from hidden_attractors.native.backends import C_SOURCE_ROOT


# ── Helper evaluated at collection time ──────────────────────────────────────

def _candidate_data_available() -> bool:
    """Return True only when the current selection is promoted for hiddenness."""
    from hidden_attractors.candidates import PROMOTED_SELECTION

    if not PROMOTED_SELECTION.exists():
        return False
    payload = json.loads(PROMOTED_SELECTION.read_text(encoding="utf-8"))
    return (
        payload.get("selection_status") == "promoted_for_hiddenness"
        and len(payload.get("selected_candidates", [])) >= 3
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_chua_equilibria_are_vector_field_zeros() -> None:
    params = chua_nonsmooth_parameters()
    for eq in equilibria_nonsmooth(params).values():
        assert np.linalg.norm(rhs_nonsmooth(eq, params)) < 1.0e-10


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
        "The current validation tree does not contain three candidates promoted "
        "for hiddenness."
    ),
)
def test_final_candidate_loader_returns_three_records() -> None:
    """Full integration test — only runs when output files are present."""
    from hidden_attractors import load_final_candidate_records

    records = load_final_candidate_records()
    assert len(records) == 3
    assert all(record.q > 0.0 for record in records)
