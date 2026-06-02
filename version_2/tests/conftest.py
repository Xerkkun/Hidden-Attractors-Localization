"""Test fixtures shared across the repository."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from uuid import uuid4

import pytest


_TEST_ARTIFACTS_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "test_artifacts"


def _safe_test_name(node_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", node_name).strip("._") or "test"


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    """Create temporary paths with inherited ACLs on Windows.

    Pytest's built-in fixture creates directories with mode 0700. On some
    Windows ACL configurations that produces directories which cannot be
    reopened by the test process. Let Windows inherit the repository ACL
    instead and keep all transient test data under the ignored output tree.
    """

    _TEST_ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    path = _TEST_ARTIFACTS_ROOT / f"{_safe_test_name(request.node.name)}_{uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def valid_run_metadata() -> dict:
    """Minimal complete metadata accepted for strong hiddenness promotion."""

    return {
        "schema_version": "1.0",
        "run_id": "test-run",
        "workflow": "pytest",
        "system": "test-system",
        "created_at_utc": "2026-06-02T00:00:00+00:00",
        "numerical_contract": {
            "q": 0.9998,
            "h": 0.01,
            "t_final": 100.0,
            "t_burn": 20.0,
            "memory": {
                "mode": "full",
                "M": None,
                "memory_window_steps": None,
                "memory_window_time": None,
                "is_full_caputo": True,
            },
            "integrator": {"name": "abm", "backend": "python", "caputo": True},
        },
        "software": {
            "python_version": "3.test",
            "platform": "test",
            "package_version": "test",
            "numpy_version": "test",
            "git_commit": "test",
            "working_tree_dirty": False,
            "git_diff_sha256": None,
        },
        "parameters": {"alpha": 1.0},
        "lure": {
            "matrix": [[-1.0]],
            "input_vector": [1.0],
            "output_vector": [1.0],
            "scalar_nonlinearity": "psi",
            "transfer_convention": "test",
            "harmonic_condition": "test",
        },
        "seed": {
            "candidate_id": "candidate-test",
            "family": "test",
            "x0": [1.0],
            "source": "pytest",
            "parameters": {},
        },
        "random_seed": 7,
        "random_seed_policy": "fixed_reproducible",
        "provenance": {},
        "extra": {},
    }
