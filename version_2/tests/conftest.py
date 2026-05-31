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
