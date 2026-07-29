"""Shared fixtures for synthetic software-performance benchmarks.

The inputs in this directory are deliberately small, round-number fixtures.
They exercise public code paths but are not validation cases and carry no
claim about the dynamics of a physical system.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def performance_parameters():
    """Return neutral coefficients used only to time implementation paths."""
    from hidden_attractors.models.chua import ChuaParameters

    return ChuaParameters(
        alpha=8.0,
        beta=12.0,
        gamma=0.1,
        m0=-0.2,
        m1=-1.0,
    )


@pytest.fixture(scope="session")
def frac_backend(performance_parameters):
    """Build the native trajectory backend with synthetic coefficients."""
    from hidden_attractors.native.backends import FractionalChuaBackend

    try:
        backend = FractionalChuaBackend.build()
    except Exception as exc:
        pytest.skip(
            "FractionalChuaBackend unavailable "
            f"({exc}); install a supported C compiler"
        )
    backend.set_params(performance_parameters)
    return backend


@pytest.fixture(scope="session")
def basin_backend(performance_parameters):
    """Build the native classifier backend with synthetic coefficients."""
    from hidden_attractors.native.backends import BasinBackend

    try:
        backend = BasinBackend.build()
    except Exception as exc:
        pytest.skip(
            f"BasinBackend unavailable ({exc}); install a supported C compiler"
        )
    backend.set_params(performance_parameters)
    return backend
