"""
benchmarks/conftest.py
======================
Shared fixtures and helpers for the benchmark suite.

Run benchmarks with:
    python -m pytest benchmarks/ -v --benchmark-only           # pytest-benchmark
    python benchmarks/bench_efork_single_trajectory.py        # standalone

The benchmarks are intentionally *not* collected by the normal test suite
(testpaths = ["tests"]).  They live in a separate directory and are only
executed explicitly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# ── Canonical problem parameters ─────────────────────────────────────────────
#
# These match the values used in the project's scientific reports.  Changing
# them changes what is being measured, so treat them as fixed constants.

Q = 0.9998       # Caputo fractional order
H = 0.005        # integration step  (seconds)
LM = 10.0        # finite-memory window length  (seconds)
T_FINAL_SHORT = 50.0    # "quick" scenario: detects obvious regressions fast
T_FINAL_LONG = 200.0    # "long" scenario: realistic basin-sweep trajectory

SEED_CANONICAL = np.array([0.1, 0.0, 0.0])   # canonical hidden seed region


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def chua_params():
    """Return the canonical piecewise Chua parameters."""
    from hidden_attractors.models.chua import chua_piecewise_parameters
    return chua_piecewise_parameters()


@pytest.fixture(scope="session")
def chua_system():
    """Return the registered 'chua-fractional' ChaoticSystem."""
    from hidden_attractors.systems import get_system
    return get_system("chua-fractional")


@pytest.fixture(scope="session")
def integer_chua_system():
    """Return the registered 'chua-integer' ChaoticSystem."""
    from hidden_attractors.systems import get_system
    return get_system("chua-integer")


@pytest.fixture(scope="session")
def frac_backend():
    """Build (compile if needed) and return a FractionalChuaBackend."""
    from hidden_attractors.native.backends import FractionalChuaBackend
    try:
        return FractionalChuaBackend.build()
    except Exception as exc:
        pytest.skip(f"FractionalChuaBackend unavailable ({exc}); install gcc or set ALLOW_NO_OPENMP=1")


@pytest.fixture(scope="session")
def basin_backend():
    """Build (compile if needed) and return a BasinBackend."""
    from hidden_attractors.native.backends import BasinBackend
    try:
        return BasinBackend.build()
    except Exception as exc:
        pytest.skip(f"BasinBackend unavailable ({exc}); install gcc or set ALLOW_NO_OPENMP=1")
