from __future__ import annotations

import sys
import pytest
import numpy as np
from pathlib import Path

# Add workspace root and version_2 to sys.path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root / "version_2") not in sys.path:
    sys.path.insert(0, str(workspace_root / "version_2"))

from hidden_attractors.integrations.selector import validate_integrator_compatibility, integrate


def test_validate_integrator_compatibility():
    # Valid combos
    assert validate_integrator_compatibility("abm", 0.95) == "abm"
    assert validate_integrator_compatibility("efork3", 0.98) == "efork3"
    assert validate_integrator_compatibility("rk4", 1.0) == "rk4"
    assert validate_integrator_compatibility("heun", 1.0) == "heun"
    assert validate_integrator_compatibility("efork_q1", 1.0) == "efork_q1"

    # Redirection warning
    with pytest.warns(UserWarning, match="redirects to the integer-order"):
        assert validate_integrator_compatibility("efork3", 1.0) == "efork3"

    # Invalid q ranges
    with pytest.raises(ValueError, match="must be in"):
        validate_integrator_compatibility("efork3", 0.0)
    with pytest.raises(ValueError, match="must be in"):
        validate_integrator_compatibility("efork3", 1.2)

    # Incompatible combos
    with pytest.raises(ValueError, match="requires q < 1"):
        validate_integrator_compatibility("abm", 1.0)
    with pytest.raises(ValueError, match="only supports integer-order"):
        validate_integrator_compatibility("rk4", 0.95)
    with pytest.raises(ValueError, match="only supports integer-order"):
        validate_integrator_compatibility("heun", 0.99)


def test_integrate_dispatch():
    # Simple linear RHS: dx/dt = -x
    def rhs(t, x):
        return -x

    # Integer integration
    t, x, status = integrate(rhs, np.array([1.0]), q=1.0, h=0.01, t_final=1.0, integrator="heun")
    assert status == "ok"
    assert len(t) == 101
    assert np.allclose(x[-1], np.exp(-1.0), rtol=1e-2)

    # Fractional integration (using Python fallback or general solver)
    t_frac, x_frac, status_frac = integrate(
        rhs, np.array([1.0]), q=0.98, h=0.01, t_final=0.5, integrator="efork3", use_c_backend=False
    )
    assert status_frac == "ok"
    assert len(t_frac) == 51
