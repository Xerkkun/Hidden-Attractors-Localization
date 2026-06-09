from __future__ import annotations

import sys
from pathlib import Path
import pytest

# Add version_2 to path
version_2_dir = Path(__file__).resolve().parents[2]
if str(version_2_dir) not in sys.path:
    sys.path.insert(0, str(version_2_dir))

from hidden_attractors.integrations.selector import validate_integrator_compatibility
from hidden_attractors.workflows.config_loader import load_config

def test_continuation_order_validation():
    # ABM is fractional-only
    with pytest.raises(ValueError, match="requires q < 1"):
        validate_integrator_compatibility("abm", 1.0)

    # RK4 is integer-only
    with pytest.raises(ValueError, match="only supports integer-order systems"):
        validate_integrator_compatibility("rk4", 0.95)

    # Efork3 handles both
    assert validate_integrator_compatibility("efork3", 0.95) == "efork3"
    
    # Efork3 at q=1 redirects to efork_q1
    with pytest.warns(UserWarning, match="redirects to the integer-order"):
        assert validate_integrator_compatibility("efork3", 1.0) == "efork3"
