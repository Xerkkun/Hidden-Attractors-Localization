import sys
from pathlib import Path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import numpy as np
import pytest
from src.systems.chua_saturation import ChuaSaturationSystem

def test_saturation_breakpoints():
    system = ChuaSaturationSystem()
    
    # Under A = 2.0 (which is > 1.0)
    A = 2.0
    breakpoints = system.describing_function_breakpoints(A)
    
    theta_c = np.arccos(1.0 / A)
    expected = sorted([0.0, theta_c, np.pi - theta_c, np.pi])
    
    assert len(breakpoints) == 4
    assert np.allclose(breakpoints, expected)
