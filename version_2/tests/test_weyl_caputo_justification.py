"""Unit tests for Weyl-Caputo justification and transfer function evaluations."""

from __future__ import annotations

import numpy as np
import pytest

from hidden_attractors.seed_generation.lure import WEYL_CAPUTO_NOTE, lure_transfer_function
from hidden_attractors.systems import get_system


def test_weyl_caputo_note():
    """Verify presence and text of the Weyl-Caputo note."""
    assert "Weyl-Caputo Note:" in WEYL_CAPUTO_NOTE
    assert "s = (j*omega)^q" in WEYL_CAPUTO_NOTE
    assert "omega^q * exp(j * q * pi / 2)" in WEYL_CAPUTO_NOTE


def test_prohibit_integer_evaluation_for_fractional_system():
    """Verify that evaluating a fractional system with q=1.0 is prohibited."""
    system = get_system("chua-nonsmooth").lure
    
    # Evaluating with q=1.0 should raise ValueError as it is a fractional system
    with pytest.raises(ValueError, match="Prohibited evaluating fractional Lur'e system"):
        lure_transfer_function(1.0, 1.0, system)


def test_fractional_evaluation():
    """Verify that W(j*omega) evaluates using fractional frequency (j*omega)^q."""
    system = get_system("chua-nonsmooth").lure
    
    # We can check that evaluating with q=0.9998 does not raise and evaluates successfully
    val = lure_transfer_function(1.0, 0.9998, system)
    assert isinstance(val, complex)
    assert np.isfinite(val)
