import sys
from pathlib import Path
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import pytest
from src.contracts import validate_contracts

def test_syntax_validation_succeeds_when_q_none():
    # When resolved=False (early validation), having q_seed = None and q_dynamics = None is syntactically fine
    config = {
        "integrator": "abm",
        "seed_mode": "fractional",
        "continuation_mode": "fractional",
        "memory_policy": "full_caputo",
        "memory_mode": "full",
        "transfer_convention": "standard",
        "harmonic_condition": "1_minus_WN",
        "q_seed": None,
        "q_dynamics": None
    }
    # Should not raise ValueError
    validate_contracts(config, resolved=False)

def test_resolved_validation_rejects_abm_for_integer():
    # q_dynamics_effective = 1.0 with integrator = abm should be rejected when resolved=True
    config = {
        "integrator": "abm",
        "seed_mode": "integer",
        "continuation_mode": "integer",
        "memory_policy": "none",
        "transfer_convention": "standard",
        "harmonic_condition": "1_minus_WN",
        "q_seed": 1.0,
        "q_dynamics": 1.0
    }
    with pytest.raises(ValueError, match="ABM integrator is not allowed for integer-order dynamics"):
        validate_contracts(config, resolved=True)

def test_resolved_validation_rejects_heun_for_fractional():
    # q_dynamics_effective < 1.0 with integrator = heun should be rejected when resolved=True
    config = {
        "integrator": "heun",
        "seed_mode": "fractional",
        "continuation_mode": "fractional",
        "memory_policy": "full_caputo",
        "memory_mode": "full",
        "transfer_convention": "standard",
        "harmonic_condition": "1_minus_WN",
        "q_seed": 0.9,
        "q_dynamics": 0.9
    }
    with pytest.raises(ValueError, match="is not allowed for fractional-order dynamics"):
        validate_contracts(config, resolved=True)

def test_resolved_validation_rejects_fractional_continuation_with_integer_order():
    # continuation_mode = "fractional" with q_continuation_effective = 1.0 should be rejected when resolved=True
    config = {
        "integrator": "abm",
        "seed_mode": "fractional",
        "continuation_mode": "fractional",
        "memory_policy": "full_caputo",
        "memory_mode": "full",
        "transfer_convention": "standard",
        "harmonic_condition": "1_minus_WN",
        "q_seed": 0.9,
        "q_dynamics": 0.9,
        "q_continuation": 1.0
    }
    with pytest.raises(ValueError, match="Fractional continuation mode cannot be run with integer-order continuation dynamics"):
        validate_contracts(config, resolved=True)
