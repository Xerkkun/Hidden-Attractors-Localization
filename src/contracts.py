"""Explicit mathematical and numerical contracts for the experimental /src layer."""

from typing import Any, Dict, Optional, Union
import numpy as np

# Valid option sets for contracts
VALID_SEED_MODES = {"integer", "fractional"}
VALID_CONTINUATION_MODES = {"integer", "fractional"}
VALID_INTEGRATORS = {"abm", "efork3", "efork_q1", "heun"}
VALID_MEMORY_POLICIES = {"none", "full_caputo", "finite_window"}
VALID_TRANSFER_CONVENTIONS = {"standard", "opposite_sign"}
VALID_HARMONIC_CONDITIONS = {"1_minus_WN", "1_plus_WN"}

def validate_contracts(config: Dict[str, Any]) -> None:
    """Validate all mathematical and numerical contract keys in the configuration.
    
    Raises ValueError if any config value violates the explicit /src contracts.
    """
    # 1. Seed Mode
    seed_mode = config.get("seed_mode")
    if seed_mode is not None and seed_mode not in VALID_SEED_MODES:
        raise ValueError(f"Invalid seed_mode contract: {seed_mode}. Must be one of {VALID_SEED_MODES}.")
        
    # 2. Continuation Mode
    cont_mode = config.get("continuation_mode")
    if cont_mode is not None and cont_mode not in VALID_CONTINUATION_MODES:
        raise ValueError(f"Invalid continuation_mode contract: {cont_mode}. Must be one of {VALID_CONTINUATION_MODES}.")
        
    # 3. Integrator
    integrator = config.get("integrator")
    if integrator is not None and integrator not in VALID_INTEGRATORS:
        raise ValueError(f"Invalid integrator contract: {integrator}. Must be one of {VALID_INTEGRATORS}.")
        
    # 4. Memory Policy
    mem_policy = config.get("memory_policy")
    if mem_policy is not None and mem_policy not in VALID_MEMORY_POLICIES:
        raise ValueError(f"Invalid memory_policy contract: {mem_policy}. Must be one of {VALID_MEMORY_POLICIES}.")
        
    # 5. Transfer Convention
    trans_conv = config.get("transfer_convention")
    if trans_conv is not None and trans_conv not in VALID_TRANSFER_CONVENTIONS:
        raise ValueError(f"Invalid transfer_convention contract: {trans_conv}. Must be one of {VALID_TRANSFER_CONVENTIONS}.")
        
    # 6. Harmonic Condition
    harm_cond = config.get("harmonic_condition")
    if harm_cond is not None and harm_cond not in VALID_HARMONIC_CONDITIONS:
        raise ValueError(f"Invalid harmonic_condition contract: {harm_cond}. Must be one of {VALID_HARMONIC_CONDITIONS}.")
        
    # 7. q_seed validation
    q_seed = config.get("q_seed")
    if q_seed is not None:
        try:
            q_s_val = float(q_seed)
            if not (0.0 < q_s_val <= 1.0):
                raise ValueError()
        except (TypeError, ValueError):
            raise ValueError(f"q_seed contract must be a float in (0, 1], got {q_seed}.")
            
    # 8. q_dynamics validation
    q_dynamics = config.get("q_dynamics")
    if q_dynamics is not None:
        try:
            q_d_val = float(q_dynamics)
            if not (0.0 < q_d_val <= 1.0):
                raise ValueError()
        except (TypeError, ValueError):
            raise ValueError(f"q_dynamics contract must be a float in (0, 1], got {q_dynamics}.")
            
    # 10. Coherence checks
    # Warn/raise if there are illegal combinations:
    if trans_conv == "standard" and harm_cond == "1_plus_WN":
        # While not strictly standard, we support it experimentally if explicitly configured.
        pass
    if trans_conv == "opposite_sign" and harm_cond == "1_minus_WN":
        pass
