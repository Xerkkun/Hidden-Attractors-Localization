"""Explicit mathematical and numerical contracts for the package."""

from typing import Any, Dict, Optional, Union
import numpy as np

VALID_SEED_MODES = {"integer", "fractional"}
VALID_CONTINUATION_MODES = {"integer", "fractional"}
VALID_INTEGRATORS = {"abm", "efork3", "efork_q1", "heun", "adm_wu2023", "rk4"}
VALID_MEMORY_POLICIES = {"none", "full_caputo", "finite_window"}
VALID_TRANSFER_CONVENTIONS = {"standard", "opposite_sign"}
VALID_HARMONIC_CONDITIONS = {"1_minus_WN", "1_plus_WN"}

def validate_contracts(config: Dict[str, Any], resolved: bool = False) -> None:
    """Validate all mathematical and numerical contract keys in the configuration.
    
    If resolved is False: performs early syntactic validation.
    If resolved is True: performs strong dynamic compatibility coherence checks.
    """
    seed_mode = config.get("seed_mode")
    if seed_mode is not None and seed_mode not in VALID_SEED_MODES:
        raise ValueError(f"Invalid seed_mode contract: {seed_mode}. Must be one of {VALID_SEED_MODES}.")
        
    cont_mode = config.get("continuation_mode")
    if cont_mode is not None and cont_mode not in VALID_CONTINUATION_MODES:
        raise ValueError(f"Invalid continuation_mode contract: {cont_mode}. Must be one of {VALID_CONTINUATION_MODES}.")
        
    integrator = config.get("integrator")
    if integrator is not None and integrator not in VALID_INTEGRATORS:
        raise ValueError(f"Invalid integrator contract: {integrator}. Must be one of {VALID_INTEGRATORS}.")
        
    mem_policy = config.get("memory_policy")
    if mem_policy is not None and mem_policy not in VALID_MEMORY_POLICIES:
        raise ValueError(f"Invalid memory_policy contract: {mem_policy}. Must be one of {VALID_MEMORY_POLICIES}.")
        
    trans_conv = config.get("transfer_convention")
    if trans_conv is not None and trans_conv not in VALID_TRANSFER_CONVENTIONS:
        raise ValueError(f"Invalid transfer_convention contract: {trans_conv}. Must be one of {VALID_TRANSFER_CONVENTIONS}.")
        
    harm_cond = config.get("harmonic_condition")
    if harm_cond is not None and harm_cond not in VALID_HARMONIC_CONDITIONS:
        raise ValueError(f"Invalid harmonic_condition contract: {harm_cond}. Must be one of {VALID_HARMONIC_CONDITIONS}.")
        
    q_seed = config.get("q_seed")
    if q_seed is not None:
        try:
            q_s_val = float(q_seed)
            if not (0.0 < q_s_val <= 1.0):
                raise ValueError()
        except (TypeError, ValueError):
            raise ValueError(f"q_seed contract must be a float in (0, 1], got {q_seed}.")
            
    q_dynamics = config.get("q_dynamics")
    if q_dynamics is not None:
        try:
            q_d_val = float(q_dynamics)
            if not (0.0 < q_d_val <= 1.0):
                raise ValueError()
        except (TypeError, ValueError):
            raise ValueError(f"q_dynamics contract must be a float in (0, 1], got {q_dynamics}.")
            
    memory_mode = config.get("memory_mode")
    if mem_policy == "full_caputo" and memory_mode != "full":
        raise ValueError("When memory_policy is 'full_caputo', memory_mode must be 'full'.")

    if mem_policy == "finite_window":
        if memory_mode != "window":
            raise ValueError("When memory_policy is 'finite_window', memory_mode must be 'window'.")
        mem_win_len = config.get("memory_window_length")
        if mem_win_len is None or mem_win_len <= 0:
            raise ValueError("When memory_policy is 'finite_window', memory_window_length must be > 0.")

    allow_nonstandard = config.get("allow_nonstandard_sign_pairing", False)
    if trans_conv == "standard" and harm_cond == "1_plus_WN":
        if not allow_nonstandard:
            raise ValueError("Non-standard pairing: transfer_convention='standard' and harmonic_condition='1_plus_WN' is forbidden unless allow_nonstandard_sign_pairing is True.")

    if trans_conv == "opposite_sign" and harm_cond == "1_minus_WN":
        if not allow_nonstandard:
            raise ValueError("Non-standard pairing: transfer_convention='opposite_sign' and harmonic_condition='1_minus_WN' is forbidden unless allow_nonstandard_sign_pairing is True.")

    if resolved:
        q_dyn_eff = config.get("q_dynamics")
        if q_dyn_eff is None:
            dyn_mode = config.get("dynamics_mode", "fractional")
            if dyn_mode == "integer":
                q_dyn_eff = 1.0
            else:
                q_dyn_eff = config.get("q", 1.0)
                if q_dyn_eff is None:
                    q_dyn_eff = 1.0
                    
        q_cont_eff = config.get("q_continuation")
        if q_cont_eff is None:
            if config.get("continuation_mode") == "integer":
                q_cont_eff = 1.0
            else:
                q_cont_eff = q_dyn_eff

        if integrator == "abm" and abs(q_dyn_eff - 1.0) < 1e-9:
            raise ValueError("ABM integrator is not allowed for integer-order dynamics (q_dynamics = 1.0). Use 'efork_q1' or 'heun'.")

        if integrator in {"heun", "efork_q1"} and q_dyn_eff < 1.0:
            raise ValueError(f"Integrator '{integrator}' is not allowed for fractional-order dynamics (q_dynamics < 1.0). Use 'abm' or 'efork3'.")

        if cont_mode == "integer" and integrator == "abm":
            raise ValueError("ABM integrator is not allowed for integer continuation. Use 'efork_q1' or 'heun'.")

        if cont_mode == "fractional" and integrator in {"heun", "efork_q1"}:
            raise ValueError(f"Integrator '{integrator}' is not allowed for fractional continuation. Use 'abm' or 'efork3'.")

        if cont_mode == "fractional" and abs(q_cont_eff - 1.0) < 1e-9:
            raise ValueError("Fractional continuation mode cannot be run with integer-order continuation dynamics (q_continuation = 1.0).")

        if mem_policy == "none" and q_dyn_eff < 1.0:
            raise ValueError("memory_policy cannot be 'none' when q_dynamics < 1.0.")
