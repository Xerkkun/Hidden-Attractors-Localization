from typing import Any, Dict, List

def classify_hiddenness_verdict(
    target_hits_from_equilibria: int,
    equilibria_count: int,
    unstable_equilibria_count: int,
    seed_reached_attractor: bool,
    numerical_failures: int = 0
) -> str:
    """Classify the overall hiddenness verdict under the tested numerical contract.
    
    States:
        - "hidden_verified_under_tested_radii"
        - "compatible_with_hiddenness_under_tested_radii"
        - "self_excited_contact_detected"
        - "not_supported"
        - "numerical_failure"
    """
    if numerical_failures > 0 and target_hits_from_equilibria == 0:
        return "numerical_failure"
        
    if target_hits_from_equilibria > 0:
        # Trajectories from equilibrium neighborhoods hit the candidate attractor:
        # therefore it is self-excited!
        return "self_excited_contact_detected"
        
    if not seed_reached_attractor:
        # The seed itself failed to converge to the bounded candidate attractor
        return "not_supported"
        
    # If we had 0 target hits, and seed successfully reached the attractor:
    # Check if there are stable equilibria
    stable_count = equilibria_count - unstable_equilibria_count
    
    if stable_count == 0:
        # System has only unstable equilibria, and no neighborhood trajectories excited the attractor
        return "hidden_verified_under_tested_radii"
    else:
        # System has at least one stable equilibrium, which is compatible with hiddenness
        return "compatible_with_hiddenness_under_tested_radii"
