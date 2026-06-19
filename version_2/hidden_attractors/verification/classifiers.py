from .candidate_gate import normalize_hiddenness_label

def classify_hiddenness_verdict(
    target_hits_from_equilibria: int,
    equilibria_count: int,
    unstable_equilibria_count: int,
    seed_reached_attractor: bool,
    numerical_failures: int = 0
) -> str:
    """Classify the overall hiddenness verdict under the tested numerical contract.
    
    States:
        - "compatible_with_hiddenness"
        - "self_excited"
        - "inconclusive"
    """
    if numerical_failures > 0 and target_hits_from_equilibria == 0:
        return "inconclusive"
        
    if target_hits_from_equilibria > 0:
        return "self_excited"
        
    if not seed_reached_attractor:
        return "inconclusive"
        
    return "compatible_with_hiddenness"


__all__ = ["classify_hiddenness_verdict", "normalize_hiddenness_label"]
