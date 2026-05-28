import numpy as np
from typing import Any

def validate_lure_decomposition(system: Any) -> bool:
    """Validate that the Lur'e decomposition vector field matches evaluation.
    
    Checks at random points in state space that:
        P * X + b * psi(r^T * X) == evaluate_rhs(X)
    """
    rng = np.random.default_rng(42)
    for _ in range(10):
        x = rng.uniform(-10.0, 10.0, 3)
        lure = system.lure
        rhs_lure = lure.matrix @ x + lure.input_vector * lure.nonlinearity(lure.output_vector @ x)
        rhs_eval = system.evaluate(x)
        if not np.allclose(rhs_lure, rhs_eval, atol=1e-12):
            return False
    return True
