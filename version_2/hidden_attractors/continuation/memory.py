import numpy as np
from typing import Any, Optional, Tuple

def extract_memory_window(
    times: np.ndarray,
    states: np.ndarray,
    h: float,
    memory_mode: str,
    memory_window_length: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a slice of the trajectory as prehistory.
    
    If memory_mode == "window" and memory_window_length is provided, truncates to the last M points.
    Shifts times so that the final point is exactly at t = 0.0.
    """
    t = np.asarray(times, dtype=float).copy()
    x = np.asarray(states, dtype=float).copy()
    
    if memory_mode == "window" and memory_window_length is not None:
        pts = min(len(t), int(memory_window_length))
        t = t[-pts:]
        x = x[-pts:]
        
    t = t - t[-1]
    return t, x
