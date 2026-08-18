"""Small dependency-free numerical helpers shared by internal modules."""

from __future__ import annotations

from collections.abc import Callable


def bisect_root(
    func: Callable[[float], float],
    left: float,
    right: float,
    *,
    maxiter: int = 100,
    xtol: float = 1.0e-12,
) -> float:
    """Return a scalar root from a sign-changing bracket by bisection."""

    lo = float(left)
    hi = float(right)
    flo = float(func(lo))
    fhi = float(func(hi))
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError("root is not bracketed.")
    for _ in range(int(maxiter)):
        mid = 0.5 * (lo + hi)
        fmid = float(func(mid))
        if abs(fmid) <= xtol or abs(hi - lo) <= xtol:
            return mid
        if flo * fmid <= 0.0:
            hi = mid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)
