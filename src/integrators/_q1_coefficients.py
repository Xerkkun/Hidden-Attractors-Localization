"""EFORK-3 integer-order limit coefficients (q → 1).

When the Caputo order q → 1 the three-stage EFORK-3 coefficients converge
to the following rational values (Ghoreishi et al., 2023):

    a21 = 1/2,  a31 = 1/2,  a32 = -1/4
    w1  = 2/3,  w2  = 5/3,  w3  = -4/3

These are the *only* correct coefficients to use when ``integrator="efork"``
and ``q == 1.0``.  Using Heun (standard predictor-corrector) at q=1 would
produce a numerically different scheme and must NOT be silently substituted.

These constants are duplicated from
``version_2/hidden_attractors/solvers/integer.py`` to avoid a cross-package
import dependency.  Keep both copies in sync if the values ever change.
"""

EFORK_Q1_A21: float = 0.5
EFORK_Q1_A31: float = 0.5
EFORK_Q1_A32: float = -0.25
EFORK_Q1_W1: float = 2.0 / 3.0
EFORK_Q1_W2: float = 5.0 / 3.0
EFORK_Q1_W3: float = -4.0 / 3.0

__all__ = [
    "EFORK_Q1_A21",
    "EFORK_Q1_A31",
    "EFORK_Q1_A32",
    "EFORK_Q1_W1",
    "EFORK_Q1_W2",
    "EFORK_Q1_W3",
]
