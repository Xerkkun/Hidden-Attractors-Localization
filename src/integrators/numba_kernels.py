"""Numba JIT-compiled kernels for the integer-order (q=1.0) EFORK-3 integrator.

Bypasses the Python interpreter overhead in the inner integration loop,
providing 10×–50× speedup over pure-Python NumPy for the q=1.0 path.

Requires numba >= 0.63.0 (Python 3.14 compatible).

Architecture
------------
The module is structured in two layers:

1. ``@njit`` kernel (``efork3_q1_integrate_numba``):
   Pure numeric function compiled to native machine code. Accepts only
   scalars, float64 arrays, and integer flags — no Python objects.

2. Python wrapper (``integrate_efork3_q1_numba``):
   Accepts the standard ``system`` object + dicts/lists, extracts the
   numeric parameters, calls the kernel, and maps integer status codes
   back to status strings. Returns ``None`` if Numba is not installed or
   the system type is unknown (caller falls back to Python path).

Usage
-----
This module is called automatically from ``general.py`` when
``q == 1.0``. Users do not need to import it directly.
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# Optional Numba import — graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False
    # Define a no-op decorator so the rest of the module can be parsed
    def njit(*args, **kwargs):  # type: ignore[misc]
        def _decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _decorator

# ---------------------------------------------------------------------------
# Integer constants visible to both @njit world and Python world
# ---------------------------------------------------------------------------
# RHS / nonlinearity type flags
PSI_SATURATION: int = 0   # psi(sigma) = (m0 - m1) * clip(sigma, -1, 1)
PSI_ARCTAN:     int = 1   # psi(sigma) = (n - m) * arctan(sigma)
PSI_POLYNOMIAL: int = 2   # psi(sigma) = coeff * (sigma^3 - sigma)

# Status codes (mapped back to strings by the Python wrapper)
STATUS_OK:               int = 0
STATUS_DIVERGED:         int = 1   # hard norm threshold exceeded
STATUS_DIVERGED_EARLY:   int = 2   # progressive divergence detected
STATUS_CONVERGED_EQ:     int = 3   # trajectory settled at equilibrium

_STATUS_TO_STR = {
    STATUS_OK:             "ok",
    STATUS_DIVERGED:       "diverged",
    STATUS_DIVERGED_EARLY: "diverged_early",
    STATUS_CONVERGED_EQ:   "converged_equilibrium_early",
}


# ---------------------------------------------------------------------------
# Layer 1 — @njit inner kernel (compiled to machine code)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True, nogil=True)
def _eval_rhs_njit(
    x: np.ndarray,
    P: np.ndarray,
    b: np.ndarray,
    psi_coeff: float,
    psi_kind: int,
) -> np.ndarray:
    """Evaluate the Chua system RHS for 3-dimensional state vector x.

    The Chua Lur'e form is:  dx/dt = P @ x + b * psi(r^T x)
    with r = [1, 0, 0], so sigma = x[0].

    Parameters
    ----------
    x         : float64[3]   current state
    P         : float64[3,3] linear part matrix (C-contiguous)
    b         : float64[3]   nonlinearity input vector
    psi_coeff : float        (m0 - m1) for saturation, (n - m) for arctan
    psi_kind  : int          PSI_SATURATION or PSI_ARCTAN
    """
    sigma = x[0]

    # Evaluate nonlinearity
    if psi_kind == PSI_SATURATION:
        # Saturación: psi(sigma) = psi_coeff * clip(sigma, -1, 1)
        if sigma > 1.0:
            sigma_c = 1.0
        elif sigma < -1.0:
            sigma_c = -1.0
        else:
            sigma_c = sigma
        psi = psi_coeff * sigma_c
    elif psi_kind == PSI_ARCTAN:
        # Arctan: psi(sigma) = psi_coeff * atan(sigma)
        psi = psi_coeff * math.atan(sigma)
    else:
        # Polynomial: psi(sigma) = psi_coeff * (sigma**3 - sigma)
        psi = psi_coeff * (sigma**3 - sigma)

    # Manual 3×3 matrix-vector multiply + b*psi (avoids NumPy overhead for dim=3)
    out = np.empty(3, dtype=np.float64)
    out[0] = P[0, 0]*x[0] + P[0, 1]*x[1] + P[0, 2]*x[2] + b[0] * psi
    out[1] = P[1, 0]*x[0] + P[1, 1]*x[1] + P[1, 2]*x[2] + b[1] * psi
    out[2] = P[2, 0]*x[0] + P[2, 1]*x[1] + P[2, 2]*x[2] + b[2] * psi
    return out


@njit(cache=True, fastmath=True, nogil=True)
def efork3_q1_integrate_numba(
    x0: np.ndarray,
    P: np.ndarray,
    b: np.ndarray,
    psi_coeff: float,
    psi_kind: int,
    h: float,
    n_steps: int,
    # Hard divergence (outer boundary)
    divergence_norm_hard: float,
    # Early-stop parameters
    es_enabled: bool,
    div_enabled: bool,
    div_norm_es: float,
    div_consec_threshold: int,
    div_growth_factor: float,
    eq_enabled: bool,
    eq_tol: float,
    eq_deriv_tol: float,
    eq_consec_threshold: int,
    eq_min_t: float,
    equilibria: np.ndarray,   # shape (N_eq, 3); (0, 3) if none
):
    """EFORK-3 (q→1 limit) integration loop compiled to native code.

    Implements the three-stage explicit fractional Runge-Kutta scheme at
    the q=1.0 integer limit (Ghoreishi et al., 2023).  All early-stopping
    logic (divergence, equilibrium convergence) is included and runs at
    native speed.

    Parameters
    ----------
    x0                   : float64[3]      initial condition
    P                    : float64[3,3]    system matrix (C-contiguous)
    b                    : float64[3]      nonlinearity vector
    psi_coeff            : float           nonlinearity coefficient
    psi_kind             : int             PSI_SATURATION or PSI_ARCTAN
    h                    : float           step size
    n_steps              : int             number of integration steps
    divergence_norm_hard : float           hard upper norm limit (> 0); -1 = disabled
    es_enabled           : bool            master early-stop switch
    div_enabled          : bool            enable divergence early-stop
    div_norm_es          : float           soft divergence norm threshold
    div_consec_threshold : int             consecutive steps above threshold to trigger
    div_growth_factor    : float           norm growth factor for progressive divergence
    eq_enabled           : bool            enable equilibrium convergence early-stop
    eq_tol               : float           state distance tolerance to equilibrium
    eq_deriv_tol         : float           derivative norm tolerance at equilibrium
    eq_consec_threshold  : int             consecutive steps within tolerance to trigger
    eq_min_t             : float           minimum integration time before eq check
    equilibria           : float64[N,3]    equilibrium points to check against

    Returns
    -------
    x_arr    : float64[n_steps+1, 3]   full trajectory (all steps allocated)
    last_idx : int                      last valid index in x_arr
    status   : int                      STATUS_* integer code
    """
    # EFORK-3 Butcher tableau coefficients for q → 1
    # (Ghoreishi et al., 2023 — see _q1_coefficients.py)
    A21 =  0.5
    A31 =  0.5
    A32 = -0.25
    W1  =  2.0 / 3.0
    W2  =  5.0 / 3.0
    W3  = -4.0 / 3.0

    # Allocate full trajectory
    x_arr = np.zeros((n_steps + 1, 3), dtype=np.float64)
    x_arr[0, 0] = x0[0]
    x_arr[0, 1] = x0[1]
    x_arr[0, 2] = x0[2]

    x = x0.copy()
    status   = STATUS_OK
    last_idx = 0

    n_eqs            = equilibria.shape[0]
    eq_consec_counts = np.zeros(n_eqs, dtype=np.int64)

    div_consec_count    = 0
    growth_consec_count = 0
    prev_norm           = -1.0

    for n in range(n_steps):
        t_next = (n + 1) * h

        # --- EFORK-3 stages ---
        k1 = h * _eval_rhs_njit(x, P, b, psi_coeff, psi_kind)
        k2 = h * _eval_rhs_njit(x + A21 * k1, P, b, psi_coeff, psi_kind)
        k3 = h * _eval_rhs_njit(x + A31 * k1 + A32 * k2, P, b, psi_coeff, psi_kind)
        x_next = x + W1 * k1 + W2 * k2 + W3 * k3

        # --- Hard divergence check ---
        norm = math.sqrt(x_next[0]**2 + x_next[1]**2 + x_next[2]**2)
        if divergence_norm_hard > 0.0 and norm > divergence_norm_hard:
            x_arr[n + 1, 0] = x_next[0]
            x_arr[n + 1, 1] = x_next[1]
            x_arr[n + 1, 2] = x_next[2]
            last_idx = n + 1
            status   = STATUS_DIVERGED
            break

        # Accept step
        x[0] = x_next[0]
        x[1] = x_next[1]
        x[2] = x_next[2]
        x_arr[n + 1, 0] = x[0]
        x_arr[n + 1, 1] = x[1]
        x_arr[n + 1, 2] = x[2]
        last_idx = n + 1

        # --- Early-stop checks ---
        if es_enabled:
            # 1. Divergence by soft norm threshold / progressive growth
            if div_enabled:
                if norm > div_norm_es:
                    div_consec_count += 1
                else:
                    div_consec_count = 0

                if prev_norm >= 0.0 and norm > div_growth_factor * prev_norm:
                    growth_consec_count += 1
                else:
                    growth_consec_count = 0

                prev_norm = norm

                if (div_consec_count >= div_consec_threshold or
                        growth_consec_count >= div_consec_threshold):
                    status = STATUS_DIVERGED_EARLY
                    break
            else:
                prev_norm = norm

            # 2. Equilibrium convergence (check once per step if enough time elapsed)
            if eq_enabled and n_eqs > 0 and t_next >= eq_min_t:
                # Evaluate derivative at current x_next (reuse _eval_rhs_njit)
                deriv    = _eval_rhs_njit(x, P, b, psi_coeff, psi_kind)
                deriv_nm = math.sqrt(deriv[0]**2 + deriv[1]**2 + deriv[2]**2)

                triggered = False
                for k in range(n_eqs):
                    dx0 = x[0] - equilibria[k, 0]
                    dx1 = x[1] - equilibria[k, 1]
                    dx2 = x[2] - equilibria[k, 2]
                    diff_nm = math.sqrt(dx0**2 + dx1**2 + dx2**2)

                    if diff_nm < eq_tol and deriv_nm < eq_deriv_tol:
                        eq_consec_counts[k] += 1
                    else:
                        eq_consec_counts[k] = 0

                    if eq_consec_counts[k] >= eq_consec_threshold:
                        status    = STATUS_CONVERGED_EQ
                        triggered = True
                        break

                if triggered:
                    break
        else:
            prev_norm = norm

    return x_arr, last_idx, status


# ---------------------------------------------------------------------------
# Layer 2 — Python wrapper (handles objects, dicts, strings)
# ---------------------------------------------------------------------------

def integrate_efork3_q1_numba(
    system,
    x0,
    h: float,
    t_final: float,
    divergence_norm=120.0,
    early_stop_config=None,
    equilibria=None,
):
    """Python entry point for the Numba-accelerated EFORK-3 q=1.0 integrator.

    Parameters
    ----------
    system          : ChuaSaturationSystem | ChuaArctanSystem
                      System object (used only to extract numeric params).
    x0              : array-like, shape (3,)
    h               : float   step size
    t_final         : float   integration end time
    divergence_norm : float   hard divergence norm (passed to kernel)
    early_stop_config : dict | None
    equilibria      : list[np.ndarray] | None   equilibrium points

    Returns
    -------
    (t_arr, x_arr, status_str) on success, or ``None`` if Numba is
    unavailable or the system type is not supported (signals the caller
    to use the Python fallback path).
    """
    if not NUMBA_AVAILABLE:
        return None

    # Resolve system type and extract numeric parameters
    # Import here to avoid circular imports at module level
    try:
        from ..systems.chua_saturation import ChuaSaturationSystem
        from ..systems.chua_arctan import ChuaArctanSystem
        from ..systems.chua_polynomial import ChuaPolynomialSystem
    except ImportError:
        return None

    if isinstance(system, ChuaSaturationSystem):
        psi_kind  = PSI_SATURATION
        psi_coeff = float(system.m0 - system.m1)
    elif isinstance(system, ChuaArctanSystem):
        psi_kind  = PSI_ARCTAN
        psi_coeff = float(system.n - system.m)
    elif isinstance(system, ChuaPolynomialSystem):
        psi_kind  = PSI_POLYNOMIAL
        psi_coeff = float(system.coeff)
    else:
        # Unknown system type — fall back to Python path
        return None

    # Ensure contiguous float64 arrays (Numba requires C-contiguous)
    P    = np.ascontiguousarray(system.P, dtype=np.float64)
    b    = np.ascontiguousarray(system.b, dtype=np.float64)
    x0_a = np.asarray(x0, dtype=np.float64)

    n_steps = int(np.ceil(t_final / h))

    # Parse early-stop config (mirrors the logic in general.py)
    esc      = early_stop_config if early_stop_config is not None else {}
    es_en    = bool(esc.get("enabled", True))
    div_en   = bool(esc.get("divergence_enabled",
                            esc.get("divergence", {}).get("enabled", True)))
    div_nm   = float(esc.get("divergence_norm",
                             esc.get("divergence", {}).get("norm", 80.0)))
    div_con  = int(esc.get("divergence_consecutive_steps",
                           esc.get("divergence", {}).get("consecutive_steps", 5)))
    div_gr   = float(esc.get("divergence_growth_factor",
                             esc.get("divergence", {}).get("growth_factor", 1.25)))
    eq_en    = bool(esc.get("equilibrium_enabled",
                            esc.get("equilibrium", {}).get("enabled", True)))
    eq_tol   = float(esc.get("equilibrium_tol",
                             esc.get("equilibrium", {}).get("tol", 1e-3)))
    eq_drv   = float(esc.get("equilibrium_derivative_tol",
                             esc.get("equilibrium", {}).get("derivative_tol", 1e-4)))
    eq_con   = int(esc.get("equilibrium_consecutive_steps",
                           esc.get("equilibrium", {}).get("consecutive_steps", 200)))
    eq_mt    = float(esc.get("equilibrium_min_time",
                             esc.get("equilibrium", {}).get("min_time", 5.0)))

    div_hard = float(divergence_norm) if divergence_norm is not None else -1.0

    # Build equilibria array for the kernel (shape (N_eq, 3))
    if equilibria and len(equilibria) > 0:
        eq_arr = np.array(
            [np.asarray(e, dtype=np.float64) for e in equilibria],
            dtype=np.float64
        )
    else:
        eq_arr = np.zeros((0, 3), dtype=np.float64)

    # Call the @njit kernel
    x_arr, last_idx, status_code = efork3_q1_integrate_numba(
        x0_a, P, b, psi_coeff, psi_kind, h, n_steps,
        div_hard,
        es_en, div_en, div_nm, div_con, div_gr,
        eq_en, eq_tol, eq_drv, eq_con, eq_mt,
        eq_arr,
    )

    # Build time array and map status code to string
    t_arr      = np.arange(last_idx + 1, dtype=np.float64) * h
    status_str = _STATUS_TO_STR.get(int(status_code), "ok")

    return t_arr, x_arr[:last_idx + 1], status_str


# ---------------------------------------------------------------------------
# Benchmark utility (for quick manual verification)
# ---------------------------------------------------------------------------

def benchmark(n_steps: int = 50_000, n_trials: int = 5):
    """Compare Python-loop vs Numba kernel for a standard Chua integration.

    Prints wall-clock times and speedup ratio.  Run with::

        python -c "from src.integrators.numba_kernels import benchmark; benchmark()"
    """
    import time

    try:
        from ..systems.chua_saturation import ChuaSaturationSystem
        from . import general as gen_mod
    except ImportError:
        print("[benchmark] Could not import system/integrator modules.")
        return

    system = ChuaSaturationSystem()
    x0     = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    h      = 0.01
    t_final = n_steps * h

    # --- Warm-up (triggers JIT compilation) ---
    print(f"[benchmark] Numba available: {NUMBA_AVAILABLE}")
    if NUMBA_AVAILABLE:
        print("[benchmark] Warming up JIT (first call compiles to native code)...")
        t0 = time.perf_counter()
        integrate_efork3_q1_numba(system, x0, h, t_final)
        print(f"[benchmark] JIT compile+run: {time.perf_counter() - t0:.2f}s")

    # --- Numba timing ---
    numba_times = []
    if NUMBA_AVAILABLE:
        for _ in range(n_trials):
            t0 = time.perf_counter()
            integrate_efork3_q1_numba(system, x0, h, t_final)
            numba_times.append(time.perf_counter() - t0)
        avg_numba = sum(numba_times) / len(numba_times)
        print(f"[benchmark] Numba  avg ({n_trials} trials, {n_steps} steps): {avg_numba*1000:.2f} ms")

    # --- Python timing ---
    from .general import integrate_general
    python_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        integrate_general(
            rhs=lambda t, x: system.evaluate_rhs(x),
            x0=x0, q=1.0, h=h, t_final=t_final,
            integrator="efork", system=None,  # force Python path
        )
        python_times.append(time.perf_counter() - t0)
    avg_python = sum(python_times) / len(python_times)
    print(f"[benchmark] Python avg ({n_trials} trials, {n_steps} steps): {avg_python*1000:.2f} ms")

    if NUMBA_AVAILABLE and avg_numba > 0:
        print(f"[benchmark] Speedup: {avg_python / avg_numba:.1f}×")
