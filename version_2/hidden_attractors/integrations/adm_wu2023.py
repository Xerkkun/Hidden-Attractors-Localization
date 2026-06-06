"""ADM (Adomian Decomposition Method) integrator for the fractional Chua arctan system.

Reproduces the numerical method used in Wu et al. (2023):
    "Hidden attractors in a new fractional-order Chua system with arctan
     nonlinearity and its DSP implementation."

System equations (Caputo order q)
----------------------------------
    D^q x = -alpha*(1+m)*x + alpha*y - alpha*(n-m)*arctan(x)
    D^q y =  x - y + z
    D^q z = -beta*y - gamma*z

ADM 4th-order local update (Ruta B)
-------------------------------------
This integrator uses a **local** Adomian decomposition at each step.  The
update is a polynomial in h^q whose coefficients are computed from the current
state only.  There is **no accumulated Caputo memory** across steps.

    x_{n+1} = C1^0
             + C1^1 * h^q   / Gamma(q+1)
             + C1^2 * h^2q  / Gamma(2q+1)
             + C1^3 * h^3q  / Gamma(3q+1)
             + C1^4 * h^4q  / Gamma(4q+1)

and similarly for y (C2^k) and z (C3^k).

Scientific boundary
--------------------
This integrator is a **reproduction of the ADM method** described in Wu et al.
(2023).  It is NOT equivalent to the Adams-Bashforth-Moulton (ABM) or EFORK-3
methods, which implement Caputo full-memory integration.

- ADM local: each step depends only on the *current* state.  No convolution
  with the full history of the trajectory is performed.
- ABM / EFORK-3: each step accumulates the full Caputo kernel sum from t=0 to
  t_n.  This is the mathematically rigorous Caputo integral definition.

If ADM produces a chaotic-looking trajectory while ABM/EFORK-3 produce a
periodic orbit from the same initial conditions, this difference reflects the
distinction between the two numerical methods, NOT a confirmed difference in
the dynamics of the Caputo fractional system.  It does NOT prove the existence
of a hidden attractor under the full-memory Caputo definition.

Interface
---------
    adm_wu2023_integrate(params, x0, q, h, N, divergence_norm) ->
        (times, states, status, info)

    params: dict or object with keys: alpha, beta, gamma, m, n
    x0:     array-like (3,)
    q:      Caputo order (0 < q <= 1)
    h:      step size
    N:      number of steps
    divergence_norm: float, halt if norm(X) exceeds this value
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Gamma factor cache
# ---------------------------------------------------------------------------

def _gamma_factors(q: float) -> Dict[str, float]:
    """Pre-compute all Gamma values needed for the 4th-order ADM coefficients."""
    g = {
        "gq1":  math.gamma(q + 1.0),            # Gamma(q+1)
        "g2q1": math.gamma(2.0 * q + 1.0),       # Gamma(2q+1)
        "g3q1": math.gamma(3.0 * q + 1.0),       # Gamma(3q+1)
        "g4q1": math.gamma(4.0 * q + 1.0),       # Gamma(4q+1)
        "g2q2": math.gamma(2.0 * (q + 1.0)),     # Gamma(2(q+1)) = Gamma(2q+2)
        "g3q3": math.gamma(3.0 * (q + 1.0)),     # Gamma(3(q+1)) = Gamma(3q+3)
    }
    # Powers of h: h^q, h^2q, h^3q, h^4q  — computed per step, not here.
    return g


# ---------------------------------------------------------------------------
# Single ADM step
# ---------------------------------------------------------------------------

def _adm_step(
    x0: float, y0: float, z0: float,
    alpha: float, beta: float, gamma: float,
    m: float, n: float,
    q: float, h: float,
    gf: Dict[str, float],
) -> Tuple[float, float, float]:
    """Advance one step of the 4th-order ADM scheme.

    Parameters
    ----------
    x0, y0, z0 : Current state (C^0 components).
    alpha, beta, gamma, m, n : System parameters.
    q : Caputo order.
    h : Step size.
    gf : Pre-computed Gamma factor dictionary from :func:`_gamma_factors`.

    Returns
    -------
    x1, y1, z1 : State at the next step.

    Notes
    -----
    Nonlinear term derivative at x0 (arctan decomposition):
        Let s = x0^2 + 1
        d/dx  arctan(x)|x=x0 = 1/s                               → g0
        d2/dx2 arctan(x)|x=x0 = -2*x0/s^2                        → g1_raw  (unnorm.)
        d3/dx3 arctan(x)|x=x0 = 8*x0^2/s^3 - 2/s^2              → g2_raw  (unnorm.)
    """
    nm = float(n) - float(m)          # (n - m)
    s  = x0 * x0 + 1.0               # x0^2 + 1
    s2 = s * s
    s3 = s2 * s

    # First derivative of arctan at x0
    g0 = 1.0 / s

    # Second-order term (raw second derivative of arctan, not divided by 2!)
    # Used in the Adomian polynomial A1 = d/du[arctan(u)]|_{u=x0} * C1^1
    # The factor (1/2!) appears in A2 for (C1^1)^2 * g1 term.
    g1_raw = -2.0 * x0 / s2          # d^2/dx^2 arctan(x0) [raw]

    # Third-order term (raw third derivative of arctan)
    g2_raw = 8.0 * x0 * x0 / s3 - 2.0 / s2   # d^3/dx^3 arctan(x0) [raw]

    # --------------------------------------------------------------------------
    # C^1 coefficients  (first Adomian component)
    # --------------------------------------------------------------------------
    C1_1 = -alpha * (1.0 + m) * x0 + alpha * y0 - alpha * nm * math.atan(x0)
    C2_1 = x0 - y0 + z0
    C3_1 = -beta * y0 - gamma * z0

    # --------------------------------------------------------------------------
    # C^2 coefficients
    # For the arctan nonlinearity, the second Adomian polynomial is:
    #   A1(x0, C1_1) = d/dx[arctan(x)]|x=x0 * C1_1 = g0 * C1_1
    # --------------------------------------------------------------------------
    C1_2 = (-alpha * (1.0 + m) * C1_1
             + alpha * C2_1
             - alpha * nm * g0 * C1_1)
    C2_2 = C1_1 - C2_1 + C3_1
    C3_2 = -beta * C2_1 - gamma * C3_1

    # --------------------------------------------------------------------------
    # C^3 coefficients
    # Adomian polynomial A2:
    #   A2 = g0 * C1_2 + (1/2!) * g1_raw * (C1_1)^2 * Gamma(2q+1)/Gamma(2(q+1))
    #
    # Paper (Wu 2023, Eq. 22): ratio = Gamma(2q+1) / Gamma_2(q+1)
    # where Gamma_2(q+1) := Gamma(2*(q+1)) = Gamma(2q+2)  [NOT Gamma(q+1)^2]
    # --------------------------------------------------------------------------
    ratio_A2 = gf["g2q1"] / gf["g2q2"]     # Gamma(2q+1) / Gamma(2q+2) = 1/(2q+1)

    arctan_A2 = (g0 * C1_2
                 + 0.5 * g1_raw * C1_1 * C1_1 * ratio_A2)

    C1_3 = (-alpha * (1.0 + m) * C1_2
             + alpha * C2_2
             - alpha * nm * arctan_A2)
    C2_3 = C1_2 - C2_2 + C3_2
    C3_3 = -beta * C2_2 - gamma * C3_2

    # --------------------------------------------------------------------------
    # C^4 coefficients
    # Adomian polynomial A3:
    #   A3 = g0 * C1_3
    #        + g1_raw * C1_1 * C1_2 * Gamma(3q+1) / (Gamma(q+1)*Gamma(2q+1))
    #        + (1/6) * g2_raw * (C1_1)^3 * Gamma(3q+1) / Gamma(3(q+1))
    #
    # Here the Gamma ratios arise from the Cauchy product of fractional power
    # series terms in the Adomian decomposition.
    # --------------------------------------------------------------------------
    ratio_A3a = gf["g3q1"] / (gf["gq1"] * gf["g2q1"])  # Gamma(3q+1)/(Gamma(q+1)*Gamma(2q+1))
    ratio_A3b = gf["g3q1"] / gf["g3q3"]                 # Gamma(3q+1)/Gamma(3(q+1))=Gamma(3q+3)  [Wu2023 Eq.23]

    arctan_A3 = (g0 * C1_3
                 + g1_raw * C1_1 * C1_2 * ratio_A3a
                 + (1.0 / 6.0) * g2_raw * C1_1 * C1_1 * C1_1 * ratio_A3b)

    C1_4 = (-alpha * (1.0 + m) * C1_3
             + alpha * C2_3
             - alpha * nm * arctan_A3)
    C2_4 = C1_3 - C2_3 + C3_3
    C3_4 = -beta * C2_3 - gamma * C3_3

    # --------------------------------------------------------------------------
    # Series sum: x_{n+1} = sum_{k=0}^{4} C^k * h^{kq} / Gamma(kq+1)
    # --------------------------------------------------------------------------
    hq  = h ** q
    h2q = hq * hq
    h3q = h2q * hq
    h4q = h3q * hq

    x1 = (x0
          + C1_1 * hq  / gf["gq1"]
          + C1_2 * h2q / gf["g2q1"]
          + C1_3 * h3q / gf["g3q1"]
          + C1_4 * h4q / gf["g4q1"])

    y1 = (y0
          + C2_1 * hq  / gf["gq1"]
          + C2_2 * h2q / gf["g2q1"]
          + C2_3 * h3q / gf["g3q1"]
          + C2_4 * h4q / gf["g4q1"])

    z1 = (z0
          + C3_1 * hq  / gf["gq1"]
          + C3_2 * h2q / gf["g2q1"]
          + C3_3 * h3q / gf["g3q1"]
          + C3_4 * h4q / gf["g4q1"])

    return x1, y1, z1


# ---------------------------------------------------------------------------
# Public integrator
# ---------------------------------------------------------------------------

def adm_wu2023_integrate(
    params: Any,
    x0: np.ndarray,
    q: float,
    h: float,
    N: int,
    divergence_norm: float = 120.0,
) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]:
    """Integrate the Wu2023 fractional Chua arctan system using the 4th-order ADM.

    Parameters
    ----------
    params : object or dict with attributes/keys: alpha, beta, gamma, m, n.
    x0 : array-like (3,) — initial condition [x, y, z].
    q : Caputo order, 0 < q <= 1.
    h : Step size.
    N : Number of integration steps (t_final = N * h).
    divergence_norm : Stop if ||X|| > divergence_norm.

    Returns
    -------
    times : ndarray (M,) — time points.
    states : ndarray (M, 3) — state trajectory [x, y, z].
    status : str — one of "ok", "diverged", "nonfinite".
    info : dict — metadata including integrator_class and scientific_label.

    Notes
    -----
    This is a **local ADM** integrator: each step uses only the current
    state, with no Caputo history accumulation.  It reproduces the method
    of Wu et al. (2023) for comparison purposes.

    Do NOT use this as a substitute for full-memory Caputo integration
    (ABM or EFORK-3) when rigorous fractional-order analysis is required.
    """
    # ── Parameter extraction ─────────────────────────────────────────────────
    if isinstance(params, dict):
        alpha = float(params["alpha"])
        beta  = float(params["beta"])
        gamma = float(params["gamma"])
        m_raw = params["m"] if "m" in params else params.get("a1")
        if m_raw is None:
            raise ValueError("ADM Wu2023 parameters require either 'm' or 'a1'.")
        m = float(m_raw)
        if "n" in params and params["n"] is not None:
            n_param = float(params["n"])
        elif "a2" in params and params["a2"] is not None:
            n_param = m + float(params["a2"])
        else:
            raise ValueError("ADM Wu2023 parameters require either 'n' or 'a2'.")
    else:
        alpha = float(params.alpha)
        beta  = float(params.beta)
        gamma = float(params.gamma)
        m_raw = getattr(params, "m", None)
        if m_raw is None:
            m_raw = getattr(params, "a1", None)
        if m_raw is None:
            raise ValueError("ADM Wu2023 parameters require either 'm' or 'a1'.")
        m = float(m_raw)
        n_raw = getattr(params, "n", None)
        if n_raw is not None:
            n_param = float(n_raw)
        else:
            a2_raw = getattr(params, "a2", None)
            if a2_raw is None:
                raise ValueError("ADM Wu2023 parameters require either 'n' or 'a2'.")
            n_param = m + float(a2_raw)

    q    = float(q)
    h    = float(h)
    N    = int(N)

    if not (0.0 < q <= 1.0):
        raise ValueError(f"ADM Wu2023 requires 0 < q <= 1, got q={q}.")
    if h <= 0.0:
        raise ValueError(f"Step size h must be positive, got h={h}.")
    if N <= 0:
        raise ValueError(f"Number of steps N must be positive, got N={N}.")

    # ── Pre-compute Gamma factors ─────────────────────────────────────────────
    gf = _gamma_factors(q)

    # ── Allocate arrays ───────────────────────────────────────────────────────
    x0_arr = np.asarray(x0, dtype=float).ravel()
    if x0_arr.size != 3:
        raise ValueError(f"Initial condition x0 must have 3 elements, got {x0_arr.size}.")

    times  = np.empty(N + 1, dtype=float)
    states = np.empty((N + 1, 3), dtype=float)

    times[0]    = 0.0
    states[0]   = x0_arr

    status = "ok"
    last_n = 0

    cx, cy, cz = float(x0_arr[0]), float(x0_arr[1]), float(x0_arr[2])

    # ── Main integration loop ─────────────────────────────────────────────────
    for step_index in range(N):
        nx, ny, nz = _adm_step(
            cx, cy, cz,
            alpha, beta, gamma, m, n_param,
            q, h, gf,
        )

        # ── Finite check ─────────────────────────────────────────────────────
        if not (math.isfinite(nx) and math.isfinite(ny) and math.isfinite(nz)):
            status = "nonfinite"
            break

        # ── Divergence check ─────────────────────────────────────────────────
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm > divergence_norm:
            times[step_index + 1]  = (step_index + 1) * h
            states[step_index + 1] = [nx, ny, nz]
            last_n = step_index + 1
            status = "diverged"
            break

        times[step_index + 1]  = (step_index + 1) * h
        states[step_index + 1] = [nx, ny, nz]
        last_n = step_index + 1
        cx, cy, cz = nx, ny, nz

    info: Dict[str, Any] = {
        "integrator": "adm_wu2023",
        "integrator_class": "adm_local_reproduction",
        "scientific_label": (
            "ADM reproduction of Wu et al. 2023. "
            "Local update only — no Caputo full-memory history. "
            "NOT equivalent to ABM or EFORK-3 Caputo integration."
        ),
        "hidden_verified": False,
        "q": q,
        "h": h,
        "N": N,
        "steps_completed": last_n,
        "t_final_reached": float(times[last_n]),
        "divergence_norm_threshold": divergence_norm,
        "caputo_memory": "none — local ADM step only",
    }

    return times[: last_n + 1], states[: last_n + 1], status, info


# ---------------------------------------------------------------------------
# Convenience: integrate from dict config
# ---------------------------------------------------------------------------

def adm_wu2023_integrate_from_config(
    config: Dict[str, Any],
    x0: np.ndarray,
    label: str = "unnamed",
) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, Any]]:
    """Wrap :func:`adm_wu2023_integrate` reading params/options from a config dict.

    Keys read: alpha, beta, gamma, m, n, q, h, N (or t_final), divergence_norm.
    """
    h = float(config.get("h", 0.01))
    if "N" in config:
        N = int(config["N"])
    else:
        t_final = float(config.get("t_final", 100.0))
        N = int(math.ceil(t_final / h))

    times, states, status, info = adm_wu2023_integrate(
        params=config,
        x0=x0,
        q=float(config.get("q", 0.99)),
        h=h,
        N=N,
        divergence_norm=float(config.get("divergence_norm", 120.0)),
    )
    info["label"] = label
    return times, states, status, info


# ---------------------------------------------------------------------------
# RHS helper (for external consistency checks)
# ---------------------------------------------------------------------------

def rhs_chua_arctan(x: np.ndarray, params: Any) -> np.ndarray:
    """Evaluate the RHS vector field F(X) of the Wu2023 system.

    F1 = -alpha*(1+m)*x + alpha*y - alpha*(n-m)*arctan(x)
    F2 =  x - y + z
    F3 = -beta*y - gamma*z

    Used in unit tests to verify consistency with ChuaArctanSystem.evaluate_rhs.
    """
    if isinstance(params, dict):
        alpha = float(params["alpha"])
        beta  = float(params["beta"])
        gamma = float(params["gamma"])
        m_raw = params["m"] if "m" in params else params.get("a1")
        if m_raw is None:
            raise ValueError("ADM Wu2023 parameters require either 'm' or 'a1'.")
        m = float(m_raw)
        if "n" in params and params["n"] is not None:
            n = float(params["n"])
        elif "a2" in params and params["a2"] is not None:
            n = m + float(params["a2"])
        else:
            raise ValueError("ADM Wu2023 parameters require either 'n' or 'a2'.")
    else:
        alpha = float(params.alpha)
        beta  = float(params.beta)
        gamma = float(params.gamma)
        m_raw = getattr(params, "m", None)
        if m_raw is None:
            m_raw = getattr(params, "a1", None)
        if m_raw is None:
            raise ValueError("ADM Wu2023 parameters require either 'm' or 'a1'.")
        m = float(m_raw)
        n_raw = getattr(params, "n", None)
        if n_raw is not None:
            n = float(n_raw)
        else:
            a2_raw = getattr(params, "a2", None)
            if a2_raw is None:
                raise ValueError("ADM Wu2023 parameters require either 'n' or 'a2'.")
            n = m + float(a2_raw)

    xv, yv, zv = float(x[0]), float(x[1]), float(x[2])
    nm = n - m
    f1 = -alpha * (1.0 + m) * xv + alpha * yv - alpha * nm * math.atan(xv)
    f2 = xv - yv + zv
    f3 = -beta * yv - gamma * zv
    return np.array([f1, f2, f3], dtype=float)
