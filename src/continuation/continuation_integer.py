import numpy as np
from typing import Any, Dict, List, Optional, Sequence
from ..integrators.abm import caputo_abm_integrate
from ..integrators.efork import efork_integrate


def run_integer_continuation(
    system: Any,
    seed_x0: np.ndarray,
    k_gain: float,
    lambda_values: Sequence[float],
    h: float,
    t_transient: float = 30.0,
    t_keep: float = 30.0,
    div_threshold: float = 120.0,
    integrator: str = "abm",
    early_stop_config: Optional[Dict] = None,
    equilibria: Optional[List[np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    """Execute integer-order parameter continuation for parameter eta (lambda_values).

    Parameters
    ----------
    system : Lur'e system with attributes P, b, r, psi.
    seed_x0 : Initial condition.
    k_gain : DF linearisation gain.
    lambda_values : Sequence of eta values to sweep.
    h : Integration step size.
    t_transient : Transient duration per step (seconds).
    t_keep : Keep duration per step (seconds).
    div_threshold : Hard divergence norm limit.
    integrator : "abm" or "efork".
    early_stop_config : Early-stop configuration dict.
    equilibria : List of equilibrium arrays for convergence detection.

    Returns
    -------
    List of step dicts with keys: lambda_value, x_in, x_out, trajectory, status,
    used_c_backend, rhs_source, n_steps, t_end, max_norm, x_in_norm, x_out_norm,
    early_stop_reason.
    """
    x_in = np.asarray(seed_x0, dtype=float).copy()
    steps: List[Dict[str, Any]] = []

    # Deformed system: D_t^1 X = P0 X + eta * b * phi(r^T X)
    p0 = system.P + k_gain * np.outer(system.b, system.r)

    for eta in lambda_values:
        eta_f = float(eta)
        x_in_norm = float(np.linalg.norm(x_in))

        def rhs(x, _eta=eta_f):
            sigma = float(system.r @ x)
            delta = float(system.psi(sigma)) - k_gain * sigma
            return p0 @ x + _eta * system.b * delta

        # ── 1. Transient stage ─────────────────────────────────────────────
        if integrator == "abm":
            t_tr, x_tr, status_tr = caputo_abm_integrate(
                rhs, x_in, q=1.0, h=h, t_final=t_transient,
                divergence_norm=div_threshold, system=system,
                early_stop_config=early_stop_config,
                equilibria=equilibria,
            )
        else:  # efork
            t_tr, x_tr, status_tr = efork_integrate(
                system, x_in, q=1.0, h=h, t_final=t_transient,
                k=k_gain, eps=eta_f,
                divergence_norm=div_threshold,
                early_stop_config=early_stop_config,
                equilibria=equilibria,
            )

        if status_tr != "ok":
            x_out = x_tr[-1].copy() if len(x_tr) > 0 else x_in.copy()
            x_out_norm = float(np.linalg.norm(x_out))
            max_norm = float(np.max(np.linalg.norm(x_tr, axis=1))) if len(x_tr) > 0 else x_in_norm
            traj = np.column_stack((t_tr, x_tr)) if len(t_tr) > 0 else np.empty((0, 1 + len(x_in)))
            steps.append(_make_step_dict(
                eta=eta_f, x_in=x_in, x_out=x_out, trajectory=traj,
                status=status_tr,
                n_steps=len(t_tr), t_end=float(t_tr[-1]) if len(t_tr) > 0 else 0.0,
                max_norm=max_norm, x_in_norm=x_in_norm, x_out_norm=x_out_norm,
                early_stop_reason=status_tr,
            ))
            break

        x_mid = x_tr[-1].copy()

        # ── 2. Keep stage ──────────────────────────────────────────────────
        if integrator == "abm":
            t_kp, x_kp, status_kp = caputo_abm_integrate(
                rhs, x_mid, q=1.0, h=h, t_final=t_keep,
                divergence_norm=div_threshold, system=system,
                early_stop_config=early_stop_config,
                equilibria=equilibria,
            )
        else:  # efork
            t_kp, x_kp, status_kp = efork_integrate(
                system, x_mid, q=1.0, h=h, t_final=t_keep,
                k=k_gain, eps=eta_f,
                divergence_norm=div_threshold,
                early_stop_config=early_stop_config,
                equilibria=equilibria,
            )

        x_out = x_kp[-1].copy()
        x_out_norm = float(np.linalg.norm(x_out))
        max_norm = float(np.max(np.linalg.norm(x_kp, axis=1))) if len(x_kp) > 0 else x_in_norm
        traj = np.column_stack((t_kp, x_kp)) if len(t_kp) > 0 else np.empty((0, 1 + len(x_in)))

        steps.append(_make_step_dict(
            eta=eta_f, x_in=x_in, x_out=x_out, trajectory=traj,
            status=status_kp,
            n_steps=len(t_tr) + len(t_kp),
            t_end=float(t_kp[-1]) if len(t_kp) > 0 else 0.0,
            max_norm=max_norm, x_in_norm=x_in_norm, x_out_norm=x_out_norm,
            early_stop_reason=status_kp if status_kp != "ok" else "",
        ))

        if status_kp != "ok":
            break

        x_in = x_out

    return steps


def _make_step_dict(
    eta: float,
    x_in: np.ndarray,
    x_out: np.ndarray,
    trajectory: np.ndarray,
    status: str,
    n_steps: int,
    t_end: float,
    max_norm: float,
    x_in_norm: float,
    x_out_norm: float,
    early_stop_reason: str,
) -> Dict[str, Any]:
    """Build a standardised integer continuation step record."""
    return {
        "lambda_value": eta,
        "x_in": x_in.copy(),
        "x_out": x_out.copy(),
        "trajectory": trajectory,
        "status": status,
        "used_c_backend": False,
        "rhs_source": "python",
        "n_steps": n_steps,
        "t_end": t_end,
        "max_norm": max_norm,
        "x_in_norm": x_in_norm,
        "x_out_norm": x_out_norm,
        "early_stop_reason": early_stop_reason,
    }
