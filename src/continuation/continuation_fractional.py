import numpy as np
from typing import Any, Dict, List, Sequence, Optional
from ..integrators.fractional_c import fractional_integrate


def run_fractional_continuation(
    system: Any,
    seed_x0: np.ndarray,
    k_gain: float,
    lambda_values: Sequence[float],
    h: float,
    memory_mode: str = "full",
    memory_window_length: Optional[int] = None,
    div_threshold: float = 120.0,
    integrator: str = "abm",
    use_c_backend: bool = True,
    # Per-step time control (resolved by caller before calling)
    t_transient: float = 30.0,
    t_keep: float = 30.0,
    # Optional harmonic prehistory (passed as pre-built arrays)
    history_times: Optional[np.ndarray] = None,
    history_states: Optional[np.ndarray] = None,
    # Early stop for continuation steps
    early_stop_config: Optional[Dict] = None,
    # Equilibria list for early stop detection
    equilibria: Optional[List[np.ndarray]] = None,
    # Backend policy
    require_c_backend: bool = True,
    allow_python_fallback: bool = False,
) -> List[Dict[str, Any]]:
    """Execute fractional-order parameter continuation for parameter eta (lambda_values).

    Parameters
    ----------
    system : Lur'e system object with P, b, r, q, psi attributes.
    seed_x0 : Initial condition (modal seed).
    k_gain : DF linearisation gain k from harmonic balance.
    lambda_values : Sequence of eta values in [0, 1].
    h : Integration step size.
    memory_mode : "full" or "window" fractional memory.
    memory_window_length : Window length (only used when memory_mode="window").
    div_threshold : Hard divergence norm limit.
    integrator : "abm" or "efork".
    use_c_backend : Whether to attempt the C native backend.
    t_transient : Transient duration for each step (seconds).
    t_keep : Keep duration for each step (seconds).
    history_times : Pre-built harmonic prehistory time array (shape (L,)).
    history_states : Pre-built harmonic prehistory state array (shape (L, dim)).
    early_stop_config : Early-stop configuration dict (see configs.py).
    equilibria : List of equilibrium arrays for convergence detection.
    require_c_backend : If True, raise on C backend failure (no silent fallback).
    allow_python_fallback : If True, fall back to Python ABM on C failure.

    Returns
    -------
    List of step dicts with keys: lambda_value, x_in, x_out, trajectory, status,
    used_c_backend, rhs_source, n_steps, t_end, max_norm, x_in_norm, x_out_norm,
    early_stop_reason.
    """
    x_in = np.asarray(seed_x0, dtype=float).copy()
    steps: List[Dict[str, Any]] = []

    # Carry pre-built prehistory for the first step; shifted after each step
    hist_t = history_times
    hist_x = history_states

    # Deformed system: D^q X = P0 X + eta * b * phi(r^T X)
    p0 = system.P + k_gain * np.outer(system.b, system.r)

    nsteps_tr = int(np.ceil(t_transient / h))
    nsteps_kp = int(np.ceil(t_keep / h))

    for eta in lambda_values:
        eta_f = float(eta)

        def rhs_deformed(t_val, x_val, _eta=eta_f):
            sigma = float(system.r @ x_val)
            delta = float(system.psi(sigma)) - k_gain * sigma
            return p0 @ x_val + _eta * system.b * delta

        # Only pass the registered system when eta=1 and k=0 (pure nonlinear)
        sys_to_pass = system if (abs(k_gain) < 1e-12 and abs(eta_f - 1.0) < 1e-12) else None

        x_in_norm = float(np.linalg.norm(x_in))

        # ── 1. Transient stage ─────────────────────────────────────────────
        try:
            t_tr, x_tr, status_tr, info_tr = fractional_integrate(
                rhs=rhs_deformed,
                x0=x_in,
                q=system.q,
                h=h,
                t_final=t_transient,
                method=integrator,
                memory_mode=memory_mode,
                memory_window_length=memory_window_length,
                history_times=hist_t,
                history_states=hist_x,
                system=sys_to_pass,
                use_c_backend=use_c_backend,
                divergence_norm=div_threshold,
                return_history=True,
                allow_python_fallback=allow_python_fallback,
                early_stop_config=early_stop_config,
                equilibria=equilibria,
            )
            used_c = info_tr.get("used_c_backend", False)
            rhs_src = info_tr.get("rhs_source", "python")
        except Exception as exc:
            err_msg = f"backend_failure:{exc}"
            if require_c_backend and not allow_python_fallback:
                steps.append(_make_step_dict(
                    eta=eta_f, x_in=x_in, x_out=x_in.copy(),
                    trajectory=np.empty((0, 1 + len(x_in))),
                    status=err_msg,
                    used_c=False, rhs_src="none",
                    n_steps=0, t_end=0.0, max_norm=x_in_norm,
                    x_in_norm=x_in_norm, x_out_norm=x_in_norm,
                    early_stop_reason=err_msg
                ))
                break
            raise

        # If C backend required but fell back to Python, fail explicitly
        if require_c_backend and not used_c and not allow_python_fallback:
            err_msg = "backend_failure:C backend unavailable and python fallback disabled"
            steps.append(_make_step_dict(
                eta=eta_f, x_in=x_in, x_out=x_in.copy(),
                trajectory=np.empty((0, 1 + len(x_in))),
                status=err_msg,
                used_c=False, rhs_src="python",
                n_steps=0, t_end=0.0, max_norm=x_in_norm,
                x_in_norm=x_in_norm, x_out_norm=x_in_norm,
                early_stop_reason=err_msg
            ))
            break

        if status_tr != "ok":
            t_tr_new = t_tr[-nsteps_tr:] if len(t_tr) >= nsteps_tr else t_tr
            x_tr_new = x_tr[-nsteps_tr:] if len(x_tr) >= nsteps_tr else x_tr
            x_out = x_tr[-1].copy() if len(x_tr) > 0 else x_in.copy()
            x_out_norm = float(np.linalg.norm(x_out))
            max_norm = float(np.max(np.linalg.norm(x_tr, axis=1))) if len(x_tr) > 0 else x_in_norm
            traj = np.column_stack((t_tr_new, x_tr_new)) if len(t_tr_new) > 0 else np.empty((0, 1 + len(x_in)))
            steps.append(_make_step_dict(
                eta=eta_f, x_in=x_in, x_out=x_out,
                trajectory=traj,
                status=status_tr,
                used_c=used_c, rhs_src=rhs_src,
                n_steps=len(t_tr), t_end=float(t_tr[-1]) if len(t_tr) > 0 else 0.0,
                max_norm=max_norm, x_in_norm=x_in_norm, x_out_norm=x_out_norm,
                early_stop_reason=status_tr if status_tr != "ok" else ""
            ))
            break

        x_mid = x_tr[-1].copy()

        # Prepare history for keep stage: shift so last point is at 0.0
        hist_t = t_tr - t_tr[-1]
        hist_x = x_tr

        # ── 2. Keep stage ──────────────────────────────────────────────────
        try:
            t_kp, x_kp, status_kp, info_kp = fractional_integrate(
                rhs=rhs_deformed,
                x0=x_mid,
                q=system.q,
                h=h,
                t_final=t_keep,
                method=integrator,
                memory_mode=memory_mode,
                memory_window_length=memory_window_length,
                history_times=hist_t,
                history_states=hist_x,
                system=sys_to_pass,
                use_c_backend=use_c_backend,
                divergence_norm=div_threshold,
                return_history=True,
                allow_python_fallback=allow_python_fallback,
                early_stop_config=early_stop_config,
                equilibria=equilibria,
            )
            used_c = info_kp.get("used_c_backend", False)
            rhs_src = info_kp.get("rhs_source", "python")
        except Exception as exc:
            err_msg = f"backend_failure:{exc}"
            x_out_norm = float(np.linalg.norm(x_mid))
            steps.append(_make_step_dict(
                eta=eta_f, x_in=x_in, x_out=x_mid,
                trajectory=np.empty((0, 1 + len(x_in))),
                status=err_msg,
                used_c=False, rhs_src="none",
                n_steps=len(t_tr), t_end=float(t_tr[-1]),
                max_norm=float(np.max(np.linalg.norm(x_tr, axis=1))) if len(x_tr) > 0 else x_in_norm,
                x_in_norm=x_in_norm, x_out_norm=x_out_norm,
                early_stop_reason=err_msg
            ))
            break

        x_out = x_kp[-1].copy()
        t_kp_kept = t_kp[-nsteps_kp:] if len(t_kp) >= nsteps_kp else t_kp
        x_kp_kept = x_kp[-nsteps_kp:] if len(x_kp) >= nsteps_kp else x_kp
        x_out_norm = float(np.linalg.norm(x_out))
        max_norm = float(np.max(np.linalg.norm(x_kp, axis=1))) if len(x_kp) > 0 else x_in_norm
        traj = np.column_stack((t_kp_kept, x_kp_kept)) if len(t_kp_kept) > 0 else np.empty((0, 1 + len(x_in)))

        steps.append(_make_step_dict(
            eta=eta_f, x_in=x_in, x_out=x_out,
            trajectory=traj,
            status=status_kp,
            used_c=used_c, rhs_src=rhs_src,
            n_steps=len(t_tr) + len(t_kp),
            t_end=float(t_kp[-1]) if len(t_kp) > 0 else 0.0,
            max_norm=max_norm, x_in_norm=x_in_norm, x_out_norm=x_out_norm,
            early_stop_reason=status_kp if status_kp != "ok" else ""
        ))

        if status_kp != "ok":
            break

        # Update history for next eta step: shift so last point is at 0.0
        hist_t = t_kp - t_kp[-1]
        hist_x = x_kp

        x_in = x_out

    return steps


def _make_step_dict(
    eta: float,
    x_in: np.ndarray,
    x_out: np.ndarray,
    trajectory: np.ndarray,
    status: str,
    used_c: bool,
    rhs_src: str,
    n_steps: int,
    t_end: float,
    max_norm: float,
    x_in_norm: float,
    x_out_norm: float,
    early_stop_reason: str,
) -> Dict[str, Any]:
    """Build a standardised continuation step record."""
    return {
        "lambda_value": eta,
        "x_in": x_in.copy(),
        "x_out": x_out.copy(),
        "trajectory": trajectory,
        "status": status,
        "used_c_backend": used_c,
        "rhs_source": rhs_src,
        "n_steps": n_steps,
        "t_end": t_end,
        "max_norm": max_norm,
        "x_in_norm": x_in_norm,
        "x_out_norm": x_out_norm,
        "early_stop_reason": early_stop_reason,
    }
