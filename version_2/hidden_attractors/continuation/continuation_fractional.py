import numpy as np
from typing import Any, Dict, List, Sequence, Optional
from ..integrations.fractional_c import fractional_integrate


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
    t_transient: float = 30.0,
    t_keep: float = 30.0,
    history_times: Optional[np.ndarray] = None,
    history_states: Optional[np.ndarray] = None,
    early_stop_config: Optional[Dict] = None,
    equilibria: Optional[List[np.ndarray]] = None,
    require_c_backend: bool = True,
    allow_python_fallback: bool = False,
    q: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Execute fractional-order parameter continuation for parameter eta (lambda_values)."""
    if integrator.lower() == "abm":
        return run_fractional_continuation_abm_monolithic(
            system=system,
            seed_x0=seed_x0,
            k_gain=k_gain,
            lambda_values=lambda_values,
            h=h,
            memory_mode=memory_mode,
            memory_window_length=memory_window_length,
            div_threshold=div_threshold,
            t_transient=t_transient,
            t_keep=t_keep,
            history_times=history_times,
            history_states=history_states,
            early_stop_config=early_stop_config,
            equilibria=equilibria,
            q=q,
        )

    q_effective = q if q is not None else float(system.parameters.get("q", 1.0))

    x_in = np.asarray(seed_x0, dtype=float).copy()
    steps: List[Dict[str, Any]] = []

    hist_t = history_times
    hist_x = history_states

    p0 = system.lure.matrix + k_gain * np.outer(system.lure.input_vector, system.lure.output_vector)

    nsteps_tr = int(np.ceil(t_transient / h))
    nsteps_kp = int(np.ceil(t_keep / h))

    history_pol = "finite_window" if memory_mode == "window" else "full_caputo"

    for eta in lambda_values:
        eta_f = float(eta)

        def rhs_deformed(t_val, x_val, _eta=eta_f):
            sigma = float(system.lure.output_vector @ x_val)
            delta = float(system.lure.nonlinearity(sigma)) - k_gain * sigma
            return p0 @ x_val + _eta * system.lure.input_vector * delta

        sys_to_pass = system if (abs(k_gain) < 1e-12 and abs(eta_f - 1.0) < 1e-12) else None
        x_in_norm = float(np.linalg.norm(x_in))

        # ── 1. Transient stage ─────────────────────────────────────────────
        try:
            t_tr, x_tr, status_tr, info_tr = fractional_integrate(
                rhs=rhs_deformed,
                x0=x_in,
                q=q_effective,
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
                    early_stop_reason=err_msg,
                    history_policy=history_pol,
                    carry_state_history=True,
                    carry_derivative_history=False,
                    eta_boundary_policy="right_continuous"
                ))
                break
            raise

        if require_c_backend and not used_c and not allow_python_fallback:
            err_msg = "backend_failure:C backend unavailable and python fallback disabled"
            steps.append(_make_step_dict(
                eta=eta_f, x_in=x_in, x_out=x_in.copy(),
                trajectory=np.empty((0, 1 + len(x_in))),
                status=err_msg,
                used_c=False, rhs_src="python",
                n_steps=0, t_end=0.0, max_norm=x_in_norm,
                x_in_norm=x_in_norm, x_out_norm=x_in_norm,
                early_stop_reason=err_msg,
                history_policy=history_pol,
                carry_state_history=True,
                carry_derivative_history=False,
                eta_boundary_policy="right_continuous"
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
                early_stop_reason=status_tr if status_tr != "ok" else "",
                history_policy=history_pol,
                carry_state_history=True,
                carry_derivative_history=False,
                eta_boundary_policy="right_continuous"
            ))
            break

        x_mid = x_tr[-1].copy()

        hist_t = t_tr - t_tr[-1]
        hist_x = x_tr

        # ── 2. Keep stage ──────────────────────────────────────────────────
        try:
            t_kp, x_kp, status_kp, info_kp = fractional_integrate(
                rhs=rhs_deformed,
                x0=x_mid,
                q=q_effective,
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
                early_stop_reason=err_msg,
                history_policy=history_pol,
                carry_state_history=True,
                carry_derivative_history=False,
                eta_boundary_policy="right_continuous"
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
            early_stop_reason=status_kp if status_kp != "ok" else "",
            history_policy=history_pol,
            carry_state_history=True,
            carry_derivative_history=False,
            eta_boundary_policy="right_continuous"
        ))

        if status_kp != "ok":
            break

        hist_t = t_kp - t_kp[-1]
        hist_x = x_kp

        x_in = x_out

    return steps


def run_fractional_continuation_abm_monolithic(
    system: Any,
    seed_x0: np.ndarray,
    k_gain: float,
    lambda_values: Sequence[float],
    h: float,
    memory_mode: str = "full",
    memory_window_length: Optional[int] = None,
    div_threshold: float = 120.0,
    t_transient: float = 30.0,
    t_keep: float = 30.0,
    history_times: Optional[np.ndarray] = None,
    history_states: Optional[np.ndarray] = None,
    early_stop_config: Optional[Dict] = None,
    equilibria: Optional[List[np.ndarray]] = None,
    q: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Monolithic Python ABM fractional continuation."""
    from scipy.special import gamma
    from ..integrations.abm import eval_rhs

    x0_arr = np.asarray(seed_x0, dtype=float)
    dim = x0_arr.size
    h = float(h)
    q_effective = q if q is not None else float(system.parameters.get("q", 1.0))
    q = float(q_effective)

    nsteps_tr = int(np.ceil(t_transient / h))
    nsteps_kp = int(np.ceil(t_keep / h))
    steps_per_stage = nsteps_tr + nsteps_kp
    num_stages = len(lambda_values)
    total_new_steps = num_stages * steps_per_stage

    if history_times is not None and history_states is not None:
        history_times = np.asarray(history_times, dtype=float)
        history_states = np.asarray(history_states, dtype=float)
        K = len(history_times)
    else:
        K = 1
        history_times = np.array([0.0])
        history_states = x0_arr.reshape(1, dim)

    total_capacity = K + total_new_steps
    t_arr = np.zeros(total_capacity, dtype=float)
    x_arr = np.zeros((total_capacity, dim), dtype=float)
    f_arr = np.zeros((total_capacity, dim), dtype=float)

    t_arr[:K] = history_times
    x_arr[:K] = history_states

    p0 = system.lure.matrix + k_gain * np.outer(system.lure.input_vector, system.lure.output_vector)

    def get_stage_rhs(eta_val: float):
        def rhs_deformed(t_val, x_val):
            sigma = float(system.lure.output_vector @ x_val)
            delta = float(system.lure.nonlinearity(sigma)) - k_gain * sigma
            return p0 @ x_val + eta_val * system.lure.input_vector * delta
        return rhs_deformed

    first_rhs = get_stage_rhs(float(lambda_values[0]))
    for j in range(K):
        f_arr[j] = eval_rhs(first_rhs, t_arr[j], history_states[j])

    for step_idx in range(total_new_steps):
        t_arr[K + step_idx] = t_arr[K - 1] + (step_idx + 1) * h

    powers = np.arange(total_capacity + 2, dtype=float)
    pow_q = powers ** q
    pow_q1 = powers ** (q + 1.0)
    hq = h ** q
    pred_scale = hq / float(gamma(q + 1.0))
    val_gq2 = float(gamma(q + 2.0))
    corr_scale = hq / val_gq2 if abs(val_gq2) > 1e-15 else 0.0

    esc = early_stop_config if early_stop_config is not None else {}
    es_enabled = esc.get("enabled", True)
    div_enabled = esc.get("divergence_enabled", esc.get("divergence", {}).get("enabled", True))
    div_norm = esc.get("divergence_norm", esc.get("divergence", {}).get("norm", 80.0))
    div_consec = esc.get("divergence_consecutive_steps", esc.get("divergence", {}).get("consecutive_steps", 5))
    div_growth = esc.get("divergence_growth_factor", esc.get("divergence", {}).get("growth_factor", 1.25))
    eq_enabled = esc.get("equilibrium_enabled", esc.get("equilibrium", {}).get("enabled", True))
    eq_t = esc.get("equilibrium_tol", esc.get("equilibrium", {}).get("tol", 1e-3))
    eq_deriv = esc.get("equilibrium_derivative_tol", esc.get("equilibrium", {}).get("derivative_tol", 1e-4))
    eq_consec = esc.get("equilibrium_consecutive_steps", esc.get("equilibrium", {}).get("consecutive_steps", 200))
    eq_min_t = esc.get("equilibrium_min_time", esc.get("equilibrium", {}).get("min_time", 5.0))

    steps_records = []
    curr_n = K - 1

    for stage_idx, eta in enumerate(lambda_values):
        eta_f = float(eta)
        rhs_curr = get_stage_rhs(eta_f)

        x_in = x_arr[curr_n].copy()
        x_in_norm = float(np.linalg.norm(x_in))

        transient_start_idx = curr_n + 1
        transient_end_idx = curr_n + nsteps_tr

        keep_start_idx = transient_end_idx + 1
        keep_end_idx = transient_end_idx + nsteps_kp

        status = "ok"
        div_consec_count = 0
        growth_consec_count = 0
        prev_norm = -1.0
        eq_consec_counts = [0] * len(equilibria) if equilibria else []

        early_stopped = False
        stop_reason = ""
        last_integrated_n = curr_n

        for local_step in range(steps_per_stage):
            n = curr_n + local_step
            t_n1 = t_arr[n] + h

            if memory_mode == "window" and memory_window_length is not None:
                s_idx = max(0, n - int(memory_window_length) + 1)
            else:
                s_idx = 0

            j_range = np.arange(s_idx, n + 1)
            b_weights = pow_q[n + 1 - j_range] - pow_q[n - j_range]
            predictor = x_arr[s_idx] + pred_scale * (b_weights @ f_arr[s_idx: n + 1])

            try:
                fp = eval_rhs(rhs_curr, t_n1, predictor)
            except Exception as exc:
                status = f"solver_exception:{exc}"
                stop_reason = status
                early_stopped = True
                break

            n_prime = n - s_idx
            a0 = float(n_prime) ** (q + 1.0) - (float(n_prime) - q) * (float(n_prime) + 1.0) ** q
            if n_prime > 0:
                mid_indices = n - np.arange(s_idx + 1, n + 1)
                a_mid = (pow_q1[mid_indices + 2]
                         + pow_q1[mid_indices]
                         - 2.0 * pow_q1[mid_indices + 1])
                a_weights = np.concatenate(([a0], a_mid))
            else:
                a_weights = np.array([a0])

            corrected = x_arr[s_idx] + corr_scale * ((a_weights @ f_arr[s_idx: n + 1]) + fp)
            norm = np.linalg.norm(corrected)

            if div_threshold is not None and norm > div_threshold:
                status = "diverged"
                stop_reason = status
                x_arr[n + 1] = corrected
                last_integrated_n = n + 1
                early_stopped = True
                break

            if not np.all(np.isfinite(corrected)):
                status = "nonfinite_solution"
                stop_reason = status
                early_stopped = True
                break

            x_arr[n + 1] = corrected
            t_arr[n + 1] = t_n1

            try:
                f_arr[n + 1] = eval_rhs(rhs_curr, t_n1, corrected)
            except Exception as exc:
                status = f"solver_exception:{exc}"
                stop_reason = status
                early_stopped = True
                break

            last_integrated_n = n + 1

            if es_enabled:
                if div_enabled:
                    if norm > div_norm:
                        div_consec_count += 1
                    else:
                        div_consec_count = 0
                    if prev_norm >= 0.0:
                        if norm > div_growth * prev_norm:
                            growth_consec_count += 1
                        else:
                            growth_consec_count = 0
                    prev_norm = norm
                    if div_consec_count >= div_consec or growth_consec_count >= div_consec:
                        status = "diverged_early"
                        stop_reason = status
                        early_stopped = True
                        break
                else:
                    prev_norm = norm

                if eq_enabled and equilibria and t_n1 >= eq_min_t:
                    converged_eq_idx = -1
                    for k, eq in enumerate(equilibria):
                        diff_norm = np.linalg.norm(corrected - eq)
                        try:
                            deriv_norm = np.linalg.norm(eval_rhs(rhs_curr, t_n1, corrected))
                        except Exception:
                            deriv_norm = 9999.0

                        if diff_norm < eq_t and deriv_norm < eq_deriv:
                            eq_consec_counts[k] += 1
                        else:
                            eq_consec_counts[k] = 0

                        if eq_consec_counts[k] >= eq_consec:
                            converged_eq_idx = k
                            break
                    if converged_eq_idx != -1:
                        status = "converged_equilibrium_early"
                        stop_reason = status
                        early_stopped = True
                        break
            else:
                prev_norm = norm

        if early_stopped:
            stage_times = t_arr[transient_start_idx: last_integrated_n + 1]
            stage_states = x_arr[transient_start_idx: last_integrated_n + 1]

            traj_times = stage_times
            traj_states = stage_states

            x_out = x_arr[last_integrated_n].copy()
            x_out_norm = float(np.linalg.norm(x_out))
            max_norm = float(np.max(np.linalg.norm(stage_states, axis=1))) if len(stage_states) > 0 else x_in_norm
            traj = np.column_stack((traj_times, traj_states)) if len(traj_times) > 0 else np.empty((0, 1 + dim))

            step_record = _make_step_dict(
                eta=eta_f, x_in=x_in, x_out=x_out,
                trajectory=traj,
                status=status,
                used_c=False, rhs_src="python_native",
                n_steps=len(stage_times), t_end=float(stage_times[-1]) if len(stage_times) > 0 else 0.0,
                max_norm=max_norm, x_in_norm=x_in_norm, x_out_norm=x_out_norm,
                early_stop_reason=stop_reason,
                history_policy="finite_window" if memory_mode == "window" else "full_caputo",
                carry_state_history=True,
                carry_derivative_history=True,
                eta_boundary_policy="right_continuous"
            )
            steps_records.append(step_record)
            break
        else:
            keep_times = t_arr[keep_start_idx: keep_end_idx + 1]
            keep_states = x_arr[keep_start_idx: keep_end_idx + 1]

            x_out = x_arr[keep_end_idx].copy()
            x_out_norm = float(np.linalg.norm(x_out))

            all_stage_states = x_arr[transient_start_idx: keep_end_idx + 1]
            max_norm = float(np.max(np.linalg.norm(all_stage_states, axis=1)))
            traj = np.column_stack((keep_times, keep_states))

            step_record = _make_step_dict(
                eta=eta_f, x_in=x_in, x_out=x_out,
                trajectory=traj,
                status="ok",
                used_c=False, rhs_src="python_native",
                n_steps=steps_per_stage, t_end=float(keep_times[-1]),
                max_norm=max_norm, x_in_norm=x_in_norm, x_out_norm=x_out_norm,
                early_stop_reason="",
                history_policy="finite_window" if memory_mode == "window" else "full_caputo",
                carry_state_history=True,
                carry_derivative_history=True,
                eta_boundary_policy="right_continuous"
            )
            steps_records.append(step_record)

            curr_n = keep_end_idx
            x_in = x_out

    return steps_records


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
    history_policy: str = "full_caputo",
    carry_state_history: bool = True,
    carry_derivative_history: bool = False,
    eta_boundary_policy: str = "right_continuous",
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
        "history_policy": history_policy,
        "carry_state_history": carry_state_history,
        "carry_derivative_history": carry_derivative_history,
        "eta_boundary_policy": eta_boundary_policy,
    }
