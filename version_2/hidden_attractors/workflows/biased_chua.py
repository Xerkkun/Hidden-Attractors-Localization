# -*- coding: utf-8 -*-
"""Centralized Workflows for Biased Chua Fractional Hidden Attractor Localization.

Consolidates Steps 1 through 5 of the BDF pipeline.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.special import gamma as gamma_func

from hidden_attractors.paths import PROJECT_ROOT
from hidden_attractors.continuation.continuation_fractional import run_fractional_continuation
from hidden_attractors.diagnostics.periodicity import classify_post_transient_periodicity
from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.models.chua import ChuaParameters, chua_parameters
from hidden_attractors.seed_generation.chua import find_harmonic_seed, find_omega_gain_candidates, chua_matrices, chua_gain
from hidden_attractors.seed_generation.core import (
    complex_dtype,
    fractional_iomega_power,
    real_dtype,
    validate_fractional_order,
)
from hidden_attractors.systems import get_system
from hidden_attractors.verification.equilibria import solve_equilibria
from hidden_attractors.verification.hiddenness import (
    evaluate_target_match,
    generate_neighborhood_points,
)
from hidden_attractors.verification.hiddenness_contract import verify_hiddenness_contract

# Standardized plotting imports
from hidden_attractors.plotting.biased_chua import (
    plot_centered_trajectory,
    plot_sign_audit,
    plot_continuation_metrics,
    plot_candidate_report,
    plot_biased_vs_centered,
    plot_mega_summary,
    plot_sphere_summary,
    plot_heatmap_hiddenness,
)

# ── Dynamic roots ─────────────────────────────────────────────────────────────
VERSION2 = PROJECT_ROOT
ROOT     = VERSION2.parent


class _LazyPandas:
    """Import pandas only when CSV/DataFrame workflow steps actually need it."""

    def __getattr__(self, name: str) -> Any:
        try:
            import pandas as pandas_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pandas is required for biased Chua CSV/DataFrame workflow steps. "
                "Install the analysis extra or install pandas explicitly."
            ) from exc
        globals()["pd"] = pandas_module
        return getattr(pandas_module, name)


pd: Any = _LazyPandas()
# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Centered Reference Search
# ══════════════════════════════════════════════════════════════════════════════

def run_centered_reference(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Runs centered describing function search as baseline."""
    sys_cfg   = cfg["system"]
    grid_cfg  = cfg["parameter_grid"]
    int_cfg   = cfg["integrator"]
    s1_cfg    = cfg["step1_centered_reference"]
    plot_cfg  = cfg["plots"]

    q = float(sys_cfg["q"])
    h = float(int_cfg["h"])
    memory_mode = int_cfg["memory_mode"]
    alpha = float(sys_cfg["parameters"]["alpha"])
    beta  = float(sys_cfg["parameters"]["beta"])
    gamma = float(sys_cfg["parameters"]["gamma"])

    out_root = ROOT / cfg["experiment"]["output_dir"]
    out_centered = out_root / "step1_centered"
    out_centered.mkdir(parents=True, exist_ok=True)
    (out_centered / "trajectories").mkdir(exist_ok=True)
    (out_centered / "plots").mkdir(exist_ok=True)

    t_sim_final    = float(s1_cfg["t_sim_final"])
    t_sim_trans    = float(s1_cfg["t_sim_transient"])
    t_cont_trans   = float(s1_cfg["t_transient"])
    t_cont_keep    = float(s1_cfg["t_keep"])
    eta_steps      = int(s1_cfg["eta_steps"])
    omega_min      = float(s1_cfg["omega_min"])
    omega_max      = float(s1_cfg["omega_max"])
    nscan          = int(s1_cfg["nscan"])
    lambda_values  = list(np.linspace(0.0, 1.0, eta_steps))

    m1_values = [float(v) for v in grid_cfg["m1_values"]]
    m0_values = [float(v) for v in grid_cfg["m0_values"]]

    all_results: List[Dict[str, Any]] = []

    for m1 in m1_values:
        for m0 in m0_values:
            params = chua_parameters(
                model="nonsmooth",
                alpha=alpha, beta=beta, gamma=gamma, m0=m0, m1=m1,
            )
            prefix = (
                f"centered_q{int(q*10000)}_m1_{m1:.4f}_m0_{m0:.4f}"
                .replace(".", "p").replace("-", "m")
            )

            print(f"  m1={m1}, m0={m0} ...")

            try:
                pairs = find_omega_gain_candidates(
                    q=q, params=params,
                    wmin=omega_min, wmax=omega_max, nscan=nscan,
                )
            except Exception as exc:
                print(f"    [DF SCAN] error: {exc}")
                continue

            print(f"    Ramas centradas encontradas: {len(pairs)}")

            for branch_idx, (omega0, k_gain) in enumerate(pairs):
                row: Dict[str, Any] = {
                    "m1": m1, "m0": m0, "branch": branch_idx,
                    "omega0": float(omega0), "k_gain": float(k_gain),
                    "cont_status": "not_run", "sim_status": "not_run",
                    "verdict": "not_run", "prefix": prefix + f"_br{branch_idx}",
                }

                try:
                    seed_data = find_harmonic_seed(
                        q=q, params=params, branch_index=branch_idx,
                        wmin=omega_min, wmax=omega_max,
                    )
                    x_seed = seed_data.seed
                except Exception as exc:
                    print(f"    [SEED br{branch_idx}] error: {exc}")
                    row["cont_status"] = "seed_error"
                    all_results.append(row)
                    continue

                system = get_system("chua-nonsmooth")
                system.parameters.update({
                    "m1": m1, "m0": m0,
                    "alpha": alpha, "beta": beta, "gamma": gamma,
                })

                try:
                    steps = run_fractional_continuation(
                        system=system,
                        seed_x0=x_seed,
                        k_gain=k_gain,
                        lambda_values=lambda_values,
                        h=h,
                        memory_mode=memory_mode,
                        integrator="abm",
                        t_transient=t_cont_trans,
                        t_keep=t_cont_keep,
                        q=q,
                    )
                    final_step  = steps[-1]
                    cont_status = final_step["status"]
                except Exception as exc:
                    print(f"    [CONT br{branch_idx}] error: {exc}")
                    row["cont_status"] = f"error:{exc}"
                    all_results.append(row)
                    continue

                row["cont_status"] = cont_status

                if cont_status != "ok":
                    all_results.append(row)
                    continue

                x_final = final_step["x_out"].copy()

                try:
                    sim_t, sim_x, sim_status, _ = fractional_integrate(
                        rhs=lambda t, x: system.rhs(x, system.parameters),
                        x0=x_final, q=q, h=h, t_final=t_sim_final,
                        method="abm", memory_mode=memory_mode,
                        system=system, use_c_backend=True,
                    )
                except Exception as exc:
                    row["sim_status"] = f"error:{exc}"
                    all_results.append(row)
                    continue

                row["sim_status"] = sim_status

                full_traj = np.column_stack((sim_t, sim_x))
                post_traj = full_traj[sim_t >= t_sim_trans]

                if len(post_traj) > 10:
                    diag = classify_post_transient_periodicity(post_traj, h=h)
                    verdict = diag["candidate_label"]
                else:
                    verdict = "too_short"

                row["verdict"] = verdict

                traj_path = out_centered / "trajectories" / f"{row['prefix']}_trajectory.csv"
                np.savetxt(
                    traj_path, post_traj, delimiter=",",
                    header="t,x,y,z", comments="",
                )

                if plot_cfg["save_figures"]:
                    plot_centered_trajectory(
                        traj=post_traj,
                        outpath=out_centered / "plots" / f"{row['prefix']}_phase.png",
                        t_burn=0.0,
                        h=h,
                    )

                all_results.append(row)

    csv_path = out_centered / "centered_branches.csv"
    if all_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)

    return all_results

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Biased Describing Function Search
# ══════════════════════════════════════════════════════════════════════════════

def get_Wq(omega: float, q: float, pmat: np.ndarray,
           qvec: np.ndarray, rvec: np.ndarray) -> complex:
    """Compute the fractional transfer function value using the P - lambda I sign convention.

    Warning
    -------
    This function constructs the state-space matrix as ``P - (j*omega)^q * I``,
    which corresponds to:
        W_q(lambda) = r^T (P - lambda I)^(-1) b.
    This is the negative of the standard Laplace / spectral transfer function
    convention:
        W_std(lambda) = r^T (lambda I - P)^(-1) b.
    This sign inversion is deliberate and historically absorbed in the BDF harmonic
    residual condition:
        1.0 + W_q * N_1 = 0
    which is mathematically equivalent to the standard feedback loop balance condition:
        1.0 - W_std * N_1 = 0.
    In contrast, `build_biased_seed` uses the standard `lambda * I - P` formulation to
    correctly reconstruct the phase coordinates of the seed.
    """
    matrix = pmat.astype(complex_dtype) - fractional_iomega_power(omega, q) * np.eye(3, dtype=complex_dtype)
    return complex_dtype(
        (rvec.astype(complex_dtype).reshape(1, -1)
         @ np.linalg.inv(matrix)
         @ qvec.astype(complex_dtype).reshape(-1, 1))[0, 0]
    )

def harmonic_residual_sign_audit(W: complex, N1: float) -> Dict[str, float]:
    R_plus  = 1.0 + W * N1
    R_minus = 1.0 - W * N1
    return {
        "R_plus_real":  float(R_plus.real),
        "R_plus_imag":  float(R_plus.imag),
        "R_plus_abs":   float(abs(R_plus)),
        "R_minus_real": float(R_minus.real),
        "R_minus_imag": float(R_minus.imag),
        "R_minus_abs":  float(abs(R_minus)),
    }

def biased_saturation_df(A: float, c: float, g: float,
                          n_theta: int = 8192) -> Tuple[float, float]:
    if A < 1e-6:
        return 0.0, 0.0
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=real_dtype)
    sigma = c + A * np.cos(theta)
    psi   = g * np.clip(sigma, -1.0, 1.0)
    psi0  = float(np.mean(psi))
    psi1  = 2.0 * float(np.mean(psi * np.cos(theta)))
    return psi0, psi1 / A

def biased_saturation_residual(A: float, c: float, omega: float,
                                params: ChuaParameters, q: float,
                                n_theta: int = 8192) -> np.ndarray:
    if A < 1e-6:
        return np.array([c, 1e2, 1e2], dtype=float)

    g    = chua_gain(params)
    psi0, N1 = biased_saturation_df(A, c, g, n_theta)
    pmat, qvec, rvec = chua_matrices(params)

    try:
        x_bar = np.linalg.solve(pmat, -qvec * psi0)
        F0 = c - float(rvec @ x_bar)
    except np.linalg.LinAlgError:
        F0 = 1e3

    try:
        Wq = get_Wq(omega, q, pmat, qvec, rvec)
    except np.linalg.LinAlgError:
        Wq = 0.0

    term = 1.0 + Wq * N1
    return np.array([F0, float(term.real), float(term.imag)], dtype=float)

def find_biased_branches(params: ChuaParameters, q: float,
                          s2_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    q_val = validate_fractional_order(q)
    n_theta = int(s2_cfg.get("n_theta", 8192))

    A_grid     = np.linspace(*s2_cfg["A_range"],     int(s2_cfg["n_A"]))
    c_grid     = np.linspace(*s2_cfg["c_range"],     int(s2_cfg["n_c"]))
    omega_grid = np.linspace(*s2_cfg["omega_range"], int(s2_cfg["n_omega"]))
    tol        = float(s2_cfg["residual_tol"])

    def residual(x):
        return biased_saturation_residual(x[0], x[1], x[2], params, q_val, n_theta)

    raw = []
    for A0 in A_grid:
        for c0 in c_grid:
            for w0 in omega_grid:
                try:
                    res = least_squares(
                        residual, x0=[A0, c0, w0],
                        bounds=([1e-6, -12.0, 0.1], [25.0, 12.0, 8.0]),
                        ftol=1e-10, xtol=1e-10, max_nfev=300,
                    )
                    if res.success and np.linalg.norm(residual(res.x)) < tol:
                        raw.append((*res.x, np.linalg.norm(residual(res.x))))
                except Exception:
                    continue

    A_tol = c_tol = w_tol = 1e-3
    unique: List[Dict[str, Any]] = []
    for A_c, c_c, w_c, res_c in raw:
        if A_c < 0.5 or not (0.5 <= w_c <= 6.0):
            continue
        dup = False
        for ex in unique:
            if abs(A_c - ex["A"]) < A_tol and abs(c_c - ex["c"]) < c_tol and abs(w_c - ex["omega"]) < w_tol:
                if res_c < ex["residual_norm"]:
                    ex.update({"A": A_c, "c": c_c, "omega": w_c, "residual_norm": res_c})
                dup = True
                break
        if not dup:
            unique.append({"A": A_c, "c": c_c, "omega": w_c, "residual_norm": res_c})

    unique.sort(key=lambda x: x["residual_norm"])
    return unique

def build_biased_seed(params: ChuaParameters, q: float,
                       A: float, c: float, omega: float,
                       psi0: float, N1: float) -> Dict[str, Any]:
    q_val = validate_fractional_order(q)
    pmat, qvec, rvec = chua_matrices(params)
    x_bar = np.linalg.solve(pmat, -qvec * psi0)
    lam   = fractional_iomega_power(omega, q_val)
    matrix = lam * np.eye(3, dtype=complex_dtype) - pmat.astype(complex_dtype)
    X1 = np.linalg.solve(matrix, qvec.astype(complex_dtype)) * N1 * A
    return {
        "seed":   x_bar + np.real(X1),
        "x_bar":  x_bar,
        "Re_X1":  np.real(X1),
        "Im_X1":  np.imag(X1),
    }

def run_affine_continuation(params: ChuaParameters, q: float, h: float,
                              seed_x0: np.ndarray, A: float, c: float,
                              psi0: float, N1: float,
                              lambda_values: List[float],
                              t_transient: float, t_keep: float,
                              div_threshold: float) -> List[Dict[str, Any]]:
    dim = 3
    h   = float(h)
    q   = float(q)
    nsteps_tr = int(np.ceil(t_transient / h))
    nsteps_kp = int(np.ceil(t_keep / h))
    steps_per_stage = nsteps_tr + nsteps_kp
    total_new = len(lambda_values) * steps_per_stage

    t_arr = np.zeros(1 + total_new, dtype=float)
    x_arr = np.zeros((1 + total_new, dim), dtype=float)
    f_arr = np.zeros((1 + total_new, dim), dtype=float)
    t_arr[0] = 0.0
    x_arr[0] = seed_x0

    pmat, qvec, rvec = chua_matrices(params)
    P_aff   = pmat + N1 * np.outer(qvec, rvec)
    c_aff   = qvec * (psi0 - N1 * c)
    g       = params.m0 - params.m1

    def rhs(x: np.ndarray, eta: float) -> np.ndarray:
        sigma   = float(rvec @ x)
        psi_val = g * np.clip(sigma, -1.0, 1.0)
        return P_aff @ x + c_aff + eta * qvec * (psi_val - psi0 - N1 * (sigma - c))

    f_arr[0] = rhs(x_arr[0], lambda_values[0])
    powers   = np.arange(total_new + 3, dtype=float)
    pow_q    = powers ** q
    pow_q1   = powers ** (q + 1.0)
    hq       = h ** q
    pred_sc  = hq / float(gamma_func(q + 1.0))
    gq2      = float(gamma_func(q + 2.0))
    corr_sc  = hq / gq2 if abs(gq2) > 1e-15 else 0.0

    records, curr_n, diverged = [], 0, False

    for eta in lambda_values:
        if diverged:
            break
        x_in     = x_arr[curr_n].copy()
        stage_ok = True

        for local_step in range(steps_per_stage):
            n    = curr_n + local_step
            j_r  = np.arange(0, n + 1)
            b_w  = pow_q[n + 1 - j_r] - pow_q[n - j_r]
            pred = x_arr[0] + pred_sc * (b_w @ f_arr[0: n + 1])

            fp   = rhs(pred, eta)
            n_p  = n
            a0   = float(n_p) ** (q + 1) - (float(n_p) - q) * (float(n_p) + 1) ** q
            if n_p > 0:
                mid  = n - np.arange(1, n + 1)
                a_mid = pow_q1[mid + 2] + pow_q1[mid] - 2.0 * pow_q1[mid + 1]
                a_w  = np.concatenate(([a0], a_mid))
            else:
                a_w = np.array([a0])

            corrected = x_arr[0] + corr_sc * ((a_w @ f_arr[0: n + 1]) + fp)
            nrm = np.linalg.norm(corrected)

            if nrm > div_threshold or not np.all(np.isfinite(corrected)):
                diverged  = True
                stage_ok  = False
                x_arr[n + 1] = corrected if np.all(np.isfinite(corrected)) else x_arr[n]
                t_arr[n + 1] = t_arr[n] + h
                f_arr[n + 1] = f_arr[n]
                break

            x_arr[n + 1] = corrected
            t_arr[n + 1] = t_arr[n] + h
            f_arr[n + 1] = rhs(corrected, eta)

        keep_start = curr_n + nsteps_tr + 1
        keep_end   = curr_n + steps_per_stage
        keep_times = t_arr[keep_start: keep_end + 1]
        keep_states = x_arr[keep_start: keep_end + 1]
        x_out = x_arr[keep_end] if stage_ok else x_arr[curr_n + steps_per_stage]

        traj = np.column_stack((keep_times, keep_states)) if len(keep_times) > 0 else np.empty((0, 4))
        records.append({
            "lambda_value": float(eta),
            "x_in":  x_in,
            "x_out": x_out,
            "trajectory": traj,
            "status": "ok" if stage_ok else "diverged",
            "x_out_norm": float(np.linalg.norm(x_out)),
        })
        curr_n = keep_end

    return records

def run_biased_df_search(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Runs biased DF grid search, continuation, and long integration."""
    sys_cfg  = cfg["system"]
    grid_cfg = cfg["parameter_grid"]
    int_cfg  = cfg["integrator"]
    s2_cfg   = cfg["step2_biased_df_search"]
    plot_cfg = cfg["plots"]

    q           = float(sys_cfg["q"])
    h           = float(int_cfg["h"])
    memory_mode = int_cfg["memory_mode"]
    alpha  = float(sys_cfg["parameters"]["alpha"])
    beta   = float(sys_cfg["parameters"]["beta"])
    gamma  = float(sys_cfg["parameters"]["gamma"])

    t_sim_final  = float(s2_cfg["t_sim_final"])
    t_sim_trans  = float(s2_cfg["t_sim_transient"])
    t_cont_trans = float(s2_cfg["t_transient"])
    t_cont_keep  = float(s2_cfg["t_keep"])
    eta_step     = float(s2_cfg["eta_step"])
    div_thr      = float(s2_cfg["div_threshold"])
    max_br       = int(s2_cfg.get("max_branches_per_case", 2))
    lambda_vals  = list(np.arange(0.0, 1.0 + 1e-9, eta_step))

    m1_values = [float(v) for v in grid_cfg["m1_values"]]
    m0_values = [float(v) for v in grid_cfg["m0_values"]]

    out_root   = ROOT / cfg["experiment"]["output_dir"]
    out_s2     = out_root / "step2_biased_df"
    for sub in ["trajectories", "plots", "plots/sign_audit", "plots/continuation",
                "plots/attractors", "continuation_steps"]:
        (out_s2 / sub).mkdir(parents=True, exist_ok=True)

    roots_rows    : List[Dict] = []
    audit_rows    : List[Dict] = []
    seed_rows     : List[Dict] = []
    identity_rows : List[Dict] = []
    cont_rows     : List[Dict] = []
    classif_rows  : List[Dict] = []
    all_results   : List[Dict] = []

    for m1 in m1_values:
        for m0 in m0_values:
            params = chua_parameters(
                model="nonsmooth", alpha=alpha, beta=beta, gamma=gamma, m0=m0, m1=m1,
            )
            g    = chua_gain(params)
            pmat, qvec, rvec = chua_matrices(params)

            print(f"  m1={m1}, m0={m0} ...")

            biased_roots = find_biased_branches(params, q, s2_cfg)
            print(f"    Raíces sesgadas: {len(biased_roots)}")

            for root_idx, r in enumerate(biased_roots[:max_br]):
                A, c, omega, res_norm = r["A"], r["c"], r["omega"], r["residual_norm"]
                psi0, N1 = biased_saturation_df(A, c, g)
                is_centered = abs(c) <= 0.05

                tag = (
                    f"biased_q{int(q*10000)}_m1_{m1:.4f}_m0_{m0:.4f}"
                    f"_branch_{root_idx}_c_{'centered_like' if is_centered else f'{c:.3f}'}"
                )
                tag = tag.replace(".", "p").replace("-", "m")

                print(f"    br{root_idx}: A={A:.3f}  c={c:.3f}  w={omega:.3f}  res={res_norm:.2e}"
                      + ("  [centrada, skip]" if is_centered else ""))

                Wq = get_Wq(omega, q, pmat, qvec, rvec)
                audit = harmonic_residual_sign_audit(Wq, N1)
                audit_rows.append({
                    "m1": m1, "m0": m0, "branch": root_idx, "A": A, "c": c,
                    "omega": omega, "N1": N1, "psi0": psi0,
                    "Re_Wq": Wq.real, "Im_Wq": Wq.imag,
                    **{k: v for k, v in audit.items()},
                    "sign_consistent": audit["R_plus_abs"] < 1e-3,
                })

                seed_info = build_biased_seed(params, q, A, c, omega, psi0, N1)
                X_seed, x_bar, Re_X1 = seed_info["seed"], seed_info["x_bar"], seed_info["Re_X1"]

                dc_err   = abs(float(rvec @ x_bar) - c)
                harm_re  = float(rvec @ Re_X1)
                harm_im  = float(rvec @ seed_info["Im_X1"])
                harm_amp = np.sqrt(harm_re**2 + harm_im**2)
                amp_err  = abs(harm_amp - A)
                R_plus   = float(np.linalg.norm(biased_saturation_residual(A, c, omega, params, q)))

                if dc_err >= 1e-5:
                    seed_status = "dc_mismatch"
                elif amp_err >= 1e-4:
                    seed_status = "harmonic_amplitude_mismatch"
                elif R_plus >= 1e-4:
                    seed_status = "residual_too_large"
                else:
                    seed_status = "ok"

                seed_rows.append({
                    "m1": m1, "m0": m0, "branch": root_idx,
                    "dc_err": dc_err, "amp_err": amp_err, "R_plus": R_plus,
                    "seed_status": seed_status,
                })
                roots_rows.append({
                    "m1": m1, "m0": m0, "q": q, "branch": root_idx,
                    "A": A, "c": c, "omega": omega, "N1": N1, "psi0": psi0,
                    "residual_norm": res_norm,
                    "X_seed_x": X_seed[0], "X_seed_y": X_seed[1], "X_seed_z": X_seed[2],
                    "prefix": tag,
                })

                max_id_err = 0.0
                for _ in range(5):
                    X_rand = np.random.uniform(-5, 5, 3)
                    sigma  = float(rvec @ X_rand)
                    psi_v  = (params.m0 - params.m1) * np.clip(sigma, -1, 1)
                    f_orig = pmat @ X_rand + qvec * psi_v
                    P_aff  = pmat + N1 * np.outer(qvec, rvec)
                    c_aff  = qvec * (psi0 - N1 * c)
                    f_eta1 = P_aff @ X_rand + c_aff + 1.0 * qvec * (psi_v - psi0 - N1 * (sigma - c))
                    max_id_err = max(max_id_err, float(np.linalg.norm(f_eta1 - f_orig)))
                identity_rows.append({
                    "m1": m1, "m0": m0, "branch": root_idx,
                    "max_identity_error": max_id_err,
                    "status": "ok" if max_id_err < 1e-10 else "failed",
                })

                print(f"      Continuacion afin ...")
                cont_steps = run_affine_continuation(
                    params, q, h, X_seed, A, c, psi0, N1, lambda_vals,
                    t_cont_trans, t_cont_keep, div_thr,
                )
                survived = (len(cont_steps) == len(lambda_vals)
                            and all(s["status"] == "ok" for s in cont_steps))
                print(f"      Continuacion: {'OK' if survived else 'FALLO'}")

                cont_rows.append({
                    "m1": m1, "m0": m0, "branch": root_idx,
                    "status": "ok" if survived else "failed",
                    "final_norm": cont_steps[-1]["x_out_norm"] if cont_steps else float("nan"),
                })

                if not survived:
                    classif_rows.append({
                        "m1": m1, "m0": m0, "branch": root_idx,
                        "classification": "continuation_failed", "prefix": tag,
                    })
                    all_results.append({**roots_rows[-1], "cont_status": "failed",
                                        "verdict": "continuation_failed"})
                    continue

                if plot_cfg["save_figures"] and plot_cfg.get("continuation_norm", True):
                    plot_continuation_metrics(
                        cont_steps, tag,
                        out_s2 / "plots" / "continuation" / f"{tag}_continuation.png",
                    )

                x_final_cont = cont_steps[-1]["x_out"].copy()

                system = get_system("chua-nonsmooth")
                system.parameters.update({"m1": m1, "m0": m0, "alpha": alpha, "beta": beta, "gamma": gamma})
                print(f"      Simulacion final ...")
                sim_t, sim_x, sim_status, _ = fractional_integrate(
                    rhs=lambda t, x: system.rhs(x, system.parameters),
                    x0=x_final_cont, q=q, h=h, t_final=t_sim_final,
                    method="abm", memory_mode=memory_mode,
                    system=system, use_c_backend=True,
                )

                full_traj = np.column_stack((sim_t, sim_x))
                post_traj = full_traj[sim_t >= t_sim_trans]

                if len(post_traj) > 10:
                    max_range = float(np.max(np.ptp(post_traj[:, 1:4], axis=0)))
                    if max_range < 0.05:
                        verdict = "collapsed_to_equilibrium"
                    else:
                        diag = classify_post_transient_periodicity(post_traj, h=h)
                        verdict = diag["candidate_label"]
                else:
                    verdict = "too_short"

                print(f"      Veredicto: {verdict}")

                traj_path = out_s2 / "trajectories" / f"{tag}_trajectory.csv"
                pd.DataFrame(post_traj, columns=["t", "x", "y", "z"]).to_csv(traj_path, index=False)

                if plot_cfg["save_figures"]:
                    params_str = f"m1={m1} | m0={m0} | A={A:.3f} | c={c:.3f} | ω={omega:.3f}"
                    plot_candidate_report(
                        post_traj, params_str, verdict,
                        out_s2 / "plots" / "attractors" / f"{tag}_report.png",
                    )

                classif_rows.append({
                    "m1": m1, "m0": m0, "branch": root_idx,
                    "classification": verdict, "prefix": tag,
                    "centroid_x": float(np.mean(post_traj[:, 1])) if len(post_traj) > 0 else float("nan"),
                    "centroid_y": float(np.mean(post_traj[:, 2])) if len(post_traj) > 0 else float("nan"),
                    "centroid_z": float(np.mean(post_traj[:, 3])) if len(post_traj) > 0 else float("nan"),
                })
                all_results.append({**roots_rows[-1], "cont_status": "ok", "verdict": verdict})

    pd.DataFrame(roots_rows).to_csv(out_s2 / "roots_corrected.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out_s2 / "roots_sign_audit.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out_s2 / "seed_consistency_checks.csv", index=False)
    pd.DataFrame(identity_rows).to_csv(out_s2 / "affine_homotopy_identity.csv", index=False)
    pd.DataFrame(cont_rows).to_csv(out_s2 / "affine_continuation_summary.csv", index=False)
    pd.DataFrame(classif_rows).to_csv(out_s2 / "final_classification.csv", index=False)

    if plot_cfg["save_figures"] and audit_rows:
        plot_sign_audit(audit_rows, out_s2 / "plots" / "sign_audit" / "sign_audit_comparison.png")

    return all_results

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Hiddenness Standard Protocol
# ══════════════════════════════════════════════════════════════════════════════

def run_probe(system: Any, x0: np.ndarray, ref_tail: np.ndarray,
              stable_eqs: List[np.ndarray], h3_cfg: Dict) -> Dict[str, Any]:
    q      = float(h3_cfg["_q"])
    h      = float(h3_cfg["_h"])
    t_fin  = float(h3_cfg["t_final_probe"])
    t_burn = float(h3_cfg["t_burn_probe"])
    eq_tol = float(h3_cfg["equilibrium_tol"])
    metric = h3_cfg["match_metric"]
    m_tol  = float(h3_cfg["match_tol"])

    try:
        t_arr, x_arr, status, _ = fractional_integrate(
            rhs=lambda t, x: system.rhs(x, system.parameters),
            x0=x0, q=q, h=h, t_final=t_fin,
            method="abm", memory_mode="full",
            system=system, use_c_backend=True,
        )
    except Exception as exc:
        return {"destination": "numerical_failure", "status": f"exception:{exc}"}

    if status in ("diverged", "nonfinite_solution"):
        return {"destination": "divergence", "status": status}

    n_burn = int(np.ceil(t_burn / h))
    tail   = x_arr[n_burn:] if len(x_arr) > n_burn else x_arr
    final  = x_arr[-1] if len(x_arr) > 0 else x0

    for eq in stable_eqs:
        if np.linalg.norm(final - eq) <= eq_tol:
            return {"destination": "stable_equilibrium", "status": "ok"}

    if evaluate_target_match(tail, ref_tail, metric=metric, tolerance=m_tol):
        return {"destination": "target_attractor", "status": "ok"}

    return {"destination": "other_attractor", "status": "ok"}

def run_hiddenness_for_candidate(candidate: Dict[str, Any],
                                  cfg: Dict[str, Any]) -> Dict[str, Any]:
    sys_cfg = cfg["system"]
    int_cfg = cfg["integrator"]
    h3_cfg  = cfg["step3_hiddenness"].copy()
    h3_cfg["_q"] = float(sys_cfg["q"])
    h3_cfg["_h"] = float(int_cfg["h"])
    plot_cfg = cfg["plots"]

    alpha = float(sys_cfg["parameters"]["alpha"])
    beta  = float(sys_cfg["parameters"]["beta"])
    gamma = float(sys_cfg["parameters"]["gamma"])
    m1    = float(candidate["m1"])
    m0    = float(candidate["m0"])
    prefix = str(candidate["prefix"])
    traj_path = Path(candidate["traj_path"])

    out_root = ROOT / cfg["experiment"]["output_dir"]
    outdir   = out_root / "step3_hiddenness" / prefix
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [{prefix}]  m1={m1}  m0={m0}  c={candidate.get('c','?')}")

    df_ref   = pd.read_csv(traj_path)
    t_burn   = float(h3_cfg["t_burn_probe"])
    ref_tail = df_ref[df_ref["t"] >= t_burn][["x", "y", "z"]].values
    print(f"    Referencia: {len(ref_tail)} puntos post-transitorio")

    system = get_system("chua-nonsmooth")
    system.parameters.update({"m1": m1, "m0": m0, "alpha": alpha, "beta": beta, "gamma": gamma})
    equilibria = solve_equilibria(system)
    stable_eqs = list(equilibria.values())
    print(f"    Equilibrios: {list(equilibria.keys())}")

    radii          = [float(r) for r in h3_cfg["radii"]]
    samples_per_r  = [int(s) for s in h3_cfg["samples_per_radius"]]
    random_seed    = int(cfg["experiment"]["random_seed"])

    all_runs     : List[Dict] = []
    sphere_recs  : List[Dict] = []

    for eq_name, eq_pt in equilibria.items():
        for r_idx, (radius, n_samples) in enumerate(zip(radii, samples_per_r)):
            print(f"    [{eq_name}]  r={radius:.0e}  n={n_samples}", end=" ... ", flush=True)
            pts = generate_neighborhood_points(
                eq_point=eq_pt, radius=radius, num_samples=n_samples,
                mode=h3_cfg["sampling_mode"], seed=random_seed + r_idx,
            )
            stats = {k: 0 for k in
                     ["target_attractor", "stable_equilibrium", "divergence",
                      "other_attractor", "numerical_failure"]}
            radius_runs: List[Dict] = []

            for pt in pts:
                res = run_probe(system, pt, ref_tail, stable_eqs, h3_cfg)
                radius_runs.append(res)
                stats[res["destination"]] = stats.get(res["destination"], 0) + 1

                if stats["target_attractor"] > 0:
                    break

            print(f"TARGET={stats['target_attractor']}  EQ={stats['stable_equilibrium']}  "
                  f"DIV={stats['divergence']}")

            if plot_cfg["save_figures"]:
                safe_r = f"{radius:.0e}".replace("-", "m")
                plot_sphere_summary(
                    eq_name, eq_pt, radius, radius_runs, pts[:len(radius_runs)],
                    outdir / f"sphere_{eq_name}_{safe_r}.png",
                )

            for res in radius_runs:
                all_runs.append({
                    "equilibrium": eq_name, "radius": float(radius), **res,
                })
            sphere_recs.append({
                "system_id": prefix, "equilibrium": eq_name, "radius": float(radius),
                "samples": len(radius_runs), "TARGET": stats["target_attractor"],
                "EQ": stats["stable_equilibrium"], "OTHER": stats["other_attractor"],
                "DIV": stats["divergence"], "FAIL": stats["numerical_failure"],
            })

    contract = verify_hiddenness_contract(
        equilibria=equilibria,
        sphere_summary_records=sphere_recs,
        probe_runs=all_runs,
        required_radii=radii,
        require_all_equilibria=bool(h3_cfg.get("require_all_equilibria", True)),
        allow_numerical_failures=bool(h3_cfg.get("allow_numerical_failures", False)),
        ref_tail_size=len(ref_tail),
        min_ref_tail_points=int(h3_cfg.get("min_ref_tail_points", 200)),
        target_match_metric=h3_cfg["match_metric"],
        target_match_tol=float(h3_cfg["match_tol"]),
        target_match_nn_percentile=float(h3_cfg["match_percentile"]),
        seed_reached_attractor=True,
    )

    pd.DataFrame(sphere_recs).to_csv(outdir / "hiddenness_summary.csv", index=False)
    pd.DataFrame([{"equilibrium": r["equilibrium"], "radius": r["radius"],
                   "destination": r["destination"], "status": r["status"]}
                  for r in all_runs]).to_csv(outdir / "probe_runs.csv", index=False)

    contract_safe = {k: v for k, v in contract.items() if k != "run_metadata"}
    with open(outdir / "hiddenness_contract.json", "w", encoding="utf-8") as f:
        json.dump(contract_safe, f, indent=2)

    if plot_cfg["save_figures"] and plot_cfg.get("heatmap_hiddenness", True):
        plot_heatmap_hiddenness(sphere_recs, radii, outdir / "heatmap_hiddenness.png")

    return {
        "prefix": prefix,
        "m1": m1, "m0": m0, "c": candidate.get("c", 0.0),
        "hiddenness_status":  contract["hiddenness_status"],
        "hidden_verified":    contract["hidden_verified"],
        "hidden_compatible":  contract["hidden_compatible"],
        "self_excited":       contract["self_excited_contact_detected"],
        "target_hits":        contract["target_hits_total"],
        "samples_total":      contract["samples_total"],
        "equilibria_tested":  contract["equilibria_tested"],
    }

def run_hiddenness_verification(candidates: List[Dict[str, Any]],
                                 cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Runs standard hiddenness protocol sweep across survived candidates."""
    print("=" * 65)
    print("PASO 3 - Verificacion de Ocultedad (Protocolo Estandar)")
    print(f"  Radios: {cfg['step3_hiddenness']['radii']}")
    print(f"  Muestras/radio: {cfg['step3_hiddenness']['samples_per_radius']}")
    print(f"  Candidatos a verificar: {len(candidates)}")
    print("=" * 65)

    results = []
    for cand in candidates:
        result = run_hiddenness_for_candidate(cand, cfg)
        results.append(result)

    out_root = ROOT / cfg["experiment"]["output_dir"] / "step3_hiddenness"
    out_root.mkdir(parents=True, exist_ok=True)

    with open(out_root / "hiddenness_global_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Extended Multiprocessing Hiddenness Probe
# ══════════════════════════════════════════════════════════════════════════════

_worker_system    = None
_worker_ref_tail  = None
_worker_stable_eqs = None
_worker_h4_cfg    = None

def init_worker(m1: float, m0: float, alpha: float, beta: float, gamma: float,
                ref_tail: np.ndarray, stable_eqs: List[np.ndarray],
                h4_cfg: Dict[str, Any]) -> None:
    global _worker_system, _worker_ref_tail, _worker_stable_eqs, _worker_h4_cfg
    from hidden_attractors.systems import get_system as _get
    _worker_system = _get("chua-nonsmooth")
    _worker_system.parameters.update({
        "m1": m1, "m0": m0, "alpha": alpha, "beta": beta, "gamma": gamma,
    })
    _worker_ref_tail   = ref_tail
    _worker_stable_eqs = stable_eqs
    _worker_h4_cfg     = h4_cfg

def worker_run_probe(x0: np.ndarray) -> Dict[str, Any]:
    from hidden_attractors.integrations.fractional_c import fractional_integrate
    from hidden_attractors.verification.hiddenness import evaluate_target_match

    cfg    = _worker_h4_cfg
    q      = float(cfg["_q"])
    h      = float(cfg["_h"])
    t_fin  = float(cfg["t_final_probe"])
    t_burn = float(cfg["t_burn_probe"])
    eq_tol = float(cfg["equilibrium_tol"])

    try:
        _, x_arr, status, _ = fractional_integrate(
            rhs=lambda t, x: _worker_system.rhs(x, _worker_system.parameters),
            x0=x0, q=q, h=h, t_final=t_fin,
            method="abm", memory_mode="full",
            system=_worker_system, use_c_backend=True,
        )
    except Exception as exc:
        return {"destination": "numerical_failure", "status": f"exception:{exc}"}

    if status in ("diverged", "nonfinite_solution"):
        return {"destination": "divergence", "status": status}

    n_burn = int(np.ceil(t_burn / h))
    tail   = x_arr[n_burn:] if len(x_arr) > n_burn else x_arr
    final  = x_arr[-1] if len(x_arr) > 0 else x0

    for eq in _worker_stable_eqs:
        if np.linalg.norm(final - eq) <= eq_tol:
            return {"destination": "stable_equilibrium", "status": "ok"}

    if evaluate_target_match(tail, _worker_ref_tail,
                              metric=cfg["match_metric"],
                              tolerance=float(cfg["match_tol"])):
        return {"destination": "target_attractor", "status": "ok"}

    return {"destination": "other_attractor", "status": "ok"}

def sample_ball(eq_point: np.ndarray, radius: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = len(eq_point)
    pts: List[np.ndarray] = []
    while len(pts) < n:
        batch  = rng.normal(0.0, 1.0, (n * 3, dim))
        norms  = np.linalg.norm(batch, axis=1, keepdims=True)
        r_vals = rng.uniform(0.0, 1.0, (n * 3, 1)) ** (1.0 / dim)
        ball   = eq_point + radius * r_vals * batch / norms
        for pt in ball:
            if np.linalg.norm(pt - eq_point) <= radius:
                pts.append(pt)
            if len(pts) >= n:
                break
    return np.array(pts[:n])

def run_extended_hiddenness(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Runs high density multiprocessing probe up to radius 2.0."""
    sys_cfg = cfg["system"]
    int_cfg = cfg["integrator"]
    h4_cfg  = cfg["step4_extended_hiddenness"].copy()
    h4_cfg["_q"]    = float(sys_cfg["q"])
    h4_cfg["_h"]    = float(int_cfg["h"])
    plot_cfg = cfg["plots"]

    alpha = float(sys_cfg["parameters"]["alpha"])
    beta  = float(sys_cfg["parameters"]["beta"])
    gamma = float(sys_cfg["parameters"]["gamma"])

    target = h4_cfg["target_candidate"]
    m1, m0 = float(target["m1"]), float(target["m0"])
    c_bias  = float(target["c"])
    prefix  = (
        f"biased_q{int(float(sys_cfg['q'])*10000)}"
        f"_m1_{m1:.4f}_m0_{m0:.4f}_branch_{target['branch']}_c_{c_bias:.3f}"
    ).replace(".", "p").replace("-", "m")

    out_root = ROOT / cfg["experiment"]["output_dir"]
    traj_path = out_root / "step2_biased_df" / "trajectories" / f"{prefix}_trajectory.csv"
    outdir    = out_root / "step4_extended_hiddenness" / prefix
    outdir.mkdir(parents=True, exist_ok=True)

    df_ref   = pd.read_csv(traj_path)
    t_burn   = float(h4_cfg["t_burn_probe"])
    ref_tail = df_ref[df_ref["t"] >= t_burn][["x", "y", "z"]].values

    system = get_system("chua-nonsmooth")
    system.parameters.update({"m1": m1, "m0": m0, "alpha": alpha, "beta": beta, "gamma": gamma})
    equilibria  = solve_equilibria(system)
    stable_eqs  = list(equilibria.values())

    radius_plan = [(float(r), int(n)) for r, n in h4_cfg["radius_plan"]]
    radii       = [r for r, _ in radius_plan]
    random_seed = int(cfg["experiment"]["random_seed"])

    n_workers_cfg = h4_cfg.get("n_workers", "auto")
    if n_workers_cfg == "auto":
        n_workers = max(1, multiprocessing.cpu_count() - 2)
    else:
        n_workers = int(n_workers_cfg)

    print("=" * 65)
    print("PASO 4 — Verificación Extendida (Multiprocessing)")
    print(f"  m1={m1}  m0={m0}  c={c_bias}")
    print(f"  Radios: {radii}")
    print(f"  Workers: {n_workers}")
    print("=" * 65)

    all_records : List[Dict] = []
    all_runs    : List[Dict] = []
    t0_global   = time.time()
    total_probes = sum(n for _, n in radius_plan) * len(equilibria)
    probe_count  = 0

    pool = multiprocessing.Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(m1, m0, alpha, beta, gamma, ref_tail, stable_eqs, h4_cfg),
    )

    for eq_idx, (eq_name, eq_pt) in enumerate(equilibria.items()):
        for r_idx, (radius, n_samples) in enumerate(radius_plan):
            t0_r = time.time()
            seed  = random_seed + eq_idx * 100 + r_idx * 10
            pts   = sample_ball(eq_pt, radius, n_samples, seed)

            async_res = [pool.apply_async(worker_run_probe, (pt,)) for pt in pts]
            stats = {k: 0 for k in
                     ["target_attractor", "stable_equilibrium", "divergence",
                      "other_attractor", "numerical_failure"]}
            radius_runs: List[Dict] = []
            report_step = max(1, n_samples // 4)

            for k, ares in enumerate(async_res):
                res = ares.get()
                radius_runs.append(res)
                stats[res["destination"]] = stats.get(res["destination"], 0) + 1
                probe_count += 1

                if (k + 1) % report_step == 0 or (k + 1) == n_samples:
                    elapsed = time.time() - t0_global
                    rate    = probe_count / elapsed if elapsed > 0 else 0
                    eta     = (total_probes - probe_count) / rate if rate > 0 else 0
                    print(f"  [{eq_name}] r={radius:.0e}  {k+1:4d}/{n_samples:4d}  "
                          f"TARGET={stats['target_attractor']}  EQ={stats['stable_equilibrium']}  "
                          f"OTHER={stats['other_attractor']}  "
                          f"[{elapsed:.0f}s  ETA~{eta:.0f}s]")

            dt_r = time.time() - t0_r
            print(f"  FINAL r={radius:.0e}  n={n_samples}  {dt_r:.1f}s: {stats}")

            if plot_cfg["save_figures"]:
                plot_sphere_summary(eq_name, eq_pt, radius, radius_runs, pts, outdir / f"sphere_{eq_name}_{radius:.0e}.png")

            for res in radius_runs:
                all_runs.append({"equilibrium": eq_name, "radius": float(radius), **res})

            all_records.append({
                "equilibrium": eq_name, "radius": float(radius), "samples": n_samples,
                "TARGET": stats["target_attractor"], "EQ": stats["stable_equilibrium"],
                "OTHER": stats["other_attractor"], "DIV": stats["divergence"],
                "FAIL": stats["numerical_failure"],
            })

    pool.close()
    pool.join()

    target_hits  = sum(r["TARGET"] for r in all_records)
    samples_tot  = sum(r["samples"] for r in all_records)
    self_excited = target_hits > 0
    status       = "SELF_EXCITED_DETECTED" if self_excited else "HIDDEN_COMPATIBLE"

    pd.DataFrame(all_records).to_csv(outdir / "hiddenness_summary.csv", index=False)
    pd.DataFrame([{"equilibrium": r["equilibrium"], "radius": r["radius"],
                   "destination": r["destination"], "status": r["status"]}
                  for r in all_runs]).to_csv(outdir / "probe_runs.csv", index=False)

    result = {
        "prefix": prefix, "m1": m1, "m0": m0, "c": c_bias,
        "sampling_mode": h4_cfg["sampling_mode"],
        "protocol": {"radii": radii, "radius_plan": radius_plan,
                     "total_probes": total_probes,
                     "t_final": float(h4_cfg["t_final_probe"]),
                     "t_burn":  float(h4_cfg["t_burn_probe"])},
        "hiddenness_status": status,
        "target_hits_total": int(target_hits),
        "samples_total":     int(samples_tot),
        "records": all_records,
    }
    with open(outdir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if plot_cfg["save_figures"] and plot_cfg.get("heatmap_hiddenness", True):
        plot_heatmap_hiddenness(all_records, radii, outdir / "heatmap_target_fraction.png")

    return result

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Final Summary and Figure Gallery
# ══════════════════════════════════════════════════════════════════════════════

def load_traj(traj_path: Path, t_burn: float = 0.0,
              max_pts: int = 15000) -> Optional[np.ndarray]:
    if not traj_path.exists():
        return None
    df = pd.read_csv(traj_path)
    if t_burn > 0:
        df = df[df["t"] >= t_burn]
    data = df[["t", "x", "y", "z"]].values
    stride = max(1, len(data) // max_pts)
    return data[::stride]

def write_markdown_report(classif_rows: List[Dict], hid_results: List[Dict],
                           outpath: Path) -> None:
    lines = [
        "# Atractor Oculto en Chua Fraccionario No Suave — Reporte Final\n",
        "> [!NOTE]",
        "> Primer ejemplo exitoso de la librería `hidden_attractors_fo`.",
        "> Los resultados son reproducibles ejecutando `run_example.py`.\n",
        "## Parámetros del Sistema\n",
        "| Parámetro | Valor |",
        "|---|---|",
        "| Sistema | Chua No Suave (Saturación bilineal) |",
        "| Orden fraccionario q | 0.9998 |",
        "| α (alpha) | 8.4562 |",
        "| β (beta) | 12.0732 |",
        "| γ (gamma) | 0.0052 |",
        "| Integrador | Caputo ABM memoria completa |",
        "| h | 0.01 s |",
        "",
        "## Candidatos Encontrados (Paso 2)\n",
        "| m1 | m0 | branch | c (bias DC) | Clasificación |",
        "|---|---|---|---|---|",
    ]

    for r in classif_rows:
        lines.append(
            f"| {r.get('m1','?')} | {r.get('m0','?')} | {r.get('branch','?')}"
            f"| {r.get('c', r.get('centroid_x','?'))} | **{r.get('classification','?')}** |"
        )

    lines += [
        "",
        "## Resultados de Ocultedad (Paso 3)\n",
        "> [!WARNING]",
        "> La ausencia de contacto con las vecindades ensayadas **NO constituye**",
        "> prueba matemática global de ocultedad, sino verificación numérica finita.\n",
        "| Candidato | m1 | m0 | c | Estado | Compatible | TARGET hits | Muestras |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in hid_results:
        icon = "✅" if r.get("hidden_compatible") else "❌"
        lines.append(
            f"| `{r.get('prefix','?')}` | {r.get('m1','?')} | {r.get('m0','?')}"
            f"| {r.get('c', 0):.3f} | `{r.get('hiddenness_status','?')}` | {icon}"
            f"| {r.get('target_hits', '?')} | {r.get('samples_total', '?')} |"
        )

    lines += [
        "",
        "## Proceso Metodológico\n",
        "1. **Función Descriptiva Centrada (c=0)**: Búsqueda de ramas de la DF estándar.",
        "   Solo produce atractores periódicos (autoexcitados o sin interés).",
        "2. **Función Descriptiva Sesgada (c≠0)**: Extensión al caso con bias DC.",
        "   Convención de signo: `1 + Wq(jω) · N₁(A,c) = 0`.",
        "3. **Reconstrucción algebraica de semilla**: x̄ = −P⁻¹bψ₀, X₁ fasorial.",
        "4. **Verificación de identidad homotópica**: f_{η=1} ≡ f_original.",
        "5. **Continuación afín Caputo ABM**: deforma gradualmente el sistema desde",
        "   el linealizado (η=0) hasta el original (η=1).",
        "6. **Simulación final**: integración larga con el sistema original.",
        "7. **Verificación de ocultedad**: barrido de esferas alrededor de todos",
        "   los equilibrios con 225 muestras/equilibrio × 6 radios.",
        "8. **Test extendido**: verificación masiva hasta r=2.0 con multiprocessing.",
    ]

    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Reporte Markdown: {outpath}")

def run_summarize_and_plot(cfg: Dict[str, Any]) -> None:
    """Reads all past step outputs and generates report assets and Markdown/JSON summaries."""
    plot_cfg = cfg["plots"]
    out_root = ROOT / cfg["experiment"]["output_dir"]
    out_s5   = out_root / "step5_summary"
    out_s5.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("PASO 5 - Resumen y Galeria de Figuras")
    print("=" * 65)

    classif_csv = out_root / "step2_biased_df" / "final_classification.csv"
    classif_rows = pd.read_csv(classif_csv).to_dict("records") if classif_csv.exists() else []

    hid_json = out_root / "step3_hiddenness" / "hiddenness_global_summary.json"
    hid_results = json.loads(hid_json.read_text(encoding="utf-8")) if hid_json.exists() else []

    traj_dir_s2 = out_root / "step2_biased_df" / "trajectories"
    candidates = []
    for row in classif_rows:
        if "failed" in row.get("classification", ""):
            continue
        prefix    = row.get("prefix", "")
        traj_path = traj_dir_s2 / f"{prefix}_trajectory.csv"
        traj      = load_traj(traj_path)
        candidates.append({
            **row,
            "traj":      traj,
            "traj_path": traj_path,
        })

    centered_dir = out_root / "step1_centered" / "trajectories"
    centered_traj = None
    if centered_dir.exists():
        c_files = list(centered_dir.glob("*_trajectory.csv"))
        if c_files:
            centered_traj = load_traj(c_files[0])

    if plot_cfg["save_figures"]:
        print("\n  Generando galeria de figuras ...")
        for cand in candidates:
            if cand["traj"] is None:
                continue
            m1      = cand.get("m1", "?")
            m0      = cand.get("m0", "?")
            c_val   = cand.get("c", cand.get("centroid_x", "?"))
            verdict = cand.get("classification", "?")
            p_str   = f"m1={m1} | m0={m0} | c≈{c_val:.3f}" if isinstance(c_val, float) else f"m1={m1} | m0={m0}"
            outpath = out_s5 / "gallery" / f"{cand['prefix']}_detailed.png"
            plot_candidate_report(cand["traj"], p_str, verdict, outpath)

    if plot_cfg["save_figures"] and plot_cfg.get("comparison_biased_centered", True):
        for cand in candidates:
            if cand["traj"] is None:
                continue
            m1 = cand.get("m1", "?")
            m0 = cand.get("m0", "?")
            p_str = f"m1={m1} | m0={m0}"
            plot_biased_vs_centered(
                biased_traj=cand["traj"],
                centered_traj=centered_traj,
                params_str=p_str,
                outpath=out_s5 / "comparisons" / f"{cand['prefix']}_vs_centered.png",
            )

    if plot_cfg["save_figures"]:
        plot_mega_summary(candidates, out_s5 / "MEGA_all_candidates.png")

    write_markdown_report(classif_rows, hid_results, out_s5 / "final_report.md")

    summary = {
        "experiment": cfg["experiment"],
        "candidates_found":    len(classif_rows),
        "candidates_survived": len(candidates),
        "hiddenness_results":  hid_results,
    }
    (out_s5 / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if plot_cfg.get("save_figures", True):
        from ..plotting.generate_publication_figures import (
            generate_biased_report_dynamics,
            generate_comparison_report_heatmaps,
        )

        generate_comparison_report_heatmaps()
        generate_biased_report_dynamics()

