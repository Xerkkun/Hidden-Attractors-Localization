import numpy as np
from scipy.optimize import minimize, root_scalar
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from .transfer import W_eval
from .describing_function import (
    evaluate_describing_function,
    evaluate_describing_function_batch,
    solve_amplitude_from_gain,
)

@dataclass
class HarmonicCandidate:
    A0: float
    omega0: float
    k: float
    df_method_used: str
    df_warning: Optional[str]
    residual_gain: float

class CandidateList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detailed_candidates: List[HarmonicCandidate] = []

def find_harmonic_candidates(
    system: Any,
    transfer_mode: str,
    seed_strategy: str = "k_phi",
    df_residual_tol: float = 1e-2,
    omega_min: float = 0.01,
    omega_max: float = 20.0,
    amplitude_min: float = 0.01,
    amplitude_max: float = 20.0,
    grid_size_omega: int = 200,
    grid_size_amplitude: int = 200,
    root_refinement: bool = True,
    q: Optional[float] = None,
    describing_function_mode: str = "auto",
    transfer_convention: str = "standard",
    harmonic_condition: str = "1_minus_WN",
    precomputed_W_vals: Optional[np.ndarray] = None,
    precomputed_omega_grid: Optional[np.ndarray] = None,
) -> List[Tuple[float, float, float]]:
    """Find all candidate pairs (A0, omega0, k) solving the harmonic condition."""
    if q is None:
        q = getattr(system, "q", 1.0)

    candidates = CandidateList()

    if seed_strategy == "nyquist_df":
        if precomputed_W_vals is not None and precomputed_omega_grid is not None:
            ws = precomputed_omega_grid
            W_vals = precomputed_W_vals
        else:
            ws = np.linspace(omega_min, omega_max, grid_size_omega)
            W_vals = W_eval(
                ws, q, transfer_mode,
                system.P, system.b, system.r,
                transfer_convention=transfer_convention,
            )

        as_ = np.linspace(amplitude_min, amplitude_max, grid_size_amplitude)
        N_vals = evaluate_describing_function_batch(
            system, as_, mode=describing_function_mode
        )  # shape (n_A,)

        if harmonic_condition == "1_minus_WN":
            res = np.abs(W_vals[None, :] * N_vals[:, None] - 1.0)  # (n_A, n_w)
        else:
            res = np.abs(W_vals[None, :] * N_vals[:, None] + 1.0)

        best_pts = []
        inner = res[1:-1, 1:-1]
        is_local_min = (
            (inner <= res[0:-2, 0:-2]) & (inner <= res[0:-2, 1:-1]) &
            (inner <= res[0:-2, 2:  ]) & (inner <= res[1:-1, 0:-2]) &
            (inner <= res[1:-1, 2:  ]) & (inner <= res[2:  , 0:-2]) &
            (inner <= res[2:  , 1:-1]) & (inner <= res[2:  , 2:  ]) &
            (inner < 0.2)
        )
        ii, jj = np.where(is_local_min)
        for i, j in zip(ii + 1, jj + 1):
            best_pts.append((as_[i], ws[j], res[i, j]))

        for A_grid, w_grid, _ in best_pts:
            if root_refinement:
                def obj_func(z, _system=system, _q=q, _tm=transfer_mode,
                             _tc=transfer_convention, _hc=harmonic_condition,
                             _dfm=describing_function_mode):
                    A, w = z
                    if A <= 0.01 or w <= 0.01:
                        return 1e6
                    try:
                        W = W_eval(
                            w, _q, _tm,
                            _system.P, _system.b, _system.r,
                            transfer_convention=_tc,
                        )
                        N_val = evaluate_describing_function(
                            _system, A, mode=_dfm
                        ).value
                        val = W * N_val
                        if _hc == "1_minus_WN":
                            return float(np.abs(val - 1.0))
                        else:
                            return float(np.abs(val + 1.0))
                    except Exception:
                        return 1e6

                res_opt = minimize(
                    obj_func,
                    [A_grid, w_grid],
                    bounds=[(amplitude_min, amplitude_max), (omega_min, omega_max)],
                    method="L-BFGS-B",
                )
                if res_opt.success and res_opt.fun < df_residual_tol:
                    A_ref, w_ref = res_opt.x
                    df_res = evaluate_describing_function(
                        system, A_ref, mode=describing_function_mode
                    )
                    k_ref = df_res.value
                    if not any(
                        np.allclose([A_ref, w_ref], [c[0], c[1]], rtol=1e-2)
                        for c in candidates
                    ):
                        actual_residual = res_opt.fun
                        candidates.append((float(A_ref), float(w_ref), float(k_ref)))
                        candidates.detailed_candidates.append(
                            HarmonicCandidate(
                                A0=float(A_ref),
                                omega0=float(w_ref),
                                k=float(k_ref),
                                df_method_used=df_res.method,
                                df_warning=df_res.warning,
                                residual_gain=float(actual_residual),
                            )
                        )
            else:
                df_res = evaluate_describing_function(
                    system, A_grid, mode=describing_function_mode
                )
                k_grid = df_res.value
                candidates.append((float(A_grid), float(w_grid), float(k_grid)))
                candidates.detailed_candidates.append(
                    HarmonicCandidate(
                        A0=float(A_grid),
                        omega0=float(w_grid),
                        k=float(k_grid),
                        df_method_used=df_res.method,
                        df_warning=df_res.warning,
                        residual_gain=0.0,
                    )
                )

    elif seed_strategy in {"k_phi", "imw_gain"}:
        scan_n = max(grid_size_omega, 20000)
        if (
            precomputed_W_vals is not None
            and precomputed_omega_grid is not None
            and len(precomputed_omega_grid) >= scan_n
        ):
            ws = precomputed_omega_grid
            W_vals_scan = precomputed_W_vals
        else:
            ws = np.linspace(omega_min, omega_max, scan_n)
            W_vals_scan = W_eval(
                ws, q, transfer_mode,
                system.P, system.b, system.r,
                transfer_convention=transfer_convention,
            )

        ims = W_vals_scan.imag

        sign_changes = np.where(np.diff(np.sign(ims)))[0]
        valid = ~(np.isnan(ims[sign_changes]) | np.isnan(ims[sign_changes + 1]))
        sign_changes = sign_changes[valid]

        omega_roots = []
        for j in sign_changes:
            try:
                def root_f(w, _q=q, _tm=transfer_mode, _P=system.P,
                           _b=system.b, _r=system.r, _tc=transfer_convention):
                    return W_eval(w, _q, _tm, _P, _b, _r,
                                  transfer_convention=_tc).imag

                sol = root_scalar(root_f, bracket=[float(ws[j]), float(ws[j + 1])],
                                  method="bisect")
                if sol.converged:
                    omega_roots.append(sol.root)
            except Exception:
                pass

        for w0 in omega_roots:
            W0 = W_eval(
                w0, q, transfer_mode,
                system.P, system.b, system.r,
                transfer_convention=transfer_convention,
            )
            if abs(W0.real) < 1e-12:
                continue

            if harmonic_condition == "1_minus_WN":
                k_gain = 1.0 / W0.real
            else:
                k_gain = -1.0 / W0.real

            try:
                A0 = solve_amplitude_from_gain(
                    system, k_gain, amplitude_min, amplitude_max,
                    mode=describing_function_mode,
                )
                df_res = evaluate_describing_function(
                    system, A0, mode=describing_function_mode
                )
                if not any(
                    np.allclose([A0, w0], [c[0], c[1]], rtol=1e-2)
                    for c in candidates
                ):
                    candidates.append((float(A0), float(w0), float(k_gain)))
                    candidates.detailed_candidates.append(
                        HarmonicCandidate(
                            A0=float(A0),
                            omega0=float(w0),
                            k=float(k_gain),
                            df_method_used=df_res.method,
                            df_warning=df_res.warning,
                            residual_gain=abs(df_res.value - k_gain),
                        )
                    )
            except Exception:
                pass

    else:
        raise ValueError(f"Unknown seed_strategy: {seed_strategy}")

    if candidates:
        sorted_pairs = sorted(
            zip(candidates, candidates.detailed_candidates),
            key=lambda pair: pair[0][2],
        )
        candidates.clear()
        candidates.detailed_candidates.clear()
        for cand_tup, detailed_cand in sorted_pairs:
            candidates.append(cand_tup)
            candidates.detailed_candidates.append(detailed_cand)

    return candidates
