"""Search arctan Chua fractional candidates with DF seeds and continuation.

This is an exploratory search tool, not a hiddenness verifier.  It looks for
parameter sets whose describing-function seed survives numerical continuation
to eta=1 and whose post-continuation trajectory is not classified as periodic
or convergent by the finite-time diagnostics.

Continuation modes:
    abm_full    - full-memory ABM Caputo continuation.
    abm_restart - ABM per eta segment with last-point memory restart.
    adm_restart - Wu2023 local ADM recurrence per eta segment; arctan rho=1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from hidden_attractors.continuation.continuation_fractional import run_fractional_continuation
from hidden_attractors.diagnostics.periodicity import classify_post_transient_periodicity
from hidden_attractors.integrations.adm_wu2023 import adm_wu2023_integrate
from hidden_attractors.integrations.fractional_c import fractional_integrate
from hidden_attractors.models.chua import chua_parameters
from hidden_attractors.seed_generation.chua_arctan_wu2023 import find_centered_arctan_wu2023_branches
from hidden_attractors.systems.builtins import _chua_lure_system


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "arctan_full_memory_search"


def _float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _values_or_single(values_text: str | None, single_value: float) -> list[float]:
    if values_text:
        return _float_list(values_text)
    return [float(single_value)]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _system_from_params(params: dict[str, float], q: float) -> SimpleNamespace:
    payload = {"model": "arctan", **params, "q": float(q)}
    return SimpleNamespace(parameters=payload, lure=_chua_lure_system(payload))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_trajectory(path_prefix: Path, title: str, trajectory: np.ndarray) -> list[str]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    sample = trajectory
    if sample.shape[0] > 6000:
        idx = np.linspace(0, sample.shape[0] - 1, 6000, dtype=int)
        sample = sample[idx]
    t, x, y, z = sample[:, 0], sample[:, 1], sample[:, 2], sample[:, 3]
    written: list[str] = []

    fig = plt.figure(figsize=(8.0, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, linewidth=0.45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    png = path_prefix.with_name(path_prefix.name + "_phase3d.png")
    fig.savefig(png, dpi=220)
    plt.close(fig)
    written.append(str(png.relative_to(ROOT)))

    fig, axes = plt.subplots(3, 1, figsize=(10.0, 7.0), sharex=True)
    for ax, label, values in zip(axes, ("x", "y", "z"), (x, y, z)):
        ax.plot(t, values, linewidth=0.55)
        ax.set_ylabel(label)
        ax.grid(True, linestyle=":", linewidth=0.6)
    axes[-1].set_xlabel("t")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    png = path_prefix.with_name(path_prefix.name + "_timeseries.png")
    fig.savefig(png, dpi=220)
    plt.close(fig)
    written.append(str(png.relative_to(ROOT)))
    return written


def _score_periodicity(periodicity: dict[str, Any]) -> float:
    metrics = periodicity.get("component_metrics", [])
    if not metrics:
        return 0.0
    entropy = float(np.mean([float(row.get("spectral_entropy", 0.0)) for row in metrics]))
    range_score = float(np.mean([math.log1p(float(row.get("range", 0.0))) for row in metrics]))
    periodic_penalty = 2.0 * int(periodicity.get("periodic_post_transient", False))
    return entropy + 0.15 * range_score - periodic_penalty


def _metric_summary(periodicity: dict[str, Any]) -> dict[str, float]:
    metrics = periodicity.get("component_metrics", [])
    if not metrics:
        return {
            "max_component_range": 0.0,
            "mean_component_range": 0.0,
            "mean_spectral_entropy": 0.0,
            "max_spectral_entropy": 0.0,
            "max_fft_dominant_power_ratio": 0.0,
        }
    ranges = [float(row.get("range", 0.0)) for row in metrics]
    entropies = [float(row.get("spectral_entropy", 0.0)) for row in metrics]
    ratios = [float(row.get("fft_dominant_power_ratio", 0.0)) for row in metrics]
    return {
        "max_component_range": float(max(ranges)),
        "mean_component_range": float(np.mean(ranges)),
        "mean_spectral_entropy": float(np.mean(entropies)),
        "max_spectral_entropy": float(max(entropies)),
        "max_fft_dominant_power_ratio": float(max(ratios)),
    }


def _step_record(
    *,
    eta: float,
    x_in: np.ndarray,
    x_out: np.ndarray,
    trajectory: np.ndarray,
    status: str,
    method: str,
    memory_mode: str,
    memory_policy: str,
    caputo_history_accumulated: bool,
) -> dict[str, Any]:
    return {
        "lambda_value": float(eta),
        "x_in": np.asarray(x_in, dtype=float),
        "x_out": np.asarray(x_out, dtype=float),
        "trajectory": np.asarray(trajectory, dtype=float),
        "status": status,
        "integrator": method,
        "memory_mode": memory_mode,
        "memory_policy": memory_policy,
        "caputo_history_accumulated": bool(caputo_history_accumulated),
    }


def _run_abm_restart_continuation(
    *,
    system: SimpleNamespace,
    seed_x0: np.ndarray,
    k_gain: float,
    eta_values: np.ndarray,
    h: float,
    q: float,
    t_transient: float,
    t_keep: float,
    divergence_norm: float,
) -> list[dict[str, Any]]:
    """ABM continuation with a last-point restart at every eta stage."""
    x_in = np.asarray(seed_x0, dtype=float).copy()
    steps: list[dict[str, Any]] = []
    p0 = system.lure.matrix + float(k_gain) * np.outer(system.lure.input_vector, system.lure.output_vector)
    n_keep = int(np.ceil(float(t_keep) / float(h)))
    t_total = float(t_transient) + float(t_keep)
    for eta in eta_values:
        eta_f = float(eta)

        def rhs_deformed(_t: float, state: np.ndarray, _eta: float = eta_f) -> np.ndarray:
            sigma = float(system.lure.output_vector @ state)
            delta = float(system.lure.nonlinearity(sigma)) - float(k_gain) * sigma
            return p0 @ state + _eta * system.lure.input_vector * delta

        times, states, status, _info = fractional_integrate(
            rhs=rhs_deformed,
            x0=x_in,
            q=q,
            h=h,
            t_final=t_total,
            method="abm",
            memory_mode="full",
            memory_window_length=None,
            system=None,
            use_c_backend=False,
            divergence_norm=divergence_norm,
            return_history=True,
            allow_python_fallback=True,
        )
        trajectory = np.column_stack((times[-n_keep:], states[-n_keep:])) if len(times) else np.empty((0, 4))
        x_out = states[-1].copy() if len(states) else x_in.copy()
        steps.append(_step_record(
            eta=eta_f,
            x_in=x_in,
            x_out=x_out,
            trajectory=trajectory,
            status=status,
            method="ABM",
            memory_mode="restart",
            memory_policy="last_point_restart",
            caputo_history_accumulated=False,
        ))
        x_in = x_out
        if status != "ok":
            break
    return steps


def _run_adm_restart_continuation(
    *,
    params: dict[str, float],
    seed_x0: np.ndarray,
    k_gain: float,
    eta_values: np.ndarray,
    h: float,
    q: float,
    t_transient: float,
    t_keep: float,
    divergence_norm: float,
) -> list[dict[str, Any]]:
    """Local ADM continuation with last-point restarts.

    This reuses the Wu2023 ADM recurrence and is therefore restricted to
    arctan(rho*x) with rho=1.
    """
    if abs(float(params.get("rho", 1.0)) - 1.0) > 1.0e-12:
        raise ValueError("adm_restart currently supports only rho=1 because the Wu2023 ADM recurrence is for atan(x).")
    x_in = np.asarray(seed_x0, dtype=float).copy()
    steps: list[dict[str, Any]] = []
    n_total = int(np.ceil((float(t_transient) + float(t_keep)) / float(h)))
    n_keep = int(np.ceil(float(t_keep) / float(h)))
    for eta in eta_values:
        eta_f = float(eta)
        m_eff = float(params["a1"]) + float(k_gain) * (1.0 - eta_f)
        a2_eff = eta_f * float(params["a2"])
        adm_params = {
            "alpha": float(params["alpha"]),
            "beta": float(params["beta"]),
            "gamma": float(params["gamma"]),
            "m": m_eff,
            "n": m_eff + a2_eff,
        }
        times, states, status, _info = adm_wu2023_integrate(
            params=adm_params,
            x0=x_in,
            q=q,
            h=h,
            N=n_total,
            divergence_norm=divergence_norm,
        )
        trajectory = np.column_stack((times[-n_keep:], states[-n_keep:])) if len(times) else np.empty((0, 4))
        x_out = states[-1].copy() if len(states) else x_in.copy()
        steps.append(_step_record(
            eta=eta_f,
            x_in=x_in,
            x_out=x_out,
            trajectory=trajectory,
            status=status,
            method="ADM_WU2023",
            memory_mode="restart",
            memory_policy="last_point_restart_local_adm",
            caputo_history_accumulated=False,
        ))
        x_in = x_out
        if status != "ok":
            break
    return steps


def _run_configured_continuation(
    *,
    args: argparse.Namespace,
    system: SimpleNamespace,
    params: dict[str, float],
    branch: Any,
    eta_values: np.ndarray,
    q: float,
) -> list[dict[str, Any]]:
    method = str(args.continuation_method)
    if method == "abm_full":
        return run_fractional_continuation(
            system=system,
            seed_x0=branch.seed,
            k_gain=branch.gain,
            lambda_values=eta_values,
            h=float(args.h),
            memory_mode="full",
            integrator="abm",
            use_c_backend=False,
            require_c_backend=False,
            allow_python_fallback=True,
            t_transient=float(args.t_transient),
            t_keep=float(args.t_keep),
            q=q,
            div_threshold=float(args.divergence_norm),
        )
    if method == "abm_restart":
        return _run_abm_restart_continuation(
            system=system,
            seed_x0=branch.seed,
            k_gain=branch.gain,
            eta_values=eta_values,
            h=float(args.h),
            q=q,
            t_transient=float(args.t_transient),
            t_keep=float(args.t_keep),
            divergence_norm=float(args.divergence_norm),
        )
    if method == "adm_restart":
        return _run_adm_restart_continuation(
            params=params,
            seed_x0=branch.seed,
            k_gain=branch.gain,
            eta_values=eta_values,
            h=float(args.h),
            q=q,
            t_transient=float(args.t_transient),
            t_keep=float(args.t_keep),
            divergence_norm=float(args.divergence_norm),
        )
    raise ValueError(f"unsupported continuation method: {method}")


def run_search(args: argparse.Namespace) -> dict[str, Any]:
    if float(args.h) > 0.01:
        raise ValueError("h must be <= 0.01 for this search contract.")
    if str(args.memory_mode) != "full" and str(args.continuation_method) == "abm_full":
        raise ValueError("abm_full requires memory_mode='full'.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = out_dir / "trajectories"
    figure_dir = out_dir / "figures"

    alpha_values = _values_or_single(args.alpha_values, float(args.alpha))
    beta_values = _values_or_single(args.beta_values, float(args.beta))
    gamma_values = _values_or_single(args.gamma_values, float(args.gamma))
    accepted_labels = set(str(label) for label in args.accepted_labels.split(",") if label)
    q = float(args.q)
    eta_values = np.linspace(0.0, 1.0, int(args.n_eta) + 1)
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    cases_run = 0

    stop = False
    for alpha in alpha_values:
        for beta in beta_values:
            for gamma in gamma_values:
                for a1 in _float_list(args.a1_values):
                    for a2 in _float_list(args.a2_values):
                        for rho in _float_list(args.rho_values):
                            if int(args.max_cases) > 0 and cases_run >= int(args.max_cases):
                                stop = True
                                break
                            params_obj = chua_parameters(
                                model="arctan",
                                alpha=alpha,
                                beta=beta,
                                gamma=gamma,
                                a1=a1,
                                a2=a2,
                                rho=rho,
                            )
                            params = {
                                "alpha": alpha,
                                "beta": beta,
                                "gamma": gamma,
                                "a1": a1,
                                "a2": a2,
                                "rho": rho,
                            }
                            try:
                                branches = find_centered_arctan_wu2023_branches(
                                    q=q,
                                    params=params_obj,
                                    nscan=int(args.nscan),
                                    transfer_mode=args.transfer_mode,
                                )
                            except Exception as exc:
                                rows.append({
                                    "status": "seed_error",
                                    "error": str(exc),
                                    **params,
                                })
                                continue
                            if not branches:
                                rows.append({"status": "no_df_branch", **params})
                                continue

                            system = _system_from_params(params, q)
                            for branch in branches[: int(args.max_branches)]:
                                cases_run += 1
                                case_id = (
                                    f"alpha_{alpha:g}_beta_{beta:g}_gamma_{gamma:g}_"
                                    f"a1_{a1:g}_a2_{a2:g}_rho_{rho:g}_"
                                    f"branch_{branch.branch_index}"
                                ).replace("-", "m").replace(".", "p")
                                try:
                                    steps = _run_configured_continuation(
                                        args=args,
                                        system=system,
                                        params=params,
                                        branch=branch,
                                        eta_values=eta_values,
                                        q=q,
                                    )
                                except Exception as exc:
                                    rows.append({
                                        "case_id": case_id,
                                        "status": "continuation_error",
                                        "continuation_method": args.continuation_method,
                                        "error": str(exc),
                                        "branch": branch.branch_index,
                                        "omega": branch.omega,
                                        "k": branch.gain,
                                        "A": branch.amplitude,
                                        **params,
                                    })
                                    continue
                                final = steps[-1]
                                trajectory = np.asarray(final["trajectory"], dtype=float)
                                if trajectory.size:
                                    shifted = trajectory.copy()
                                    shifted[:, 0] = shifted[:, 0] - shifted[0, 0]
                                else:
                                    shifted = trajectory.reshape(0, 4)
                                periodicity = classify_post_transient_periodicity(
                                    shifted,
                                    h=float(args.h),
                                    config={
                                        "t_transient": 0.0,
                                        "require_two_components": True,
                                        "entropy_min": float(args.entropy_min),
                                        "dominant_ratio_max": float(args.dominant_ratio_max),
                                        "relaxed_dominant_ratio": float(args.relaxed_dominant_ratio),
                                        "freq_drift_max": float(args.freq_drift_max),
                                        "min_range": float(args.min_range),
                                        "divergence_norm": float(args.divergence_norm),
                                    },
                                ) if shifted.shape[0] else {"candidate_label": "empty_trajectory"}
                                score = _score_periodicity(periodicity)
                                metric_summary = _metric_summary(periodicity)
                                final_eta = float(final["lambda_value"])
                                accepted = (
                                    final.get("status") == "ok"
                                    and abs(final_eta - 1.0) < 1.0e-12
                                    and str(periodicity.get("candidate_label")) in accepted_labels
                                )
                                candidate_label = str(periodicity.get("candidate_label"))
                                if accepted and candidate_label == "chaotic_candidate_pending_robustness":
                                    row_status = "chaotic_candidate"
                                    candidate_tier = "strong_chaotic_screen"
                                elif accepted:
                                    row_status = "nonperiodic_candidate_exploratory"
                                    candidate_tier = "weak_nonperiodic_screen"
                                else:
                                    row_status = "rejected_or_inconclusive"
                                    candidate_tier = ""
                                traj_path = ""
                                if accepted or bool(args.save_all_trajectories):
                                    traj_path_obj = trajectory_dir / f"{case_id}_final.csv"
                                    traj_path_obj.parent.mkdir(parents=True, exist_ok=True)
                                    np.savetxt(
                                        traj_path_obj,
                                        shifted,
                                        delimiter=",",
                                        header="t,x,y,z",
                                        comments="",
                                    )
                                    traj_path = str(traj_path_obj.relative_to(ROOT))
                                figures: list[str] = []
                                if accepted and bool(args.plot) and shifted.shape[0]:
                                    figures = _plot_trajectory(
                                        figure_dir / case_id,
                                        f"{case_id} | {periodicity.get('candidate_label')}",
                                        shifted,
                                    )
                                row = {
                                    "case_id": case_id,
                                    "status": row_status,
                                    "candidate_tier": candidate_tier,
                                    "continuation_status": final.get("status"),
                                    "final_eta": final_eta,
                                    "candidate_label": candidate_label,
                                    "periodic_post_transient": periodicity.get("periodic_post_transient"),
                                    "diverged_post_transient": periodicity.get("diverged_post_transient"),
                                    "n_periodic_components": periodicity.get("n_periodic_components"),
                                    "score": score,
                                    **metric_summary,
                                    "q": q,
                                    "h": float(args.h),
                                    "continuation_method": args.continuation_method,
                                    "integrator": final.get("integrator", "ABM"),
                                    "memory_mode": final.get("memory_mode", "full"),
                                    "memory_policy": final.get("memory_policy", "full_history"),
                                    "caputo_history_accumulated": final.get("caputo_history_accumulated", True),
                                    "transfer_mode": args.transfer_mode,
                                    "transfer_exponent_applied": args.transfer_mode != "published_integer_laplace",
                                    "branch": branch.branch_index,
                                    "omega": branch.omega,
                                    "k": branch.gain,
                                    "A": branch.amplitude,
                                    "seed": json.dumps(branch.seed.tolist(), separators=(",", ":")),
                                    "trajectory": traj_path,
                                    "figures": json.dumps(figures, separators=(",", ":")),
                                    **params,
                                }
                                rows.append(row)
                                if accepted:
                                    candidates.append({**row, "periodicity": periodicity})
                            if int(args.max_cases) > 0 and cases_run >= int(args.max_cases):
                                stop = True
                                break
                        if stop:
                            break
                    if stop:
                        break
                if stop:
                    break
            if stop:
                break
        if stop:
            break

    ranked = sorted(rows, key=lambda item: float(item.get("score", -999.0)), reverse=True)
    strong_candidates = [
        row for row in candidates
        if row.get("candidate_label") == "chaotic_candidate_pending_robustness"
    ]
    exploratory_candidates = [
        row for row in candidates
        if row.get("candidate_label") != "chaotic_candidate_pending_robustness"
    ]
    summary = {
        "status": "completed",
        "output_dir": str(out_dir.relative_to(ROOT)),
        "cases_run": cases_run,
        "candidate_count": len(candidates),
        "strong_candidate_count": len(strong_candidates),
        "exploratory_candidate_count": len(exploratory_candidates),
        "contract": {
            "q": q,
            "h": float(args.h),
            "h_max": 0.01,
            "continuation_method": args.continuation_method,
            "integrator": {
                "abm_full": "ABM",
                "abm_restart": "ABM",
                "adm_restart": "ADM_WU2023",
            }.get(str(args.continuation_method), str(args.continuation_method)),
            "memory_mode": "full" if args.continuation_method == "abm_full" else "restart",
            "memory_policy": {
                "abm_full": "full_history",
                "abm_restart": "last_point_restart",
                "adm_restart": "last_point_restart_local_adm",
            }.get(str(args.continuation_method), "unknown"),
            "caputo_history_accumulated": args.continuation_method == "abm_full",
            "transfer_mode": args.transfer_mode,
            "accepted_labels": sorted(accepted_labels),
            "hiddenness_tested": False,
        },
        "top_rows": ranked[: min(10, len(ranked))],
        "candidates": candidates,
    }
    _write_csv(out_dir / "candidate_scan.csv", rows)
    _write_csv(out_dir / "candidate_scan_ranked.csv", ranked)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=_json_default) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--q", type=float, default=0.99)
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--memory-mode", default="full")
    parser.add_argument("--continuation-method", default="abm_full", choices=["abm_full", "abm_restart", "adm_restart"])
    parser.add_argument("--transfer-mode", default="published_integer_laplace", choices=["published_integer_laplace", "fractional_spectral"])
    parser.add_argument("--alpha", type=float, default=8.4562)
    parser.add_argument("--beta", type=float, default=12.0732)
    parser.add_argument("--gamma", type=float, default=0.0052)
    parser.add_argument("--alpha-values", default=None)
    parser.add_argument("--beta-values", default=None)
    parser.add_argument("--gamma-values", default=None)
    parser.add_argument("--a1-values", default="0.2,0.4,0.6")
    parser.add_argument("--a2-values", default="-1.0,-1.5585,-2.0")
    parser.add_argument("--rho-values", default="1.0")
    parser.add_argument("--nscan", type=int, default=2000)
    parser.add_argument("--max-branches", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--n-eta", type=int, default=4)
    parser.add_argument("--t-transient", type=float, default=3.0)
    parser.add_argument("--t-keep", type=float, default=3.0)
    parser.add_argument("--divergence-norm", type=float, default=120.0)
    parser.add_argument("--entropy-min", type=float, default=0.25)
    parser.add_argument("--dominant-ratio-max", type=float, default=0.65)
    parser.add_argument("--relaxed-dominant-ratio", type=float, default=0.45)
    parser.add_argument("--freq-drift-max", type=float, default=0.05)
    parser.add_argument("--min-range", type=float, default=0.01)
    parser.add_argument("--accepted-labels", default="chaotic_candidate_pending_robustness")
    parser.add_argument("--save-all-trajectories", action="store_true")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.set_defaults(plot=True)
    args = parser.parse_args()
    summary = run_search(args)
    print(json.dumps({
        "status": summary["status"],
        "output_dir": summary["output_dir"],
        "cases_run": summary["cases_run"],
        "candidate_count": summary["candidate_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
