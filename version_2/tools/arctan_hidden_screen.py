"""Screen Chua-arctan parameters before equilibrium-neighborhood tests.

This exploratory stage uses full-history Caputo ABM and finite-time chaos
indicators. It never promotes a row to a hidden attractor.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hidden_attractors.analysis.spectral import spectral_diagnostics_multicoordinate
from hidden_attractors.analysis.zero_one import zero_one_multicoordinate
from hidden_attractors.diagnostics.periodicity import classify_post_transient_periodicity
from hidden_attractors.integrations.general import integrate_general
from hidden_attractors.models.chua import chua_parameters, equilibria_arctan, jacobian_arctan
from hidden_attractors.seed_generation.chua_arctan_wu2023 import (
    find_centered_arctan_wu2023_branches,
)
from hidden_attractors.systems import get_system


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "arctan_hidden_candidate_search"


def _float_list(text: str) -> list[float]:
    return [float(token.strip()) for token in text.split(",") if token.strip()]


def _seed(text: str) -> np.ndarray:
    values = _float_list(text)
    if len(values) != 3:
        raise argparse.ArgumentTypeError("seed must have three comma-separated values")
    return np.asarray(values, dtype=float)


def _range(text: str) -> tuple[float, float]:
    values = _float_list(text)
    if len(values) != 2 or values[0] > values[1]:
        raise argparse.ArgumentTypeError("range must contain ordered min,max values")
    return values[0], values[1]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_system(params: dict[str, float], q: float):
    base = get_system("chua-arctan")
    merged = dict(base.parameters)
    merged.update(params)
    merged.update({"model": "arctan", "q": q, "system_id": "chua_fractional_arctan"})
    return replace(base, parameters=merged)


def _matignon(params: dict[str, float], q: float) -> dict[str, Any]:
    model = chua_parameters(model="arctan", **params)
    threshold = q * np.pi / 2.0
    equilibria = equilibria_arctan(model)
    stability: dict[str, str] = {}
    margins: list[float] = []
    for name, state in equilibria.items():
        eigenvalues = np.linalg.eigvals(jacobian_arctan(state, model))
        margin = float(np.min(np.abs(np.angle(eigenvalues)) - threshold))
        margins.append(margin)
        stability[name] = "stable" if margin > 0.0 else "unstable"
    return {
        "equilibria_count": len(equilibria),
        "stability": stability,
        "minimum_margin": min(margins),
    }


def _poincare_cloud(tail: np.ndarray) -> dict[str, float | int]:
    if len(tail) < 3:
        return {"crossings": 0, "normalized_spread": 0.0}
    section = float(np.median(tail[:, 0]))
    left = tail[:-1, 0] - section
    right = tail[1:, 0] - section
    indices = np.flatnonzero((left < 0.0) & (right >= 0.0))
    points = []
    for index in indices:
        denominator = right[index] - left[index]
        weight = 0.0 if denominator == 0.0 else -left[index] / denominator
        points.append(tail[index, 1:3] + weight * (tail[index + 1, 1:3] - tail[index, 1:3]))
    if len(points) < 2:
        return {"crossings": len(points), "normalized_spread": 0.0}
    cloud = np.asarray(points)
    scale = float(np.linalg.norm(np.ptp(tail[:, 1:3], axis=0)))
    if len(points) > 2:
        spread = float(np.sqrt(np.trace(np.cov(cloud.T))))
    else:
        spread = float(np.linalg.norm(cloud[1] - cloud[0]))
    return {"crossings": len(points), "normalized_spread": spread / max(scale, 1.0e-12)}


def _diagnose(
    times: np.ndarray,
    states: np.ndarray,
    *,
    h: float,
    t_burn: float,
    zero_one_samples: Sequence[int],
) -> dict[str, Any]:
    tail = states[times >= t_burn]
    if len(tail) < 100 or not np.all(np.isfinite(tail)):
        return {"screen_label": "inconclusive_short_or_nonfinite", "score": -10.0}
    zero_one_runs = [
        zero_one_multicoordinate(
            times,
            states,
            t_burn,
            n_c=40,
            max_samples=sample_count,
        )
        for sample_count in zero_one_samples
    ]
    k_values = [float(run["K_global_median"]) for run in zero_one_runs]
    chaotic_votes = sum(run["state_global"] == "zero_one_chaotic_candidate" for run in zero_one_runs)
    spectral = spectral_diagnostics_multicoordinate(times, states, t_burn)
    spectral_rows = list(spectral["coordinate_results"].values())
    entropy = float(np.mean([row["spectral_entropy"] for row in spectral_rows]))
    peak = float(max(row["peak_dominance"] for row in spectral_rows))
    periodicity = classify_post_transient_periodicity(
        np.column_stack((times, states)),
        h=h,
        config={"t_transient": t_burn},
    )
    poincare = _poincare_cloud(tail)
    state_range = np.ptp(tail, axis=0)
    nontrivial = float(np.max(state_range)) >= 0.05
    section_ok = poincare["crossings"] >= 8 and poincare["normalized_spread"] >= 0.01
    robust_k = float(np.median(k_values))
    if nontrivial and chaotic_votes >= 2 and section_ok:
        label = "strong_chaos_candidate_pending_hiddenness"
    elif nontrivial and (chaotic_votes >= 1 or robust_k >= 0.5) and section_ok:
        label = "chaos_candidate_exploratory"
    elif periodicity.get("periodic_post_transient"):
        label = "regular_periodic_rejected"
    else:
        label = "inconclusive_nonperiodic"
    score = (
        robust_k
        + 0.35 * entropy
        + 0.5 * min(float(poincare["normalized_spread"]), 1.0)
        + 0.15 * min(float(poincare["crossings"]) / 30.0, 1.0)
        - 0.25 * float(peak > 0.6)
    )
    return {
        "screen_label": label,
        "score": float(score),
        "zero_one_K_robust_median": robust_k,
        "zero_one_K_by_sample_count": k_values,
        "zero_one_states": [run["state_global"] for run in zero_one_runs],
        "zero_one_chaotic_votes": chaotic_votes,
        "spectral_state": spectral["state_global"],
        "spectral_entropy_mean": entropy,
        "peak_dominance_max": peak,
        "periodicity_label": periodicity["candidate_label"],
        "poincare_crossings": poincare["crossings"],
        "poincare_normalized_spread": poincare["normalized_spread"],
        "state_range": state_range.tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.h > 0.01:
        raise ValueError("search contract requires h <= 0.01")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.output_dir or DEFAULT_OUTPUT_ROOT / f"run_{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    trajectory_dir = out / "trajectories"
    trajectory_dir.mkdir(exist_ok=True)
    a1_values = _float_list(args.a1_values) if args.a1_values else [args.a1]
    a2_values = _float_list(args.a2_values) if args.a2_values else [args.a2]
    rho_values = _float_list(args.rho_values) if args.rho_values else [args.rho]
    base_params = {"gamma": args.gamma}
    if args.random_cases > 0:
        rng = np.random.default_rng(args.random_seed)
        bounds = {
            "q": _range(args.q_range),
            "alpha": _range(args.alpha_range),
            "beta": _range(args.beta_range),
            "gamma": _range(args.gamma_range),
            "a1": _range(args.a1_range),
            "a2": _range(args.a2_range),
            "rho": _range(args.rho_range),
        }
        seed_bounds = [_range(text) for text in (args.x0_range, args.y0_range, args.z0_range)]
        cases = []
        for _ in range(args.random_cases):
            values = {
                name: float(rng.uniform(low, high))
                for name, (low, high) in bounds.items()
            }
            initial_condition = (
                np.asarray(
                    [rng.uniform(low, high) for low, high in seed_bounds],
                    dtype=float,
                )
                if args.randomize_initial_condition
                else np.asarray(args.seed, dtype=float)
            )
            cases.append(
                (
                    values["q"],
                    values["alpha"],
                    values["beta"],
                    values["gamma"],
                    values["a1"],
                    values["a2"],
                    values["rho"],
                    initial_condition,
                    "random" if args.randomize_initial_condition else "fixed",
                )
            )
    else:
        cases = []
        for q in _float_list(args.q_values):
            for alpha in _float_list(args.alpha_values):
                for beta in _float_list(args.beta_values):
                    for a1 in a1_values:
                        for a2 in a2_values:
                            for rho in rho_values:
                                if args.seed_strategy == "df_branches":
                                    params = chua_parameters(
                                        model="arctan",
                                        alpha=alpha,
                                        beta=beta,
                                        gamma=args.gamma,
                                        a1=a1,
                                        a2=a2,
                                        rho=rho,
                                    )
                                    branches = find_centered_arctan_wu2023_branches(
                                        q=q,
                                        params=params,
                                        transfer_mode=(
                                            "published_integer"
                                            if np.isclose(q, 1.0)
                                            else "fractional_spectral"
                                        ),
                                        nscan=args.df_nscan,
                                    )
                                    for branch in branches:
                                        cases.append(
                                            (
                                                q,
                                                alpha,
                                                beta,
                                                args.gamma,
                                                a1,
                                                a2,
                                                rho,
                                                np.asarray(branch.seed, dtype=float),
                                                f"df_branch_{branch.branch_index}",
                                            )
                                        )
                                else:
                                    cases.append(
                                        (
                                            q,
                                            alpha,
                                            beta,
                                            args.gamma,
                                            a1,
                                            a2,
                                            rho,
                                            np.asarray(args.seed, dtype=float),
                                            "fixed",
                                        )
                                    )
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    sample_counts = [int(value) for value in _float_list(args.zero_one_samples)]
    rows: list[dict[str, Any]] = []
    for index, (q, alpha, beta, gamma, a1, a2, rho, initial_condition, seed_source) in enumerate(cases):
        params = {
            "gamma": gamma,
            "alpha": alpha,
            "beta": beta,
            "a1": a1,
            "a2": a2,
            "rho": rho,
        }
        system = _build_system(params, q)
        if args.integrator == "auto":
            integrator = "heun" if np.isclose(q, 1.0) else "abm"
        else:
            integrator = args.integrator
        if np.isclose(q, 1.0) and integrator not in {"heun", "rk4", "efork3"}:
            raise ValueError("q=1 requires heun, rk4, efork3, or --integrator auto")
        if not np.isclose(q, 1.0) and integrator not in {"abm", "efork3"}:
            raise ValueError("q<1 requires abm, efork3, or --integrator auto")
        memory_window_length = (
            args.memory_window_steps
            if args.memory_mode == "window" and args.memory_window_steps > 0
            else None
        )
        if args.memory_mode == "window" and memory_window_length is None and not np.isclose(q, 1.0):
            raise ValueError("--memory-window-steps must be positive for windowed q<1 runs")
        times, states, status = integrate_general(
            lambda _time, state, active=system: active.evaluate(state),
            initial_condition,
            q=q,
            h=args.h,
            t_final=args.t_final,
            integrator=integrator,
            memory_mode=args.memory_mode,
            memory_window_length=memory_window_length,
            system=system,
            use_c_backend=True,
            divergence_norm=args.divergence_norm,
        )
        diagnostics = (
            _diagnose(
                times,
                states,
                h=args.h,
                t_burn=args.t_burn,
                zero_one_samples=sample_counts,
            )
            if status == "ok"
            else {"screen_label": status, "score": -20.0}
        )
        local = _matignon(params, q)
        case_id = (
            f"q{q:.4f}_alpha{alpha:.4f}_beta{beta:.4f}_"
            f"g{gamma:.4f}_a1{a1:.3f}_a2{a2:.3f}_rho{rho:.3f}_{seed_source}"
        ).replace("-", "m").replace(".", "p")
        row = {
            "case_index": index,
            "case_id": case_id,
            "q": q,
            "alpha": alpha,
            "beta": beta,
            **params,
            "h": args.h,
            "t_final": args.t_final,
            "t_burn": args.t_burn,
            "integration_status": status,
            "integrator": integrator,
            "memory_mode": "not_applicable" if np.isclose(q, 1.0) else args.memory_mode,
            "memory_window_steps": (
                None if np.isclose(q, 1.0) else memory_window_length
            ),
            "seed": json.dumps(initial_condition.tolist(), separators=(",", ":")),
            "seed_source": seed_source,
            "max_norm": float(np.max(np.linalg.norm(states, axis=1))) if len(states) else float("nan"),
            "equilibria_count": local["equilibria_count"],
            "equilibrium_stability": json.dumps(local["stability"], sort_keys=True),
            "matignon_minimum_margin": local["minimum_margin"],
            **diagnostics,
            "zero_one_K_by_sample_count": json.dumps(
                diagnostics.get("zero_one_K_by_sample_count", []), separators=(",", ":")
            ),
            "zero_one_states": json.dumps(
                diagnostics.get("zero_one_states", []), separators=(",", ":")
            ),
            "state_range": json.dumps(diagnostics.get("state_range", []), separators=(",", ":")),
            "hiddenness_status": "not_tested",
        }
        rows.append(row)
        if str(row["screen_label"]).startswith(("strong_", "chaos_")):
            np.savez_compressed(
                trajectory_dir / f"{case_id}.npz",
                times=times,
                states=states,
                seed=initial_condition,
            )
        print(
            f"[{index + 1}/{len(cases)}] {case_id}: "
            f"{row['screen_label']} score={float(row['score']):.4f}",
            flush=True,
        )
    ranked = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
    _write_csv(out / "candidate_screen.csv", rows)
    _write_csv(out / "candidate_screen_ranked.csv", ranked)
    summary = {
        "schema_version": "1.0",
        "stage": "arctan_fractional_chaos_screen",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out.relative_to(ROOT)),
        "cases_run": len(rows),
        "strong_candidates": sum(
            row["screen_label"] == "strong_chaos_candidate_pending_hiddenness" for row in rows
        ),
        "exploratory_candidates": sum(
            row["screen_label"] == "chaos_candidate_exploratory" for row in rows
        ),
        "numerical_contract": {
            "derivative": (
                "integer_ode"
                if rows and all(np.isclose(float(row["q"]), 1.0) for row in rows)
                else "Caputo"
            ),
            "integrator": (
                rows[0]["integrator"]
                if rows and len({str(row["integrator"]) for row in rows}) == 1
                else "mixed"
            ),
            "memory_mode": (
                "not_applicable"
                if rows and all(np.isclose(float(row["q"]), 1.0) for row in rows)
                else args.memory_mode
            ),
            "memory_policy": (
                "not_applicable"
                if rows and all(np.isclose(float(row["q"]), 1.0) for row in rows)
                else ("full_history" if args.memory_mode == "full" else "finite_window")
            ),
            "memory_window_steps": (
                None if args.memory_mode == "full" else args.memory_window_steps
            ),
            "caputo_history_accumulated": bool(
                rows
                and not all(np.isclose(float(row["q"]), 1.0) for row in rows)
                and args.memory_mode == "full"
            ),
            "h": args.h,
            "t_final": args.t_final,
            "t_burn": args.t_burn,
            "hiddenness_tested": False,
        },
        "seed_policy": (
            "uniform_random_per_case"
            if args.randomize_initial_condition
            else args.seed_strategy
        ),
        "seed": None if args.randomize_initial_condition else args.seed,
        "top_rows": ranked[: min(12, len(ranked))],
        "scientific_boundary": (
            "Finite-time screening only. Independent robustness and all-equilibria "
            "neighborhood tests are required before a hiddenness-compatible label."
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_jsonable) + chr(10),
        encoding="utf-8",
    )
    (out / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=_jsonable) + chr(10),
        encoding="utf-8",
    )
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--q-values", default="0.91,0.94,0.97,0.99")
    parser.add_argument("--alpha-values", default="8.2,8.3,8.4,8.5,8.6,8.7")
    parser.add_argument("--beta-values", default="11.5,11.7,11.9,12.1,12.3,12.5")
    parser.add_argument("--gamma", type=float, default=0.0052)
    parser.add_argument("--a1", type=float, default=0.4)
    parser.add_argument("--a2", type=float, default=-1.5585)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--a1-values", default=None)
    parser.add_argument("--a2-values", default=None)
    parser.add_argument("--rho-values", default=None)
    parser.add_argument("--random-cases", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260623)
    parser.add_argument("--q-range", default="0.90,0.999")
    parser.add_argument("--alpha-range", default="6.0,12.0")
    parser.add_argument("--beta-range", default="8.0,16.0")
    parser.add_argument("--gamma-range", default="0.001,0.08")
    parser.add_argument("--a1-range", default="-0.1,0.8")
    parser.add_argument("--a2-range", default="-3.5,-0.8")
    parser.add_argument("--rho-range", default="0.5,2.0")
    parser.add_argument("--seed", type=_seed, default=np.array([13.0, 0.7, -19.0]))
    parser.add_argument("--seed-strategy", choices=["fixed", "df_branches"], default="fixed")
    parser.add_argument("--df-nscan", type=int, default=5000)
    parser.add_argument("--randomize-initial-condition", action="store_true")
    parser.add_argument("--x0-range", default="-25,25")
    parser.add_argument("--y0-range", default="-8,8")
    parser.add_argument("--z0-range", default="-35,35")
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument(
        "--integrator",
        choices=["auto", "abm", "efork3", "heun", "rk4"],
        default="auto",
    )
    parser.add_argument("--memory-mode", choices=["full", "window"], default="full")
    parser.add_argument("--memory-window-steps", type=int, default=0)
    parser.add_argument("--t-final", type=float, default=80.0)
    parser.add_argument("--t-burn", type=float, default=40.0)
    parser.add_argument("--zero-one-samples", default="500,800,1200")
    parser.add_argument("--divergence-norm", type=float, default=120.0)
    parser.add_argument("--max-cases", type=int, default=0)
    return parser


def main() -> None:
    summary = run(make_parser().parse_args())
    run_dir = ROOT / str(summary["output_dir"])
    if (run_dir / "publication_figure_inputs.json").exists():
        from hidden_attractors.plotting.generate_publication_figures import (
            generate_all_publication_figures,
        )

        generate_all_publication_figures(str(run_dir), {})
    print(json.dumps(summary, indent=2, default=_jsonable))


if __name__ == "__main__":
    main()
