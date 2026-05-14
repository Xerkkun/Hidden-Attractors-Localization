#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

import chua_initial_cond as chua
from extended_search_utils import chua_ic_params, json_safe, write_csv
from lure_candidate_manifest import load_config


ROOT = Path(__file__).resolve().parent
TARGET_Q = 0.9998
OUTPUT_DIR = ROOT / "outputs" / "lure_biased_multiparam_q09998"

HIDDENNESS_LABELS = {
    "candidate_hidden_like",
    "compatible_with_hiddenness_under_tested_radii",
    "not_supported_by_Eplus_neighborhood_test",
    "not_supported_by_refined_neighborhood_test",
    "strongest_current_candidate_but_not_verified",
}

EVAL_FIELDS = [
    "candidate_id",
    "q",
    "A",
    "sigma0",
    "omega",
    "N_re",
    "N_im",
    "W_re",
    "W_im",
    "residual_re",
    "residual_im",
    "residual_abs",
    "rho_H",
    "rhoH_class",
    "harmonic_energy_ratio",
    "seed_x",
    "seed_y",
    "seed_z",
    "seed_reconstruction_status",
    "q_consistent",
    "source_hint",
    "notes",
    "evaluation_stage",
    "candidate_status",
]

SEED_FIELDS = [
    "candidate_id",
    "seed_id",
    "q",
    "A",
    "sigma0",
    "omega",
    "phi",
    "x0",
    "y0",
    "z0",
    "sigma_check",
    "sigma_error",
    "reconstruction_method",
    "valid_seed",
]


def q_tag(q: float) -> str:
    return f"{float(q):.5f}".replace("-", "m").replace(".", "p")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def require_q09998(cfg: Dict[str, Any], *, context: str) -> float:
    q = float(cfg.get("q", cfg.get("frac_order", float("nan"))))
    frac = float(cfg.get("frac_order", q))
    if not math.isfinite(q) or not math.isfinite(frac):
        raise ValueError(f"{context}: q and frac_order must be explicit.")
    if abs(q - TARGET_Q) > 5e-10 or abs(frac - TARGET_Q) > 5e-10:
        raise ValueError(f"{context}: q mismatch. Expected q=frac_order={TARGET_Q}, got q={q}, frac_order={frac}.")
    if bool(cfg.get("enforce_q_consistency", True)) is not True:
        raise ValueError(f"{context}: enforce_q_consistency must be true for this exploration.")
    return q


def ensure_outdir(cfg: Dict[str, Any]) -> Path:
    root = Path(cfg.get("outputs", {}).get("root", "outputs/lure_biased_multiparam_q09998"))
    if not root.is_absolute():
        root = ROOT / root
    expected = OUTPUT_DIR.resolve()
    if root.resolve() != expected:
        raise ValueError(f"Output dir must be {rel(expected)} for this exploration, got {rel(root)}.")
    forbidden = [
        ROOT / "outputs" / "lure_route",
        ROOT / "outputs" / "extended_search" / "corrida1",
        ROOT / "runs_machado_sweep_fast",
    ]
    for path in forbidden:
        try:
            root.resolve().relative_to(path.resolve())
        except ValueError:
            continue
        raise ValueError(f"Refusing to write inside protected historical path: {rel(path)}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def latin_hypercube(bounds: Sequence[Tuple[float, float]], n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    d = len(bounds)
    if n <= 0:
        return np.empty((0, d), dtype=float)
    u = (rng.random((n, d)) + np.arange(n).reshape(n, 1)) / n
    for j in range(d):
        rng.shuffle(u[:, j])
    out = np.empty_like(u)
    for j, (lo, hi) in enumerate(bounds):
        out[:, j] = float(lo) + u[:, j] * (float(hi) - float(lo))
    return out.astype(float)


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def source_hint_points(cfg: Dict[str, Any]) -> np.ndarray:
    search = cfg["search"]
    manifest = Path(search.get("source_hint_manifest", "outputs/lure_route/lure_candidates_manifest.csv"))
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    rows = read_csv_rows(manifest)
    points: List[Tuple[float, float, float, str]] = []
    for row in rows:
        q_old = finite_float(row.get("q"))
        if not math.isfinite(q_old) or abs(q_old - TARGET_Q) <= 5e-10:
            continue
        A_old = finite_float(row.get("A"))
        omega_old = finite_float(row.get("omega"))
        sigma_old = finite_float(row.get("sigma0"), 0.0)
        if not (math.isfinite(A_old) and math.isfinite(omega_old)):
            continue
        cid = row.get("candidate_id", "old_lure")
        for a_scale in (0.85, 1.0, 1.15):
            for w_scale in (0.92, 1.0, 1.08):
                for s_shift in (-1.0, 0.0, 1.0):
                    A = min(max(A_old * a_scale, float(search["A_min"])), float(search["A_max"]))
                    omega = min(max(omega_old * w_scale, float(search["omega_min"])), float(search["omega_max"]))
                    sigma0 = min(max(sigma_old + s_shift, float(search["sigma0_min"])), float(search["sigma0_max"]))
                    points.append((A, sigma0, omega, f"source_hint_only_q_mismatch:{cid}"))
    if not points:
        return np.empty((0, 4), dtype=object)
    out = np.empty((len(points), 4), dtype=object)
    for i, row in enumerate(points):
        out[i, :] = row
    return out


class Quadrature:
    def __init__(self, K: int, n_quad: int):
        self.K = int(K)
        self.n_quad = int(n_quad)
        if self.K < 2:
            raise ValueError("K_rhoH must be at least 2.")
        if self.n_quad < max(64, 8 * self.K):
            raise ValueError("quadrature_points is too small for K_rhoH.")
        theta = np.linspace(0.0, 2.0 * np.pi, self.n_quad, endpoint=False, dtype=float)
        self.theta = theta
        self.costheta = np.cos(theta)
        self.cos_table = np.vstack([np.cos(k * theta) for k in range(self.K + 1)])
        self.sin_table = np.vstack([np.sin(k * theta) for k in range(self.K + 1)])


def psi_vectorized(sigma: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    if chua.chua_model(p) == "arctan":
        return float(p["a2"]) * np.arctan(float(p["rho"]) * sigma)
    return float(chua.chua_gain_A(p)) * np.clip(sigma, -1.0, 1.0)


def fourier_y(A: float, sigma0: float, p: Dict[str, Any], quad: Quadrature) -> Tuple[np.ndarray, float]:
    sigma = float(sigma0) + float(A) * quad.costheta
    y = psi_vectorized(sigma, p)
    ak = (2.0 / quad.n_quad) * (quad.cos_table[1:, :] @ y)
    bk = (2.0 / quad.n_quad) * (quad.sin_table[1:, :] @ y)
    Y = ak - 1j * bk
    return Y.astype(np.complex128), float(np.mean(y))


def classify_rho(rho_H: float) -> str:
    if not math.isfinite(rho_H):
        return "rhoH_invalid"
    if rho_H < 0.15:
        return "rhoH_priority_lt_0p15"
    if rho_H < 0.30:
        return "rhoH_usable_lt_0p30"
    return "rhoH_high_reject"


def evaluate_point(
    *,
    candidate_id: str,
    A: float,
    sigma0: float,
    omega: float,
    q: float,
    p: Dict[str, Any],
    quad: Quadrature,
    source_hint: str,
    stage: str,
) -> Dict[str, Any]:
    notes: List[str] = []
    try:
        if A <= 1.0e-4 or omega <= 0.0:
            raise ValueError("A and omega must be positive.")
        Y, _y_mean = fourier_y(A, sigma0, p, quad)
        Y1 = complex(Y[0])
        N = Y1 / float(A)
        W1 = complex(chua.W_frac(omega, q, p))
        residual = 1.0 + W1 * N
        denom = abs(W1) * abs(Y1) + 1.0e-14
        higher = 0.0
        higher_energy = 0.0
        total_energy = abs(Y1) ** 2
        for idx in range(2, quad.K + 1):
            Yk = complex(Y[idx - 1])
            Wk = complex(chua.W_frac(idx * float(omega), q, p))
            higher += abs(Wk) * abs(Yk)
            higher_energy += abs(Yk) ** 2
            total_energy += abs(Yk) ** 2
        rho_H = float(higher / denom)
        harmonic_energy_ratio = float(higher_energy / max(total_energy, 1.0e-300))
        q_consistent = abs(float(q) - TARGET_Q) <= 5e-10
        return {
            "candidate_id": candidate_id,
            "q": float(q),
            "A": float(A),
            "sigma0": float(sigma0),
            "omega": float(omega),
            "N_re": float(np.real(N)),
            "N_im": float(np.imag(N)),
            "W_re": float(np.real(W1)),
            "W_im": float(np.imag(W1)),
            "residual_re": float(np.real(residual)),
            "residual_im": float(np.imag(residual)),
            "residual_abs": float(abs(residual)),
            "rho_H": rho_H,
            "rhoH_class": classify_rho(rho_H),
            "harmonic_energy_ratio": harmonic_energy_ratio,
            "seed_x": "",
            "seed_y": "",
            "seed_z": "",
            "seed_reconstruction_status": "not_reconstructed_for_raw_evaluation",
            "q_consistent": bool(q_consistent),
            "source_hint": source_hint,
            "notes": " ".join(notes),
            "evaluation_stage": stage,
            "candidate_status": "candidate_hidden_like",
        }
    except Exception as exc:
        return {
            "candidate_id": candidate_id,
            "q": float(q),
            "A": float(A),
            "sigma0": float(sigma0),
            "omega": float(omega),
            "N_re": "",
            "N_im": "",
            "W_re": "",
            "W_im": "",
            "residual_re": "",
            "residual_im": "",
            "residual_abs": float("nan"),
            "rho_H": float("nan"),
            "rhoH_class": "evaluation_failed",
            "harmonic_energy_ratio": float("nan"),
            "seed_x": "",
            "seed_y": "",
            "seed_z": "",
            "seed_reconstruction_status": f"evaluation_failed:{exc}",
            "q_consistent": abs(float(q) - TARGET_Q) <= 5e-10,
            "source_hint": source_hint,
            "notes": str(exc),
            "evaluation_stage": stage,
            "candidate_status": "candidate_hidden_like",
        }


def sample_stage0(cfg: Dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    search = cfg["search"]
    n_samples = int(args.n_samples if args.n_samples is not None else search.get("n_samples", 5000))
    bounds = [
        (float(search["A_min"]), float(search["A_max"])),
        (float(search["sigma0_min"]), float(search["sigma0_max"])),
        (float(search["omega_min"]), float(search["omega_max"])),
    ]
    lhs = latin_hypercube(bounds, n_samples, int(search.get("random_seed", 20260514)))
    hints = source_hint_points(cfg)
    if hints.size == 0:
        out = np.empty((lhs.shape[0], 4), dtype=object)
        out[:, :3] = lhs
        out[:, 3] = "latin_hypercube_q09998"
        return out
    out = np.empty((lhs.shape[0] + hints.shape[0], 4), dtype=object)
    out[: lhs.shape[0], :3] = lhs
    out[: lhs.shape[0], 3] = "latin_hypercube_q09998"
    out[lhs.shape[0] :, :] = hints
    return out


def local_refine(
    rows: Sequence[Dict[str, Any]],
    cfg: Dict[str, Any],
    q: float,
    p: Dict[str, Any],
    quad: Quadrature,
) -> List[Dict[str, Any]]:
    try:
        from scipy.optimize import minimize
    except Exception as exc:
        return [
            {
                **dict(row),
                "candidate_id": f"refine_skipped_{i:04d}",
                "evaluation_stage": "S1_refinement_skipped",
                "notes": f"scipy unavailable: {exc}",
            }
            for i, row in enumerate(rows)
        ]

    search = cfg["search"]
    bounds = [
        (float(search["A_min"]), float(search["A_max"])),
        (float(search["sigma0_min"]), float(search["sigma0_max"])),
        (float(search["omega_min"]), float(search["omega_max"])),
    ]
    top = [
        row
        for row in rows
        if math.isfinite(finite_float(row.get("residual_abs"))) and bool(row.get("q_consistent", False))
    ]
    top.sort(key=lambda r: (finite_float(r.get("residual_abs")), finite_float(r.get("rho_H"), 1.0e9)))
    top = top[: int(search.get("local_refine_top", 100))]
    refined: List[Dict[str, Any]] = []

    def objective(x: np.ndarray) -> float:
        A, sigma0, omega = [float(v) for v in x]
        for val, (lo, hi) in zip((A, sigma0, omega), bounds):
            if val < lo or val > hi:
                return 1.0e3 + sum(abs(val - np.clip(val, lo, hi)) for val, (lo, hi) in zip((A, sigma0, omega), bounds))
        row = evaluate_point(
            candidate_id="objective",
            A=A,
            sigma0=sigma0,
            omega=omega,
            q=q,
            p=p,
            quad=quad,
            source_hint="local_refine_objective",
            stage="S1_objective",
        )
        residual = finite_float(row.get("residual_abs"), 1.0e3)
        rho = finite_float(row.get("rho_H"), 1.0e3)
        return residual + 0.05 * max(0.0, rho - float(search.get("rhoH_priority", 0.15))) ** 2

    for idx, row in enumerate(top):
        x0 = np.array([finite_float(row["A"]), finite_float(row["sigma0"]), finite_float(row["omega"])], dtype=float)
        try:
            result = minimize(
                objective,
                x0,
                method="Powell",
                bounds=bounds,
                options={"maxiter": 80, "xtol": 1.0e-5, "ftol": 1.0e-5, "disp": False},
            )
            x = np.asarray(result.x if result.success else x0, dtype=float)
            source_hint = str(row.get("source_hint", ""))
            notes = f"S1 from {row.get('candidate_id', '')}; optimizer_success={bool(result.success)}; fun={float(result.fun):.6g}"
        except Exception as exc:
            x = x0
            source_hint = str(row.get("source_hint", ""))
            notes = f"S1 optimizer failed: {exc}"
        refined_row = evaluate_point(
            candidate_id=f"lure_biased_refine_tmp_{idx:04d}",
            A=float(x[0]),
            sigma0=float(x[1]),
            omega=float(x[2]),
            q=q,
            p=p,
            quad=quad,
            source_hint=source_hint,
            stage="S1_local_refine",
        )
        refined_row["notes"] = f"{refined_row.get('notes', '')} {notes}".strip()
        refined.append(refined_row)
    return refined


def candidate_filter(rows: Sequence[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    search = cfg["search"]
    residual_keep = float(search.get("residual_keep", 0.05))
    rho_keep = float(search.get("rhoH_keep", 0.30))
    candidates = []
    seen: set[Tuple[int, int, int]] = set()
    for row in rows:
        residual = finite_float(row.get("residual_abs"))
        rho = finite_float(row.get("rho_H"))
        A = finite_float(row.get("A"))
        omega = finite_float(row.get("omega"))
        if not (math.isfinite(residual) and math.isfinite(rho) and math.isfinite(A) and math.isfinite(omega)):
            continue
        if residual >= residual_keep or rho >= rho_keep or A <= 1.0e-4 or omega <= 0.0:
            continue
        key = (round(A, 8), round(finite_float(row.get("sigma0")), 8), round(omega, 8))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(dict(row))
    candidates.sort(key=lambda r: (finite_float(r.get("residual_abs")), finite_float(r.get("rho_H"))))
    for rank, row in enumerate(candidates, start=1):
        row["candidate_id"] = f"lure_biased_q_{q_tag(float(row['q']))}_rank_{rank:04d}"
        if rank == 1:
            row["candidate_status"] = "strongest_current_candidate_but_not_verified"
        else:
            row["candidate_status"] = "candidate_hidden_like"
        row["notes"] = (
            f"{row.get('notes', '')} first_harmonic_describing_function_seed_only; "
            "not_hidden_verified; validate_by_causal_caputo_integration"
        ).strip()
    return candidates


def reconstruct_seed_bank(candidates: Sequence[Dict[str, Any]], cfg: Dict[str, Any], p: Dict[str, Any]) -> List[Dict[str, Any]]:
    phases = [float(v) for v in cfg.get("seeds", {}).get("phases", [0.0])]
    q = float(cfg["q"])
    _P, _b, r = chua.chua_matrices(p)
    seed_rows: List[Dict[str, Any]] = []
    for cand in candidates:
        valid_count = 0
        first_seed: np.ndarray | None = None
        first_status = "not_attempted"
        for idx, phi in enumerate(phases):
            seed_id = f"{cand['candidate_id']}_phi_{idx:02d}"
            try:
                seed, _xbar, _V, _fourier = chua.reconstruct_biased_lure_seed(
                    q,
                    p,
                    float(cand["A"]),
                    float(cand["sigma0"]),
                    float(cand["omega"]),
                    theta=phi,
                    K=int(cfg["search"].get("K_rhoH", 20)),
                    n_quad=int(cfg["search"].get("quadrature_points", 4096)),
                )
                sigma_target = float(cand["sigma0"]) + float(cand["A"]) * math.cos(phi)
                r_vec = np.asarray(r, dtype=float)
                sigma_before = float(r_vec @ np.asarray(seed, dtype=float))
                r_norm2 = float(r_vec @ r_vec)
                if r_norm2 <= 1.0e-300:
                    raise RuntimeError("r vector has zero norm; cannot enforce sigma constraint.")
                seed = np.asarray(seed, dtype=float) + ((sigma_target - sigma_before) / r_norm2) * r_vec
                sigma_check = float(r_vec @ np.asarray(seed, dtype=float))
                sigma_error = sigma_check - sigma_target
                valid = bool(np.all(np.isfinite(seed)) and abs(sigma_error) <= 1.0e-6)
                if valid:
                    valid_count += 1
                    if first_seed is None:
                        first_seed = np.asarray(seed, dtype=float)
                        first_status = "ok"
                row = {
                    "candidate_id": cand["candidate_id"],
                    "seed_id": seed_id,
                    "q": q,
                    "A": float(cand["A"]),
                    "sigma0": float(cand["sigma0"]),
                    "omega": float(cand["omega"]),
                    "phi": phi,
                    "x0": float(seed[0]),
                    "y0": float(seed[1]),
                    "z0": float(seed[2]),
                    "sigma_check": sigma_check,
                    "sigma_error": sigma_error,
                    "reconstruction_method": "reconstruct_biased_lure_seed_plus_sigma_hyperplane_projection",
                    "valid_seed": valid,
                }
            except Exception as exc:
                row = {
                    "candidate_id": cand["candidate_id"],
                    "seed_id": seed_id,
                    "q": q,
                    "A": float(cand["A"]),
                    "sigma0": float(cand["sigma0"]),
                    "omega": float(cand["omega"]),
                    "phi": phi,
                    "x0": "",
                    "y0": "",
                    "z0": "",
                    "sigma_check": "",
                    "sigma_error": "",
                    "reconstruction_method": f"failed:{exc}",
                    "valid_seed": False,
                }
                if first_status == "not_attempted":
                    first_status = f"failed:{exc}"
            seed_rows.append(row)
        if first_seed is not None:
            cand["seed_x"], cand["seed_y"], cand["seed_z"] = [float(v) for v in first_seed]
        cand["seed_reconstruction_status"] = f"{first_status}; valid_phase_count={valid_count}"
    return seed_rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2), encoding="utf-8")


def ensure_empty_downstream_files(outdir: Path) -> None:
    write_csv(
        outdir / "continuation_summary.csv",
        [],
        [
            "candidate_id",
            "seed_id",
            "route_id",
            "q",
            "eta",
            "sigma0_current",
            "step_index",
            "h",
            "memory_length",
            "memory_points",
            "memory_carried",
            "bounded",
            "diverged",
            "equilibrium_hit",
            "final_class",
            "final_x",
            "final_y",
            "final_z",
            "range_x",
            "range_y",
            "range_z",
            "fft_peak",
            "psd_entropy",
            "notes",
        ],
    )
    write_csv(outdir / "continuation_paths.csv", [], ["candidate_id", "seed_id", "route_id", "step_index", "q", "eta", "sigma0_current", "x", "y", "z"])
    write_csv(outdir / "continuation_survivors.csv", [], ["candidate_id", "seed_id", "route_id", "q", "final_class", "memory_carried", "continuation_reliability", "final_x", "final_y", "final_z", "notes"])
    write_csv(outdir / "early_equilibrium_filter_raw.csv", [], ["candidate_id", "seed_id", "equilibrium_id", "direction_label", "rho", "q", "h", "memory_length", "t_final", "final_class", "target_hit", "hiddenness_status", "notes"])
    write_csv(outdir / "early_equilibrium_filter_summary.csv", [], ["candidate_id", "n_Eplus_TARGET", "n_E0_TARGET", "n_Eminus_TARGET", "hiddenness_status", "notes"])
    write_csv(outdir / "robustness_survivors.csv", [], ["candidate_id", "case_id", "q", "h", "memory_length", "t_final", "bounded", "final_class", "range_x", "range_y", "range_z", "mean_tail", "var_tail", "fft_peak", "psd_entropy", "robust_attractor", "notes"])
    write_csv(outdir / "lure_biased_vs_machado_comparison.csv", [], ["candidate_id", "machado_candidate_id", "q_lure", "q_machado", "likely_same_attractor_as_machado", "distinct_candidate", "notes"])


def load_q_audit_summary() -> Dict[str, Any]:
    path = ROOT / "outputs" / "q_audit" / "q_audit_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_report(
    outdir: Path,
    cfg: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    all_rows: Sequence[Dict[str, Any]],
    seed_rows: Sequence[Dict[str, Any]],
    files_written: Sequence[str],
    run_metadata: Dict[str, Any],
) -> None:
    q_audit = load_q_audit_summary()
    best_by_residual = sorted(candidates, key=lambda r: finite_float(r.get("residual_abs")))[:10]
    best_by_rho = sorted(candidates, key=lambda r: finite_float(r.get("rho_H")))[:10]
    lines = [
        "# Biased Lure multiparameter exploration q=0.9998",
        "",
        "This report is conservative: no `hidden_verified` status is declared.",
        "",
        "The describing-function calculation is a first-harmonic heuristic used to generate seeds. "
        "It is not treated as a proof of exact Caputo limit cycles; validation must use causal Caputo integration with memory.",
        "",
        "## q consistency",
        "",
        f"- q_global: `{float(cfg['q']):.5f}`",
        f"- frac_order: `{float(cfg['frac_order']):.5f}`",
        f"- enforce_q_consistency: `{bool(cfg.get('enforce_q_consistency', True))}`",
        f"- q_consistency_status: `ok_q_0p9998`",
        "",
        "## q audit summary",
        "",
    ]
    if q_audit:
        for key, value in q_audit.get("classification_counts", {}).items():
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- q audit has not been run yet. Run `python audit_and_homogenize_q.py`.")
    lines.extend(
        [
            "",
            "## Execution scope",
            "",
            f"- configured_S0_n_samples: `{run_metadata.get('configured_n_samples')}`",
            f"- executed_S0_lhs_samples: `{run_metadata.get('executed_lhs_samples')}`",
            f"- source_hint_samples: `{run_metadata.get('source_hint_samples')}`",
            f"- local_refine_top: `{run_metadata.get('local_refine_top')}`",
            f"- execution_scope: `{run_metadata.get('execution_scope')}`",
        ]
    )
    lines.extend(["", "## Best candidates by residual", ""])
    if best_by_residual:
        lines.append("| candidate_id | A | sigma0 | omega | residual_abs | rho_H | status |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in best_by_residual:
            lines.append(
                f"| `{row['candidate_id']}` | {float(row['A']):.6g} | {float(row['sigma0']):.6g} | "
                f"{float(row['omega']):.6g} | {float(row['residual_abs']):.6g} | {float(row['rho_H']):.6g} | "
                f"{row.get('candidate_status', 'candidate_hidden_like')} |"
            )
    else:
        lines.append("No candidates passed the residual/rho_H filters.")
    lines.extend(["", "## Best candidates by rho_H", ""])
    if best_by_rho:
        lines.append("| candidate_id | residual_abs | rho_H | rhoH_class |")
        lines.append("|---|---:|---:|---|")
        for row in best_by_rho:
            lines.append(f"| `{row['candidate_id']}` | {float(row['residual_abs']):.6g} | {float(row['rho_H']):.6g} | {row['rhoH_class']} |")
    else:
        lines.append("No candidates passed the residual/rho_H filters.")
    lines.extend(
        [
            "",
            "## Continuation survivors",
            "",
            "Not executed by this search stage unless continuation is enabled with explicit simulation flags.",
            "",
            "## Candidates discarded by E+",
            "",
            "Not executed by this search stage. The continuation module writes the E+ filter outputs.",
            "",
            "## Robust survivors",
            "",
            "Not executed by this search stage. Robustness is reserved for candidates that pass the early E+ filter.",
            "",
            "## Machado comparison",
            "",
            "Not executed by this search stage. The continuation module compares survivors with `branch_0_mu_4p00000_theta_0p00000` if artifacts exist.",
            "",
            "## Warnings",
            "",
            "- Do not reuse q=0.99 or q=0.999 files as numerical inputs for this exploration.",
            "- Historical q-mismatched Lure results can only act as `source_hint_only_q_mismatch` zones.",
            "- `candidate_hidden_like` only means the seed passed the first-harmonic residual and rho_H filters.",
            "- Hiddenness remains unverified until equilibrium-neighborhood tests pass under the reported radii, memory and discretization.",
        ]
    )
    (outdir / "lure_biased_multiparam_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "q_global": float(cfg["q"]),
        "q_consistency_status": "ok_q_0p9998",
        "n_candidates_raw": len(all_rows),
        "n_candidates_filtered": len(candidates),
        "n_seeds": len(seed_rows),
        "n_continuation_survivors": 0,
        "n_passed_Eplus_filter": 0,
        "n_robust_survivors": 0,
        "best_candidate_id": candidates[0]["candidate_id"] if candidates else "",
        "best_candidate_status": candidates[0].get("candidate_status", "") if candidates else "",
        "hidden_verified": False,
        "run_metadata": run_metadata,
        "allowed_conservative_labels": sorted(HIDDENNESS_LABELS),
        "files_written": list(files_written),
        "q_audit_summary": q_audit,
    }
    write_json(outdir / "lure_biased_multiparam_summary.json", summary)


def run_search(config_path: str | Path, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(config_path)
    q = require_q09998(cfg, context=str(config_path))
    outdir = ensure_outdir(cfg)
    if args.resume and (outdir / "biased_lure_candidates.csv").exists() and (outdir / "biased_lure_seed_bank.csv").exists():
        candidates = read_csv_rows(outdir / "biased_lure_candidates.csv")
        seed_rows = read_csv_rows(outdir / "biased_lure_seed_bank.csv")
        all_rows = read_csv_rows(outdir / "biased_lure_all_evaluations.csv")
        return {
            "cfg": cfg,
            "outdir": outdir,
            "candidates": candidates,
            "seed_rows": seed_rows,
            "all_rows": all_rows,
            "files_written": [rel(outdir / "biased_lure_candidates.csv"), rel(outdir / "biased_lure_seed_bank.csv")],
        }

    p = chua_ic_params(cfg)
    quad = Quadrature(int(cfg["search"].get("K_rhoH", 20)), int(cfg["search"].get("quadrature_points", 4096)))
    points = sample_stage0(cfg, args)
    configured_n_samples = int(cfg["search"].get("n_samples", 5000))
    executed_lhs_samples = int(args.n_samples if args.n_samples is not None else configured_n_samples)
    source_hint_samples = int(sum(1 for item in points if str(item[3]).startswith("source_hint_only_q_mismatch")))
    run_metadata = {
        "configured_n_samples": configured_n_samples,
        "executed_lhs_samples": executed_lhs_samples,
        "source_hint_samples": source_hint_samples,
        "total_stage0_points": int(len(points)),
        "local_refine_top": int(cfg["search"].get("local_refine_top", 100)),
        "quadrature_points": int(cfg["search"].get("quadrature_points", 4096)),
        "K_rhoH": int(cfg["search"].get("K_rhoH", 20)),
        "execution_scope": "smoke_override" if args.n_samples is not None and int(args.n_samples) != configured_n_samples else "configured_full_search",
    }
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(points):
        A, sigma0, omega = float(item[0]), float(item[1]), float(item[2])
        source_hint = str(item[3])
        rows.append(
            evaluate_point(
                candidate_id=f"lure_biased_s0_{idx:06d}",
                A=A,
                sigma0=sigma0,
                omega=omega,
                q=q,
                p=p,
                quad=quad,
                source_hint=source_hint,
                stage="S0_latin_hypercube_or_hint",
            )
        )
        if (idx + 1) % 1000 == 0:
            print(f"S0 evaluated {idx + 1}/{len(points)}", flush=True)
    refined = local_refine(rows, cfg, q, p, quad)
    all_rows = rows + refined
    candidates = candidate_filter(all_rows, cfg)
    seed_rows = reconstruct_seed_bank(candidates, cfg, p)

    files_written: List[str] = []
    write_csv(outdir / "biased_lure_all_evaluations.csv", all_rows, EVAL_FIELDS)
    files_written.append(rel(outdir / "biased_lure_all_evaluations.csv"))
    write_csv(outdir / "biased_lure_candidates.csv", candidates, EVAL_FIELDS)
    files_written.append(rel(outdir / "biased_lure_candidates.csv"))
    write_json(outdir / "biased_lure_candidates.json", {"q": q, "records": candidates})
    files_written.append(rel(outdir / "biased_lure_candidates.json"))
    write_csv(outdir / "biased_lure_seed_bank.csv", seed_rows, SEED_FIELDS)
    files_written.append(rel(outdir / "biased_lure_seed_bank.csv"))
    write_json(outdir / "search_run_metadata.json", run_metadata)
    files_written.append(rel(outdir / "search_run_metadata.json"))
    ensure_empty_downstream_files(outdir)
    files_written.extend(
        [
            rel(outdir / "continuation_summary.csv"),
            rel(outdir / "continuation_paths.csv"),
            rel(outdir / "continuation_survivors.csv"),
            rel(outdir / "early_equilibrium_filter_raw.csv"),
            rel(outdir / "early_equilibrium_filter_summary.csv"),
            rel(outdir / "robustness_survivors.csv"),
            rel(outdir / "lure_biased_vs_machado_comparison.csv"),
        ]
    )
    write_report(outdir, cfg, candidates, all_rows, seed_rows, files_written, run_metadata)
    files_written.extend([rel(outdir / "lure_biased_multiparam_report.md"), rel(outdir / "lure_biased_multiparam_summary.json")])
    elapsed = time.time() - t0
    print(f"biased_lure_search_elapsed_sec={elapsed:.3f}", flush=True)
    return {
        "cfg": cfg,
        "outdir": outdir,
        "candidates": candidates,
        "seed_rows": seed_rows,
        "all_rows": all_rows,
        "files_written": files_written,
    }


def print_console_summary(result: Dict[str, Any]) -> None:
    candidates = result["candidates"]
    seed_rows = result["seed_rows"]
    all_rows = result["all_rows"]
    files_written = result["files_written"]
    print(f"q_global={TARGET_Q:.5f}")
    print("q_consistency_status=ok_q_0p9998")
    print(f"n_candidates_raw={len(all_rows)}")
    print(f"n_candidates_filtered={len(candidates)}")
    print(f"n_seeds={len(seed_rows)}")
    print("n_continuation_survivors=0")
    print("n_passed_Eplus_filter=0")
    print("n_robust_survivors=0")
    print(f"best_candidate_id={candidates[0]['candidate_id'] if candidates else ''}")
    print(f"best_candidate_status={candidates[0].get('candidate_status', '') if candidates else ''}")
    print("files_written=" + ";".join(files_written))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Biased Lure multiparameter search fixed at q=0.9998.")
    parser.add_argument("--config", required=True, help="Path to configs/lure_biased_multiparam_q09998.yaml")
    parser.add_argument("--resume", action="store_true", help="Reuse existing search and seed-bank outputs if present.")
    parser.add_argument("--force", action="store_true", help="Forwarded to long simulation stages when requested.")
    parser.add_argument("--run-continuation", action="store_true", help="Run the long continuation/filter/robustness module after search.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override S0 sample count for smoke runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_search(args.config, args)
    cfg = result["cfg"]
    if args.run_continuation or bool(cfg.get("continuation", {}).get("enabled", False)):
        from lure_biased_multiparam_continuation import run_continuation_pipeline

        cont_result = run_continuation_pipeline(args.config, resume=args.resume, force=args.force)
        result["files_written"].extend(cont_result.get("files_written", []))
    print_console_summary(result)


if __name__ == "__main__":
    main()
