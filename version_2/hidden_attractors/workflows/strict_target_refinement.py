"""Strict target-reference refinement for basin and equilibrium-ball rows.

This workflow re-integrates selected initial conditions and accepts a
``target_positive`` or ``target_negative`` label only when finite-time
trajectory geometry is close to the selected target reference, separated from
the symmetric target counterpart, and separated from deterministic
equilibrium-neighborhood controls.

Mathematical warning:
    The output is a stricter numerical classifier under a recorded contract. It
    does not prove hiddenness. A target hit from an equilibrium neighborhood is
    evidence against hiddenness under the tested contract.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis.trajectory import cloud_median_distance, trajectory_metrics
from ..basins import CLASS_LABELS, class_label
from ..io import append_csv, read_csv_rows, read_json, write_csv, write_json
from ..native.backends import BasinBackend, FractionalChuaBackend
from ..parallel import force_single_openmp_thread_current_process, force_single_openmp_thread_env
from ..paths import PROJECT_ROOT
from ..reproducibility import (
    collect_lure_metadata,
    collect_run_metadata,
    collect_seed_metadata,
    write_run_metadata,
)
from ..systems import get_system
from .protocol import sample_uniform_ball


LEGACY_ROOT = PROJECT_ROOT / "tools" / "legacy"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from danca2017_chua_abm_replication import DancaChuaConfig, caputo_abm_integrate, chua_rhs_factory  # noqa: E402


ROOT_OUTPUTS = PROJECT_ROOT.parent / "outputs"
DEFAULT_PROJECT_CANDIDATE = "lure_biased_q_0p99980_rank_0001"
DEFAULT_REFINED_LABELS = {
    **CLASS_LABELS,
    6: "bounded_other",
    7: "ambiguous_target_pair",
    8: "control_like",
}
STRICT_FIELDS = [
    "case_index",
    "mode",
    "candidate_id",
    "equilibrium_id",
    "radius",
    "sample_id",
    "ix",
    "iy",
    "x0",
    "y0",
    "z0",
    "old_class_label",
    "class_id",
    "class_label",
    "refined_status",
    "best_reference",
    "best_score",
    "score_positive",
    "score_negative",
    "reference_margin",
    "best_control",
    "best_control_score",
    "control_margin",
    "cloud_norm_positive",
    "cloud_norm_negative",
    "section_norm_positive",
    "section_norm_negative",
    "range_rel_positive",
    "range_rel_negative",
    "fft_rel_positive",
    "fft_rel_negative",
    "bounded",
    "diverged",
    "equilibrium_like",
    "noncollapsed_variance",
    "range_x",
    "range_y",
    "range_z",
    "var_x_tail",
    "var_y_tail",
    "var_z_tail",
    "fft_peak",
    "section_points",
    "elapsed_sec",
]


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _tokens(value: Any) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "off", "none"}


def _seed_from_row(row: dict[str, Any]) -> np.ndarray:
    if all(str(row.get(k, "")) != "" for k in ("x0", "y0", "z0")):
        return np.array([_float(row["x0"]), _float(row["y0"]), _float(row["z0"])], dtype=float)
    raw = str(row.get("x0", ""))
    sep = ";" if ";" in raw else ","
    vals = [_float(part) for part in raw.split(sep) if part.strip()]
    if len(vals) != 3:
        raise ValueError(f"cannot parse x0 from row: {raw!r}")
    return np.asarray(vals, dtype=float)


def _analysis_start(contract: dict[str, Any], analysis: dict[str, Any]) -> float:
    return max(float(contract.get("t_burn", 0.0)), float(analysis["tail_fraction_start"]) * float(contract["t_final"]))


def _score_against(payload: dict[str, Any], ref: dict[str, Any], weights: dict[str, float]) -> dict[str, float]:
    ref_range = np.asarray(ref["range_vec"], dtype=float)
    denom = max(float(np.linalg.norm(ref_range)), 1.0e-12)
    cloud = cloud_median_distance(payload["tail_sample"], np.asarray(ref["tail_sample"], dtype=float))
    cloud_norm = cloud / denom if math.isfinite(cloud) else float("inf")
    range_rel = float(np.linalg.norm(np.asarray(payload["range_vec"], dtype=float) - ref_range) / denom)
    ref_fft = float(ref.get("fft_peak", float("nan")))
    fft = float(payload.get("fft_peak", float("nan")))
    fft_rel = float(abs(fft - ref_fft) / max(abs(ref_fft), 1.0e-12)) if math.isfinite(ref_fft) and math.isfinite(fft) else float("nan")
    section = cloud_median_distance(payload["section"], np.asarray(ref["section"], dtype=float))
    section_norm = section / denom if math.isfinite(section) else float("nan")
    score = float(weights["cloud"] * cloud_norm + weights["range"] * range_rel)
    if math.isfinite(fft_rel):
        score += float(weights["fft"] * fft_rel)
    if math.isfinite(section_norm):
        score += float(weights["section"] * section_norm)
    return {
        "score": score,
        "cloud_norm": cloud_norm,
        "range_rel": range_rel,
        "fft_rel": fft_rel,
        "section_norm": section_norm,
    }


def _metric_ok(value: float, threshold: float, *, missing_ok: bool = False) -> bool:
    if not math.isfinite(value):
        return bool(missing_ok)
    return float(value) <= float(threshold)


def _classify_strict(
    traj: np.ndarray,
    cfg: dict[str, Any],
    refs: dict[str, dict[str, Any]],
    controls: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    contract = cfg["contract"]
    analysis = cfg["analysis"]
    metrics, payload = trajectory_metrics(
        traj,
        h=float(contract["sample_h"]),
        t_start=_analysis_start(contract, analysis),
        divergence_norm=float(analysis["divergence_norm"]),
        equilibrium_tol=float(analysis["equilibrium_tol"]),
        max_section_points=int(analysis["max_section_points"]),
        max_cloud_points=int(analysis["max_cloud_points"]),
    )
    if metrics["diverged"]:
        return {**metrics, "class_id": 3, "class_label": "infinity", "refined_status": "diverged", "best_reference": "", "best_score": float("nan")}
    if metrics["equilibrium_like"]:
        return {**metrics, "class_id": 0, "class_label": "equilibrium", "refined_status": "equilibrium_like", "best_reference": "", "best_score": float("nan")}
    if not metrics["bounded"]:
        return {**metrics, "class_id": 4, "class_label": "unknown", "refined_status": "not_bounded_not_diverged", "best_reference": "", "best_score": float("nan")}
    if not metrics["noncollapsed_variance"]:
        return {**metrics, "class_id": 4, "class_label": "unknown", "refined_status": "collapsed_or_low_variance", "best_reference": "", "best_score": float("nan")}

    weights = {k: float(v) for k, v in analysis["score_weights"].items()}
    pos = _score_against(payload, refs["positive"], weights)
    neg = _score_against(payload, refs["negative"], weights)
    if pos["score"] <= neg["score"]:
        best_label, best, other = "positive", pos, neg
    else:
        best_label, best, other = "negative", neg, pos
    reference_margin = float(other["score"] - best["score"])
    close_to_target = bool(
        _metric_ok(best["score"], float(analysis["max_score"]))
        and _metric_ok(best["cloud_norm"], float(analysis["max_cloud_norm"]))
        and _metric_ok(best["range_rel"], float(analysis["max_range_rel"]))
        and _metric_ok(best["fft_rel"], float(analysis["max_fft_rel"]), missing_ok=True)
        and _metric_ok(best["section_norm"], float(analysis["max_section_norm"]), missing_ok=True)
    )
    margin_ok = reference_margin >= float(analysis["min_reference_margin"])

    best_control_label = ""
    best_control_score = float("nan")
    control_margin = float("inf")
    control_ok = True
    if controls:
        control_scores = {label: _score_against(payload, control, weights)["score"] for label, control in controls.items()}
        finite = {label: score for label, score in control_scores.items() if math.isfinite(score)}
        if finite:
            best_control_label, best_control_score = min(finite.items(), key=lambda item: item[1])
            control_margin = float(best_control_score - best["score"])
            control_ok = control_margin >= float(analysis["min_control_margin"])

    class_id = 1 if best_label == "positive" else 2
    if not close_to_target:
        class_id = 6
        status = "bounded_noncollapsed_reference_mismatch"
    elif not margin_ok:
        class_id = 7
        status = "ambiguous_target_pair"
    elif not control_ok:
        class_id = 8
        status = "target_not_separated_from_equilibrium_controls"
    else:
        status = "matched_target_reference"
    return {
        **metrics,
        "class_id": class_id,
        "class_label": DEFAULT_REFINED_LABELS[class_id],
        "refined_status": status,
        "best_reference": best_label,
        "best_score": best["score"],
        "score_positive": pos["score"],
        "score_negative": neg["score"],
        "reference_margin": reference_margin,
        "best_control": best_control_label,
        "best_control_score": best_control_score,
        "control_margin": control_margin,
        "cloud_norm_positive": pos["cloud_norm"],
        "cloud_norm_negative": neg["cloud_norm"],
        "section_norm_positive": pos["section_norm"],
        "section_norm_negative": neg["section_norm"],
        "range_rel_positive": pos["range_rel"],
        "range_rel_negative": neg["range_rel"],
        "fft_rel_positive": pos["fft_rel"],
        "fft_rel_negative": neg["fft_rel"],
    }


def _project_integrator(cfg: dict[str, Any]) -> tuple[Any, Any]:
    eq_backend = BasinBackend.build(output_name=f"strict_refine_project_eq_{os.getpid()}")
    backend = FractionalChuaBackend.build(output_name=f"strict_refine_project_{os.getpid()}")
    contract = cfg["contract"]

    def integrate(seed: np.ndarray) -> np.ndarray:
        return backend.integrate_efork3(
            seed,
            q=float(contract["q"]),
            h=float(contract["h"]),
            Lm=float(contract["memory_length"]),
            t_final=float(contract["t_final"]),
        )

    return eq_backend, integrate


def _danca_integrator(cfg: dict[str, Any]) -> tuple[Any, Any]:
    contract = cfg["contract"]
    dcfg = DancaChuaConfig(
        q=float(contract["q"]),
        h=float(contract["h"]),
        t_final=float(contract["t_final"]),
        transient=float(contract["t_burn"]),
        equilibrium_tol=float(cfg["analysis"]["equilibrium_tol"]),
        divergence_norm=float(cfg["analysis"]["divergence_norm"]),
        store_stride=int(contract.get("store_stride", 1)),
    )
    rhs = chua_rhs_factory(dcfg)

    def integrate(seed: np.ndarray) -> np.ndarray:
        traj, _meta = caputo_abm_integrate(
            rhs,
            np.asarray(seed, dtype=float),
            q=dcfg.q,
            h=dcfg.h,
            t_final=dcfg.t_final,
            divergence_norm=dcfg.divergence_norm,
            store_stride=dcfg.store_stride,
        )
        return traj

    return BasinBackend.build(output_name=f"strict_refine_danca_eq_{os.getpid()}"), integrate


def _reference_payloads(cfg: dict[str, Any], integrate: Any) -> dict[str, dict[str, Any]]:
    analysis = cfg["analysis"]
    contract = cfg["contract"]
    pos_seed = np.asarray(cfg["reference"]["positive_seed"], dtype=float)
    refs: dict[str, dict[str, Any]] = {}
    for label, seed in {"positive": pos_seed, "negative": -pos_seed}.items():
        traj = integrate(seed)
        _metrics, payload = trajectory_metrics(
            traj,
            h=float(contract["sample_h"]),
            t_start=_analysis_start(contract, analysis),
            divergence_norm=float(analysis["divergence_norm"]),
            equilibrium_tol=float(analysis["equilibrium_tol"]),
            max_section_points=int(analysis["max_section_points"]),
            max_cloud_points=int(analysis["max_cloud_points"]),
        )
        refs[label] = payload
    return refs


def _control_payloads(cfg: dict[str, Any], eq_backend: Any, integrate: Any) -> dict[str, dict[str, Any]]:
    analysis = cfg["analysis"]
    control_cfg = analysis.get("negative_controls", {})
    if not _truthy(control_cfg.get("enabled", True)):
        return {}
    radius = float(control_cfg.get("radius", 1.0e-4))
    eq_names = _tokens(control_cfg.get("equilibria", "E0,E+,E-"))
    if radius <= 0.0 or not eq_names:
        return {}
    equilibria = eq_backend.equilibria()
    controls: dict[str, dict[str, Any]] = {}
    rng = np.random.default_rng(int(control_cfg.get("random_seed", 20260524)))
    count = int(control_cfg.get("samples_per_equilibrium", 12))
    for eq_name in eq_names:
        if eq_name not in equilibria:
            continue
        center = np.asarray(equilibria[eq_name], dtype=float)
        for sample_id, point in enumerate(sample_uniform_ball(center, radius, count, rng)):
            traj = integrate(point)
            _metrics, payload = trajectory_metrics(
                traj,
                h=float(cfg["contract"]["sample_h"]),
                t_start=_analysis_start(cfg["contract"], analysis),
                divergence_norm=float(analysis["divergence_norm"]),
                equilibrium_tol=float(analysis["equilibrium_tol"]),
                max_section_points=int(analysis["max_section_points"]),
                max_cloud_points=int(analysis["max_cloud_points"]),
            )
            controls[f"{eq_name}:ball_{sample_id:04d}"] = payload
    return controls


def _load_project_candidate_from_sphere(source_cfg: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for cand in source_cfg.get("candidates", []):
        if str(cand.get("candidate_id")) == str(candidate_id):
            return cand
    raise ValueError(f"candidate_id {candidate_id!r} not found in sphere config")


def _contract_and_reference(mode: str, source_dir: Path, args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray, str]:
    if mode == "basin-project":
        src = read_json(source_dir / "positive_x_basin_config.json")
        project = src["project"]
        contract = {
            "backend": "project_efork",
            "q": float(project["q"]),
            "h": float(project["h"]),
            "sample_h": float(project["h"]),
            "memory_length": float(project["Lm"]),
            "t_final": float(args.t_final) if float(args.t_final) > 0 else float(project["t_final"]),
            "t_burn": float(args.t_burn) if float(args.t_burn) >= 0 else float(project["t_burn"]),
        }
        return contract, np.asarray(project["seed"], dtype=float), str(project.get("candidate_id", DEFAULT_PROJECT_CANDIDATE))
    if mode == "basin-danca":
        src = read_json(source_dir / "positive_x_basin_config.json")
        danca = src["danca"]
        h = float(danca["h"])
        stride = int(danca.get("store_stride", 1))
        contract = {
            "backend": "danca_abm",
            "q": float(danca["q"]),
            "h": h,
            "sample_h": h * stride,
            "memory_length": "full_caputo_history",
            "t_final": float(args.t_final) if float(args.t_final) > 0 else float(danca["t_final"]),
            "t_burn": float(args.t_burn) if float(args.t_burn) >= 0 else float(danca["transient"]),
            "store_stride": stride,
        }
        return contract, np.asarray(danca["seed"], dtype=float), "danca2017_reference"
    if mode == "sphere-project":
        src = read_json(source_dir / "top3_sphere_robustness_config.json")
        sphere = src["sphere_contract"]
        cand = _load_project_candidate_from_sphere(src, args.candidate_id)
        h = float(sphere["h"])
        contract = {
            "backend": "project_efork",
            "q": float(sphere["q"]),
            "h": h,
            "sample_h": h,
            "memory_length": float(sphere["memory_length"]),
            "t_final": float(args.t_final) if float(args.t_final) > 0 else float(sphere["t_final"]),
            "t_burn": float(args.t_burn) if float(args.t_burn) >= 0 else float(sphere["t_burn"]),
        }
        return contract, np.asarray(cand.get("seed", cand.get("robust_start")), dtype=float), str(cand["candidate_id"])
    if mode == "sphere-danca":
        run_cfg = read_json(source_dir / "run_config.json")
        ref = read_json(source_dir / "danca_reference_summary.json")
        h = float(run_cfg["h"])
        stride = int(run_cfg.get("store_stride", 1))
        contract = {
            "backend": "danca_abm",
            "q": float(run_cfg["q"]),
            "h": h,
            "sample_h": h * stride,
            "memory_length": "full_caputo_history",
            "t_final": float(args.t_final) if float(args.t_final) > 0 else float(run_cfg["t_final"]),
            "t_burn": float(args.t_burn) if float(args.t_burn) >= 0 else float(run_cfg["transient"]),
            "store_stride": stride,
        }
        return contract, np.asarray(ref["best_seed"]["x0"], dtype=float), "danca2017_reference"
    raise ValueError(f"unsupported mode {mode!r}")


def _default_source_csv(mode: str, source_dir: Path) -> Path:
    names = {
        "basin-project": "project_best_basin_grid.csv",
        "basin-danca": "danca_basin_grid_raw.csv",
        "sphere-project": "sphere_raw.csv",
        "sphere-danca": "danca_hiddenness_raw.csv",
    }
    return source_dir / names[mode]


def _source_label(row: dict[str, Any], mode: str) -> str:
    if mode == "sphere-danca":
        return str(row.get("class_label", row.get("class", "")))
    return str(row.get("class_label", ""))


def _normalize_plan_row(row: dict[str, Any], mode: str, candidate_id: str, case_index: int) -> dict[str, Any]:
    seed = _seed_from_row(row)
    eq = row.get("equilibrium_id", row.get("eq_id", ""))
    radius = row.get("radius", row.get("delta", ""))
    sample_id = row.get("sample_id", "")
    if sample_id == "" and row.get("case_id"):
        sample_id = str(row["case_id"]).split("_")[-1]
    return {
        "case_index": row.get("case_index", case_index),
        "mode": mode,
        "candidate_id": row.get("candidate_id", candidate_id),
        "equilibrium_id": eq,
        "radius": radius,
        "sample_id": sample_id,
        "ix": row.get("ix", ""),
        "iy": row.get("iy", ""),
        "x0": float(seed[0]),
        "y0": float(seed[1]),
        "z0": float(seed[2]),
        "old_class_label": _source_label(row, mode),
    }


def make_config(outdir: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_dir).resolve()
    mode = str(args.mode)
    contract, positive_seed, candidate_id = _contract_and_reference(mode, source_dir, args)
    source_csv = Path(args.source_csv).resolve() if args.source_csv else _default_source_csv(mode, source_dir)
    source_rows = read_csv_rows(source_csv)
    allowed = set(_tokens(args.source_labels))
    eq_filter = set(_tokens(args.equilibrium_filter))
    selected: list[dict[str, Any]] = []
    for idx, row in enumerate(source_rows):
        if mode == "sphere-project" and str(row.get("candidate_id")) != str(candidate_id):
            continue
        if allowed and _source_label(row, mode) not in allowed:
            continue
        eq = str(row.get("equilibrium_id", row.get("eq_id", "")))
        if eq_filter and eq not in eq_filter:
            continue
        selected.append(_normalize_plan_row(row, mode, candidate_id, idx))
    if int(args.max_rows) > 0:
        selected = selected[: int(args.max_rows)]
    write_csv(root / "strict_refinement_plan.csv", selected)
    cfg = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "strict_target_reference_refinement",
        "mode": mode,
        "source_dir": str(source_dir),
        "source_csv": str(source_csv),
        "source_labels": sorted(allowed),
        "equilibrium_filter": sorted(eq_filter),
        "candidate_id": candidate_id,
        "reference": {
            "positive_seed": positive_seed.tolist(),
            "negative_seed_policy": "symmetry: negative_seed = -positive_seed",
        },
        "contract": contract,
        "analysis": {
            "tail_fraction_start": float(args.tail_fraction_start),
            "divergence_norm": float(args.divergence_norm),
            "equilibrium_tol": float(args.equilibrium_tol),
            "max_cloud_points": int(args.max_cloud_points),
            "max_section_points": int(args.max_section_points),
            "max_score": float(args.max_score),
            "max_cloud_norm": float(args.max_cloud_norm),
            "max_range_rel": float(args.max_range_rel),
            "max_fft_rel": float(args.max_fft_rel),
            "max_section_norm": float(args.max_section_norm),
            "min_reference_margin": float(args.min_reference_margin),
            "min_control_margin": float(args.min_control_margin),
            "negative_controls": {
                "enabled": not bool(args.disable_negative_controls),
                "radius": float(args.negative_control_radius),
                "equilibria": str(args.negative_control_equilibria),
                "sampling_mode": "ball",
                "samples_per_equilibrium": 12,
                "random_seed": 20260524,
            },
            "score_weights": {
                "cloud": float(args.weight_cloud),
                "range": float(args.weight_range),
                "fft": float(args.weight_fft),
                "section": float(args.weight_section),
            },
        },
        "chunks": int(args.chunks),
        "planned_rows": len(selected),
    }
    full_history = contract["memory_length"] == "full_caputo_history"
    memory_time = None if full_history else float(contract["memory_length"])
    lure_system = get_system("chua-nonsmooth")
    run_metadata = collect_run_metadata(
        run_id=root.name,
        workflow="strict_target_refinement",
        system="fractional_nonsmooth_chua",
        q=float(contract["q"]),
        h=float(contract["h"]),
        t_final=float(contract["t_final"]),
        t_burn=float(contract["t_burn"]),
        memory_mode="full" if full_history else "finite_window",
        M=None if full_history else int(round(float(memory_time) / float(contract["h"]))),
        memory_window_steps=None if full_history else int(round(float(memory_time) / float(contract["h"]))),
        memory_window_time=memory_time,
        is_full_caputo=full_history,
        integrator_name="abm" if contract["backend"] == "danca_abm" else "efork3",
        integrator_backend="native",
        caputo=True,
        parameters=lure_system.parameters,
        lure=collect_lure_metadata(
            lure_system.lure,
            transfer_convention="c^T(A - sI)^(-1)b",
            harmonic_condition="strict target-reference refinement only",
        ),
        seed=collect_seed_metadata(
            {
                "candidate_id": candidate_id,
                "family": "continued_candidate",
                "x0": positive_seed,
            },
            source=str(source_dir),
        ),
        random_seed=20260524,
        random_seed_policy="fixed_reproducible",
    )
    cfg["run_metadata"] = write_run_metadata(root / "run_metadata.json", run_metadata)
    write_json(root / "strict_refinement_config.json", cfg)
    return cfg


def _integrator_for_cfg(cfg: dict[str, Any]) -> tuple[Any, Any]:
    if cfg["contract"]["backend"] == "danca_abm":
        return _danca_integrator(cfg)
    return _project_integrator(cfg)


def run_chunk(outdir: str | Path, chunk_id: int, chunks: int) -> Path:
    force_single_openmp_thread_current_process()
    root = Path(outdir)
    cfg = read_json(root / "strict_refinement_config.json")
    plan = read_csv_rows(root / "strict_refinement_plan.csv")
    eq_backend, integrate = _integrator_for_cfg(cfg)
    refs = _reference_payloads(cfg, integrate)
    controls = _control_payloads(cfg, eq_backend, integrate)
    path = root / f"strict_refined_chunk_{int(chunk_id):03d}.csv"
    if path.exists():
        path.unlink()
    done = root / f"strict_refined_chunk_{int(chunk_id):03d}.done"
    if done.exists():
        done.unlink()
    rows = 0
    for item in plan:
        idx = _int(item["case_index"])
        if idx % int(chunks) != int(chunk_id):
            continue
        start = time.time()
        try:
            traj = integrate(np.array([_float(item["x0"]), _float(item["y0"]), _float(item["z0"])], dtype=float))
            out = _classify_strict(traj, cfg, refs, controls)
        except Exception as exc:
            out = {
                "class_id": 5,
                "class_label": "numerical_failure",
                "refined_status": f"exception:{exc!r}",
                "best_reference": "",
                "best_score": float("nan"),
            }
        final = {**item, **out, "elapsed_sec": time.time() - start}
        append_csv(path, final, STRICT_FIELDS)
        rows += 1
        if rows % 25 == 0:
            print(f"strict {cfg['mode']} chunk {chunk_id}: {rows} rows", flush=True)
    write_json(done, {"chunk_id": int(chunk_id), "rows": rows, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return path


def aggregate(outdir: str | Path, *, wait: bool = False, poll_sec: float = 60.0) -> Path:
    root = Path(outdir)
    cfg = read_json(root / "strict_refinement_config.json")
    chunks = int(cfg["chunks"])
    while wait:
        if all((root / f"strict_refined_chunk_{idx:03d}.done").exists() for idx in range(chunks)):
            break
        time.sleep(float(poll_sec))
    rows: list[dict[str, str]] = []
    for idx in range(chunks):
        rows.extend(read_csv_rows(root / f"strict_refined_chunk_{idx:03d}.csv"))
    rows.sort(key=lambda r: _int(r.get("case_index", 0)))
    write_csv(root / "strict_refined_rows.csv", rows, STRICT_FIELDS)

    counts = Counter(str(row.get("class_label", "")) for row in rows)
    target_hits = counts.get("target_positive", 0) + counts.get("target_negative", 0)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("equilibrium_id", "")), str(row.get("radius", "")))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (eq, radius), sub in sorted(grouped.items(), key=lambda item: (item[0][0], _float(item[0][1], 0.0))):
        sub_counts = Counter(str(row.get("class_label", "")) for row in sub)
        summary_rows.append(
            {
                "candidate_id": cfg["candidate_id"],
                "mode": cfg["mode"],
                "equilibrium_id": eq,
                "radius": radius,
                "n_refined": len(sub),
                "n_target_positive": sub_counts.get("target_positive", 0),
                "n_target_negative": sub_counts.get("target_negative", 0),
                "n_bounded_other": sub_counts.get("bounded_other", 0),
                "n_ambiguous_target_pair": sub_counts.get("ambiguous_target_pair", 0),
                "n_control_like": sub_counts.get("control_like", 0),
                "n_infinity": sub_counts.get("infinity", 0),
                "n_equilibrium": sub_counts.get("equilibrium", 0),
                "n_unknown": sub_counts.get("unknown", 0),
                "n_numerical_failure": sub_counts.get("numerical_failure", 0),
            }
        )
    write_csv(root / "strict_refinement_summary.csv", summary_rows)
    decision = {
        "candidate_id": cfg["candidate_id"],
        "mode": cfg["mode"],
        "planned_rows": int(cfg["planned_rows"]),
        "refined_rows": len(rows),
        "target_hits_after_strict_refinement": int(target_hits),
        "status": "partial" if len(rows) != int(cfg["planned_rows"]) else "ok",
        "hiddenness_interpretation": "rejected_self_excited_contact" if target_hits else "compatible_with_hiddenness_under_tested_radii",
        "notes": [
            "Only selected source rows were re-integrated and reclassified.",
            "A strict target hit is numerical evidence of target-like basin membership, not a proof of hiddenness.",
            "For equilibrium-neighborhood rows, strict target hits are evidence against hiddenness under the tested contract.",
        ],
    }
    write_json(root / "strict_refinement_decision.json", decision)
    return root / "strict_refinement_decision.json"


def launch(outdir: str | Path, args: argparse.Namespace) -> Path:
    root = Path(outdir)
    cfg = make_config(root, args)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = force_single_openmp_thread_env(os.environ.copy())
    env["PYTHONUNBUFFERED"] = "1"
    script = Path(args.script_path).resolve()
    launched: list[dict[str, Any]] = []
    for idx in range(int(args.chunks)):
        cmd = [sys.executable, str(script), "--job", "chunk", "--output-dir", str(root), "--chunk-id", str(idx), "--chunks", str(args.chunks)]
        stdout = (logs / f"chunk_{idx:03d}.out").open("ab")
        stderr = (logs / f"chunk_{idx:03d}.err").open("ab")
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
        launched.append({"job": f"chunk_{idx:03d}", "pid": proc.pid, "cmd": cmd})
    cmd = [sys.executable, str(script), "--job", "aggregate", "--output-dir", str(root), "--wait"]
    stdout = (logs / "aggregate.out").open("ab")
    stderr = (logs / "aggregate.err").open("ab")
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
    launched.append({"job": "aggregate", "pid": proc.pid, "cmd": cmd})
    manifest = {"output_dir": str(root), "mode": cfg["mode"], "planned_rows": cfg["planned_rows"], "chunks": int(cfg["chunks"]), "launched": launched}
    write_json(root / "launch_manifest.json", manifest)
    print(manifest, flush=True)
    return root / "launch_manifest.json"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict target-reference refinement for basin and equilibrium-ball outputs.")
    parser.add_argument("--job", choices=["launch", "chunk", "aggregate"], default="launch")
    parser.add_argument("--mode", choices=["basin-project", "basin-danca", "sphere-project", "sphere-danca"], required=False, default="basin-project")
    parser.add_argument("--source-dir", default=str(ROOT_OUTPUTS / "equilibrium_zoom_Eplus_160_20260517"))
    parser.add_argument("--source-csv", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--candidate-id", default=DEFAULT_PROJECT_CANDIDATE)
    parser.add_argument("--source-labels", default="unknown,target_positive,target_negative")
    parser.add_argument("--equilibrium-filter", default="")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--t-final", type=float, default=0.0)
    parser.add_argument("--t-burn", type=float, default=-1.0)
    parser.add_argument("--tail-fraction-start", type=float, default=0.5)
    parser.add_argument("--divergence-norm", type=float, default=120.0)
    parser.add_argument("--equilibrium-tol", type=float, default=1.0e-3)
    parser.add_argument("--max-cloud-points", type=int, default=700)
    parser.add_argument("--max-section-points", type=int, default=250)
    parser.add_argument("--max-score", type=float, default=0.45)
    parser.add_argument("--max-cloud-norm", type=float, default=0.35)
    parser.add_argument("--max-range-rel", type=float, default=0.60)
    parser.add_argument("--max-fft-rel", type=float, default=0.35)
    parser.add_argument("--max-section-norm", type=float, default=0.50)
    parser.add_argument("--min-reference-margin", type=float, default=0.10)
    parser.add_argument("--min-control-margin", type=float, default=0.10)
    parser.add_argument("--negative-control-radius", type=float, default=1.0e-4)
    parser.add_argument("--negative-control-equilibria", default="E0,E+,E-")
    parser.add_argument("--disable-negative-controls", action="store_true")
    parser.add_argument("--weight-cloud", type=float, default=1.0)
    parser.add_argument("--weight-range", type=float, default=0.35)
    parser.add_argument("--weight-fft", type=float, default=0.20)
    parser.add_argument("--weight-section", type=float, default=0.35)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--script-path", default=str(PROJECT_ROOT / "tools" / "cli" / "strict_target_refinement.py"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if args.mode == "sphere-danca" and args.source_labels == "unknown,target_positive,target_negative":
        args.source_labels = "bounded_nontrivial"
    outdir = Path(args.output_dir).resolve() if args.output_dir else ROOT_OUTPUTS / f"strict_target_refinement_{args.mode}_{int(time.time())}"
    if args.job == "launch":
        launch(outdir, args)
    elif args.job == "chunk":
        run_chunk(outdir, int(args.chunk_id), int(args.chunks))
    elif args.job == "aggregate":
        aggregate(outdir, wait=bool(args.wait))


if __name__ == "__main__":
    main()
