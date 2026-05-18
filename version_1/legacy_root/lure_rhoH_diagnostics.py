#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

import chua_initial_cond as chua
from harmonic_diagnostics import rho_h_diagnostic
from lure_candidate_manifest import (
    DEFAULT_CONFIG,
    ROOT,
    as_float,
    as_int,
    csv_value,
    json_safe,
    load_config,
    official_chua_params,
    read_csv_rows,
    read_json,
    resolve_path,
)


RHO_FIELDS = [
    "candidate_id",
    "q",
    "branch_index",
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
    "K",
    "harmonic_energy_ratio",
    "rhoH_class",
]

COMPARISON_FIELDS = [
    "lure_candidate_id",
    "machado_candidate_id",
    "q",
    "final_state_distance",
    "range_distance",
    "mean_tail_distance",
    "var_tail_distance",
    "fft_peak_distance",
    "psd_entropy_distance",
    "target_label_match",
    "likely_same_attractor",
    "machado_source",
]


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k, "")) for k in fields})


def rho_class(value: float, good: float, marginal: float) -> str:
    if not math.isfinite(value):
        return "missing"
    if value < good:
        return "good"
    if value < marginal:
        return "marginal"
    return "poor"


def manifest_path(cfg: Dict[str, Any]) -> Path:
    return ROOT / str(cfg.get("manifest", {}).get("output_dir", "outputs/lure_route")) / "lure_candidates_manifest.csv"


def finite_candidate(row: Dict[str, Any]) -> bool:
    return all(math.isfinite(as_float(row.get(k))) for k in ["A", "omega", "q"])


def compute_rows(cfg: Dict[str, Any], manifest_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rho_cfg = cfg.get("rhoH", {})
    K = int(rho_cfg.get("K", 20))
    n_quad = int(rho_cfg.get("quadrature_points", 8192))
    sigma0_default = float(rho_cfg.get("sigma0_default", 0.0))
    good = float(rho_cfg.get("threshold_good", 0.1))
    marginal = float(rho_cfg.get("threshold_marginal", 0.3))
    p = official_chua_params()
    rows: List[Dict[str, Any]] = []
    for mrow in manifest_rows:
        cid = str(mrow.get("candidate_id", ""))
        q = as_float(mrow.get("q"))
        A = as_float(mrow.get("A"))
        omega = as_float(mrow.get("omega"))
        sigma0 = as_float(mrow.get("sigma0"), sigma0_default)
        branch = as_int(mrow.get("branch_index"), 0)
        if not (math.isfinite(A) and math.isfinite(omega) and math.isfinite(q)):
            rows.append({
                "candidate_id": cid,
                "q": q,
                "branch_index": branch,
                "A": A,
                "sigma0": sigma0,
                "omega": omega,
                "rhoH_class": "missing",
                "K": K,
            })
            continue
        try:
            diag = rho_h_diagnostic(
                candidate_id=cid,
                df_family="lure_classic",
                A=A,
                sigma0=sigma0,
                omega=omega,
                q=q,
                p=p,
                K=K,
                n_quad=n_quad,
                threshold=good,
            )
            row = {
                "candidate_id": cid,
                "q": q,
                "branch_index": branch,
                "A": A,
                "sigma0": sigma0,
                "omega": omega,
                "N_re": diag.get("N_re"),
                "N_im": diag.get("N_im"),
                "W_re": diag.get("W_re"),
                "W_im": diag.get("W_im"),
                "residual_re": diag.get("residual_re"),
                "residual_im": diag.get("residual_im"),
                "residual_abs": diag.get("residual_abs"),
                "rho_H": diag.get("rho_H"),
                "K": K,
                "harmonic_energy_ratio": diag.get("harmonic_energy_ratio"),
                "rhoH_class": rho_class(as_float(diag.get("rho_H")), good, marginal),
                "_fourier": diag.get("fourier"),
            }
        except Exception as exc:
            row = {
                "candidate_id": cid,
                "q": q,
                "branch_index": branch,
                "A": A,
                "sigma0": sigma0,
                "omega": omega,
                "rhoH_class": "missing",
                "K": K,
                "_error": str(exc),
            }
        rows.append(row)
    return rows


def plot_outputs(outdir: Path, rows: Sequence[Dict[str, Any]]) -> List[str]:
    plotdir = outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    files: List[str] = []
    finite = [r for r in rows if math.isfinite(as_float(r.get("rho_H")))]
    if finite:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.scatter([as_float(r.get("q")) for r in finite], [as_float(r.get("rho_H")) for r in finite], s=30)
        ax.set_xlabel("q")
        ax.set_ylabel("rho_H")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = plotdir / "lure_rhoH_vs_q.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(str(path))

        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.scatter([as_float(r.get("A")) for r in finite], [as_float(r.get("rho_H")) for r in finite], s=30)
        ax.set_xlabel("A")
        ax.set_ylabel("rho_H")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = plotdir / "lure_rhoH_vs_A.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(str(path))

    for row in rows:
        fourier = row.get("_fourier")
        if not isinstance(fourier, dict):
            continue
        coeffs = fourier.get("coefficients", {})
        if not coeffs:
            continue
        ks = sorted(int(k) for k in coeffs.keys())
        vals = [abs(complex(coeffs[k]["Y"])) for k in ks]
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.bar(ks, vals, color="#2563eb")
        ax.set_xlabel("k")
        ax.set_ylabel("|Y_k(A,sigma0)|")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        safe = str(row["candidate_id"]).replace("/", "_").replace("\\", "_")
        path = plotdir / f"lure_harmonic_spectrum_{safe}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(str(path))
    return files


def vec(row: Dict[str, Any], keys: Sequence[str]) -> np.ndarray:
    return np.asarray([as_float(row.get(k)) for k in keys], dtype=float)


def scan_machado_records(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    roots = cfg.get("source", {}).get("search_roots", ["."])
    for root_value in roots:
        root = resolve_path(root_value, ROOT)
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            name = path.name
            if name not in {"df_seed_comparison_summary.json", "machado_sweep_summary.json"}:
                continue
            try:
                data = read_json(path)
            except Exception:
                continue
            for rec in data.get("records", []):
                if not isinstance(rec, dict):
                    continue
                method = str(rec.get("method", "")).lower()
                slug = str(rec.get("slug", rec.get("candidate_id", ""))).lower()
                if "machado" not in method and "machado" not in slug and "mu_" not in slug:
                    continue
                records.append({
                    "candidate_id": rec.get("candidate_id", rec.get("slug", "")),
                    "q": as_float(rec.get("q", data.get("frac_order"))),
                    "final_x": vec({"x": ""}, []),
                    "final_state": rec.get("final_state_eps1", []),
                    "range_x": rec.get("range_x", rec.get("shape_diagnostics", {}).get("range_x", "")),
                    "range_y": rec.get("range_y", rec.get("shape_diagnostics", {}).get("range_y", "")),
                    "range_z": rec.get("range_z", rec.get("shape_diagnostics", {}).get("range_z", "")),
                    "mean_x_tail": rec.get("mean_x_tail", ""),
                    "mean_y_tail": rec.get("mean_y_tail", ""),
                    "mean_z_tail": rec.get("mean_z_tail", ""),
                    "var_x_tail": rec.get("var_x_tail", ""),
                    "var_y_tail": rec.get("var_y_tail", ""),
                    "var_z_tail": rec.get("var_z_tail", ""),
                    "fft_peak": rec.get("fft_peak", ""),
                    "psd_entropy": rec.get("psd_entropy", ""),
                    "target_label": rec.get("target_label", rec.get("hiddenness_status", "")),
                    "source": str(path),
                })
    return records


def comparison_rows(cfg: Dict[str, Any], manifest_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    machado = scan_machado_records(cfg)
    rows: List[Dict[str, Any]] = []
    for lure in manifest_rows:
        lfinal = vec(lure, ["final_x", "final_y", "final_z"])
        lrange = vec(lure, ["range_x", "range_y", "range_z"])
        for mach in machado:
            if math.isfinite(as_float(lure.get("q"))) and math.isfinite(as_float(mach.get("q"))):
                if abs(as_float(lure.get("q")) - as_float(mach.get("q"))) > 5e-4:
                    continue
            mfinal = np.asarray(mach.get("final_state", [np.nan, np.nan, np.nan]), dtype=float)
            if mfinal.size < 3:
                continue
            mfinal = mfinal[:3]
            final_dist = float(np.linalg.norm(lfinal - mfinal)) if np.all(np.isfinite(lfinal)) and np.all(np.isfinite(mfinal)) else float("nan")
            mrange = vec(mach, ["range_x", "range_y", "range_z"])
            range_dist = float(np.linalg.norm(lrange - mrange)) if np.all(np.isfinite(lrange)) and np.all(np.isfinite(mrange)) else float("nan")
            same = (
                (math.isfinite(final_dist) and final_dist < 1.0)
                or (math.isfinite(range_dist) and range_dist < 0.5)
            )
            rows.append({
                "lure_candidate_id": lure.get("candidate_id", ""),
                "machado_candidate_id": mach.get("candidate_id", ""),
                "q": lure.get("q", ""),
                "final_state_distance": final_dist,
                "range_distance": range_dist,
                "mean_tail_distance": "",
                "var_tail_distance": "",
                "fft_peak_distance": "",
                "psd_entropy_distance": "",
                "target_label_match": str(lure.get("hiddenness_status", "")) == str(mach.get("target_label", "")),
                "likely_same_attractor": bool(same),
                "machado_source": mach.get("source", ""),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute rho_H diagnostics for Lure manifest candidates.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    cfg = load_config(args.config)
    outdir = ROOT / str(cfg.get("manifest", {}).get("output_dir", "outputs/lure_route"))
    rows_manifest = read_csv_rows(manifest_path(cfg))
    rows = compute_rows(cfg, rows_manifest)
    public = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    csv_path = outdir / "lure_rhoH_diagnostics.csv"
    write_csv(csv_path, public, RHO_FIELDS)
    plot_files = plot_outputs(outdir, rows)
    comp = comparison_rows(cfg, rows_manifest)
    files_written = [str(csv_path), *plot_files]
    if comp:
        comp_path = outdir / "lure_vs_machado_comparison.csv"
        write_csv(comp_path, comp, COMPARISON_FIELDS)
        files_written.append(str(comp_path))
    print("candidate_id,q,branch_index,priority_class,target_total,target_from_Eminus,target_from_Eplus,rho_H,rhoH_class,hiddenness_status,should_refine,files_written", flush=True)
    by_id = {r["candidate_id"]: r for r in public}
    files = ";".join(files_written)
    for m in rows_manifest:
        r = by_id.get(m.get("candidate_id", ""), {})
        print(",".join([
            str(m.get("candidate_id", "")),
            str(m.get("q", "")),
            str(m.get("branch_index", "")),
            str(m.get("priority_class", "")),
            str(m.get("target_total", "")),
            str(m.get("target_from_Eminus", "")),
            str(m.get("target_from_Eplus", "")),
            str(r.get("rho_H", "")),
            str(r.get("rhoH_class", "")),
            str(m.get("hiddenness_status", "")),
            str(m.get("should_refine", "")),
            files,
        ]), flush=True)


if __name__ == "__main__":
    main()
