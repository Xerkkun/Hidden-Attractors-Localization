from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import chua_initial_cond as chua
from biased_describing_function import search_biased_candidates
from corrida1_refined_verification import run_corrida1_refined_verification
from debug_branch1_failures import run_debug_branch1_failures
from equilibria_analysis import analyze_equilibria
from extended_search_utils import chua_ic_params, json_safe, load_config, timestamped_output_dir, write_csv
from harmonic_diagnostics import centered_candidates_from_nyquist, plot_harmonic_spectrum, rho_h_diagnostic, write_rho_outputs
from multiparameter_continuation import run_multiparameter_continuation
from seed_cloud_search import run_seed_cloud_search


def centered_seed_candidate(row: Dict[str, Any], cfg: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    omega = float(row["omega"])
    W = chua.W_frac(omega, float(cfg["q"]), p)
    k = float(-1.0 / np.real(W))
    xseed, _v, _eig = chua.build_fractional_seed(float(cfg["q"]), p, omega, k, float(row["A"]))
    return {
        "candidate_id": row["candidate_id"],
        "df_family": row["df_family"] + "_centered",
        "A": float(row["A"]),
        "sigma0": 0.0,
        "omega": omega,
        "q": float(cfg["q"]),
        "mu": row.get("mu", ""),
        "residual_abs": row.get("residual_abs", ""),
        "rho_H": row.get("rho_H", ""),
        "seed_x": float(xseed[0]),
        "seed_y": float(xseed[1]),
        "seed_z": float(xseed[2]),
    }


def build_rho_diagnostics(cfg: Dict[str, Any], p: Dict[str, Any], outdir: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    centered_seed_rows: List[Dict[str, Any]] = []
    for cand in centered_candidates_from_nyquist(cfg, p):
        diag = rho_h_diagnostic(
            candidate_id=cand["candidate_id"],
            df_family=cand["df_family"],
            A=float(cand["A"]),
            sigma0=0.0,
            omega=float(cand["omega"]),
            q=float(cfg["q"]),
            p=p,
            mu=cand.get("mu"),
            K=int(cfg["rho_H"].get("K", 10)),
            n_quad=int(cfg["rho_H"].get("n_quad", 4096)),
            threshold=float(cfg["rho_H"].get("threshold", 0.1)),
            machado_branch=int(cfg.get("machado", {}).get("branch", 0)),
            machado_eps=float(cfg.get("machado", {}).get("zero_eps", 1e-12)),
        )
        rows.append(diag)
        seed_source = {**cand, **diag}
        try:
            centered_seed_rows.append(centered_seed_candidate(seed_source, cfg, p))
        except Exception:
            pass
    return rows, centered_seed_rows


def write_report(
    cfg: Dict[str, Any],
    outdir: Path,
    rho_rows: List[Dict[str, Any]],
    biased_rows: List[Dict[str, Any]],
    continuation: List[Dict[str, Any]],
    seed_rows: List[Dict[str, Any]],
) -> None:
    survived = [r for r in continuation if r.get("survived")]
    seed_candidates = [r for r in seed_rows if r.get("final_class") in {"candidate_hidden_like", "candidate_bounded_nontrivial"}]
    best_rho = sorted([r for r in rho_rows if np.isfinite(float(r.get("rho_H", np.nan)))], key=lambda r: float(r["rho_H"]))[:8]
    best_biased = sorted([r for r in biased_rows if np.isfinite(float(r.get("residual_abs", np.nan)))], key=lambda r: float(r["residual_abs"]))[:8]
    lines = [
        "# Extended Chua Fractional Non-Smooth Search",
        "",
        "La funcion descriptiva y el balance armonico se usan como aproximaciones de regimen oscilatorio estacionario tipo Weyl para generar semillas. Las semillas se validan despues en el sistema causal de Caputo con memoria desde t0.",
        "",
        "## Parametros",
        "",
        f"- q: `{cfg['q']}`",
        f"- h: `{cfg.get('h')}`",
        f"- Lm: `{cfg.get('Lm')}`",
        f"- modelo: `{cfg.get('model', 'piecewise')}`",
        "",
        "Convencion de transferencia: el repositorio usa `W_code(lambda)=r^T(P-lambda I)^(-1)b`, con `lambda=(j omega)^q`. Los residuos reportados usan `1 + W_code N` para respetar el signo historico del codigo.",
        "",
        "## Conteos",
        "",
        f"- diagnosticos rho_H: `{len(rho_rows)}`",
        f"- candidatos sesgados guardados: `{len(biased_rows)}`",
        f"- candidatos que sobrevivieron continuacion: `{len(survived)}`",
        f"- semillas de nube simuladas: `{len(seed_rows)}`",
        f"- candidatos de nube para revisar: `{len(seed_candidates)}`",
        "",
        "## Mejores rho_H",
        "",
    ]
    for row in best_rho:
        lines.append(f"- `{row['candidate_id']}`: rho_H={float(row['rho_H']):.6g}, residual={float(row['residual_abs']):.6g}")
    lines.extend(["", "## Mejores residuos DF sesgada", ""])
    for row in best_biased:
        lines.append(f"- `{row['candidate_id']}`: family={row['df_family']}, A={float(row['A']):.6g}, sigma0={float(row['sigma0']):.6g}, omega={float(row['omega']):.6g}, residual={float(row['residual_abs']):.6g}, rho_H={float(row['rho_H']):.6g}")
    lines.extend([
        "",
        "## Advertencias Sobre Memoria",
        "",
        "- La continuacion multiparametrica usa `FractionalHistory` y transporta una ventana discreta `ceil(Lm/h)+1` cuando el integrador Python EFORK recibe `history`.",
        "- Si q cambia entre etapas, la ventana se conserva como heuristica numerica de memoria corta; no es una prueba de existencia de una orbita exacta de Caputo.",
        "- No se declaran atractores ocultos en este flujo. Las etiquetas emitidas son `candidate_hidden_like`, `candidate_bounded_nontrivial` y `requires_basin_verification`.",
        "",
        "## Candidatos Para Simulacion Larga",
        "",
    ])
    for row in best_biased[:10]:
        lines.append(f"- `{row['candidate_id']}` requiere simulacion larga y verificacion de cuenca.")
    for row in seed_candidates[:10]:
        lines.append(f"- `{row['seed_id']}` desde nube: {row['final_class']} requiere verificacion de cuenca.")
    (Path(outdir) / "extended_search_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console_table(candidates: List[Dict[str, Any]]) -> None:
    fields = ["candidate_id", "df_family", "A", "sigma0", "omega", "mu", "q", "residual_abs", "rho_H", "continuation_survived", "final_class", "memory_carried"]
    print(",".join(fields), flush=True)
    for row in candidates[:20]:
        print(",".join(str(row.get(k, "")) for k in fields), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chua_fractional_nonsmooth.yaml")
    parser.add_argument("--mode", default="extended_search", choices=["extended_search", "debug_branch1_failures", "corrida1_refined_verification"])
    args = parser.parse_args()
    if args.mode == "debug_branch1_failures":
        run_debug_branch1_failures(args.config)
        return
    if args.mode == "corrida1_refined_verification":
        run_corrida1_refined_verification(args.config)
        return
    cfg = load_config(args.config)
    outdir = timestamped_output_dir(cfg)
    p = chua_ic_params(cfg)
    chua.PARAMS = p
    chua.QORD = np.float64(float(cfg["q"]))

    eq_rows = analyze_equilibria(cfg, p, outdir)
    equilibria = [np.asarray(r["point"], dtype=float) for r in eq_rows]

    rho_rows, centered_seed_rows = build_rho_diagnostics(cfg, p, outdir)
    biased_rows, biased_all_rows = search_biased_candidates(cfg, p, outdir)
    for row in biased_all_rows:
        rho_rows.append({**row, "df_type": "biased"})
    write_rho_outputs(rho_rows, outdir)
    for row in sorted(rho_rows, key=lambda r: float(r.get("rho_H", np.inf)))[: int(cfg["rho_H"].get("spectrum_plots", 6))]:
        if "fourier" in row:
            plot_harmonic_spectrum(row, outdir)

    continuation_candidates = sorted(
        centered_seed_rows + biased_rows,
        key=lambda r: (float(r.get("residual_abs", np.inf)) if str(r.get("residual_abs", "")) else np.inf, float(r.get("rho_H", np.inf)) if str(r.get("rho_H", "")) else np.inf),
    )
    cont_results = run_multiparameter_continuation(continuation_candidates, cfg, p, equilibria, outdir)
    seed_rows = run_seed_cloud_search(cfg, p, equilibria, outdir)

    by_id = {r["candidate_id"]: r for r in continuation_candidates}
    table_rows = []
    for result in cont_results:
        base = dict(by_id.get(result["candidate_id"], {}))
        base["continuation_survived"] = bool(result.get("survived"))
        base["final_class"] = result.get("final_class", "")
        base["memory_carried"] = bool(result.get("memory_carried"))
        table_rows.append(base)
    write_report(cfg, outdir, rho_rows, biased_rows, cont_results, seed_rows)
    (outdir / "extended_search_manifest.json").write_text(json.dumps(json_safe({
        "config": cfg,
        "output_dir": str(outdir),
        "rho_count": len(rho_rows),
        "biased_count": len(biased_rows),
        "continuation_count": len(cont_results),
        "seed_cloud_count": len(seed_rows),
    }), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Extended search outputs: {outdir}", flush=True)
    print_console_table(table_rows)


if __name__ == "__main__":
    main()
