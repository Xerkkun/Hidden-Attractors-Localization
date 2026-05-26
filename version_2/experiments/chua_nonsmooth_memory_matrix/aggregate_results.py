#!/usr/bin/env python3
"""Aggregate the isolated memory-matrix evidence without changing its verdicts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import VERSION2_ROOT, experiment_spec, git_commit, is_ok_status, json_result, read_csv_rows, status_counts

from hidden_attractors.io import read_json, write_csv


REQUIRED_COLUMNS = (
    "exp_id",
    "seed_family",
    "transfer_power",
    "continuation_family",
    "continuation_memory",
    "continuation_ic_policy",
    "continuation_solver",
    "hiddenness_integrator",
    "hiddenness_memory",
    "reached_candidate_from_seed",
    "any_equilibrium_ball_hit",
    "hiddenness_status",
    "periodicity_status",
    "boundedness_status",
    "failure_rate",
)


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    """Load JSON when present; unfinished task directories remain pending."""

    return read_json(path) if path.exists() else None


def _merge_verdict(summaries: list[dict[str, Any]]) -> str:
    """Merge plus/minus decisions conservatively for one solver-memory cell."""

    statuses = {str(item.get("hiddenness_status", "inconclusive")) for item in summaries}
    if "not_hidden_under_tested_radii" in statuses:
        return "not_hidden_under_tested_radii"
    if "numerical_failure" in statuses:
        return "numerical_failure"
    if statuses == {"compatible_with_hiddenness_under_tested_radii"} and len(summaries) >= 2:
        return "compatible_with_hiddenness_under_tested_radii"
    return "inconclusive"


def build_comparison(root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate sign-specific hiddenness summaries into comparison rows."""

    continuation = {row["cache_key"]: row for row in read_csv_rows(root / "tasks" / "continuation_tasks.csv")}
    grouped: dict[tuple[str, str, str, str], list[tuple[dict[str, str], dict[str, Any] | None]]] = defaultdict(list)
    for task in read_csv_rows(root / "tasks" / "hiddenness_tasks.csv"):
        key = (task["exp_id"], task["continuation_cache_key"], task["hiddenness_integrator"], task["hiddenness_memory"])
        grouped[key].append((task, _read_optional_json(root / task["output_dir"] / "hiddenness_summary.json")))
    output: list[dict[str, Any]] = []
    for (exp_id, cont_key, integrator, memory), entries in sorted(grouped.items()):
        spec = experiment_spec(manifest, exp_id)
        cont = continuation[cont_key]
        summaries = [summary for _task, summary in entries if summary is not None]
        cont_summary = _read_optional_json(root / cont["output_dir"] / "continuation_summary.json")
        if not summaries:
            verdict = "pending"
            reached = ""
            hit = ""
            periodicity = "pending"
            boundedness = "pending"
            failure_rate: Any = ""
        else:
            verdict = _merge_verdict(summaries)
            reached = all(bool(item.get("reached_candidate_from_seed", False)) for item in summaries)
            hit = any(bool(item.get("any_equilibrium_ball_hit", False)) for item in summaries)
            periodicity_values = sorted({str(item.get("reference_periodicity_status", "")) for item in summaries})
            bounded_values = sorted({str(item.get("reference_boundedness_status", "")) for item in summaries})
            periodicity = ";".join(value for value in periodicity_values if value)
            boundedness = ";".join(value for value in bounded_values if value)
            failure_rate = sum(float(item.get("failure_rate", 1.0)) for item in summaries) / len(summaries)
        output.append(
            {
                "exp_id": exp_id,
                "seed_family": spec["seed_family"],
                "transfer_power": spec["transfer_power"],
                "continuation_family": spec["continuation_family"],
                "continuation_memory": spec["continuation_memory"],
                "continuation_ic_policy": spec["continuation_ic_policy"],
                "continuation_solver": cont["continuation_solver"],
                "hiddenness_integrator": integrator,
                "hiddenness_memory": memory,
                "reached_candidate_from_seed": reached,
                "any_equilibrium_ball_hit": hit,
                "hiddenness_status": verdict,
                "periodicity_status": periodicity,
                "boundedness_status": boundedness,
                "failure_rate": failure_rate,
                "continuation_status": "pending" if cont_summary is None else cont_summary.get("status", "failed"),
                "git_commit": git_commit(),
                "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
    return output


def _comparison_answer(rows: list[dict[str, Any]], field: str, left: str | tuple[str, ...], right: str | tuple[str, ...]) -> str:
    """Summarize available completed cells for one requested contrast."""

    completed = [row for row in rows if row["hiddenness_status"] not in {"pending", "numerical_failure"}]
    left_values = (left,) if isinstance(left, str) else left
    right_values = (right,) if isinstance(right, str) else right
    left_rows = [row for row in completed if str(row.get(field)) in left_values]
    right_rows = [row for row in completed if str(row.get(field)) in right_values]
    if not left_rows or not right_rows:
        return "Pendiente de celdas completadas comparables bajo ambos niveles."
    left_hits = sum(bool(row["any_equilibrium_ball_hit"]) for row in left_rows)
    right_hits = sum(bool(row["any_equilibrium_ball_hit"]) for row in right_rows)
    left_label = "/".join(left_values)
    right_label = "/".join(right_values)
    return f"Celdas completadas: {left_label}={len(left_rows)} (contactos={left_hits}); {right_label}={len(right_rows)} (contactos={right_hits}). Interpretar junto con periodicidad y fallos."


def write_markdown(root: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the explicit five-question interpretation requested for the matrix."""

    statuses = status_counts(rows, "hiddenness_status")
    text = [
        "# Comparacion experimental Chua no suave: semilla, continuacion y memoria",
        "",
        "Este reporte agrega pruebas numericas externas al workflow validado. Una semilla DF es heuristica; una ausencia de contactos en bolas probadas solo es `compatible_with_hiddenness_under_tested_radii`.",
        "",
        "## Contrato base",
        "",
        f"- `q_target={manifest['contract']['q_target']}`, `h={manifest['contract']['h']}`, `t_final={manifest['contract']['t_final']}`, `t_burn={manifest['contract']['t_burn']}`, `Lm={manifest['contract']['memory_length']}`.",
        "- La semilla `integer_like` usa `W(s)` con `q=1`; la semilla `fractional` usa `W_q(i omega)` en rama principal. Ambas usan DF clasica, sin Machado/FDF.",
        "- La continuacion fraccionaria ABM o EFORK transporta historia; la continuacion entera transporta solo el ultimo punto.",
        "- ABM expone una cadena causal de continuacion: `full` devuelve historia completa y `truncated` devuelve exactamente su ventana `Lm`.",
        "",
        "## Resumen de estados",
        "",
    ]
    text.extend([f"- `{key}`: {value}" for key, value in sorted(statuses.items())])
    text.extend(
        [
            "",
            "## Preguntas comparativas",
            "",
            "### a) Que cambia al usar W(s) contra W_q(s)?",
            "",
            _comparison_answer(rows, "transfer_power", "s", "s^q"),
            "",
            "La diferencia primaria medible antes de integrar esta en `omega`, ganancia, amplitud y condicion inicial guardadas en `shared/seeds/`; no equivale por si sola a diferencia de atractor.",
            "",
            "### b) Que cambia al usar continuacion entera contra fraccionaria?",
            "",
            _comparison_answer(rows, "continuation_family", "integer_like_q1", "fractional_caputo"),
            "",
            "La primera evoluciona el sistema `q=1`; la segunda evoluciona el candidato Caputo con `q_target` y memoria efectiva. Solo la prueba posterior usa siempre el sistema fraccionario objetivo.",
            "",
            "### c) Que cambia al usar ultimo punto contra ventana de memoria?",
            "",
            _comparison_answer(rows, "continuation_ic_policy", "last_point_only", ("history_window", "history_window_truncated")),
            "",
            "`last_point_only` no conserva prehistoria; `history_window` o `history_window_truncated` preserva una parte documentada del estado numerico fraccionario.",
            "",
            "### d) Que cambia al usar memoria completa contra truncada?",
            "",
            _comparison_answer(rows, "hiddenness_memory", "full", "truncated"),
            "",
            "En EFORK, `full` se implementa con un horizonte que no trunca el tiempo integrado; en ABM, `full` usa la suma completa y `truncated` la aproximacion reiniciada de ventana finita.",
            "",
            "### e) Que cambia entre ABM y EFORK?",
            "",
            "En continuacion:",
            "",
            _comparison_answer(rows, "continuation_solver", "abm", "efork"),
            "",
            "En la integracion objetivo de ocultedad:",
            "",
            _comparison_answer(rows, "hiddenness_integrator", "abm", "efork"),
            "",
            "La tabla permite contrastar tanto el integrador usado para la prueba de ocultedad como, por las columnas de continuacion, la ruta ABM frente a EFORK. ABM `full` conserva valores de toda la historia; EFORK `full` conserva la historia efectiva dentro del horizonte no truncante declarado.",
            "",
        ]
    )
    path = root / "reports" / "experiment_matrix_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text), encoding="utf-8")


def cache_report(root: Path) -> dict[str, Any]:
    """Report which task caches are complete, missing, failed, or reusable."""

    sections: dict[str, list[dict[str, Any]]] = {"shared": [], "continuation": [], "hiddenness": [], "figures": []}
    shared_rows = read_csv_rows(root / "tasks" / "shared_cache_tasks.csv")
    for row in shared_rows:
        status_file = root / "shared" / "status" / row["cache_name"] / "status.json"
        status = _read_optional_json(status_file)
        outputs = [root / path for path in row["outputs"].split(";")]
        sections["shared"].append({"task_id": row["task_id"], "cache_key": row["cache_key"], "status": "pending" if status is None else status.get("status"), "reusable": is_ok_status(status_file, outputs)})
    for name, csv_name in (("continuation", "continuation_tasks.csv"), ("hiddenness", "hiddenness_tasks.csv")):
        for row in read_csv_rows(root / "tasks" / csv_name):
            status_file = root / row["output_dir"] / "status.json"
            status = _read_optional_json(status_file)
            outputs = [Path(item) for item in (status or {}).get("outputs", []) if not str(item).endswith("/")]
            required = [root / row["output_dir"] / item for item in outputs]
            sections[name].append({"task_id": row["task_id"], "cache_key": row["cache_key"], "status": "pending" if status is None else status.get("status"), "reusable": is_ok_status(status_file, required)})
    for row in read_csv_rows(root / "tasks" / "figure_tasks.csv"):
        status_file = root / "figures" / row["exp_id"] / "status.json"
        status = _read_optional_json(status_file)
        outputs = [root / "figures" / row["exp_id"] / f"{stem}.png" for stem in ("seed_nyquist", "continuation_path", "phase3d_candidate", "equilibrium_neighborhood_trajectories", "basin_slices_xy_xz", "time_series", "fft_psd", "hiddenness_summary")]
        sections["figures"].append({"task_id": row["task_id"], "status": "pending" if status is None else status.get("status"), "reusable": is_ok_status(status_file, outputs)})
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "skip_rule": "A task is reused only when its task-specific status.json has status=ok; --force recomputes it.",
        "sections": sections,
    }


def main() -> None:
    """Write CSV, Markdown, and cache reuse summaries from current artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(VERSION2_ROOT / "outputs/chua_nonsmooth_fractional_memory_matrix"))
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    manifest = read_json(root / "experiment_matrix.json")
    rows = build_comparison(root, manifest)
    fields = list(REQUIRED_COLUMNS) + ["continuation_status", "git_commit", "date"]
    write_csv(root / "reports" / "aggregate_comparison.csv", rows, fields)
    write_markdown(root, manifest, rows)
    json_result(root / "reports" / "cache_reuse_report.json", cache_report(root))
    print(f"Wrote aggregate reports under {root / 'reports'}")


if __name__ == "__main__":
    main()
