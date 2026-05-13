#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "lure_candidate_route.yaml"

MANIFEST_FIELDS = [
    "candidate_id",
    "source_type",
    "source_path",
    "model_kind",
    "df_family",
    "q",
    "branch_index",
    "omega",
    "k",
    "A",
    "phase",
    "sigma0",
    "seed_x",
    "seed_y",
    "seed_z",
    "final_x",
    "final_y",
    "final_z",
    "final_norm",
    "continuation_status",
    "bounded",
    "hiddenness_status",
    "target_total",
    "target_from_Eminus",
    "target_from_E0",
    "target_from_Eplus",
    "total_samples",
    "target_fraction",
    "blocking_equilibrium",
    "blocking_radius",
    "source_hidden_summary_json",
    "source_hidden_check_csv",
    "source_final_attractor_csv",
    "source_continuation_json",
    "notes",
    "priority_class",
    "should_refine",
]

KNOWN_PATTERNS = [
    "df_seed_comparison_summary.json",
    "df_seed_comparison_summary.csv",
    "df_candidate_summary_*.json",
    "unified_pipeline_summary.json",
    "unified_continuation_summary.json",
    "continuation_summary*.json",
    "hidden_target_summary.json",
    "hidden_target_check*.csv",
    "final_attractor*.csv",
    "final_attractor*.npz",
    "continuation_summary*.json",
]


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("{"):
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except Exception as exc:
        data = simple_yaml_load(text)
        if not isinstance(data, dict):
            raise RuntimeError(f"No se pudo leer {path}. Instala PyYAML o usa JSON.") from exc
        return data
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("La configuracion debe ser un diccionario.")
    return data


def parse_scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if text in {"null", "None", "~"}:
        return None
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def simple_yaml_load(text: str) -> Dict[str, Any]:
    """Read the small YAML subset used by configs/lure_candidate_route.yaml."""
    raw_lines: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        raw_lines.append((indent, raw.strip()))

    def parse_block(index: int, indent: int) -> Tuple[Any, int]:
        if index >= len(raw_lines):
            return {}, index
        if raw_lines[index][1].startswith("- "):
            out_list: List[Any] = []
            while index < len(raw_lines):
                cur_indent, item = raw_lines[index]
                if cur_indent < indent or not item.startswith("- "):
                    break
                if cur_indent > indent:
                    break
                body = item[2:].strip()
                index += 1
                if body:
                    out_list.append(parse_scalar(body))
                elif index < len(raw_lines):
                    child, index = parse_block(index, raw_lines[index][0])
                    out_list.append(child)
                else:
                    out_list.append(None)
            return out_list, index

        out_dict: Dict[str, Any] = {}
        while index < len(raw_lines):
            cur_indent, item = raw_lines[index]
            if cur_indent < indent or item.startswith("- "):
                break
            if cur_indent > indent:
                break
            if ":" not in item:
                index += 1
                continue
            key, value = item.split(":", 1)
            key = key.strip()
            value = value.strip()
            index += 1
            if value:
                out_dict[key] = parse_scalar(value)
            elif index < len(raw_lines):
                child, index = parse_block(index, raw_lines[index][0])
                out_dict[key] = child
            else:
                out_dict[key] = {}
        return out_dict, index

    parsed, _ = parse_block(0, raw_lines[0][0] if raw_lines else 0)
    if not isinstance(parsed, dict):
        raise ValueError("La configuracion debe ser un diccionario.")
    return parsed


def csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return ";".join(str(float(x)) for x in value.ravel())
    if isinstance(value, (list, tuple)):
        return ";".join(str(x) for x in value)
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k, "")) for k in fields})


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def as_float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "si", "on", "ok", "bounded"}:
        return True
    if text in {"0", "false", "no", "off", "failed", "divergent"}:
        return False
    return None


def q_tag(q: float) -> str:
    return f"{float(q):.5f}".replace("-", "m").replace(".", "p")


def candidate_id_for(q: float, branch: int, suffix: str = "") -> str:
    base = f"lure_q_{q_tag(q)}_branch_{int(branch)}"
    return base + suffix


def resolve_path(path_value: Any, base: Path | None = None) -> Path:
    raw = str(path_value or "").strip()
    if not raw:
        return Path("")
    path = Path(raw)
    if path.exists():
        return path
    if path.is_absolute():
        return path
    if "\\" in raw:
        win = PureWindowsPath(raw)
        if win.is_absolute():
            return Path(raw)
        parts = [p for p in win.parts if p not in {"\\", "/"} and not p.endswith(":")]
        candidate = (base or ROOT).joinpath(*parts)
        if candidate.exists():
            return candidate
    candidate = (base or ROOT) / path
    return candidate


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def vector3(value: Any) -> Tuple[float, float, float]:
    if isinstance(value, dict):
        return (as_float(value.get("x")), as_float(value.get("y")), as_float(value.get("z")))
    if isinstance(value, str):
        parts = [p for p in value.replace(";", ",").split(",") if p.strip()]
        if len(parts) >= 3:
            return (as_float(parts[0]), as_float(parts[1]), as_float(parts[2]))
    try:
        seq = list(value)
    except Exception:
        return (float("nan"), float("nan"), float("nan"))
    seq = seq + [float("nan")] * (3 - len(seq))
    return (as_float(seq[0]), as_float(seq[1]), as_float(seq[2]))


def hidden_counts(summary: Dict[str, Any] | None) -> Dict[str, Any]:
    if not summary:
        return {
            "target_total": 0,
            "target_from_Eminus": 0,
            "target_from_E0": 0,
            "target_from_Eplus": 0,
            "total_samples": 0,
            "target_fraction": float("nan"),
            "blocking_equilibrium": "",
            "blocking_radius": "",
            "source_hidden_check_csv": "",
        }
    rows = list(summary.get("summary_by_radius", []))
    target_by_eq = {"E-": 0, "E0": 0, "E+": 0}
    total_samples = 0
    best_eq = ""
    best_radius: Any = ""
    best_hits = 0
    for row in rows:
        eq = str(row.get("equilibrium", row.get("eq", "")))
        target = as_int(row.get("TARGET", row.get("target", 0)))
        row_total = sum(as_int(row.get(k, 0)) for k in ["EQ", "DIV", "TARGET", "OTHER", "UNKNOWN"])
        total_samples += row_total
        if eq in target_by_eq:
            target_by_eq[eq] += target
        if target > best_hits:
            best_hits = target
            best_eq = eq
            best_radius = row.get("radius", "")
    target_total = as_int(summary.get("total_target_hits", sum(target_by_eq.values())))
    if target_total == 0:
        target_total = sum(target_by_eq.values())
    if total_samples <= 0:
        total_samples = as_int(summary.get("n_detail_rows", 0))
    files = summary.get("files", {}) if isinstance(summary.get("files"), dict) else {}
    return {
        "target_total": int(target_total),
        "target_from_Eminus": int(target_by_eq["E-"]),
        "target_from_E0": int(target_by_eq["E0"]),
        "target_from_Eplus": int(target_by_eq["E+"]),
        "total_samples": int(total_samples),
        "target_fraction": float(target_total / total_samples) if total_samples > 0 else float("nan"),
        "blocking_equilibrium": best_eq if best_hits > 0 else "",
        "blocking_radius": best_radius if best_hits > 0 else "",
        "source_hidden_check_csv": files.get("csv_out", ""),
    }


def source_hidden_summary(path_value: Any) -> Tuple[str, Dict[str, Any] | None]:
    path = resolve_path(path_value)
    if path and path.exists() and path.is_file():
        try:
            return str(path), read_json(path)
        except Exception:
            return str(path), None
    return str(path_value or ""), None


def priority_for(row: Dict[str, Any], divergence_norm: float) -> Tuple[str, bool]:
    final_norm = as_float(row.get("final_norm"))
    bounded = as_bool(row.get("bounded"))
    target_total = as_int(row.get("target_total", 0))
    target_fraction = as_float(row.get("target_fraction"))
    has_hidden = bool(str(row.get("source_hidden_summary_json", "")).strip()) or str(row.get("hiddenness_status", "")).strip()
    has_minimal = bool(str(row.get("source_continuation_json", "")).strip()) or np.all(np.isfinite([
        as_float(row.get("seed_x")),
        as_float(row.get("seed_y")),
        as_float(row.get("seed_z")),
        as_float(row.get("final_x")),
        as_float(row.get("final_y")),
        as_float(row.get("final_z")),
    ]))
    if bounded is False or (math.isfinite(final_norm) and final_norm > float(divergence_norm)):
        return "invalid_or_divergent", False
    if not has_hidden or not has_minimal:
        return "insufficient_data", True
    if target_total >= 10 or (math.isfinite(target_fraction) and target_fraction >= 0.10):
        return "contact_strong", False
    if 0 < target_total < 10 and (not math.isfinite(target_fraction) or target_fraction < 0.10):
        return "contact_weak", True
    if target_total == 0 and bounded is True:
        return "no_target_under_sample", True
    return "insufficient_data", True


def import_pipeline():
    spec = importlib.util.spec_from_file_location("lure_pipeline_mod", str(ROOT / "unified_nyquist_hidden_pipeline.py"))
    if spec is None or spec.loader is None:
        raise ImportError("No se pudo cargar unified_nyquist_hidden_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lure_pipeline_mod"] = module
    spec.loader.exec_module(module)
    return module


def official_chua_params() -> Dict[str, Any]:
    pipeline = import_pipeline()
    return pipeline.chua_ic_params_from_config(pipeline.CONFIG)


def recompute_missing(row: Dict[str, Any]) -> None:
    q = as_float(row.get("q"))
    branch = as_int(row.get("branch_index"), 0)
    needs = [row.get(k, "") in {"", None} or not math.isfinite(as_float(row.get(k))) for k in ["omega", "k", "A"]]
    seed_missing = not np.all(np.isfinite([as_float(row.get("seed_x")), as_float(row.get("seed_y")), as_float(row.get("seed_z"))]))
    if not (math.isfinite(q) and (any(needs) or seed_missing)):
        return
    try:
        import chua_initial_cond as chua

        p = official_chua_params()
        pairs = chua.find_omega_k_candidates(q, p, wmin=chua.WMIN, wmax=chua.WMAX, nscan=chua.NSCAN)
        pairs = [pair for pair in pairs if chua.is_describing_gain_compatible(pair[1], p)]
        if branch < 0 or branch >= len(pairs):
            return
        omega, k = pairs[branch][:2]
        A = chua.solve_amplitude_from_k(k, p)
        seed, _v, _eig = chua.build_fractional_seed(q, p, omega, k, A)
        row["omega"] = row.get("omega") if math.isfinite(as_float(row.get("omega"))) else float(omega)
        row["k"] = row.get("k") if math.isfinite(as_float(row.get("k"))) else float(k)
        row["A"] = row.get("A") if math.isfinite(as_float(row.get("A"))) else float(A)
        if seed_missing:
            row["seed_x"], row["seed_y"], row["seed_z"] = [float(v) for v in seed]
    except Exception as exc:
        row["notes"] = (str(row.get("notes", "")) + f"; recompute_missing_failed={exc}").strip("; ")


def normalize_record(
    *,
    record: Dict[str, Any],
    source_type: str,
    source_path: Path,
    summary: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    method = str(record.get("method", record.get("describing_function", ""))).strip().lower()
    slug = str(record.get("slug", "")).strip().lower()
    path_text = str(record.get("output_dir", source_path.parent)).lower().replace("\\", "/")
    if not (method in {"classic", "lure", "nyquist_df", ""} or slug == "classic" or "/classic" in path_text):
        return None
    if "machado" in method or "machado" in slug or "/machado" in path_text:
        return None
    chosen = record.get("chosen_branch", {})
    if not isinstance(chosen, dict):
        chosen = {}
    if str(chosen.get("describing_function", method or "classic")).lower() not in {"classic", "lure", "nyquist_df"}:
        return None
    q = as_float(record.get("q", record.get("frac_order", (summary or {}).get("frac_order"))))
    branch = as_int(record.get("branch_index", chosen.get("branch_index", (summary or {}).get("branch_index", 0))))
    seed = vector3(record.get("seed", chosen.get("seed", record.get("xseed", []))))
    final_state = vector3(record.get("final_state_eps1", record.get("final_state", record.get("target_seed", []))))
    final_norm = float(np.linalg.norm(np.asarray(final_state, dtype=float))) if np.all(np.isfinite(final_state)) else float("nan")
    hidden_path_text, hidden = source_hidden_summary(record.get("hidden_summary_json", ""))
    hcounts = hidden_counts(hidden)
    outputs = record.get("outputs", {}) if isinstance(record.get("outputs"), dict) else {}
    cont = record.get("continuation_summary_json", record.get("source_continuation_json", ""))
    row: Dict[str, Any] = {
        "candidate_id": "",
        "source_type": source_type,
        "source_path": str(source_path),
        "model_kind": ((summary or {}).get("model", {}) or {}).get("kind", record.get("model_kind", "")) if isinstance((summary or {}).get("model", {}), dict) else (summary or {}).get("model", ""),
        "df_family": "lure_classic",
        "q": q,
        "branch_index": branch,
        "omega": record.get("omega0", record.get("omega", chosen.get("omega0", chosen.get("omega", "")))),
        "k": record.get("k0", record.get("k", chosen.get("k", ""))),
        "A": record.get("a0", record.get("A", chosen.get("a0", ""))),
        "phase": record.get("phase", record.get("theta", chosen.get("theta", 0.0))),
        "sigma0": record.get("sigma0", 0.0),
        "seed_x": seed[0],
        "seed_y": seed[1],
        "seed_z": seed[2],
        "final_x": final_state[0],
        "final_y": final_state[1],
        "final_z": final_state[2],
        "final_norm": final_norm,
        "continuation_status": record.get("status", "ok" if cont else ""),
        "bounded": bool(math.isfinite(final_norm) and final_norm <= 1.0e5) if math.isfinite(final_norm) else "",
        "hiddenness_status": record.get("hiddenness_status", (hidden or {}).get("hiddenness_status", "")),
        "target_total": hcounts["target_total"] if hidden is not None else record.get("total_target_hits", ""),
        "target_from_Eminus": hcounts["target_from_Eminus"],
        "target_from_E0": hcounts["target_from_E0"],
        "target_from_Eplus": hcounts["target_from_Eplus"],
        "total_samples": hcounts["total_samples"],
        "target_fraction": hcounts["target_fraction"],
        "blocking_equilibrium": hcounts["blocking_equilibrium"],
        "blocking_radius": hcounts["blocking_radius"],
        "source_hidden_summary_json": hidden_path_text,
        "source_hidden_check_csv": hcounts["source_hidden_check_csv"],
        "source_final_attractor_csv": outputs.get("final_attractor_csv", record.get("source_final_attractor_csv", "")),
        "source_continuation_json": str(cont),
        "notes": record.get("notes", "classic Lure candidate extracted from existing results"),
    }
    recompute_missing(row)
    return row


def parse_df_seed_summary(path: Path) -> List[Dict[str, Any]]:
    try:
        data = read_json(path)
    except Exception:
        return []
    rows = []
    for record in data.get("records", []):
        if isinstance(record, dict):
            row = normalize_record(record=record, source_type="df_compare_summary", source_path=path, summary=data)
            if row is not None:
                rows.append(row)
    return rows


def parse_candidate_summary(path: Path) -> List[Dict[str, Any]]:
    try:
        data = read_json(path)
    except Exception:
        return []
    record = data.get("candidate")
    if not isinstance(record, dict):
        return []
    return [row for row in [normalize_record(record=record, source_type="candidate_summary", source_path=path, summary=data)] if row is not None]


def parse_pipeline_or_continuation(path: Path) -> List[Dict[str, Any]]:
    try:
        data = read_json(path)
    except Exception:
        return []
    df = data.get("describing_function", {})
    if not isinstance(df, dict):
        nested = data.get("nyquist_and_continuation", {})
        if isinstance(nested, dict):
            df = nested.get("describing_function", {})
    method = str(df.get("method", data.get("describing_function", "classic"))).lower()
    if "machado" in method:
        return []
    chosen = data.get("chosen_branch", {})
    cont = data.get("continuation", {})
    if not isinstance(chosen, dict) or not isinstance(cont, dict):
        nested = data.get("nyquist_and_continuation", {})
        if isinstance(nested, dict):
            chosen = nested.get("chosen_branch", chosen)
            cont = nested.get("continuation", cont)
    if not isinstance(chosen, dict):
        return []
    record = {
        "method": method if method else "classic",
        "frac_order": data.get("frac_order"),
        "chosen_branch": chosen,
        "seed": chosen.get("seed"),
        "final_state_eps1": cont.get("final_state_eps1"),
        "status": "ok" if cont.get("final_state_eps1") else "",
        "continuation_summary_json": str(path),
    }
    return [row for row in [normalize_record(record=record, source_type="pipeline_summary", source_path=path, summary=data)] if row is not None]


def discover_files(search_roots: Iterable[Any]) -> List[Path]:
    seen: set[Path] = set()
    files: List[Path] = []
    for root_value in search_roots:
        root = resolve_path(root_value, ROOT)
        if not root.exists():
            continue
        for pattern in KNOWN_PATTERNS:
            for path in root.rglob(pattern):
                if path.is_file():
                    rp = path.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        files.append(path)
    return sorted(files, key=lambda p: str(p).lower())


def collect_existing(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_cfg = cfg.get("source", {})
    roots = source_cfg.get("search_roots", ["."])
    files = discover_files(roots)
    rows: List[Dict[str, Any]] = []
    consumed_dirs: set[Path] = set()
    for path in files:
        if path.name == "df_seed_comparison_summary.json":
            parsed = parse_df_seed_summary(path)
            rows.extend(parsed)
            for row in parsed:
                out_dir = Path(str(row.get("source_final_attractor_csv", ""))).parent
                if str(out_dir):
                    consumed_dirs.add(out_dir.resolve())
    for path in files:
        if path.name.startswith("df_candidate_summary_"):
            if path.parent.resolve() in consumed_dirs:
                continue
            rows.extend(parse_candidate_summary(path))
    for path in files:
        if path.suffix.lower() == ".json" and path.name in {"unified_pipeline_summary.json", "unified_continuation_summary.json"}:
            rows.extend(parse_pipeline_or_continuation(path))
        elif path.suffix.lower() == ".json" and path.name.startswith("continuation_summary"):
            if path.parent.resolve() in consumed_dirs:
                continue
            rows.extend(parse_pipeline_or_continuation(path))
    return rows


def q_values_from_config(cfg: Dict[str, Any]) -> List[float]:
    q_values = cfg.get("q_values", [0.99, 0.999, 0.9998])
    if q_values == "auto_from_previous_report":
        found: List[float] = []
        for path in discover_files(cfg.get("source", {}).get("search_roots", ["."])):
            if path.suffix.lower() != ".json":
                continue
            try:
                data = read_json(path)
            except Exception:
                continue
            q = as_float(data.get("frac_order", data.get("q")))
            if math.isfinite(q):
                found.append(q)
        if found:
            return sorted(set(round(x, 8) for x in found))
        return [0.99, 0.999, 0.9998]
    return [float(v) for v in q_values]


def reproduce_candidates(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    pipeline = import_pipeline()
    rows: List[Dict[str, Any]] = []
    outdir = ROOT / str(cfg.get("manifest", {}).get("output_dir", "outputs/lure_route")) / "reproduced"
    for q in q_values_from_config(cfg):
        q = float(q)
        base_cfg = copy.deepcopy(pipeline.CONFIG)
        base_cfg["frac_order"] = q
        pipeline.ensure_current_chua_params(base_cfg)
        raw_pairs = pipeline.chua_ic.find_omega_k_candidates(
            q,
            pipeline.chua_ic.PARAMS,
            wmin=pipeline.chua_ic.WMIN,
            wmax=pipeline.chua_ic.WMAX,
            nscan=pipeline.chua_ic.NSCAN,
        )
        pairs = [pair for pair in raw_pairs if pipeline.chua_ic.is_describing_gain_compatible(pair[1], pipeline.chua_ic.PARAMS)]
        for branch in range(len(pairs)):
            cid = candidate_id_for(q, branch)
            candidate_dir = outdir / f"q_{q_tag(q)}" / f"branch_{branch}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            local_cfg = copy.deepcopy(base_cfg)
            local_cfg["branch_index"] = branch
            slug = f"lure_classic_q_{q_tag(q)}_branch_{branch}"
            local_cfg["outputs"] = pipeline.df_compare_outputs(base_cfg, candidate_dir, slug)
            local_cfg["verify_hidden"]["runtime_dir"] = candidate_dir / "hidden_verify"
            local_cfg["verify_hidden"]["config_path"] = candidate_dir / "hidden_verify" / "config_hidden_verify_frac.json"
            local_cfg["native_dir"] = candidate_dir / "native"
            pipeline.synchronize_runtime_contract(local_cfg)
            record: Dict[str, Any] = {
                "method": "classic",
                "slug": slug,
                "status": "failed",
                "output_dir": str(candidate_dir),
            }
            try:
                nyq = pipeline.compute_nyquist_seed_and_continuation(local_cfg, df_method="classic", branch_index_override=branch)
                traj = np.asarray(nyq["results"][-1]["traj"], dtype=float)
                traj_outputs = pipeline.save_df_candidate_trajectory(candidate_dir, slug, traj)
                hidden_summary = pipeline.run_hidden_verify_with_seed(local_cfg, nyq["final_state"])
                record.update({
                    "status": "ok",
                    "omega0": float(nyq["omega0"]),
                    "k0": float(nyq["k0"]),
                    "a0": float(nyq["a0"]),
                    "seed": np.asarray(nyq["xseed"], dtype=float).tolist(),
                    "final_state_eps1": np.asarray(nyq["final_state"], dtype=float).tolist(),
                    "total_target_hits": int(hidden_summary.get("total_target_hits", 0)),
                    "hiddenness_status": hidden_summary.get("hiddenness_status", ""),
                    "continuation_summary_json": str(local_cfg["outputs"]["cont_json"]),
                    "candidate_summary_json": str(local_cfg["outputs"]["summary_json"]),
                    "hidden_summary_json": str(hidden_summary["files"]["json_out"]),
                    "outputs": traj_outputs,
                    "chosen_branch": nyq["summary"]["chosen_branch"],
                    "notes": "Reproduced by lure_candidate_manifest.py from official pipeline functions.",
                })
                summary = {
                    "model": local_cfg["model"],
                    "params": local_cfg["params"],
                    "frac_order": q,
                    "candidate": record,
                    "nyquist_and_continuation": nyq["summary"],
                    "hidden_verification": hidden_summary,
                }
                Path(local_cfg["outputs"]["summary_json"]).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                record["error"] = str(exc)
            row = normalize_record(record=record, source_type="reproduced_official_pipeline", source_path=candidate_dir, summary={"frac_order": q, "model": base_cfg["model"]})
            if row is not None:
                row["candidate_id"] = cid
                rows.append(row)
    return rows


def assign_ids_and_classify(rows: List[Dict[str, Any]], cfg: Dict[str, Any], refine_contact_strong: bool = False) -> List[Dict[str, Any]]:
    divergence_norm = float(cfg.get("classification", {}).get("divergence_norm", 1.0e5))
    counts: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for row in rows:
        q = as_float(row.get("q"))
        branch = as_int(row.get("branch_index", 0))
        if math.isfinite(q):
            base = candidate_id_for(q, branch)
        else:
            base = str(row.get("candidate_id") or "lure_unknown")
        idx = counts.get(base, 0)
        counts[base] = idx + 1
        row["candidate_id"] = base if idx == 0 else f"{base}_rep{idx:02d}"
        priority, should = priority_for(row, divergence_norm)
        if refine_contact_strong and priority == "contact_strong":
            should = True
        row["priority_class"] = priority
        row["should_refine"] = bool(should)
        out.append(row)
    out.sort(key=lambda r: (as_float(r.get("q"), 999.0), as_int(r.get("branch_index"), 999), str(r.get("candidate_id"))))
    return out


def build_manifest(cfg: Dict[str, Any], source: str, refine_contact_strong: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
    source = source.strip().lower()
    files_written: List[str] = []
    rows: List[Dict[str, Any]] = []
    if source in {"auto", "existing"}:
        rows = collect_existing(cfg)
    if source == "reproduce" or (source == "auto" and not rows and bool(cfg.get("source", {}).get("reproduce_if_missing", True))):
        rows = reproduce_candidates(cfg)
    rows = assign_ids_and_classify(rows, cfg, refine_contact_strong=refine_contact_strong)
    outdir = ROOT / str(cfg.get("manifest", {}).get("output_dir", "outputs/lure_route"))
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "lure_candidates_manifest.csv"
    json_path = outdir / "lure_candidates_manifest.json"
    write_csv(csv_path, rows, MANIFEST_FIELDS)
    summary = {
        "source_mode": source,
        "rows": rows,
        "fields": MANIFEST_FIELDS,
        "notes": [
            "df_family is normalized to lure_classic for classical Lure/Nyquist describing-function candidates.",
            "Machado candidates are excluded from this manifest.",
        ],
    }
    json_path.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    files_written.extend([str(csv_path), str(json_path)])
    return rows, files_written


def print_final_table(rows: Sequence[Dict[str, Any]], files_written: Sequence[str]) -> None:
    print("candidate_id,q,branch_index,priority_class,target_total,target_from_Eminus,target_from_Eplus,rho_H,rhoH_class,hiddenness_status,should_refine,files_written", flush=True)
    files = ";".join(str(p) for p in files_written)
    for row in rows:
        print(",".join([
            str(row.get("candidate_id", "")),
            str(row.get("q", "")),
            str(row.get("branch_index", "")),
            str(row.get("priority_class", "")),
            str(row.get("target_total", "")),
            str(row.get("target_from_Eminus", "")),
            str(row.get("target_from_Eplus", "")),
            "",
            "",
            str(row.get("hiddenness_status", "")),
            str(row.get("should_refine", "")),
            files,
        ]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a normalized manifest for classical Lure candidates.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--source", choices=["auto", "existing", "reproduce"], default=None)
    parser.add_argument("--refine-contact-strong", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = args.source or str(cfg.get("source", {}).get("mode", "auto"))
    rows, files_written = build_manifest(cfg, source, refine_contact_strong=bool(args.refine_contact_strong))
    print_final_table(rows, files_written)


if __name__ == "__main__":
    main()
