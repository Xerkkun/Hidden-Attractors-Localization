#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

import chua_initial_cond as chua
import unified_nyquist_hidden_pipeline as pipe


ROOT = Path(__file__).resolve().parent
DEFAULT_CANDIDATE = "lure_q_0p99000_branch_0_rep01"
MANIFEST = ROOT / "outputs" / "lure_route" / "lure_candidates_manifest.csv"

CLASS_COLORS = {
    "target_attractor": "#dc2626",
    "TARGET": "#dc2626",
    "equilibrium_convergence": "#111827",
    "other_bounded_nontrivial": "#2563eb",
    "divergent": "#93c5fd",
    "ambiguous_long_transient": "#f97316",
    "numerical_failure": "#d1d5db",
}

EQ_COLORS = {
    "E-": "#0f172a",
    "E0": "#4b5563",
    "E+": "#7f1d1d",
}

SPHERE_COLORS = {
    "E-": "#3b82f6",
    "E0": "#9ca3af",
    "E+": "#ef4444",
}

CLASS_LABELS = {
    "target_attractor": "TARGET",
    "equilibrium_convergence": "equilibrio",
    "other_bounded_nontrivial": "otro destino acotado",
    "divergent": "divergente",
    "ambiguous_long_transient": "transitorio ambiguo",
    "numerical_failure": "fallo numerico",
}

FIG_FACE = "#cfd2d6"
AX_FACE = "#cfd2d6"
GRID_COLOR = "#8b9098"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def as_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "si", "on"}


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def resolve_path(value: Any, base: Path = ROOT) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    p = Path(text)
    if p.exists():
        return p
    q = base / text
    if q.exists():
        return q
    return p if p.is_absolute() else q


def load_candidate(candidate_id: str) -> Dict[str, str]:
    for row in read_csv_rows(MANIFEST):
        if row.get("candidate_id") == candidate_id:
            return row
    raise RuntimeError(f"No se encontro {candidate_id} en {MANIFEST}")


def load_xyz_csv(path: Path) -> np.ndarray:
    rows = read_csv_rows(path)
    if not rows:
        raise RuntimeError(f"{path} no contiene filas.")
    cols = {c.lower(): c for c in rows[0].keys()}
    xk = cols.get("x") or cols.get("final_x")
    yk = cols.get("y") or cols.get("final_y")
    zk = cols.get("z") or cols.get("final_z")
    tk = cols.get("t") or cols.get("time")
    if not (xk and yk and zk):
        raise RuntimeError(f"{path} no tiene columnas x/y/z.")
    out: List[List[float]] = []
    for idx, row in enumerate(rows):
        x, y, z = as_float(row.get(xk)), as_float(row.get(yk)), as_float(row.get(zk))
        if not np.all(np.isfinite([x, y, z])):
            continue
        t = as_float(row.get(tk), float(idx)) if tk else float(idx)
        out.append([t, x, y, z])
    if not out:
        raise RuntimeError(f"{path} no tiene puntos finitos.")
    return np.asarray(out, dtype=float)


def load_xyz_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    for key in ("traj", "trajectory", "attractor", "arr_0"):
        if key in data:
            arr = np.asarray(data[key], dtype=float)
            break
    else:
        raise RuntimeError(f"{path} no contiene una trayectoria reconocible.")
    if arr.ndim != 2:
        raise RuntimeError(f"{path} no contiene una matriz 2D.")
    if arr.shape[1] == 3:
        t = np.arange(arr.shape[0], dtype=float).reshape(-1, 1)
        arr = np.hstack([t, arr])
    if arr.shape[1] < 4:
        raise RuntimeError(f"{path} no contiene columnas t,x,y,z.")
    arr = arr[:, :4]
    return arr[np.all(np.isfinite(arr[:, 1:4]), axis=1)]


def find_candidate_attractor(candidate: Dict[str, str], output_dir: Path) -> Tuple[np.ndarray, str, bool]:
    source = resolve_path(candidate.get("source_final_attractor_csv"))
    if source and source.exists():
        return load_xyz_csv(source), str(source), False

    search_roots = [
        ROOT / "df_seed_comparison" / "classic",
        ROOT / "chua_piecewise",
        ROOT / "runs_machado_df_compare_verify2" / "chua_piecewise" / "df_seed_comparison" / "classic",
        ROOT / "runs_machado_df_compare_verify" / "chua_piecewise",
        ROOT,
    ]
    patterns = ["**/final_attractor*.csv", "**/final_attractor*.npz"]
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                try:
                    if path.suffix.lower() == ".csv":
                        return load_xyz_csv(path), str(path), False
                    if path.suffix.lower() == ".npz":
                        return load_xyz_npz(path), str(path), False
                except Exception:
                    continue

    q = as_float(candidate.get("q"), 0.99)
    h = 0.01
    Lm = 40.0
    t_final = 1500.0
    seed = np.asarray(
        [
            as_float(candidate.get("final_x"), as_float(candidate.get("seed_x"))),
            as_float(candidate.get("final_y"), as_float(candidate.get("seed_y"))),
            as_float(candidate.get("final_z"), as_float(candidate.get("seed_z"))),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(seed)):
        raise RuntimeError("No hay atractor previo ni semilla finita para reintegrar el candidato.")
    p = pipe.chua_ic_params_from_config(pipe.CONFIG)
    chua.PARAMS = p
    chua.QORD = np.float64(q)
    traj = pipe.integrate_efork3_c(seed, p, qord=q, h=h, Lm=Lm, t_total=t_final)
    path = output_dir / f"reconstructed_candidate_attractor_{candidate['candidate_id']}.csv"
    write_csv(
        path,
        [{"t": float(r[0]), "x": float(r[1]), "y": float(r[2]), "z": float(r[3])} for r in traj],
        ["t", "x", "y", "z"],
    )
    return traj, str(path), True


def parse_equilibria(rows: Sequence[Dict[str, str]]) -> Dict[str, np.ndarray]:
    eqs: Dict[str, np.ndarray] = {}
    for row in rows:
        eq_id = row.get("eq_id", "")
        vec = np.asarray([as_float(row.get("x")), as_float(row.get("y")), as_float(row.get("z"))], dtype=float)
        if eq_id and np.all(np.isfinite(vec)):
            eqs[eq_id] = vec
    return eqs


def selected_radii(raw_rows: Sequence[Dict[str, str]], direction_rows: Sequence[Dict[str, str]], candidate_id: str) -> List[float]:
    vals = [as_float(r.get("rho")) for r in raw_rows if r.get("candidate_id") == candidate_id]
    vals = [v for v in vals if math.isfinite(v) and v > 0.0]
    original = [
        as_float(r.get("radius_contact"))
        for r in direction_rows
        if r.get("candidate_id") == candidate_id and math.isfinite(as_float(r.get("radius_contact"))) and as_float(r.get("radius_contact")) > 0.0
    ]
    out: List[float] = []
    if vals:
        out.append(min(vals))
    if original:
        out.append(float(np.mean(original)))
    elif vals:
        out.append(float(np.mean(vals)))
    if vals:
        out.append(max(vals))
    unique: List[float] = []
    for v in out:
        if not any(abs(v - u) <= 1.0e-10 for u in unique):
            unique.append(v)
    return unique


def balanced_rows(rows: Sequence[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    if limit <= 0:
        return []
    chosen: List[Dict[str, str]] = []
    seen: set[int] = set()

    targets = [r for r in rows if truthy(r.get("target_hit")) or r.get("final_class") == "target_attractor"]
    for row in targets:
        if len(chosen) >= limit:
            return chosen
        chosen.append(row)
        seen.add(id(row))

    per_class_cap = {
        "equilibrium_convergence": 20,
        "other_bounded_nontrivial": 20,
        "divergent": 20,
        "ambiguous_long_transient": 20,
        "numerical_failure": 20,
    }
    for cls, cap in per_class_cap.items():
        pool = [r for r in rows if id(r) not in seen and r.get("final_class") == cls]
        if not pool:
            continue
        picked = round_robin_by_group(pool, min(cap, limit - len(chosen)))
        for row in picked:
            if len(chosen) >= limit:
                return chosen
            chosen.append(row)
            seen.add(id(row))
    return chosen


def round_robin_by_group(rows: Sequence[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("contact_id", ""), row.get("test_type", ""))].append(row)
    for key in groups:
        groups[key].sort(key=lambda r: (as_float(r.get("rho"), 0.0), as_float(r.get("eps_cone"), -1.0), as_float(r.get("phi"), -1.0)))
    ordered_keys = sorted(groups.keys(), key=lambda k: (k[0], 0 if k[1] == "line" else 1, k[1]))
    out: List[Dict[str, str]] = []
    while len(out) < limit and any(groups.values()):
        for key in ordered_keys:
            bucket = groups[key]
            if not bucket:
                continue
            out.append(bucket.pop(0))
            if len(out) >= limit:
                break
    return out


def downsample_traj(traj: np.ndarray, max_points: int) -> np.ndarray:
    arr = np.asarray(traj, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise RuntimeError("Trayectoria invalida.")
    arr = arr[:, :4]
    arr = arr[np.all(np.isfinite(arr[:, 1:4]), axis=1)]
    if arr.shape[0] <= max_points:
        return arr
    idx = np.unique(np.linspace(0, arr.shape[0] - 1, max_points).astype(int))
    return arr[idx]


def truncate_divergent(traj: np.ndarray, divergence_norm: float) -> np.ndarray:
    arr = np.asarray(traj, dtype=float)
    norms = np.linalg.norm(arr[:, 1:4], axis=1)
    bad = np.where(norms > divergence_norm)[0]
    if bad.size == 0:
        return arr
    stop = max(1, int(bad[0]) + 1)
    return arr[:stop]


def reconstruct_probe_trajectories(
    candidate_id: str,
    rows: Sequence[Dict[str, str]],
    output_dir: Path,
    divergence_norm: float = 1.0e5,
) -> Tuple[List[Dict[str, Any]], Path, bool]:
    cache_path = output_dir / f"reconstructed_probe_trajectories_{candidate_id}.npz"
    meta_path = output_dir / f"reconstructed_probe_trajectories_{candidate_id}_meta.json"
    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = np.load(cache_path, allow_pickle=True)
        out: List[Dict[str, Any]] = []
        for item in meta.get("trajectories", []):
            key = item["array_key"]
            if key in data:
                row = dict(item)
                row["traj"] = np.asarray(data[key], dtype=float)
                out.append(row)
        if out:
            return out, cache_path, True

    p = pipe.chua_ic_params_from_config(pipe.CONFIG)
    chua.PARAMS = p
    saved: Dict[str, np.ndarray] = {}
    meta_items: List[Dict[str, Any]] = []
    started = time.time()
    for idx, row in enumerate(rows):
        x0 = np.asarray([as_float(row.get("x0")), as_float(row.get("y0")), as_float(row.get("z0"))], dtype=float)
        if not np.all(np.isfinite(x0)):
            continue
        q = as_float(row.get("q"), 0.99)
        h = as_float(row.get("h"), 0.01)
        Lm = as_float(row.get("memory_length"), 40.0)
        t_final = as_float(row.get("t_final"), 1500.0)
        chua.QORD = np.float64(q)
        print(f"reconstruct {idx + 1}/{len(rows)} {candidate_id} {row.get('test_type')} {row.get('contact_id')} class={row.get('final_class')}", flush=True)
        traj = pipe.integrate_efork3_c(x0, p, qord=q, h=h, Lm=Lm, t_total=t_final)
        traj = truncate_divergent(traj, divergence_norm)
        if not np.all(np.isfinite(traj[:, 1:4])):
            raise RuntimeError("Una trayectoria reconstruida contiene valores no finitos.")
        sampled = downsample_traj(traj, 1800)
        key = f"traj_{idx:03d}"
        saved[key] = sampled
        item = {
            "array_key": key,
            "candidate_id": row.get("candidate_id", ""),
            "contact_id": row.get("contact_id", ""),
            "test_type": row.get("test_type", ""),
            "rho": as_float(row.get("rho")),
            "rho_original": as_float(row.get("rho_original")),
            "eps_cone": row.get("eps_cone", ""),
            "phi": row.get("phi", ""),
            "x0": as_float(row.get("x0")),
            "y0": as_float(row.get("y0")),
            "z0": as_float(row.get("z0")),
            "final_class": row.get("final_class", ""),
            "target_hit": truthy(row.get("target_hit")),
            "h": h,
            "memory_length": Lm,
            "t_final": t_final,
            "points_saved": int(sampled.shape[0]),
        }
        meta_items.append(item)
    np.savez_compressed(cache_path, **saved)
    meta = {
        "candidate_id": candidate_id,
        "created_elapsed_sec": time.time() - started,
        "trajectories": meta_items,
        "note": "Downsampled probe trajectories for plotting only.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    out = []
    for item in meta_items:
        item = dict(item)
        item["traj"] = saved[item["array_key"]]
        out.append(item)
    return out, cache_path, False


def sphere_mesh(center: np.ndarray, radius: float, theta_points: int = 24, phi_points: int = 48) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, math.pi, theta_points)
    phi = np.linspace(0.0, 2.0 * math.pi, phi_points)
    th, ph = np.meshgrid(theta, phi)
    x = center[0] + radius * np.sin(th) * np.cos(ph)
    y = center[1] + radius * np.sin(th) * np.sin(ph)
    z = center[2] + radius * np.cos(th)
    return x, y, z


def set_axes_equal(ax: Any, xyz: np.ndarray | None = None, margin: float = 0.05) -> None:
    if xyz is not None and xyz.size:
        mins = np.nanmin(xyz, axis=0)
        maxs = np.nanmax(xyz, axis=0)
        center = 0.5 * (mins + maxs)
        radius = 0.5 * float(np.max(maxs - mins))
        radius *= 1.0 + margin
    else:
        limits = np.asarray([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
        center = limits.mean(axis=1)
        radius = 0.5 * float(np.max(limits[:, 1] - limits[:, 0])) * (1.0 + margin)
    radius = max(radius, 1.0e-6)
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)


def style_article_axis(ax: Any) -> None:
    ax.set_facecolor(AX_FACE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(AX_FACE)
        axis.pane.set_edgecolor("#6b7280")
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"]["color"] = GRID_COLOR
        axis._axinfo["grid"]["linestyle"] = ":"
        axis._axinfo["grid"]["linewidth"] = 0.8
    ax.tick_params(colors="#111827", labelsize=8, pad=1)
    ax.xaxis.label.set_color("#111827")
    ax.yaxis.label.set_color("#111827")
    ax.zaxis.label.set_color("#111827")
    ax.set_xlabel(r"$x_1$", labelpad=5)
    ax.set_ylabel(r"$x_2$", labelpad=5)
    ax.set_zlabel(r"$x_3$", labelpad=5)


def eq_label(eq_id: str) -> str:
    return {"E-": r"$X_-^*$", "E0": r"$X_0^*$", "E+": r"$X_+^*$"}.get(eq_id, eq_id)


def save_figure(fig: Any, base: Path, formats: Sequence[str]) -> List[str]:
    files: List[str] = []
    for fmt in formats:
        path = base.with_suffix("." + fmt.lower().lstrip("."))
        fig.savefig(path, dpi=300 if fmt.lower() == "png" else None, bbox_inches="tight")
        files.append(str(path))
    return files


def add_spheres(ax: Any, eqs: Dict[str, np.ndarray], eq_ids: Sequence[str], radii: Sequence[float]) -> int:
    count = 0
    alphas = [0.06, 0.14, 0.05]
    for eq_id in eq_ids:
        if eq_id not in eqs:
            continue
        for idx, rho in enumerate(radii):
            if not math.isfinite(rho) or rho <= 0.0:
                continue
            x, y, z = sphere_mesh(eqs[eq_id], float(rho))
            ax.plot_surface(
                x,
                y,
                z,
                color=SPHERE_COLORS.get(eq_id, "#9ca3af"),
                alpha=alphas[min(idx, len(alphas) - 1)],
                linewidth=0.0,
                shade=False,
                zorder=0,
            )
            count += 1
    return count


def draw_equilibria(ax: Any, eqs: Dict[str, np.ndarray]) -> None:
    for eq_id in ["E-", "E0", "E+"]:
        if eq_id not in eqs:
            continue
        eq = eqs[eq_id]
        marker = "o" if eq_id == "E0" else "D"
        ax.scatter([eq[0]], [eq[1]], [eq[2]], s=45, marker=marker, color=EQ_COLORS[eq_id], edgecolor="white", linewidth=0.4, depthshade=False)
        ax.text(eq[0], eq[1], eq[2], " " + eq_label(eq_id), fontsize=9, color="#111827")


def draw_equilibrium_axis(ax: Any, eqs: Dict[str, np.ndarray]) -> None:
    if "E-" in eqs and "E+" in eqs:
        a, b = eqs["E-"], eqs["E+"]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#1d4ed8", lw=1.2, alpha=0.85)


def draw_candidate_H_marker(ax: Any, attractor: np.ndarray) -> None:
    xyz = np.asarray(attractor[:, 1:4], dtype=float)
    if xyz.shape[0] < 10:
        return
    idx = int(0.62 * (xyz.shape[0] - 1))
    p0 = xyz[idx]
    p1 = xyz[min(idx + max(8, xyz.shape[0] // 80), xyz.shape[0] - 1)]
    vec = p1 - p0
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-12:
        vec = np.array([0.0, 0.0, 1.0])
        norm = 1.0
    vec = vec / norm
    ax.quiver(p0[0], p0[1], p0[2], vec[0], vec[1], vec[2], length=1.2, color="#111827", arrow_length_ratio=0.25, linewidth=0.8)
    ax.text(p0[0] + 0.8, p0[1] + 0.15, p0[2] + 0.45, r"$H$", fontsize=11, color="#111827")


def draw_probe_trajs(ax: Any, probes: Sequence[Dict[str, Any]], *, initial_only: bool = False) -> None:
    for item in probes:
        traj = np.asarray(item["traj"], dtype=float)
        if initial_only:
            n = min(traj.shape[0], 260)
            traj = traj[:n]
        cls = item.get("final_class", "")
        color = CLASS_COLORS.get(cls, "#6b7280")
        alpha = 0.28 if cls == "divergent" else 0.55
        ax.plot(traj[:, 1], traj[:, 2], traj[:, 3], color=color, lw=0.65, alpha=alpha)
        ax.scatter([item["x0"]], [item["y0"]], [item["z0"]], color=color, s=10, alpha=0.9, depthshade=False)


def draw_contacts(ax: Any, eq_minus: np.ndarray, direction_rows: Sequence[Dict[str, str]], candidate_id: str) -> None:
    for row in direction_rows:
        if row.get("candidate_id") != candidate_id:
            continue
        p = np.asarray([as_float(row.get("x_contact")), as_float(row.get("y_contact")), as_float(row.get("z_contact"))], dtype=float)
        v = np.asarray([as_float(row.get("v_x")), as_float(row.get("v_y")), as_float(row.get("v_z"))], dtype=float)
        radius = as_float(row.get("radius_contact"))
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(v)) or not math.isfinite(radius):
            continue
        ax.plot([eq_minus[0], p[0]], [eq_minus[1], p[1]], [eq_minus[2], p[2]], color="#111827", lw=0.8, alpha=0.85)
        ax.scatter([p[0]], [p[1]], [p[2]], marker="*", s=55, color="#f59e0b", edgecolor="#111827", linewidth=0.35, depthshade=False)


def draw_raw_initial_points(ax: Any, raw_rows: Sequence[Dict[str, str]], candidate_id: str, *, size: float = 9.0, alpha: float = 0.7) -> None:
    sub = [r for r in raw_rows if r.get("candidate_id") == candidate_id]
    if not sub:
        return
    xs = [as_float(r.get("x0")) for r in sub]
    ys = [as_float(r.get("y0")) for r in sub]
    zs = [as_float(r.get("z0")) for r in sub]
    colors = [CLASS_COLORS.get(r.get("final_class", ""), "#6b7280") for r in sub]
    ax.scatter(xs, ys, zs, s=size, c=colors, alpha=alpha, edgecolor="#111827", linewidth=0.25, depthshade=False)


def draw_outcome_fan(ax: Any, raw_rows: Sequence[Dict[str, str]], candidate_id: str, eq_center: np.ndarray, *, max_segments: int = 240) -> None:
    sub = [r for r in raw_rows if r.get("candidate_id") == candidate_id]
    if len(sub) > max_segments:
        step = max(1, len(sub) // max_segments)
        sub = sub[::step][:max_segments]
    segments_by_class: Dict[str, List[np.ndarray]] = defaultdict(list)
    for row in sub:
        x0 = np.asarray([as_float(row.get("x0")), as_float(row.get("y0")), as_float(row.get("z0"))], dtype=float)
        final = np.asarray([as_float(row.get("final_x")), as_float(row.get("final_y")), as_float(row.get("final_z"))], dtype=float)
        if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(final)):
            continue
        direction = final - x0
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-12:
            continue
        direction = direction / norm
        end = x0 + 0.085 * direction
        segments_by_class[row.get("final_class", "")].append(np.vstack([x0, end]))
    for cls, segments in segments_by_class.items():
        color = CLASS_COLORS.get(cls, "#64748b")
        alpha = 0.18 if cls == "other_bounded_nontrivial" else 0.28
        lc = Line3DCollection(segments, colors=color, linewidths=0.55, alpha=alpha)
        ax.add_collection3d(lc)


def legend_handles(include_target: bool, include_sphere: bool) -> List[Any]:
    handles: List[Any] = [
        Line2D([0], [0], color="#16a34a", lw=1.5, label="candidato"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=EQ_COLORS["E-"], markeredgecolor="white", markersize=7, label="E-"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=EQ_COLORS["E0"], markeredgecolor="white", markersize=7, label="E0"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=EQ_COLORS["E+"], markeredgecolor="white", markersize=7, label="E+"),
    ]
    if include_sphere:
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#93c5fd", alpha=0.45, markersize=8, label="esfera de prueba"))
    handles.extend(
        [
            Line2D([0], [0], color="#2563eb", lw=1.2, label="trayectorias desde vecindad"),
            Line2D([0], [0], color="#2563eb", lw=1.2, label="otro destino acotado"),
        ]
    )
    if include_target:
        handles.append(Line2D([0], [0], color="#dc2626", lw=1.3, label="TARGET"))
    return handles


def plot_overview(
    output_dir: Path,
    candidate_id: str,
    attractor: np.ndarray,
    eqs: Dict[str, np.ndarray],
    probes: Sequence[Dict[str, Any]],
    radii: Sequence[float],
    direction_rows: Sequence[Dict[str, str]],
    formats: Sequence[str],
    show_spheres: bool,
    show_contacts: bool,
    all_equilibria: bool,
) -> Tuple[List[str], int]:
    fig = plt.figure(figsize=(7.2, 5.7), facecolor=FIG_FACE)
    ax = fig.add_subplot(111, projection="3d")
    style_article_axis(ax)
    attr = downsample_traj(attractor, 7000)
    ax.plot(attr[:, 1], attr[:, 2], attr[:, 3], color="#12c51a", lw=2.6, alpha=0.95)
    draw_equilibrium_axis(ax, eqs)
    draw_equilibria(ax, eqs)
    draw_candidate_H_marker(ax, attr)
    sphere_count = 0
    if show_spheres:
        sphere_count = add_spheres(ax, eqs, ["E-", "E0", "E+"] if all_equilibria else ["E-"], radii)
    draw_probe_trajs(ax, probes, initial_only=False)
    if show_contacts and "E-" in eqs:
        draw_contacts(ax, eqs["E-"], direction_rows, candidate_id)
    ax.view_init(elev=19, azim=-69)
    pts = [attr[:, 1:4]]
    pts.extend(np.asarray(item["traj"])[:, 1:4] for item in probes[: min(len(probes), 30)])
    pts.append(np.vstack(list(eqs.values())))
    set_axes_equal(ax, np.vstack(pts))
    ax.legend(handles=legend_handles(any(item.get("target_hit") for item in probes), show_spheres), fontsize=7, loc="upper left", frameon=True)
    files = save_figure(fig, output_dir / f"lure_hiddenness_3d_overview_{candidate_id}", formats)
    plt.close(fig)
    return files, sphere_count


def plot_zoom(
    output_dir: Path,
    candidate_id: str,
    eqs: Dict[str, np.ndarray],
    probes: Sequence[Dict[str, Any]],
    radii: Sequence[float],
    direction_rows: Sequence[Dict[str, str]],
    raw_rows: Sequence[Dict[str, str]],
    formats: Sequence[str],
    zoom_eq: str,
    show_spheres: bool,
    show_contacts: bool,
) -> Tuple[List[str], int]:
    if zoom_eq not in eqs:
        raise RuntimeError(f"No existe {zoom_eq} en equilibria_used.csv")
    fig = plt.figure(figsize=(6.4, 5.5), facecolor=FIG_FACE)
    ax = fig.add_subplot(111, projection="3d")
    style_article_axis(ax)
    draw_equilibria(ax, eqs)
    sphere_count = add_spheres(ax, eqs, [zoom_eq], radii) if show_spheres else 0
    if zoom_eq in eqs:
        draw_outcome_fan(ax, raw_rows, candidate_id, eqs[zoom_eq])
        draw_raw_initial_points(ax, raw_rows, candidate_id, size=7.0, alpha=0.82)
    draw_probe_trajs(ax, probes, initial_only=True)
    if show_contacts and "E-" in eqs:
        draw_contacts(ax, eqs["E-"], direction_rows, candidate_id)
    ax.view_init(elev=20, azim=-42)
    center = eqs[zoom_eq]
    span = max(0.018, 1.7 * max(radii or [0.01]))
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.legend(handles=legend_handles(any(item.get("target_hit") for item in probes), show_spheres), fontsize=7, loc="upper left", frameon=True)
    eq_name = zoom_eq.replace("+", "plus").replace("-", "minus")
    files = save_figure(fig, output_dir / f"lure_hiddenness_3d_zoom_{eq_name}_{candidate_id}", formats)
    plt.close(fig)
    return files, sphere_count


def draw_overview_panel(ax: Any, attractor: np.ndarray, eqs: Dict[str, np.ndarray], probes: Sequence[Dict[str, Any]], radii: Sequence[float]) -> None:
    style_article_axis(ax)
    attr = downsample_traj(attractor, 4500)
    ax.plot(attr[:, 1], attr[:, 2], attr[:, 3], color="#12c51a", lw=3.0, alpha=0.96)
    draw_equilibrium_axis(ax, eqs)
    draw_equilibria(ax, eqs)
    draw_candidate_H_marker(ax, attr)
    add_spheres(ax, eqs, ["E-"], radii)
    draw_probe_trajs(ax, probes[:20], initial_only=False)
    ax.view_init(elev=19, azim=-69)
    pts = [attr[:, 1:4], np.vstack(list(eqs.values()))]
    pts.extend(np.asarray(item["traj"])[:, 1:4] for item in probes[: min(len(probes), 20)])
    set_axes_equal(ax, np.vstack(pts))


def draw_zoom_panel(
    ax: Any,
    candidate_id: str,
    eqs: Dict[str, np.ndarray],
    probes: Sequence[Dict[str, Any]],
    radii: Sequence[float],
    direction_rows: Sequence[Dict[str, str]],
    raw_rows: Sequence[Dict[str, str]],
) -> None:
    style_article_axis(ax)
    draw_equilibria(ax, eqs)
    add_spheres(ax, eqs, ["E-"], radii)
    if "E-" in eqs:
        draw_outcome_fan(ax, raw_rows, candidate_id, eqs["E-"])
        draw_raw_initial_points(ax, raw_rows, candidate_id, size=7.0, alpha=0.82)
    draw_probe_trajs(ax, probes, initial_only=True)
    if "E-" in eqs:
        draw_contacts(ax, eqs["E-"], direction_rows, candidate_id)
    ax.view_init(elev=22, azim=-43)
    center = eqs["E-"]
    span = max(0.018, 1.7 * max(radii or [0.01]))
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)


def plot_article(
    output_dir: Path,
    candidate_id: str,
    attractor: np.ndarray,
    eqs: Dict[str, np.ndarray],
    probes: Sequence[Dict[str, Any]],
    radii: Sequence[float],
    direction_rows: Sequence[Dict[str, str]],
    raw_rows: Sequence[Dict[str, str]],
    formats: Sequence[str],
) -> List[str]:
    fig = plt.figure(figsize=(11.6, 6.0), facecolor=FIG_FACE)
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    draw_overview_panel(ax1, attractor, eqs, probes, radii)
    draw_zoom_panel(ax2, candidate_id, eqs, probes, radii, direction_rows, raw_rows)
    ax1.text2D(0.50, -0.08, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold", ha="center")
    ax2.text2D(0.50, -0.08, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold", ha="center")
    arrow = FancyArrowPatch(
        (0.47, 0.42),
        (0.56, 0.55),
        transform=fig.transFigure,
        arrowstyle="->",
        mutation_scale=14,
        lw=1.2,
        linestyle=(0, (4, 2)),
        color="#111827",
    )
    fig.add_artist(arrow)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.09, top=0.98, wspace=0.02)
    files = save_figure(fig, output_dir / f"lure_hiddenness_3d_article_style_{candidate_id}", formats)
    plt.close(fig)
    return files


def plot_direction_geometry(
    output_dir: Path,
    candidate_id: str,
    raw_rows: Sequence[Dict[str, str]],
    eqs: Dict[str, np.ndarray],
    radii: Sequence[float],
    direction_rows: Sequence[Dict[str, str]],
    formats: Sequence[str],
) -> Tuple[List[str], int]:
    if "E-" not in eqs:
        raise RuntimeError("No existe E- en equilibria_used.csv")
    eq_minus = eqs["E-"]
    mean_radius = radii[1] if len(radii) >= 2 else (radii[0] if radii else 0.01)
    fig = plt.figure(figsize=(6.3, 5.5), facecolor=FIG_FACE)
    ax = fig.add_subplot(111, projection="3d")
    style_article_axis(ax)
    ax.scatter([eq_minus[0]], [eq_minus[1]], [eq_minus[2]], s=55, color=EQ_COLORS["E-"], edgecolor="white", linewidth=0.4, depthshade=False)
    ax.text(eq_minus[0], eq_minus[1], eq_minus[2], " E-", fontsize=8)
    add_spheres(ax, eqs, ["E-"], [mean_radius])
    draw_contacts(ax, eq_minus, direction_rows, candidate_id)
    sub = [r for r in raw_rows if r.get("candidate_id") == candidate_id]
    xs = [as_float(r.get("x0")) for r in sub]
    ys = [as_float(r.get("y0")) for r in sub]
    zs = [as_float(r.get("z0")) for r in sub]
    cs = [CLASS_COLORS.get(r.get("final_class", ""), "#6b7280") for r in sub]
    ax.scatter(xs, ys, zs, s=8, c=cs, alpha=0.58, depthshade=False)
    segments = []
    for row in sub:
        if row.get("test_type") != "cone":
            continue
        p = np.asarray([as_float(row.get("x0")), as_float(row.get("y0")), as_float(row.get("z0"))], dtype=float)
        if np.all(np.isfinite(p)):
            segments.append(np.vstack([eq_minus, p]))
    if segments:
        lc = Line3DCollection(segments[:: max(1, len(segments) // 80)], colors="#64748b", linewidths=0.35, alpha=0.28)
        ax.add_collection3d(lc)
    ax.view_init(elev=20, azim=-42)
    span = max(0.018, 1.6 * max(radii or [mean_radius]))
    ax.set_xlim(eq_minus[0] - span, eq_minus[0] + span)
    ax.set_ylim(eq_minus[1] - span, eq_minus[1] + span)
    ax.set_zlim(eq_minus[2] - span, eq_minus[2] + span)
    ax.legend(handles=legend_handles(any(truthy(r.get("target_hit")) for r in sub), True), fontsize=7, loc="upper left", frameon=True)
    files = save_figure(fig, output_dir / f"lure_Eminus_contact_sphere_directions_{candidate_id}", formats)
    plt.close(fig)
    return files, 1


def plot_projection_cuts(
    output_dir: Path,
    candidate_id: str,
    attractor: np.ndarray,
    probes: Sequence[Dict[str, Any]],
    raw_rows: Sequence[Dict[str, str]],
    eqs: Dict[str, np.ndarray],
    radii: Sequence[float],
    formats: Sequence[str],
) -> List[str]:
    files: List[str] = []
    planes = [
        ("xy", 0, 1, r"$x_1$", r"$x_2$"),
        ("xz", 0, 2, r"$x_1$", r"$x_3$"),
        ("yz", 1, 2, r"$x_2$", r"$x_3$"),
    ]
    attr = downsample_traj(attractor, 7000)[:, 1:4]
    sub_raw = [r for r in raw_rows if r.get("candidate_id") == candidate_id]
    for plane_name, i, j, xlabel, ylabel in planes:
        fig, ax = plt.subplots(figsize=(6.2, 5.2), facecolor=FIG_FACE)
        ax.set_facecolor(AX_FACE)
        ax.grid(True, color=GRID_COLOR, linestyle=":", linewidth=0.8, alpha=0.8)
        ax.plot(attr[:, i], attr[:, j], color="#12c51a", lw=2.2, alpha=0.95, label="candidato")
        for item in probes:
            traj = np.asarray(item["traj"], dtype=float)[:, 1:4]
            cls = item.get("final_class", "")
            color = CLASS_COLORS.get(cls, "#6b7280")
            alpha = 0.45 if cls == "other_bounded_nontrivial" else 0.6
            ax.plot(traj[:, i], traj[:, j], color=color, lw=0.55, alpha=alpha)
        if sub_raw:
            xs = [as_float(r.get(["x0", "y0", "z0"][i])) for r in sub_raw]
            ys = [as_float(r.get(["x0", "y0", "z0"][j])) for r in sub_raw]
            cs = [CLASS_COLORS.get(r.get("final_class", ""), "#6b7280") for r in sub_raw]
            ax.scatter(xs, ys, s=9, c=cs, alpha=0.62, edgecolor="#111827", linewidth=0.2, label="condiciones iniciales")
        for eq_id in ["E-", "E0", "E+"]:
            if eq_id not in eqs:
                continue
            eq = eqs[eq_id]
            ax.scatter(eq[i], eq[j], s=42, marker="D" if eq_id != "E0" else "o", color=EQ_COLORS[eq_id], edgecolor="white", linewidth=0.4, zorder=8)
            ax.text(eq[i], eq[j], " " + eq_label(eq_id), fontsize=9, color="#111827", zorder=9)
        if "E-" in eqs:
            center = eqs["E-"]
            for idx, rho in enumerate(radii):
                if not math.isfinite(rho) or rho <= 0.0:
                    continue
                circle = plt.Circle(
                    (center[i], center[j]),
                    rho,
                    color=SPHERE_COLORS["E-"],
                    fill=False,
                    linewidth=0.8 if idx == 1 else 0.55,
                    alpha=0.65 if idx == 1 else 0.35,
                    linestyle="-" if idx == 1 else "--",
                )
                ax.add_patch(circle)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        handles = [
            Line2D([0], [0], color="#12c51a", lw=2.2, label="candidato"),
            Line2D([0], [0], color="#2563eb", lw=1.1, label="otro destino acotado"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=EQ_COLORS["E-"], markeredgecolor="white", markersize=6, label="E-"),
            Line2D([0], [0], color=SPHERE_COLORS["E-"], lw=0.8, linestyle="--", label="radios de prueba"),
        ]
        ax.legend(handles=handles, fontsize=7, frameon=True, loc="best")
        fig.tight_layout()
        files.extend(save_figure(fig, output_dir / f"lure_hiddenness_cut_{plane_name}_{candidate_id}", formats))
        plt.close(fig)
    return files


def build_report(
    path: Path,
    candidate_id: str,
    files_used: Sequence[str],
    output_files: Sequence[str],
    trajectories: Sequence[Dict[str, Any]],
    raw_rows: Sequence[Dict[str, str]],
    radii: Sequence[float],
    attractor_source: str,
    candidate_reintegrated: bool,
) -> None:
    classes = sorted({item.get("final_class", "") for item in trajectories if item.get("final_class")})
    target_hits = sum(1 for r in raw_rows if r.get("candidate_id") == candidate_id and truthy(r.get("target_hit")))
    initial_points = len([r for r in raw_rows if r.get("candidate_id") == candidate_id])
    lines = [
        "# Lure 3D Hiddenness Plot Report",
        "",
        f"- candidate_id: `{candidate_id}`",
        f"- attractor_source: `{attractor_source}`",
        f"- candidate_reintegrated: `{candidate_reintegrated}`",
        f"- reconstructed_probe_trajectories: `{len(trajectories)}`",
        f"- classes_included: `{', '.join(classes) if classes else 'none'}`",
        f"- sphere_radii: `{', '.join(f'{r:.8g}' for r in radii)}`",
        f"- initial_points_available: `{initial_points}`",
        f"- target_hits_available: `{target_hits}`",
        "",
        "This figure illustrates a finite adaptive local test around E-. It does not declare `hidden_verified`.",
        "The visual label is an attractor candidate and the decision remains compatible with ocultedad only under the finite targeted local test.",
        "",
        "## Files used",
        "",
    ]
    lines.extend(f"- `{p}`" for p in files_used)
    lines.extend(["", "## Output files", ""])
    lines.extend(f"- `{p}`" for p in output_files)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_formats(text: str) -> List[str]:
    out = []
    for part in text.split(","):
        fmt = part.strip().lower().lstrip(".")
        if fmt:
            out.append(fmt)
    return out or ["png", "pdf"]


def main() -> None:
    parser = argparse.ArgumentParser(description="3D article-style plot for Lure adaptive hiddenness diagnostics.")
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE)
    parser.add_argument("--input-dir", default=str(ROOT / "outputs" / "lure_route" / "adaptive_Eminus"))
    parser.add_argument("--previous-dir", default=str(ROOT / "outputs" / "lure_route" / "refined_Eminus_contact_weak"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "lure_route" / "adaptive_Eminus" / "plots"))
    parser.add_argument("--show-spheres", action="store_true", default=True)
    parser.add_argument("--show-contacts", action="store_true", default=True)
    parser.add_argument("--all-equilibria", action="store_true")
    parser.add_argument("--zoom-equilibrium", default="E-")
    parser.add_argument("--max-trajectories", type=int, default=80)
    parser.add_argument("--format", default="png,pdf")
    args = parser.parse_args()

    candidate_id = args.candidate_id
    input_dir = Path(args.input_dir)
    previous_dir = Path(args.previous_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = parse_formats(args.format)

    raw_path = input_dir / "adaptive_contact_raw.csv"
    directions_path = input_dir / "contact_directions.csv"
    eq_path = input_dir / "equilibria_used.csv"
    decision_path = input_dir / "adaptive_contact_decision.csv"
    previous_raw_path = previous_dir / "previous_target_reproduction_raw.csv"
    previous_summary_path = previous_dir / "previous_target_reproduction_summary.csv"

    for required in [raw_path, directions_path, eq_path, decision_path]:
        if not required.exists():
            raise RuntimeError(f"Falta archivo requerido: {required}")

    candidate = load_candidate(candidate_id)
    raw_rows = [r for r in read_csv_rows(raw_path) if r.get("candidate_id") == candidate_id]
    if not raw_rows:
        raise RuntimeError(f"No hay condiciones iniciales para {candidate_id} en {raw_path}")
    direction_rows = read_csv_rows(directions_path)
    eq_rows = read_csv_rows(eq_path)
    eqs = parse_equilibria(eq_rows)
    if not {"E-", "E0", "E+"}.issubset(set(eqs)):
        raise RuntimeError("equilibria_used.csv no contiene E-, E0 y E+.")

    attractor, attractor_source, candidate_reintegrated = find_candidate_attractor(candidate, output_dir)
    if not np.all(np.isfinite(attractor[:, 1:4])):
        raise RuntimeError("El atractor candidato contiene puntos no finitos.")

    selected = balanced_rows(raw_rows, int(args.max_trajectories))
    if not selected:
        raise RuntimeError("No se pudo seleccionar ninguna trayectoria para graficar.")
    probes, probe_cache, cache_reused = reconstruct_probe_trajectories(candidate_id, selected, output_dir)
    if not probes:
        raise RuntimeError("No se reconstruyo ninguna trayectoria finita.")

    radii = selected_radii(raw_rows, direction_rows, candidate_id)
    output_files: List[str] = []
    spheres_plotted = 0
    files, count = plot_overview(
        output_dir,
        candidate_id,
        attractor,
        eqs,
        probes,
        radii,
        direction_rows,
        formats,
        args.show_spheres,
        args.show_contacts,
        args.all_equilibria,
    )
    output_files.extend(files)
    spheres_plotted += count
    files, count = plot_zoom(
        output_dir,
        candidate_id,
        eqs,
        probes,
        radii,
        direction_rows,
        raw_rows,
        formats,
        args.zoom_equilibrium,
        args.show_spheres,
        args.show_contacts,
    )
    output_files.extend(files)
    spheres_plotted += count
    output_files.extend(plot_article(output_dir, candidate_id, attractor, eqs, probes, radii, direction_rows, raw_rows, formats))
    files, count = plot_direction_geometry(output_dir, candidate_id, raw_rows, eqs, radii, direction_rows, formats)
    output_files.extend(files)
    spheres_plotted += count
    output_files.extend(plot_projection_cuts(output_dir, candidate_id, attractor, probes, raw_rows, eqs, radii, formats))

    files_used = [
        str(raw_path),
        str(directions_path),
        str(eq_path),
        str(decision_path),
        str(attractor_source),
        str(probe_cache),
    ]
    if previous_raw_path.exists():
        files_used.append(str(previous_raw_path))
    if previous_summary_path.exists():
        files_used.append(str(previous_summary_path))

    report_path = output_dir / "lure_hiddenness_3d_plot_report.md"
    build_report(report_path, candidate_id, files_used, output_files, probes, raw_rows, radii, attractor_source, candidate_reintegrated)
    output_files.append(str(report_path))

    target_hits = sum(1 for item in probes if item.get("target_hit"))
    print(f"candidate_id,{candidate_id}")
    print(f"attractor_points,{int(downsample_traj(attractor, 7000).shape[0])}")
    print(f"probe_trajectories_plotted,{len(probes)}")
    print(f"initial_points_plotted,{len(raw_rows)}")
    print(f"spheres_plotted,{spheres_plotted}")
    print(f"target_hits_plotted,{target_hits}")
    print(f"probe_cache_reused,{cache_reused}")
    print("output_files")
    for path in output_files:
        print(path)


if __name__ == "__main__":
    main()
