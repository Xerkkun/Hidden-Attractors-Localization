from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def read_csv_numeric(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def trajectory_array(path: Path) -> np.ndarray:
    rows = read_csv_numeric(path)
    cols = rows[0].keys()
    if {"t", "x", "y", "z"}.issubset(cols):
        return np.array([[float(r["t"]), float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)
    if {"time", "x", "y", "z"}.issubset(cols):
        return np.array([[float(r["time"]), float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)
    raise ValueError(f"No reconozco columnas de trayectoria en {path}")


def find_traj(run_root: Path, branch: str, candidate_id: str) -> Path:
    branch_root = run_root / branch
    candidate_safe = safe_name(candidate_id)

    if not branch_root.exists():
        raise FileNotFoundError(f"No existe la rama: {branch_root}")

    # 1. Buscar cualquier CSV que contenga el candidate_id o su versión safe_name.
    patterns = [
        f"*{candidate_id}*.csv",
        f"*{candidate_safe}*.csv",
    ]

    for pattern in patterns:
        matches = sorted(branch_root.rglob(pattern))
        matches = [
            p for p in matches
            if "selected_candidates" not in p.name
            and "trajectory_metrics" not in p.name
            and "continuation_paths" not in p.name
            and "continuation_summary" not in p.name
        ]
        if matches:
            return matches[0]

    # 2. Si no existe un CSV con el nombre del candidato,
    # usar selected_1.csv, selected_2.csv, selected_3.csv según el rank.
    selected_path = branch_root / "selected_candidates.json"
    if selected_path.exists():
        data = json.loads(selected_path.read_text(encoding="utf-8"))
        for item in data.get("selected_candidates", []):
            if item.get("candidate_id") == candidate_id:
                rank = int(item.get("rank", 0))
                rank_patterns = [
                    f"selected_{rank}.csv",
                    f"*selected_{rank}*.csv",
                    f"rank_{rank}.csv",
                    f"*rank_{rank}*.csv",
                ]

                for pattern in rank_patterns:
                    matches = sorted(branch_root.rglob(pattern))
                    matches = [
                        p for p in matches
                        if "selected_candidates" not in p.name
                    ]
                    if matches:
                        return matches[0]

    # 3. Mostrar diagnóstico útil.
    csv_files = sorted(branch_root.rglob("*.csv"))
    preview = "\n".join(str(p) for p in csv_files[:30])

    raise FileNotFoundError(
        f"No encontré trayectoria para {candidate_id} en {branch_root}.\n"
        f"Primeros CSV encontrados:\n{preview}"
    )

def load_candidate(run_root: Path, branch: str, candidate_id: str) -> dict:
    if branch == "finite_window":
        path = run_root / branch / "selected_candidates.json"
        if not path.exists():
            path = run_root / "finite_window" / "selected_candidates.json"
    else:
        path = run_root / branch / "selected_candidates.json"
    if not path.exists():
        path = run_root / "full_history" / "selected_candidates.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data["selected_candidates"]:
        if item["candidate_id"] == candidate_id:
            return item
    raise KeyError(f"No encontré {candidate_id} en {path}")


def continuation_path(run_root: Path, branch: str, candidate_id: str) -> np.ndarray:
    path = run_root / branch / "continuation_paths.csv"
    rows = [r for r in read_csv_numeric(path) if r.get("candidate_id") == candidate_id]
    if not rows:
        return np.empty((0, 4), dtype=float)
    lam_key = "lambda" if "lambda" in rows[0] else ("eta" if "eta" in rows[0] else None)
    if lam_key:
        rows = sorted(rows, key=lambda r: float(r[lam_key]))
        return np.array([[float(r[lam_key]), float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=float)
    return np.array([[i, float(r["x"]), float(r["y"]), float(r["z"])] for i, r in enumerate(rows)], dtype=float)


def first_harmonic_reconstruction(traj: np.ndarray, tail_fraction: float = 0.85) -> np.ndarray:
    n0 = int((1.0 - tail_fraction) * len(traj))
    tail = traj[n0:, 1:4]
    tail = tail[np.all(np.isfinite(tail), axis=1)]
    n = len(tail)
    if n < 32:
        raise ValueError("Trayectoria demasiado corta para reconstrucción armónica.")
    centered = tail - tail.mean(axis=0)
    fft_x = np.fft.rfft(centered[:, 0])
    k = int(np.argmax(np.abs(fft_x[1:])) + 1)
    coeffs = np.fft.rfft(centered, axis=0)
    keep = np.zeros_like(coeffs)
    keep[0, :] = coeffs[0, :]
    keep[k, :] = coeffs[k, :]
    recon = np.fft.irfft(keep, n=n, axis=0) + tail.mean(axis=0)
    t = traj[n0:n0+n, 0:1]
    return np.hstack([t, recon])


def downsample(arr: np.ndarray, max_points: int) -> np.ndarray:
    if len(arr) <= max_points:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_points).astype(int)
    return arr[idx]


def plot_continuation_story(run_root: Path, branch: str, candidate_id: str, outdir: Path):
    traj = trajectory_array(find_traj(run_root, branch, candidate_id))
    cont = continuation_path(run_root, branch, candidate_id)

    traj_small = downsample(traj, 3500)
    first = downsample(traj[: max(50, min(len(traj), len(traj)//5))], 900)
    final = downsample(traj[int(0.15 * len(traj)):], 6000)

    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    if len(cont) > 0:
        ax.plot(cont[:, 1], cont[:, 2], cont[:, 3], "k--", lw=1.4, label="entrada epsilon")
        ax.plot(cont[:, 1], cont[:, 2], cont[:, 3], ":", color="0.45", lw=1.4, label="salida epsilon")

    ax.plot(first[:, 1], first[:, 2], first[:, 3], color="blue", lw=3.0, label="primer paso")
    ax.plot(final[:, 1], final[:, 2], final[:, 3], color="red", lw=3.0, alpha=0.95, label="paso final")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(candidate_id)
    ax.legend()
    fig.tight_layout()

    png = outdir / f"fig02d_{safe_name(candidate_id)}_continuation_story.png"
    pdf = outdir / f"fig02d_{safe_name(candidate_id)}_continuation_story.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)


def plot_linear_vs_original(run_root: Path, branch: str, candidate_id: str, outdir: Path):
    traj = trajectory_array(find_traj(run_root, branch, candidate_id))
    recon = first_harmonic_reconstruction(traj)

    original = downsample(traj[int(0.15 * len(traj)):], 6000)
    linear = downsample(recon, 1600)

    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(original[:, 1], original[:, 2], original[:, 3], color="red", lw=3.0, alpha=0.95, label="original")
    ax.plot(linear[:, 1], linear[:, 2], linear[:, 3], "--", color="purple", lw=2.5, alpha=0.9, label="linealizada")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(candidate_id)
    ax.legend()
    fig.tight_layout()

    png = outdir / f"fig03g_{safe_name(candidate_id)}_linear_vs_original_3d.png"
    pdf = outdir / f"fig03g_{safe_name(candidate_id)}_linear_vs_original_3d.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--branch", default="full_history")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_continuation_story(run_root, args.branch, args.candidate_id, outdir)
    plot_linear_vs_original(run_root, args.branch, args.candidate_id, outdir)

    print(f"Listo: {args.candidate_id} -> {outdir}")


if __name__ == "__main__":
    main()
