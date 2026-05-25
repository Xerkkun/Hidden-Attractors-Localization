#!/usr/bin/env python3
"""Generate a line-only non-smooth Chua trajectory using the corrected EFORK backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from hidden_attractors.native.backends import FractionalChuaBackend


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "validation" / "10_diagnostics" / "web_illustration"
DEFAULT_SEED = np.array([3.039383584794975, -0.241686206957716, -6.873467365218827])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--q", type=float, default=0.9998)
    parser.add_argument("--h", type=float, default=0.05)
    parser.add_argument("--memory-length", type=float, default=None)
    parser.add_argument("--t-final", type=float, default=500.0)
    parser.add_argument("--t-burn", type=float, default=250.0)
    parser.add_argument("--initial-state", type=float, nargs=3, default=DEFAULT_SEED.tolist())
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_length = args.t_final if args.memory_length is None else args.memory_length
    initial_state = np.asarray(args.initial_state, dtype=float)

    backend = FractionalChuaBackend.build(output_name="chua_frac_backend_corrected_k3")
    trajectory = backend.integrate_efork3(
        initial_state,
        q=args.q,
        h=args.h,
        Lm=memory_length,
        t_final=args.t_final,
    )
    tail = trajectory[trajectory[:, 0] >= args.t_burn]
    if tail.shape[0] < 2 or not np.all(np.isfinite(tail)):
        raise RuntimeError("The corrected EFORK trajectory did not yield a finite plotting window.")

    tail_range_norm = float(np.linalg.norm(np.ptp(tail[:, 1:4], axis=0)))
    final_norm = float(np.linalg.norm(tail[-1, 1:4]))
    classification = (
        "collapsed_toward_E0"
        if final_norm < 1.0e-5 and tail_range_norm < 1.0e-4
        else "bounded_nontrivial_unclassified"
    )
    figure_path = output_dir / "efork_corrected_trajectory_lines.png"
    fig = plt.figure(figsize=(7.4, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tail[:, 1], tail[:, 2], tail[:, 3], lw=0.42, color="#0f766e", alpha=0.96)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(r"Chua non-smooth: corrected EFORK rerun ($q=0.9998$)")
    ax.grid(True, alpha=0.24)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=240)
    plt.close(fig)

    np.savez_compressed(output_dir / "efork_corrected_trajectory_tail.npz", trajectory=tail)
    summary = {
        "purpose": "Line-only web illustration generated after aligning native EFORK memory evaluation at K2 and K3 with the published reference stages.",
        "method": "corrected native C EFORK-3 finite-memory trajectory",
        "model": "nonsmooth",
        "q": args.q,
        "h": args.h,
        "memory_length": memory_length,
        "t_final": args.t_final,
        "t_burn": args.t_burn,
        "initial_state": initial_state.tolist(),
        "stored_tail_rows": int(tail.shape[0]),
        "classification": classification,
        "final_distance_to_E0": final_norm,
        "tail_range_norm": tail_range_norm,
        "tail_ranges": {
            "x": [float(np.min(tail[:, 1])), float(np.max(tail[:, 1]))],
            "y": [float(np.min(tail[:, 2])), float(np.max(tail[:, 2]))],
            "z": [float(np.min(tail[:, 3])), float(np.max(tail[:, 3]))],
        },
        "figure": figure_path.name,
        "note": "This rerun is not published as an attractor unless a corrected EFORK trajectory remains bounded and nontrivial after transients.",
    }
    (output_dir / "efork_corrected_attractor_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
