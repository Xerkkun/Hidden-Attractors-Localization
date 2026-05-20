#!/usr/bin/env python3
"""Prepare or execute a minimal C-backed fractional Chua protocol.

The default mode writes the exact command and a JSON run contract without
launching a long numerical job. Pass ``--run`` when you want the example to
execute the maintained unified workflow.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hidden_attractors.io import timestamp, write_json
from hidden_attractors.paths import OUTPUTS
from hidden_attractors.workflows.unified_chua import LEGACY_PIPELINE, UnifiedChuaConfig, run_unified_chua


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS / "examples" / f"minimal_chua_protocol_{timestamp()}",
        help="Directory where the protocol contract and workflow outputs are written.",
    )
    parser.add_argument("--model", choices=["piecewise", "arctan"], default="piecewise")
    parser.add_argument("--q", type=float, default=0.9998)
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--memory-length", type=float, default=40.0)
    parser.add_argument("--t-transient", type=float, default=40.0)
    parser.add_argument("--t-keep", type=float, default=20.0)
    parser.add_argument("--basin-grid", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verify-nsamples", type=int, default=32)
    parser.add_argument("--run", action="store_true", help="Execute the unified workflow instead of only writing the command.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> UnifiedChuaConfig:
    return UnifiedChuaConfig(
        output_dir=args.output_dir,
        model=args.model,
        run_mode="balanced",
        q=args.q,
        h=args.h,
        memory_length=args.memory_length,
        t_transient=args.t_transient,
        t_keep=args.t_keep,
        basin_grid=(args.basin_grid, args.basin_grid),
        basin_workers=args.workers,
        bif_workers=args.workers,
        native_efork_workers=args.workers,
        verify_nsamples=args.verify_nsamples,
        spectral=False,
        psd=False,
        tisean=False,
        lyapunov=False,
        bifurcation=False,
        basin_planes=False,
        hidden_illustration=False,
        native_efork=True,
    )


def command_for(config: UnifiedChuaConfig) -> list[str]:
    return [sys.executable, str(LEGACY_PIPELINE), *config.to_argv()]


def main() -> None:
    args = parse_args()
    config = build_config(args)
    command = command_for(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    contract = {
        "purpose": "minimal_fractional_chua_protocol",
        "heavy_numerics": "delegated_to_c_backed_unified_workflow",
        "output_dir": str(args.output_dir),
        "command": command,
        "command_string": subprocess.list2cmdline(command),
        "config": {
            "model": config.model,
            "run_mode": config.run_mode,
            "q": config.q,
            "h": config.h,
            "memory_length": config.memory_length,
            "t_transient": config.t_transient,
            "t_keep": config.t_keep,
            "basin_grid": config.basin_grid,
            "workers": args.workers,
            "verify_nsamples": config.verify_nsamples,
            "native_efork": config.native_efork,
        },
    }
    write_json(args.output_dir / "minimal_chua_protocol.json", contract)
    (args.output_dir / "minimal_chua_protocol_command.txt").write_text(contract["command_string"] + "\n", encoding="utf-8")

    print(f"contract={args.output_dir / 'minimal_chua_protocol.json'}")
    print(f"command={contract['command_string']}")
    if args.run:
        run_unified_chua(config)


if __name__ == "__main__":
    main()
