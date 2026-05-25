#!/usr/bin/env python3
"""Write a minimal official protocol contract and optional stage envelope."""

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
from hidden_attractors.protocol_cli import main as protocol_main
from hidden_attractors.workflows.protocol import OFFICIAL_STAGE_ORDER, NumericalContract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS / "examples" / f"minimal_chua_protocol_{timestamp()}",
    )
    parser.add_argument("--q", type=float, default=0.9998)
    parser.add_argument("--h", type=float, default=0.02)
    parser.add_argument("--memory-length", type=float, default=40.0)
    parser.add_argument("--t-transient", type=float, default=40.0)
    parser.add_argument("--t-final", type=float, default=80.0)
    parser.add_argument("--run", action="store_true", help="Emit a seed_generation stage envelope; no long integration is run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = NumericalContract(
        q=args.q,
        h=args.h,
        t_final=args.t_final,
        t_transient=args.t_transient,
        backend="efork_c",
        memory_policy="finite_memory",
        memory_length=args.memory_length,
        hiddenness_radii=(1.0e-4, 1.0e-3),
        samples_per_radius=100,
        sample_growth_per_radius=50,
        random_seed_policy="fixed_reproducible",
        random_seed=20260524,
    )
    errors = contract.validate()
    if errors:
        raise ValueError("; ".join(errors))
    contract_path = args.output_dir / "minimal_chua_protocol.json"
    summary_path = args.output_dir / "seed_generation_summary.json"
    write_json(
        contract_path,
        {
            "system": "fractional_nonsmooth_chua",
            "official_stage_order": list(OFFICIAL_STAGE_ORDER),
            "numerical_contract": contract.to_dict(),
            "scientific_boundary": "describing-function families generate seeds only; hiddenness requires the full protocol.",
        },
    )
    command = [
        sys.executable,
        "-m",
        "hidden_attractors.protocol_cli",
        "generate-seeds",
        "--contract",
        str(contract_path),
        "--output",
        str(summary_path),
    ]
    command_text = subprocess.list2cmdline(command)
    (args.output_dir / "minimal_chua_protocol_command.txt").write_text(command_text + "\n", encoding="utf-8")
    print(f"contract={contract_path}")
    print(f"command={command_text}")
    if args.run:
        protocol_main(command[3:])


if __name__ == "__main__":
    main()
