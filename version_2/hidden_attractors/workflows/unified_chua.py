"""Library wrapper for the legacy unified fractional-Chua pipeline.

The historical pipeline is still a large script, but installed users should
not need PowerShell ``$env:...`` calls to run scientific modes.  This module
provides a typed configuration object and a console entry point that translate
explicit Python/CLI arguments into the legacy script's accepted command-line
options.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..paths import PROJECT_ROOT


LEGACY_PIPELINE = PROJECT_ROOT / "tools" / "legacy" / "unified_nyquist_hidden_pipeline.py"


@dataclass(frozen=True)
class UnifiedChuaConfig:
    """Explicit numerical contract for the unified Chua workflow."""

    output_dir: Path | None = None
    model: str = "nonsmooth"
    run_mode: str = "balanced"
    q: float = 0.9998
    q_values: Sequence[float] = field(default_factory=tuple)
    h: float | None = None
    memory_length: float | None = None
    t_transient: float | None = None
    t_keep: float | None = None
    basin_grid: tuple[int, int] | None = None
    basin_planes_grid: tuple[int, int] | None = None
    basin_z0: str | float | None = None
    basin_workers: int | None = None
    bif_workers: int | None = None
    native_efork_workers: int | None = None
    verify_nsamples: int | None = None
    verify_radii: Sequence[float] = field(default_factory=tuple)
    machado_mu: float | None = None
    machado_mu_values: Sequence[float] = field(default_factory=tuple)
    machado_sweep_max_candidates: int | None = None
    df_compare_branch_index: int | None = None
    spectral: bool | None = None
    psd: bool | None = None
    tisean: bool | None = None
    lyapunov: bool | None = None
    lyapunov_strict: bool | None = None
    bifurcation: bool | None = None
    basin_planes: bool | None = None
    hidden_illustration: bool | None = None
    style_only: bool | None = None
    native_efork: bool | None = None
    ignore_environment: bool = True

    def to_argv(self) -> list[str]:
        """Return command-line arguments for the maintained legacy entrypoint."""

        args: list[str] = []
        if self.ignore_environment:
            args.append("--ignore-env")
        args.extend(["--model", self.model, "--run-mode", self.run_mode, "--q", f"{float(self.q):.12g}"])
        if self.output_dir is not None:
            args.extend(["--output-dir", str(self.output_dir)])
        if self.q_values:
            args.extend(["--q-values", ",".join(f"{float(v):.12g}" for v in self.q_values)])
        for opt, value in (
            ("--h", self.h),
            ("--memory-length", self.memory_length),
            ("--t-transient", self.t_transient),
            ("--t-keep", self.t_keep),
            ("--basin-z0", self.basin_z0),
            ("--basin-workers", self.basin_workers),
            ("--bif-workers", self.bif_workers),
            ("--native-efork-workers", self.native_efork_workers),
            ("--verify-nsamples", self.verify_nsamples),
            ("--machado-mu", self.machado_mu),
            ("--machado-sweep-max-candidates", self.machado_sweep_max_candidates),
            ("--df-compare-branch-index", self.df_compare_branch_index),
        ):
            if value is not None:
                args.extend([opt, str(value)])
        if self.basin_grid is not None:
            args.extend(["--basin-grid", f"{int(self.basin_grid[0])}x{int(self.basin_grid[1])}"])
        if self.basin_planes_grid is not None:
            args.extend(["--basin-planes-grid", f"{int(self.basin_planes_grid[0])}x{int(self.basin_planes_grid[1])}"])
        if self.verify_radii:
            args.extend(["--verify-radii", ",".join(f"{float(v):.12g}" for v in self.verify_radii)])
        if self.machado_mu_values:
            args.extend(["--machado-mu-values", ",".join(f"{float(v):.12g}" for v in self.machado_mu_values)])
        for base, value in (
            ("spectral", self.spectral),
            ("psd", self.psd),
            ("tisean", self.tisean),
            ("lyapunov", self.lyapunov),
            ("lyapunov-strict", self.lyapunov_strict),
            ("bifurcation", self.bifurcation),
            ("basin-planes", self.basin_planes),
            ("hidden-illustration", self.hidden_illustration),
            ("style-only", self.style_only),
            ("native-efork", self.native_efork),
        ):
            if value is True:
                args.append(f"--{base}")
            elif value is False:
                args.append(f"--no-{base}")
        return args


def run_unified_chua(config: UnifiedChuaConfig, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Execute the unified pipeline with an explicit library configuration."""

    cmd = [sys.executable, str(LEGACY_PIPELINE), *config.to_argv()]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=check, text=True)


def make_parser() -> argparse.ArgumentParser:
    """Return the console parser for ``hidden-attractors-unified-chua``."""

    parser = argparse.ArgumentParser(description="Run the unified fractional Chua pipeline without PowerShell env variables.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--model",
        default="nonsmooth",
        choices=["nonsmooth", "arctan", "piecewise"],
        help="Chua model. 'piecewise' is accepted only as a legacy alias for 'nonsmooth'.",
    )
    parser.add_argument("--run-mode", default="balanced")
    parser.add_argument("--q", type=float, default=0.9998)
    parser.add_argument("--q-values", default="")
    parser.add_argument("--h", type=float)
    parser.add_argument("--memory-length", type=float)
    parser.add_argument("--t-transient", type=float)
    parser.add_argument("--t-keep", type=float)
    parser.add_argument("--basin-grid", default="")
    parser.add_argument("--basin-planes-grid", default="")
    parser.add_argument("--basin-z0")
    parser.add_argument("--basin-workers", type=int)
    parser.add_argument("--bif-workers", type=int)
    parser.add_argument("--native-efork-workers", type=int)
    parser.add_argument("--verify-nsamples", type=int)
    parser.add_argument("--verify-radii", default="")
    parser.add_argument("--machado-mu", type=float)
    parser.add_argument("--machado-mu-values", default="")
    parser.add_argument("--machado-sweep-max-candidates", type=int)
    parser.add_argument("--df-compare-branch-index", type=int)
    for name in ("spectral", "psd", "tisean", "lyapunov", "lyapunov-strict", "bifurcation", "basin-planes", "hidden-illustration", "style-only", "native-efork"):
        parser.add_argument(f"--{name}", dest=name.replace("-", "_"), action="store_true", default=None)
        parser.add_argument(f"--no-{name}", dest=name.replace("-", "_"), action="store_false")
    parser.add_argument("--use-current-env", action="store_true", help="Do not clear existing HIDDEN_ATTRACTORS_* variables before applying CLI options.")
    parser.add_argument("--print-command", action="store_true", help="Print the resolved legacy command instead of running it.")
    return parser


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _parse_grid(raw: str) -> tuple[int, int] | None:
    text = raw.strip().lower().replace("x", ",")
    if not text:
        return None
    parts = [part for part in text.split(",") if part.strip()]
    if len(parts) == 1:
        value = int(parts[0])
        return value, value
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise ValueError("grid values must be N or NxM.")


def config_from_args(args: argparse.Namespace) -> UnifiedChuaConfig:
    """Build :class:`UnifiedChuaConfig` from parsed CLI arguments."""

    return UnifiedChuaConfig(
        output_dir=args.output_dir,
        model="nonsmooth" if args.model == "piecewise" else args.model,
        run_mode=args.run_mode,
        q=float(args.q),
        q_values=_parse_float_list(args.q_values),
        h=args.h,
        memory_length=args.memory_length,
        t_transient=args.t_transient,
        t_keep=args.t_keep,
        basin_grid=_parse_grid(args.basin_grid),
        basin_planes_grid=_parse_grid(args.basin_planes_grid),
        basin_z0=args.basin_z0,
        basin_workers=args.basin_workers,
        bif_workers=args.bif_workers,
        native_efork_workers=args.native_efork_workers,
        verify_nsamples=args.verify_nsamples,
        verify_radii=_parse_float_list(args.verify_radii),
        machado_mu=args.machado_mu,
        machado_mu_values=_parse_float_list(args.machado_mu_values),
        machado_sweep_max_candidates=args.machado_sweep_max_candidates,
        df_compare_branch_index=args.df_compare_branch_index,
        spectral=args.spectral,
        psd=args.psd,
        tisean=args.tisean,
        lyapunov=args.lyapunov,
        lyapunov_strict=args.lyapunov_strict,
        bifurcation=args.bifurcation,
        basin_planes=args.basin_planes,
        hidden_illustration=args.hidden_illustration,
        style_only=args.style_only,
        native_efork=args.native_efork,
        ignore_environment=not bool(args.use_current_env),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Console entrypoint."""

    args = make_parser().parse_args(argv)
    config = config_from_args(args)
    cmd = [sys.executable, str(LEGACY_PIPELINE), *config.to_argv()]
    if args.print_command:
        print(" ".join(str(part) for part in cmd))
        return
    run_unified_chua(config)


__all__ = ["UnifiedChuaConfig", "config_from_args", "run_unified_chua"]


if __name__ == "__main__":
    main()
