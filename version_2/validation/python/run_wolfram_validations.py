"""
Runner for Wolfram Language algebraic/numerical validations.

Executes the .wl scripts in validation/wolfram/cases/ via wolframscript and
writes output files to validation/outputs/wolfram/<system_id>/.

Usage (from version_2/ root):
    python validation/python/run_wolfram_validations.py --all
    python validation/python/run_wolfram_validations.py \\
        --case validation/wolfram/cases/chua_fractional_saturation.wl \\
        --out validation/outputs/wolfram/chua_fractional_saturation

Requires wolframscript to be available in PATH.
If wolframscript is not installed, this script fails with a clear message
when called directly. In pytest, tests decorated with @pytest.mark.wolfram
use pytest.skip() instead of failing hard.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Repository-relative paths
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    """Return the absolute path to the version_2/ directory."""
    return Path(__file__).resolve().parents[2]


# Default case paths are relative to repo_root().
DEFAULT_CASES: list[str] = [
    "validation/wolfram/cases/chua_integer_saturation.wl",
    "validation/wolfram/cases/chua_fractional_saturation.wl",
    "validation/wolfram/cases/chua_fractional_arctan.wl",
]

DEFAULT_OUT_BASE = "validation/outputs/wolfram"


# ---------------------------------------------------------------------------
# wolframscript detection
# ---------------------------------------------------------------------------

def find_wolframscript() -> str | None:
    """Return the path to wolframscript, or None if not found."""
    return shutil.which("wolframscript")


def require_wolframscript() -> str:
    """Return the wolframscript executable path, raising RuntimeError if absent."""
    exe = find_wolframscript()
    if exe is None:
        raise RuntimeError(
            "wolframscript not found in PATH.\n"
            "Install Wolfram Engine (free) or Mathematica and ensure that\n"
            "wolframscript is on PATH.\n"
            "See: https://www.wolfram.com/engine/\n\n"
            "The main library does NOT require Wolfram.  These validations\n"
            "are optional and run only on demand."
        )
    return exe


# ---------------------------------------------------------------------------
# Case execution
# ---------------------------------------------------------------------------

def infer_system_id(case_path: Path) -> str:
    """Derive the system_id from the .wl filename stem."""
    return case_path.stem


def run_case(
    case_path: Path,
    out_dir: Path,
    wolframscript: str | None = None,
) -> dict:
    """Run a single Wolfram validation case and return a result dict.

    Parameters
    ----------
    case_path : Path
        Absolute path to the .wl case script.
    out_dir : Path
        Directory where the case writes its output files.
    wolframscript : str or None
        Path to wolframscript; resolved automatically if None.

    Returns
    -------
    dict
        Keys: ``case``, ``out_dir``, ``summary_path``, ``summary``.

    Raises
    ------
    RuntimeError
        If wolframscript is not found or the script returns a non-zero exit code.
    AssertionError
        If the validation_summary.json reports ``passed=false``.
    FileNotFoundError
        If no *_validation_summary.json is produced.
    """
    exe = wolframscript or require_wolframscript()
    case_path = case_path.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [exe, "-file", str(case_path), "--out", str(out_dir)]
    import os
    env = os.environ.copy()
    env["WOLFRAM_OUT"] = str(out_dir)
    completed = subprocess.run(
        cmd,
        cwd=str(repo_root()),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Wolfram validation script failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    summaries = sorted(out_dir.glob("*_validation_summary.json"))
    if not summaries:
        raise FileNotFoundError(
            f"No *_validation_summary.json was generated in {out_dir}.\n"
            "Check the wolframscript output above for errors."
        )

    summary_path = summaries[-1]
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)

    if not summary.get("passed", False):
        raise AssertionError(
            f"Wolfram validation did not pass: {summary_path}\n"
            f"{json.dumps(summary, indent=2, ensure_ascii=False)}"
        )

    return {
        "case": str(case_path),
        "out_dir": str(out_dir),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def run_cases(case_paths: Iterable[Path], base_out: Path) -> list[dict]:
    """Run multiple cases sequentially, re-using a single wolframscript lookup."""
    exe = require_wolframscript()
    results = []
    for case_path in case_paths:
        case_path = case_path.resolve()
        out_dir = base_out / infer_system_id(case_path)
        results.append(run_case(case_path, out_dir, wolframscript=exe))
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Wolfram Language algebraic/numerical validation scripts.\n"
            "wolframscript must be on PATH."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all three built-in cases (integer + fractional saturation + arctan).",
    )
    parser.add_argument(
        "--case",
        type=str,
        metavar="FILE.wl",
        help="Path to a single .wl script (absolute or relative to repo root).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT_BASE,
        metavar="DIR",
        help=(
            f"Output directory base (default: {DEFAULT_OUT_BASE}).  "
            "A subdirectory named after the system_id is created automatically."
        ),
    )
    args = parser.parse_args()

    root = repo_root()
    base_out = (root / args.out).resolve()

    if args.all:
        cases = [root / p for p in DEFAULT_CASES]
        results = run_cases(cases, base_out)
    elif args.case:
        raw = Path(args.case)
        case = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        out = base_out / infer_system_id(case)
        results = [run_case(case, out)]
    else:
        parser.error("Specify --all or --case <file.wl>.")

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
