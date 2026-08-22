"""Conservative PyPI update checks for the active console application.

Stability: internal
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import math
import ntpath
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import AbstractSet, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import Tag, sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version


DISTRIBUTION_NAME = "hidden-attractors-fo"
PYPI_JSON_URL = f"https://pypi.org/pypi/{DISTRIBUTION_NAME}/json"
PYPI_SIMPLE_INDEX_URL = "https://pypi.org/simple"
MAX_PYPI_RESPONSE_BYTES = 5 * 1024 * 1024
PIP_UPGRADE_PREFIX = (
    sys.executable,
    "-m",
    "pip",
    "--isolated",
    "install",
    "--upgrade",
    "--index-url",
    PYPI_SIMPLE_INDEX_URL,
)
SOURCE_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


class UpdateCheckError(RuntimeError):
    """Raised when PyPI cannot provide a usable release listing."""


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be a finite number greater than zero")
    return seconds


def _installed_version() -> str | None:
    candidates: list[Version] = []
    try:
        installed = metadata.version(DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        installed = None
    if isinstance(installed, str):
        try:
            candidates.append(Version(installed))
        except InvalidVersion:
            pass

    try:
        with SOURCE_PYPROJECT.open("rb") as stream:
            source_metadata = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError):
        source_metadata = None
    if isinstance(source_metadata, Mapping):
        project = source_metadata.get("project")
        if isinstance(project, Mapping) and project.get("name") == DISTRIBUTION_NAME:
            source_version = project.get("version")
            if isinstance(source_version, str):
                try:
                    candidates.append(Version(source_version))
                except InvalidVersion:
                    pass

    return str(max(candidates)) if candidates else None


def _active_python_version() -> Version:
    return Version(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


def _release_has_usable_file(
    files: object,
    *,
    python_version: Version,
    supported_tags: AbstractSet[Tag],
) -> bool:
    if not isinstance(files, list) or not files:
        return False
    for item in files:
        if not isinstance(item, Mapping) or bool(item.get("yanked", False)):
            continue
        requires_python = item.get("requires_python")
        if requires_python:
            if not isinstance(requires_python, str):
                continue
            try:
                python_specifier = SpecifierSet(requires_python)
            except InvalidSpecifier:
                continue
            if not python_specifier.contains(python_version, prereleases=True):
                continue
        package_type = item.get("packagetype")
        if package_type == "sdist":
            return True
        if package_type != "bdist_wheel":
            continue
        filename = item.get("filename")
        if not isinstance(filename, str):
            continue
        try:
            _distribution, _version, _build, wheel_tags = parse_wheel_filename(filename)
        except InvalidWheelFilename:
            continue
        if wheel_tags.intersection(supported_tags):
            return True
    return False


def _select_latest_version(
    payload: Mapping[str, object],
    *,
    allow_prereleases: bool,
    python_version: Version | None = None,
    supported_tags: AbstractSet[Tag] | None = None,
) -> str:
    releases = payload.get("releases")
    if not isinstance(releases, Mapping):
        raise UpdateCheckError("PyPI returned no release map")

    active_python = python_version or _active_python_version()
    active_tags = frozenset(supported_tags) if supported_tags is not None else frozenset(sys_tags())
    candidates: list[Version] = []
    for raw_version, files in releases.items():
        if not isinstance(raw_version, str):
            continue
        try:
            candidate = Version(raw_version)
        except InvalidVersion:
            continue
        if not allow_prereleases and (candidate.is_prerelease or candidate.is_devrelease):
            continue
        if not _release_has_usable_file(
            files,
            python_version=active_python,
            supported_tags=active_tags,
        ):
            continue
        candidates.append(candidate)

    if not candidates:
        release_kind = "usable releases" if allow_prereleases else "usable stable releases"
        raise UpdateCheckError(f"PyPI returned no {release_kind}")
    return str(max(candidates))


def _fetch_pypi_payload(timeout: float) -> Mapping[str, object]:
    request = Request(
        PYPI_JSON_URL,
        headers={"Accept": "application/json", "User-Agent": f"{DISTRIBUTION_NAME}-update-check"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS URL
            final_url = urlparse(response.geturl())
            try:
                final_port = final_url.port
            except ValueError as exc:
                raise UpdateCheckError("PyPI redirected to an invalid URL") from exc
            if (
                final_url.scheme.lower() != "https"
                or (final_url.hostname or "").lower() != "pypi.org"
                or final_url.username is not None
                or final_url.password is not None
                or final_port not in {None, 443}
            ):
                raise UpdateCheckError("PyPI redirected outside the allowed HTTPS origin")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    announced_size = int(content_length)
                except ValueError:
                    announced_size = 0
                if announced_size > MAX_PYPI_RESPONSE_BYTES:
                    raise UpdateCheckError("PyPI response exceeded the allowed size")

            raw_payload = response.read(MAX_PYPI_RESPONSE_BYTES + 1)
            if len(raw_payload) > MAX_PYPI_RESPONSE_BYTES:
                raise UpdateCheckError("PyPI response exceeded the allowed size")
            payload = json.loads(raw_payload.decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise UpdateCheckError(f"network request failed: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise UpdateCheckError("PyPI returned an invalid JSON response") from exc

    if not isinstance(payload, Mapping):
        raise UpdateCheckError("PyPI returned an unexpected response")
    return payload


def _pip_upgrade_command(latest_version: str, *, allow_prereleases: bool) -> list[str]:
    command = list(PIP_UPGRADE_PREFIX)
    if allow_prereleases:
        command.append("--pre")
    command.append(f"{DISTRIBUTION_NAME}=={latest_version}")
    return command


def _manual_command(latest_version: str | None, *, allow_prereleases: bool) -> str:
    executable = str(PIP_UPGRADE_PREFIX[0]).replace('"', '\\"')
    prerelease_option = " --pre" if allow_prereleases else ""
    requirement = (
        f"{DISTRIBUTION_NAME}=={latest_version}"
        if latest_version is not None
        else DISTRIBUTION_NAME
    )
    return (
        f'"{executable}" -m pip --isolated install --upgrade '
        f"--index-url {PYPI_SIMPLE_INDEX_URL}{prerelease_option} {requirement}"
    )


def _is_virtual_environment() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _running_from_windows_console_launcher(
    *,
    os_name: str | None = None,
    argv0: str | None = None,
) -> bool:
    """Return whether the active process is an installed Windows CLI launcher."""

    current_os = os.name if os_name is None else os_name
    current_argv0 = sys.argv[0] if argv0 is None else argv0
    launcher_name = ntpath.basename(current_argv0).lower()
    if launcher_name.endswith(".exe"):
        launcher_name = launcher_name[:-4]
    return current_os == "nt" and launcher_name == "hidden-attractors"


def _confirm_upgrade(latest_version: str) -> bool:
    try:
        answer = input(f"Upgrade {DISTRIBUTION_NAME} to {latest_version} now? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print("\nUpgrade cancelled; no changes were made.")
        return False
    return answer.strip().lower() in {"y", "yes"}


def _print_environment() -> None:
    environment = "virtual environment" if _is_virtual_environment() else "base/system environment"
    print(f"Active Python: {sys.executable}")
    print(f"Environment: {environment}")
    if not _is_virtual_environment():
        print("Note: this interpreter may require user or administrator permissions to upgrade packages.")


def _run_pip_process(command: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
    )


def _query_installed_version_subprocess(*, timeout: float) -> tuple[str | None, str | None]:
    command = [
        sys.executable,
        "-c",
        (
            "from importlib.metadata import version; "
            f"print(version({DISTRIBUTION_NAME!r}))"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=min(timeout, 30.0),
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return None, "post-install version check timed out"
    except OSError as exc:
        return None, f"post-install version check could not start: {exc}"

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "unknown error").strip()
        return None, f"post-install version check failed: {details}"
    lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    if not lines:
        return None, "post-install version check returned no version"
    return lines[-1], None


def _run_pip_upgrade(
    timeout: float,
    *,
    latest_version: str,
    allow_prereleases: bool,
) -> int:
    command = _pip_upgrade_command(latest_version, allow_prereleases=allow_prereleases)
    try:
        completed = _run_pip_process(command, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"Update failed: pip exceeded the {timeout:g}-second install timeout.", file=sys.stderr)
        print("pip may have made partial changes before the timeout; inspect the active environment.", file=sys.stderr)
        print(f"Manual command: {_manual_command(latest_version, allow_prereleases=allow_prereleases)}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Update failed: pip could not be started ({exc}).", file=sys.stderr)
        print(f"Manual command: {_manual_command(latest_version, allow_prereleases=allow_prereleases)}", file=sys.stderr)
        return 1

    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode == 0:
        observed_version, verification_error = _query_installed_version_subprocess(timeout=timeout)
        try:
            verified = (
                observed_version is not None
                and Version(observed_version) == Version(latest_version)
            )
        except InvalidVersion:
            verified = False
        if verified:
            print(f"Update verified: {DISTRIBUTION_NAME} {observed_version} is installed.")
            print("Restart Python or the CLI before using the new version.")
            return 0
        print(
            "pip exited successfully, but post-install verification did not find "
            f"the requested version {latest_version} (observed: {observed_version or 'unknown'}).",
            file=sys.stderr,
        )
        if verification_error:
            print(verification_error, file=sys.stderr)
        print(f"Manual command: {_manual_command(latest_version, allow_prereleases=allow_prereleases)}", file=sys.stderr)
        return 1

    stderr = (completed.stderr or "").strip()
    if stderr:
        print(stderr, file=sys.stderr)
    lowered = stderr.lower()
    if any(token in lowered for token in ("permission denied", "access is denied", "not permitted")):
        print(
            "The active environment denied write access. Activate a writable virtual environment "
            "or run the manual command with permissions appropriate for that environment.",
            file=sys.stderr,
        )
    else:
        print(
            "pip did not complete the upgrade. Check connectivity, the active environment, and its permissions.",
            file=sys.stderr,
        )
    print("pip may have made partial changes before failing; inspect the active environment.", file=sys.stderr)
    print(f"Manual command: {_manual_command(latest_version, allow_prereleases=allow_prereleases)}", file=sys.stderr)
    return completed.returncode or 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hidden-attractors update",
        description="Check PyPI and optionally update hidden-attractors-fo in the active Python environment.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check version status only; never prompt or install",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="explicitly approve the pip upgrade without an interactive prompt",
    )
    parser.add_argument(
        "--pre",
        action="store_true",
        help="include prereleases when selecting the newest PyPI version",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_seconds,
        default=10.0,
        metavar="SECONDS",
        help="PyPI request timeout (default: 10)",
    )
    parser.add_argument(
        "--install-timeout",
        type=_positive_seconds,
        default=300.0,
        metavar="SECONDS",
        help="maximum pip runtime after approval (default: 300)",
    )
    return parser


def run_update(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.check and args.yes:
        parser.error("--check and --yes cannot be used together")

    _print_environment()
    installed = _installed_version()
    print(f"Active version: {installed or 'unknown (no valid distribution or source metadata)'}")

    try:
        payload = _fetch_pypi_payload(args.timeout)
        latest = _select_latest_version(payload, allow_prereleases=args.pre)
    except UpdateCheckError as exc:
        print(f"Unable to check PyPI: {exc}", file=sys.stderr)
        print("No changes were made.", file=sys.stderr)
        print(f"Manual command: {_manual_command(None, allow_prereleases=args.pre)}", file=sys.stderr)
        return 2

    release_kind = "latest version (prereleases included)" if args.pre else "latest stable version"
    print(f"PyPI {release_kind}: {latest}")

    update_available = installed is None
    if installed is not None:
        try:
            installed_key = Version(installed)
            latest_key = Version(latest)
        except InvalidVersion as exc:
            print(f"Unable to compare versions: {exc}", file=sys.stderr)
            print(f"Manual command: {_manual_command(latest, allow_prereleases=args.pre)}", file=sys.stderr)
            return 2

        if installed_key == latest_key:
            print("Status: the active environment is up to date.")
            return 0
        if installed_key > latest_key:
            print("Status: the active environment is newer than the selected PyPI release.")
            return 0
        update_available = True

    if update_available:
        print("Status: an update is available." if installed is not None else "Status: PyPI installation is available.")

    if args.check:
        print("Check-only mode: no changes were made.")
        print(f"Manual command: {_manual_command(latest, allow_prereleases=args.pre)}")
        return 0

    approved = bool(args.yes)
    if not approved and sys.stdin.isatty():
        approved = _confirm_upgrade(latest)
    if not approved:
        print("No upgrade was approved; no changes were made.")
        print(f"Manual command: {_manual_command(latest, allow_prereleases=args.pre)}")
        return 0

    if _running_from_windows_console_launcher():
        print(
            "Upgrade not started: the active Windows console launcher must exit before "
            "pip can safely replace it.",
            file=sys.stderr,
        )
        print(
            "Close this command, then run the exact interpreter command below from a new prompt.",
            file=sys.stderr,
        )
        print(
            f"Manual command: {_manual_command(latest, allow_prereleases=args.pre)}",
            file=sys.stderr,
        )
        return 1

    print(f"Running: {_manual_command(latest, allow_prereleases=args.pre)}")
    return _run_pip_upgrade(
        args.install_timeout,
        latest_version=latest,
        allow_prereleases=args.pre,
    )


def update_cmd(argv: Sequence[str] | None = None) -> None:
    return_code = run_update(argv)
    if return_code:
        raise SystemExit(return_code)


__all__ = [
    "DISTRIBUTION_NAME",
    "MAX_PYPI_RESPONSE_BYTES",
    "PIP_UPGRADE_PREFIX",
    "PYPI_JSON_URL",
    "PYPI_SIMPLE_INDEX_URL",
    "UpdateCheckError",
    "build_parser",
    "run_update",
    "update_cmd",
]
