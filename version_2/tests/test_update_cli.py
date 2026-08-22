from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest
from packaging.tags import Tag
from packaging.version import Version

from hidden_attractors.cli import update as update_module


def _payload(**release_yanked: bool) -> dict[str, object]:
    return {
        "releases": {
            version: [
                {
                    "yanked": yanked,
                    "packagetype": "bdist_wheel",
                    "filename": f"hidden_attractors_fo-{version}-py3-none-any.whl",
                    "requires_python": ">=3.11",
                }
            ]
            for version, yanked in release_yanked.items()
        }
    }


def _write_source_pyproject(path: Path, *, name: str, version: str) -> None:
    path.write_text(
        "[project]\n"
        f'name = "{name}"\n'
        f'version = "{version}"\n',
        encoding="utf-8",
    )


@pytest.mark.cli
def test_active_version_falls_back_to_valid_adjacent_source_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write_source_pyproject(pyproject, name="hidden-attractors-fo", version="1.2.0")
    monkeypatch.setattr(update_module, "SOURCE_PYPROJECT", pyproject)
    monkeypatch.setattr(
        update_module.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(update_module.metadata.PackageNotFoundError(name)),
    )

    assert update_module._installed_version() == "1.2.0"


@pytest.mark.cli
@pytest.mark.parametrize(
    ("distribution_version", "source_version", "expected"),
    [("1.0.0", "1.2.0", "1.2.0"), ("1.3.0", "1.2.0", "1.3.0")],
)
def test_active_version_uses_highest_valid_distribution_or_source_version(
    tmp_path: Path,
    monkeypatch,
    distribution_version: str,
    source_version: str,
    expected: str,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    _write_source_pyproject(
        pyproject,
        name="hidden-attractors-fo",
        version=source_version,
    )
    monkeypatch.setattr(update_module, "SOURCE_PYPROJECT", pyproject)
    monkeypatch.setattr(update_module.metadata, "version", lambda name: distribution_version)

    assert update_module._installed_version() == expected


@pytest.mark.cli
@pytest.mark.parametrize(
    "content",
    [
        '[project]\nname = "different-project"\nversion = "9.0.0"\n',
        '[project]\nname = "hidden-attractors-fo"\nversion = "not a version"\n',
        '[project\nname = "hidden-attractors-fo"\n',
    ],
)
def test_source_version_fallback_rejects_wrong_name_or_malformed_metadata(
    tmp_path: Path, monkeypatch, content: str
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(content, encoding="utf-8")
    monkeypatch.setattr(update_module, "SOURCE_PYPROJECT", pyproject)
    monkeypatch.setattr(
        update_module.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(update_module.metadata.PackageNotFoundError(name)),
    )

    assert update_module._installed_version() is None


@pytest.mark.cli
def test_latest_version_ignores_prereleases_and_yanked_files_by_default() -> None:
    payload = _payload(**{"1.2.0": False, "1.2.1": True, "1.3.0rc1": False})

    python_version = Version("3.12")
    assert update_module._select_latest_version(
        payload,
        allow_prereleases=False,
        python_version=python_version,
    ) == "1.2.0"
    assert update_module._select_latest_version(
        payload,
        allow_prereleases=True,
        python_version=python_version,
    ) == "1.3.0rc1"


@pytest.mark.cli
def test_latest_version_requires_compatible_python_and_supported_file_type() -> None:
    payload = {
        "releases": {
            "1.2.0": [
                {
                    "yanked": False,
                    "packagetype": "sdist",
                    "requires_python": ">=3.11",
                }
            ],
            "1.3.0": [
                {
                    "yanked": False,
                    "packagetype": "bdist_wheel",
                    "requires_python": ">=3.15",
                }
            ],
            "1.4.0": [
                {
                    "yanked": False,
                    "packagetype": "bdist_egg",
                    "requires_python": ">=3.11",
                }
            ],
            "1.5.0": [
                {
                    "yanked": False,
                    "packagetype": "bdist_wheel",
                    "filename": "hidden_attractors_fo-1.5.0-cp310-cp310-win_amd64.whl",
                    "requires_python": ">=3.11",
                }
            ],
        }
    }

    assert update_module._select_latest_version(
        payload,
        allow_prereleases=False,
        python_version=Version("3.14.7"),
        supported_tags={Tag("py3", "none", "any")},
    ) == "1.2.0"


@pytest.mark.cli
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_timeout_must_be_positive_and_finite(value: str) -> None:
    with pytest.raises(Exception, match="finite number greater than zero"):
        update_module._positive_seconds(value)


@pytest.mark.cli
def test_check_only_reports_update_without_starting_pip(monkeypatch, capsys) -> None:
    monkeypatch.setattr(update_module, "_installed_version", lambda: "1.1.0")
    monkeypatch.setattr(
        update_module,
        "_fetch_pypi_payload",
        lambda timeout: _payload(**{"1.2.0": False, "1.3.0rc1": False}),
    )
    monkeypatch.setattr(
        update_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("pip must not run in check-only mode"),
    )

    assert update_module.run_update(["--check"]) == 0
    captured = capsys.readouterr()
    assert "Active version: 1.1.0" in captured.out
    assert "PyPI latest stable version: 1.2.0" in captured.out
    assert "Status: an update is available." in captured.out
    assert "no changes were made" in captured.out.lower()


@pytest.mark.cli
def test_noninteractive_default_never_starts_pip(monkeypatch, capsys) -> None:
    monkeypatch.setattr(update_module, "_installed_version", lambda: "1.1.0")
    monkeypatch.setattr(
        update_module,
        "_fetch_pypi_payload",
        lambda timeout: _payload(**{"1.2.0": False}),
    )
    monkeypatch.setattr(update_module.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(
        update_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("pip requires explicit approval"),
    )

    assert update_module.run_update([]) == 0
    captured = capsys.readouterr()
    assert "No upgrade was approved" in captured.out
    assert "Manual command:" in captured.out


@pytest.mark.cli
def test_local_release_candidate_newer_than_pypi_is_never_downgraded(monkeypatch, capsys) -> None:
    monkeypatch.setattr(update_module, "_installed_version", lambda: "1.2.0")
    monkeypatch.setattr(
        update_module,
        "_fetch_pypi_payload",
        lambda timeout: _payload(**{"1.0.0": False}),
    )
    monkeypatch.setattr(
        update_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("a newer local version must never be downgraded"),
    )

    assert update_module.run_update(["--yes"]) == 0
    captured = capsys.readouterr()
    assert "active environment is newer" in captured.out


@pytest.mark.cli
def test_yes_runs_exact_pip_upgrade_without_a_shell(monkeypatch, capsys) -> None:
    monkeypatch.setattr(update_module, "_installed_version", lambda: "1.1.0")
    monkeypatch.setattr(
        update_module,
        "_fetch_pypi_payload",
        lambda timeout: _payload(**{"1.2.0": False}),
    )
    expected = [
        update_module.sys.executable,
        "-m",
        "pip",
        "--isolated",
        "install",
        "--upgrade",
        "--index-url",
        "https://pypi.org/simple",
        "hidden-attractors-fo==1.2.0",
    ]
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="upgrade ok\n", stderr="")

    monkeypatch.setattr(update_module, "_run_pip_process", fake_run)
    monkeypatch.setattr(
        update_module,
        "_query_installed_version_subprocess",
        lambda **kwargs: ("1.2.0", None),
    )

    assert update_module.run_update(["--yes", "--install-timeout", "12"]) == 0
    assert observed["command"] == expected
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["timeout"] == 12.0
    captured = capsys.readouterr()
    assert "Update verified" in captured.out


@pytest.mark.cli
def test_windows_console_launcher_detection_is_exact() -> None:
    assert update_module._running_from_windows_console_launcher(
        os_name="nt", argv0=r"C:\venv\Scripts\hidden-attractors.exe"
    )
    assert update_module._running_from_windows_console_launcher(
        os_name="nt", argv0=r"C:\venv\Scripts\hidden-attractors"
    )
    assert update_module._running_from_windows_console_launcher(
        os_name="nt", argv0="C:/venv/Scripts/hidden-attractors.exe"
    )
    assert not update_module._running_from_windows_console_launcher(
        os_name="nt", argv0=r"C:\venv\Scripts\python.exe"
    )
    assert not update_module._running_from_windows_console_launcher(
        os_name="nt", argv0=r"C:\venv\Scripts\pytest"
    )
    assert not update_module._running_from_windows_console_launcher(
        os_name="posix", argv0="/venv/bin/hidden-attractors"
    )


@pytest.mark.cli
def test_windows_console_launcher_refuses_self_replacement(monkeypatch, capsys) -> None:
    monkeypatch.setattr(update_module, "_installed_version", lambda: "1.1.0")
    monkeypatch.setattr(
        update_module,
        "_fetch_pypi_payload",
        lambda timeout: _payload(**{"1.2.0": False}),
    )
    monkeypatch.setattr(
        update_module,
        "_running_from_windows_console_launcher",
        lambda: True,
    )
    monkeypatch.setattr(
        update_module,
        "_run_pip_process",
        lambda *args, **kwargs: pytest.fail("an active Windows launcher must not invoke pip"),
    )

    assert update_module.run_update(["--yes"]) == 1
    captured = capsys.readouterr()
    assert "active Windows console launcher must exit" in captured.err
    assert "Manual command:" in captured.err
    assert "--isolated install --upgrade" in captured.err
    assert "hidden-attractors-fo==1.2.0" in captured.err


@pytest.mark.cli
def test_pre_approval_passes_pre_flag_to_pip(monkeypatch) -> None:
    monkeypatch.setattr(update_module, "_installed_version", lambda: "1.2.0")
    monkeypatch.setattr(
        update_module,
        "_fetch_pypi_payload",
        lambda timeout: _payload(**{"1.2.0": False, "1.3.0rc1": False}),
    )
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(update_module, "_run_pip_process", fake_run)
    monkeypatch.setattr(
        update_module,
        "_query_installed_version_subprocess",
        lambda **kwargs: ("1.3.0rc1", None),
    )

    assert update_module.run_update(["--pre", "--yes"]) == 0
    assert observed["command"] == [
        update_module.sys.executable,
        "-m",
        "pip",
        "--isolated",
        "install",
        "--upgrade",
        "--index-url",
        "https://pypi.org/simple",
        "--pre",
        "hidden-attractors-fo==1.3.0rc1",
    ]


@pytest.mark.cli
def test_post_install_verifier_uses_fresh_process_and_no_shell(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="1.2.0\n", stderr="")

    monkeypatch.setattr(update_module.subprocess, "run", fake_run)

    version, error = update_module._query_installed_version_subprocess(timeout=7.0)
    assert version == "1.2.0"
    assert error is None
    command = observed["command"]
    assert isinstance(command, list)
    assert command[:2] == [update_module.sys.executable, "-c"]
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 7.0


@pytest.mark.cli
def test_pip_success_requires_exact_post_install_version(monkeypatch, capsys) -> None:
    monkeypatch.setattr(update_module, "_installed_version", lambda: "1.1.0")
    monkeypatch.setattr(
        update_module,
        "_fetch_pypi_payload",
        lambda timeout: _payload(**{"1.2.0": False}),
    )
    monkeypatch.setattr(
        update_module,
        "_run_pip_process",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="pip reported success",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        update_module,
        "_query_installed_version_subprocess",
        lambda **kwargs: ("1.1.0", None),
    )

    assert update_module.run_update(["--yes"]) == 1
    captured = capsys.readouterr()
    assert "post-install verification did not find" in captured.err
    assert "requested version 1.2.0" in captured.err


@pytest.mark.cli
def test_interactive_confirmation_is_required(monkeypatch) -> None:
    monkeypatch.setattr(update_module, "_installed_version", lambda: "1.1.0")
    monkeypatch.setattr(
        update_module,
        "_fetch_pypi_payload",
        lambda timeout: _payload(**{"1.2.0": False}),
    )
    monkeypatch.setattr(update_module.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    monkeypatch.setattr(
        update_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("declining confirmation must not start pip"),
    )

    assert update_module.run_update([]) == 0


@pytest.mark.cli
def test_offline_check_is_non_destructive_and_offers_manual_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(update_module, "_installed_version", lambda: "1.1.0")

    def fail_fetch(timeout: float):
        raise update_module.UpdateCheckError("network request failed: timed out")

    monkeypatch.setattr(update_module, "_fetch_pypi_payload", fail_fetch)
    monkeypatch.setattr(
        update_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("offline checks must not start pip"),
    )

    assert update_module.run_update(["--yes"]) == 2
    captured = capsys.readouterr()
    assert "Unable to check PyPI" in captured.err
    assert "No changes were made" in captured.err
    assert "Manual command:" in captured.err


@pytest.mark.cli
def test_pypi_response_size_is_bounded(monkeypatch) -> None:
    class OversizedResponse:
        headers = {"Content-Length": str(update_module.MAX_PYPI_RESPONSE_BYTES + 1)}

        def geturl(self):
            return update_module.PYPI_JSON_URL

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, size):
            pytest.fail("an oversized announced response must not be read")

    monkeypatch.setattr(update_module, "urlopen", lambda request, timeout: OversizedResponse())

    with pytest.raises(update_module.UpdateCheckError, match="exceeded the allowed size"):
        update_module._fetch_pypi_payload(1.0)


@pytest.mark.cli
def test_pypi_redirect_must_stay_on_exact_https_origin(monkeypatch) -> None:
    class RedirectedResponse:
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def geturl(self):
            return "https://pypi.org.evil.example/pypi/hidden-attractors-fo/json"

        def read(self, size):
            pytest.fail("a response from a foreign redirect origin must not be read")

    monkeypatch.setattr(update_module, "urlopen", lambda request, timeout: RedirectedResponse())

    with pytest.raises(update_module.UpdateCheckError, match="outside the allowed HTTPS origin"):
        update_module._fetch_pypi_payload(1.0)


@pytest.mark.cli
def test_install_timeout_reports_failure_and_manual_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(update_module, "_installed_version", lambda: "1.1.0")
    monkeypatch.setattr(
        update_module,
        "_fetch_pypi_payload",
        lambda timeout: _payload(**{"1.2.0": False}),
    )

    def timeout_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(update_module, "_run_pip_process", timeout_run)

    assert update_module.run_update(["--yes", "--install-timeout", "1"]) == 1
    captured = capsys.readouterr()
    assert "exceeded the 1-second install timeout" in captured.err
    assert "may have made partial changes" in captured.err
    assert "Manual command:" in captured.err


@pytest.mark.cli
def test_permission_failure_is_explained(monkeypatch, capsys) -> None:
    monkeypatch.setattr(update_module, "_installed_version", lambda: "1.1.0")
    monkeypatch.setattr(
        update_module,
        "_fetch_pypi_payload",
        lambda timeout: _payload(**{"1.2.0": False}),
    )
    monkeypatch.setattr(
        update_module,
        "_run_pip_process",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="Permission denied",
        ),
    )

    assert update_module.run_update(["--yes"]) == 1
    captured = capsys.readouterr()
    assert "denied write access" in captured.err
    assert "virtual environment" in captured.err
    assert "may have made partial changes" in captured.err


@pytest.mark.cli
def test_update_documentation_does_not_claim_pip_is_transactional() -> None:
    version_root = Path(__file__).resolve().parents[1]
    installation = (version_root / "docs" / "installation.md").read_text(encoding="utf-8")

    assert "may leave partial changes" in installation
    assert "failures leave the environment unchanged" not in installation
    assert "--isolated install --upgrade" in installation
    assert "--index-url https://pypi.org/simple" in installation
    assert "hidden-attractors-fo==" in installation
    assert "installed `hidden-attractors.exe` launcher" in installation
    assert "refuses to invoke pip" in installation
    assert "py -m pip install --upgrade hidden-attractors-fo" in installation


@pytest.mark.cli
def test_packaging_is_a_declared_runtime_dependency() -> None:
    version_root = Path(__file__).resolve().parents[1]
    with (version_root / "pyproject.toml").open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["dependencies"]

    assert any(item.startswith("packaging>=") for item in dependencies)


@pytest.mark.cli
def test_main_dispatch_exposes_update_help() -> None:
    from hidden_attractors.cli.main import main

    with pytest.raises(SystemExit) as exc_info:
        main(["update", "--help"])
    assert exc_info.value.code == 0
