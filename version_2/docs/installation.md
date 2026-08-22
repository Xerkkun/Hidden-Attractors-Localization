# Installation

## PyPI

```bash
python -m pip install hidden-attractors-fo
```

Install optional time-series and complexity diagnostics with:

```bash
python -m pip install "hidden-attractors-fo[analysis]"
```

Verify the library and CLI:

```bash
python -c "import hidden_attractors; print(hidden_attractors.__name__)"
hidden-attractors --help
hidden-attractors inspect systems
```

## Updating an installed release

Check the active Python environment against stable releases on PyPI without
changing anything:

```bash
hidden-attractors update --check
```

`hidden-attractors update` also starts in check mode. In an interactive
terminal it asks for confirmation before running pip; in a non-interactive
terminal it only reports the status and prints the manual command. Explicitly
approve an upgrade with:

```bash
hidden-attractors update --yes
```

On Windows, an installed `hidden-attractors.exe` launcher remains active for
the duration of this command. The CLI therefore refuses to invoke pip from
that active launcher, even with `--yes`. Exit the command and paste its exact
`sys.executable -m pip ...` command into a new terminal so the launcher is no
longer running. `--check` remains available from the launcher.

Prereleases are ignored unless `--pre` is supplied. The command uses the
currently active `sys.executable`, reports whether it is a virtual environment,
and never downgrades a local or development version that is newer than PyPI.
If the PyPI check fails before pip starts, the environment is not modified.
Once pip starts, however, installation is not transactional: a timeout,
network interruption, or permission error may leave partial changes. Inspect the
active environment before retrying. The CLI prints the exact version-pinned,
isolated command that it would run against the official PyPI index, for example:

```bash
"PYTHON" -m pip --isolated install --upgrade --index-url https://pypi.org/simple hidden-attractors-fo==VERSION_SHOWN_BY_CHECK
```

When `--pre` selected a prerelease, that printed command also contains `--pre`.
The shorter command below is a simple pip fallback, not equivalent to the
version-pinned and index-isolated command printed by the CLI:

```bash
python -m pip install --upgrade hidden-attractors-fo
```

On Windows, `py -m pip install --upgrade hidden-attractors-fo` is also a simple
fallback when `py` selects the intended environment. Prefer the exact
interpreter and pinned command printed by the CLI when environments differ.

## Development checkout

From `version_2/`:

```bash
python -m pip install -e ".[dev,analysis,docs]"
python -m pytest -q -m "hygiene or release_readiness"
```

Python 3.11, 3.12, and 3.13 are tested by the release CI matrix. Python 3.14
is declared as supported but remains outside that matrix.

## Runtime locations

Generated outputs default to `./outputs`. Override output and cache locations
with:

```bash
HIDDEN_ATTRACTORS_OUTPUT_DIR=/path/to/output
HIDDEN_ATTRACTORS_CACHE_DIR=/path/to/cache
```

Native compilation products are stored in the user cache. The installed
package directory is read-only from the library's point of view.

## Native backends

Native C backends require a compatible C compiler. Python reference backends
remain available when native compilation is unavailable. Any reported result
must record the selected backend and numerical contract.
