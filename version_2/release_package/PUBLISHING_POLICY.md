# PyPI publishing policy

The repository stores no PyPI token, API key, password, or upload credential.

Publication is performed only by `.github/workflows/publish-pypi.yml` from a
tag whose exact value is `v` followed by the version in `version_2/pyproject.toml`.
A tag push or a manual dispatch from that tag starts verification; it does not
bypass the protected `pypi` environment.

The verification job has read-only repository permissions and must complete all
of these checks:

- install the package and release tooling in a clean runner;
- validate release metadata and version consistency;
- run the test suite excluding the explicit `slow` marker;
- build both wheel and source distribution;
- run `twine check`;
- smoke-test a clean wheel installation;
- confirm that tracked files were not modified.

The build artifact is transferred to a separate publish job. Only that job has
`id-token: write`, and only for PyPI Trusted Publishing. The `pypi` GitHub
environment supplies the configured approval boundary.

PyPI receives the software distribution defined by `MANIFEST.in`. Scientific
validation datasets remain in the tagged repository and DOI archive. Publishing
the package does not alter their scientific status.

The verified 1.1.0 release-candidate manifest records
`publication_status: not_published`.
Publication must not be inferred from local builds, passing tests, or the
presence of a version string.
