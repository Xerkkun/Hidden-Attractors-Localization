# Testing

The automated suite checks public API behavior, numerical input contracts,
documentation consistency, packaging, and completed validation records.

Run the complete suite from `version_2` with the developer dependencies
installed:

```bash
python -m pytest -q
```

## Focused checks

Repository and documentation hygiene:

```bash
python -m pytest -q -m hygiene
```

Packaging and release metadata:

```bash
python -m pytest -q -m release_readiness
```

Python syntax:

```bash
python -m compileall hidden_attractors examples tests tools/cli
```

Strict documentation build:

```bash
python -m mkdocs build --strict --clean
```

## Evidence boundary

Software tests establish that implemented functions satisfy their tested
contracts. Numerical validation tests reproduce only the finite settings and
reference values stored with each completed validation record. Neither type of
test is a global proof of chaos or hiddenness.

The supported release interface is the exported Python API and the
`hidden-attractors` command. Helper scripts outside the package are not an
additional public API.
