# Hidden Attractors in Fractional-Order Systems

A Python research library for configuring, running, validating, and documenting numerical workflows for hidden-attractor candidates in integer- and fractional-order Chua/Lur'e systems.

## Installation

```bash
python -m pip install -e version_2
```

For development:

```bash
python -m pip install -e "version_2[dev,analysis,legacy]"
```

## Quick start

```bash
cd version_2
hidden-attractors --help
hidden-attractors init -e chua_fractional
hidden-attractors run -p chua_integer
```

## Repository structure

```text
version_2/                  Active Python package
version_2/hidden_attractors/ Library source code
version_2/configs/          Example configurations
version_2/examples/         Runnable examples
version_2/tests/            Automated tests
version_2/validation/       Promoted validation evidence
version_2/docs/             Documentation
```

## Documentation

* User manual: `version_2/USER_MANUAL.md`
* Quick start: `version_2/docs/quick_start.md`
* Validation evidence: `version_2/docs/validation_evidence.md`
* Claims matrix: `version_2/THESIS_CLAIMS.md`

## Citation

Citation metadata is provided in `CITATION.cff`, `.zenodo.json`, and `codemeta.json`.

Archived DOI:

```text
10.17605/OSF.IO/ZGK74
```

## License

MIT.
