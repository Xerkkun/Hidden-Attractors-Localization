# Hidden Attractors in Fractional-Order Systems

`hidden-attractors-fo` is a Python research library for reproducible numerical
workflows around hidden-attractor candidates in integer- and commensurate
Caputo fractional-order Lur'e-compatible systems. The maintained Chua examples
cover an integer reference route, a non-smooth fractional BDF/saturation route,
and a smooth arctan fractional route.

## PyPI installation

```bash
python -m pip install hidden-attractors-fo
```

The Python import name is different from the PyPI project name:

```python
import hidden_attractors
```

The installed public CLI is:

```bash
hidden-attractors --help
hidden-attractors inspect systems
hidden-attractors seed --help
```

## Development installation

From a repository checkout:

```bash
python -m pip install -e "version_2[dev,analysis,docs,legacy]"
```

## First run

```bash
cd version_2
hidden-attractors --help
hidden-attractors inspect systems
hidden-attractors validate contract --allow-pending
hidden-attractors run -p chua_integer
```

## Official examples

```bash
cd version_2
python examples/chua_integer_lure_reference/run_example.py --quick
python examples/chua_nonsmooth_biased_hidden_attractor/run_example.py --quick
python examples/chua_arctan_wu2023/run_example.py --quick
```

- Integer Chua `q=1`: reproduced software reference for the Lur'e route.
- Non-smooth fractional Chua BDF: proposed methodology; not full Danca 2017 trajectory reproduction.
- Arctan Chua Wu2023/c590: a smooth-nonlinearity validation example; Wu2023 remains bibliographic, and c590 is finite-time evidence under a local/radius-limited contract, not a global mathematical proof.

## Documentation

- User manual: `version_2/USER_MANUAL.md`
- Quick start: `version_2/docs/quick_start.md`
- API inventory: `version_2/docs/api_reference.md`
- Examples index: `version_2/docs/examples_index.md`
- Validation evidence: `version_2/docs/validation_evidence.md`
- Claims matrix: `version_2/THESIS_CLAIMS.md`
- Freeze audit: `version_2/validation/freeze_audit/`

## Evidence boundary

DF/Nyquist, continuation, plots, FFT/PSD, 0-1 tests, Poincare sections, and
Lyapunov estimates are diagnostics or seed-generation tools. Hiddenness labels require sampled local neighborhoods or basin evidence around
all equilibria under a recorded numerical contract. Large-radius spherical
contacts are reported as extended basin-geometry audits; by themselves they do
not imply a self-excited classification unless the local-radius contract records
equilibrium-neighborhood contact. These labels are not a global mathematical
proof.

## Citation

Citation metadata is provided in `CITATION.cff`, `.zenodo.json`, and
`codemeta.json`. Archived DOI:

```text
10.17605/OSF.IO/ZGK74
```

## License

MIT.
