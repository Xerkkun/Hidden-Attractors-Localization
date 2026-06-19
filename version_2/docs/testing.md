# Testing

## Smoke Checks

Comandos rápidos desde `version_2/`:

```bash
python -m compileall hidden_attractors examples tests tools/cli
python -m hidden_attractors.cli.main --help
python -m hidden_attractors.cli.main inspect --help
python -m hidden_attractors.cli.main validate --help
python -m hidden_attractors.cli.main seed --help
python -m hidden_attractors.cli.main continuation --help
```

Si el paquete está instalado en modo editable:

```bash
hidden-attractors --help
hidden-attractors inspect systems
hidden-attractors inspect candidates
hidden-attractors validate contract --allow-pending
```

## Pytest

```bash
python -m pip install -e ".[dev,analysis,legacy]"
python -m pytest -q
```

### Hygiene and CPC Readiness Tests

The project keeps a small hygiene/readiness test suite because numerical tests do not protect repository publication boundaries. These tests guard against retracking local outputs, local manuscripts, absolute paths, legacy CLI entry points, unpromoted validation outputs, and overclaimed CPC metadata.

To run these tests specifically:
```bash
python -m pytest -q -m "hygiene"
python -m pytest -q -m "cpc_readiness"
python -m pytest -q -m "not hygiene and not cpc_readiness"
```

At the current thesis-freeze audit, `validation/freeze_audit/` reports 797 passed and 34 skipped.

Current tests verify:

- Chua equilibria are zeros of the vector field;
- final candidate loading returns the expected reference records.
- the official ten-stage order and uniform JSON envelope;
- periodic pre-continuation seeds remain eligible for continuation;
- hiddenness cannot receive its strongest label without ball sampling, robust
  reproduction, all equilibria, and all basin planes;
- every executable Python and C EFORK source uses
  `K3 = a31*K1 + a32*K2`.

## Native Backends

Native backend tests should be added cautiously. Keep them short, write to a
temporary output directory, and record whether OpenMP was active.

## Validation Evidence

Tests answer whether the package still behaves as expected. Validation evidence
answers whether a scientific claim is backed by traceable numerical artifacts.

Use `outputs/` for ordinary generated run products. Promote only selected
evidence into `validation/`, following `configs/validation_contract.json`.
Each validation stage should include:

- one short `*_validation.md` interpretation;
- one `*_validation_summary.json` or equivalent summary JSON;
- CSV tables for numerical checks;
- PNG/PDF figures for visual evidence when relevant.

The final report should cite the stage summaries and selected artifacts instead
of embedding all raw data.

Run the contract checker from `version_2/` after evidence has been promoted:

```bash
hidden-attractors validate contract
```

Y, cuando se permitan etapas pendientes:

```bash
hidden-attractors validate contract --allow-pending
```

Si se necesita raíz alternativa:

```bash
hidden-attractors validate contract --validation-root path/to/validation
```

### Legacy commands no longer installed

These names may appear in old notes only; use the grouped `hidden-attractors` CLI:
- `hidden-attractors-check-validation` (deprecated; use `hidden-attractors validate contract` instead)
- `hidden-attractors-protocol` (deprecated; use `hidden-attractors protocol <stage>` instead)
