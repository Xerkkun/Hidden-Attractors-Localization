# Testing

## Smoke Checks

Comandos rápidos desde `version_2/`:

```bash
python -m compileall hidden_attractors examples tests tools/cli
python examples/quickstart_equilibria.py
python examples/list_final_candidates.py
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
hidden-attractors validate contract --help
```

## Pytest

```bash
python -m pip install -e ".[dev,analysis,legacy]"
python -m pytest -q
```

En el freeze audit actual, la suite completa reporta 797 passed y 34 skipped. El conteo oficial debe tomarse de `validation/freeze_audit/`.

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

*Nota de depreciación:* El comando antiguo `hidden-attractors-check-validation` es legacy/deprecated y no debe recomendarse como comando público. La ruta pública estable es `hidden-attractors validate contract`. Los scripts en `tools/legacy/` o wrappers antiguos no deben aparecer en smoke checks principales.
