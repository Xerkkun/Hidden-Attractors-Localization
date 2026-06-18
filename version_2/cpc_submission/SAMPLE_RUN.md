# Sample run

```bash
cd version_2
python -m pip install -e ".[dev,analysis,legacy]"
hidden-attractors --help
hidden-attractors init -e chua_fractional
hidden-attractors inspect-config -p chua_fractional
hidden-attractors validate contract --allow-pending
python -m pytest -q -m "not slow"
```

Los conteos exactos de pruebas pueden cambiar; la fuente oficial congelada esta en `validation/freeze_audit/`.
