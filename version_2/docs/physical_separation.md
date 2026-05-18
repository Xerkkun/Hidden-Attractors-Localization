# Physical separation status

La separacion fisica ya esta hecha de forma conservadora:

- `version_1/legacy_root/` contiene una copia reproducible de la raiz antigua.
- `version_2/` contiene la copia activa autocontenida.
- `version_2/pyproject.toml` instala el paquete desde `version_2/`.
- `version_2/examples/` es la ubicacion canonica para ejemplos nuevos.
- `version_2/docs/` es la ubicacion canonica para documentacion y notas de
  analisis.
- `version_2/tools/legacy/` contiene los scripts historicos migrados fuera de
  la raiz de V2.

## Verificacion realizada

Desde `version_2/`:

```bash
python -m compileall hidden_attractors examples tests tools/cli
python examples/list_final_candidates.py
python tools/cli/robustness_overlay_c_trajectories.py --help
python tools/cli/lure_top3_sphere_robustness.py --help
python tools/cli/refine_project_basin_classification.py --help
python examples/create_robustness_overlay_config.py
```

Tambien se verifico desde la carpeta superior:

```bash
python version_2/examples/list_final_candidates.py
```

## Limpieza

Si quieres conservar el repositorio Git, no borres `.git/`.

Los demas archivos y carpetas que queden fuera de `version_1/` y `version_2/`
son duplicados de la raiz vieja despues de esta separacion. Borralos solo
despues de confirmar que no necesitas una ruta antigua exacta para algun script
externo o acceso directo.
