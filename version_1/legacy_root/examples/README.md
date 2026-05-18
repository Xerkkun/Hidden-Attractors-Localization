# Ejemplos de compatibilidad

Los ejemplos canonicos de la version activa viven en `version_2/examples/`.
Esta carpeta se conserva para que comandos anteriores sigan funcionando.

Para cambios nuevos, agrega o modifica archivos en `version_2/examples/` y deja
este directorio como wrapper de compatibilidad.

## Comandos antiguos que siguen funcionando

```bash
python3 examples/list_final_candidates.py
python3 examples/create_robustness_overlay_config.py
python3 examples/aggregate_existing_robustness_overlay.py outputs/robustness_overlay_c_trajectories_20260517
python3 lure_top3_sphere_robustness.py --help
```

El workflow completo de robustez se lanza con el wrapper de la raíz:

```bash
python3 robustness_overlay_c_trajectories.py
```
