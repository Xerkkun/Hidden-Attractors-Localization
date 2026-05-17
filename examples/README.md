# Ejemplos

Estos archivos muestran cómo usar `hidden_attractors` como librería importable.
No sustituyen los workflows largos de la raíz; son puntos de entrada pequeños
para inspección, configuración y reutilización.

## Comandos

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
