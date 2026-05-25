# Version 2 analysis

Este indice marca que analisis pertenecen a la version activa y donde debe
crecer cada modificacion nueva.

## Modulos reutilizables

- `hidden_attractors.analysis.trajectory`
  - metricas de trayectoria;
  - distancias de nubes de cola;
  - FFT dominante y entropia espectral;
  - secciones de Poincare;
  - contratos de robustez `RobustnessCase`.
- `hidden_attractors.basins.classification`
  - etiquetas de clase;
  - helpers para decidir si una clase es target.
- `hidden_attractors.workflows.refined_basin`
  - reclasificacion de celdas `unknown` por geometria contra referencias.
- `hidden_attractors.workflows.robustness_overlay`
  - comparacion de trayectorias bajo cambios de `h`, `Lm` y tiempo.
- `hidden_attractors.workflows.sphere_controls`
  - adaptador compatible cuyo calculo vigente muestrea bolas interiores alrededor de equilibrios.

## Analisis integrados con CLI mantenida

- `tools/cli/robustness_overlay_c_trajectories.py`
- `tools/cli/lure_top3_sphere_robustness.py`
- `tools/cli/refine_project_basin_classification.py`

## Pendientes de migracion

- `tools/legacy/positive_x_basin_sweep.py`
  - contiene analisis importante de cuencas en `x>0`;
  - si crece, mover helpers de grilla, resumen y figuras a
    `hidden_attractors/workflows/` o `hidden_attractors/basins/`.
- `tools/legacy/plot_top3_sphere_geometry.py`
  - visualiza artefactos con nombre historico; no define el protocolo oficial;
  - si crece, mover carga/seleccion de filas y plot reusable a
    `hidden_attractors/plotting/`.

## Regla para lo nuevo

Cada analisis nuevo debe registrar:

- proposito matematico;
- contrato numerico (`q`, `h`, `Lm`, `t_final`, `t_burn`, solver/backend);
- entradas esperadas;
- salidas escritas;
- advertencia de validez cuando el resultado sea evidencia numerica y no
  prueba formal.
