# Estructura de librería

Este documento fija la organización recomendada para que el proyecto pueda
crecer como librería Python y no como una colección de scripts independientes.

## Criterio de diseño

1. El modelo matemático debe vivir separado del solver.
2. El solver debe declarar el contrato numérico: `q`, `h`, `Lm`, tiempo final y
   transitorio.
3. La función descriptiva y el balance armónico producen semillas, no
   conclusiones de ocultedad.
4. La validación por Caputo/EFORK, cuencas y vecindades de equilibrios se
   documenta como una etapa posterior.
5. Los workflows largos deben usar procesos independientes cuando se paraleliza,
   y guardar metadatos JSON/CSV por corrida.

## Paquete `hidden_attractors`

- `models`: definiciones de sistemas dinámicos, parámetros y equilibrios.
- `solvers`: interfaces para integradores fraccionarios.
- `native`: compilación y wrappers de backends C.
- `basins`: etiquetas y helpers de clasificación de cuencas.
- `analysis`: métricas de trayectoria, FFT dominante, nubes y secciones.
- `plotting`: funciones de figura reutilizables.
- `workflows`: flujos compuestos que pueden exponerse por CLI.

## Versiones

La separacion operativa queda fijada asi:

- `version_1/` documenta la etapa historica y los scripts raiz que se conservan
  para reproducibilidad.
- `version_2/` es la version activa. Las modificaciones nuevas de ejemplos y
  analisis deben quedar registradas en `version_2/examples/` y
  `version_2/analysis/`.
- El codigo reusable de V2 debe vivir en `hidden_attractors/`; los scripts raiz
  nuevos solo deben ser wrappers CLI delgados cuando haga falta mantener nombres
  de ejecucion antiguos.

## Scripts raíz

Los scripts raíz se conservan como entradas reproducibles de investigación.
Cuando un script empiece a duplicar lógica, el patrón recomendado es:

1. mover la función reusable a `hidden_attractors/`;
2. dejar el script como wrapper CLI;
3. agregar un ejemplo pequeño en `version_2/examples/`;
4. documentar el contrato numérico y las salidas.

`robustness_overlay_c_trajectories.py` y `lure_top3_sphere_robustness.py` ya
siguen ese patrón: la lógica vive en `hidden_attractors.workflows`.

## Resultados y ejemplos

Los resultados finales deben quedar bajo `outputs/final_candidate_analysis/` o
en carpetas timestamped dentro de `outputs/`. Los ejemplos no deben sobrescribir
resultados existentes; cuando escriban archivos, deben usar un directorio nuevo
o pedir explícitamente `--output-dir`.
