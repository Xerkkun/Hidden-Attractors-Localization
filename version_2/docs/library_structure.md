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

## Scripts y herramientas

La raiz de V2 se mantiene limpia como una libreria Python:

- `tools/cli/`: wrappers mantenidos para workflows publicos.
- `tools/legacy/`: scripts historicos preservados para trazabilidad.
- `artifacts/`: binarios o resultados migrados desde la estructura anterior.

Cuando un script de `tools/legacy/` empiece a duplicar lógica, el patrón
recomendado es:

1. mover la función reusable a `hidden_attractors/`;
2. dejar un wrapper CLI en `tools/cli/` si el comando importa;
3. agregar un ejemplo pequeño en `examples/`;
4. documentar el contrato numérico y las salidas.

`robustness_overlay_c_trajectories.py`, `lure_top3_sphere_robustness.py` y
`refine_project_basin_classification.py` ya siguen ese patrón: la lógica vive
en `hidden_attractors.workflows` y los wrappers viven en `tools/cli/`.

## Resultados y ejemplos

Los resultados finales deben quedar bajo `outputs/final_candidate_analysis/` o
en carpetas timestamped dentro de `outputs/`. Los ejemplos no deben sobrescribir
resultados existentes; cuando escriban archivos, deben usar un directorio nuevo
o pedir explícitamente `--output-dir`.

