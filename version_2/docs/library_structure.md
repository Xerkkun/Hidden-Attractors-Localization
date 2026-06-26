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

La raíz de version_2 se mantiene limpia como una librería Python:

- `tools/cli/`: scripts y utilidades internas obsoletas (no son entry points públicos).
- `tools/legacy/`: scripts históricos preservados para trazabilidad (uso interno/histórico).
- `artifacts/`: binarios o resultados migrados desde la estructura anterior.

El patrón recomendado para añadir o modificar la interfaz de comandos (CLI) de la biblioteca es:

1. Implementar la lógica reutilizable en `hidden_attractors/` (por ejemplo, en `hidden_attractors/workflows/`).
2. Añadir un dispatcher o subcomando en `hidden_attractors/cli/` (por ejemplo, importando el workflow e implementando el parser de argumentos).
3. Registrar el nuevo subcomando en el despachador unificado `hidden_attractors/cli/main.py` dentro del diccionario `GROUPS` y la función `main()`.
4. Añadir pruebas de integración para la CLI en la carpeta `tests/`.
5. Actualizar la referencia de la API (`docs/api_reference.md`) y los manuales de usuario.

## Resultados y ejemplos

Los resultados finales deben quedar bajo `outputs/final_candidate_analysis/` o
en carpetas timestamped dentro de `outputs/`. Los ejemplos no deben sobrescribir
resultados existentes; cuando escriban archivos, deben usar un directorio nuevo
o pedir explícitamente `--output-dir`.
