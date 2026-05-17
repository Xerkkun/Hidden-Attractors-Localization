# hidden-attractors-fo

Herramientas numéricas para localizar, reproducir y auditar candidatos a
atractores ocultos en sistemas fraccionarios tipo Chua/Lur'e.

El paquete `hidden_attractors` reúne piezas reutilizables que antes estaban
dispersas en scripts de experimento: modelos, backends C/EFORK, cargadores de
candidatos, métricas de trayectoria, clasificación de cuencas, graficación y
workflows reproducibles.

## Instalación local

Desde esta carpeta:

```bash
python3 -m pip install -e .
```

Si solo vas a ejecutar scripts desde el repositorio, no es obligatorio instalar;
Python puede importar el paquete directamente desde el directorio actual.

## Organización

- `hidden_attractors/models/`: ecuaciones, parámetros y equilibrios.
- `hidden_attractors/solvers/`: interfaces de integración fraccionaria.
- `hidden_attractors/native/`: wrappers `ctypes` para backends C existentes.
- `hidden_attractors/basins/`: etiquetas y utilidades de clasificación de cuencas.
- `hidden_attractors/analysis/`: métricas geométricas, espectrales y de sección.
- `hidden_attractors/plotting/`: figuras reutilizables.
- `hidden_attractors/workflows/`: flujos completos con CLI delgada.
- `examples/`: ejemplos pequeños que muestran el uso importable.

Los scripts históricos siguen en la raíz como workflows de investigación. La
ruta nueva es mover gradualmente la lógica común al paquete y dejar esos
scripts como entradas CLI reproducibles.

## Advertencias científicas

- Una semilla de función descriptiva o balance armónico no prueba existencia de
  un atractor de Caputo; solo produce una condición inicial a validar.
- `Lm` es parte del contrato numérico cuando se usa EFORK con memoria finita.
- Un hit `target_positive` o `target_negative` es una clasificación operacional
  bajo el contrato probado, no una prueba automática de ocultedad.
- La ocultedad se evalúa con vecindades de equilibrios y cuencas; la robustez
  evalúa persistencia bajo cambios de `h`, `Lm` y tiempo.

## Ejemplos rápidos

Listar candidatos finales:

```bash
python3 examples/list_final_candidates.py
```

Crear solo la configuración de overlays de robustez:

```bash
python3 examples/create_robustness_overlay_config.py
```

Lanzar el workflow completo de overlays con procesos independientes:

```bash
python3 robustness_overlay_c_trajectories.py
```

Lanzar controles esféricos alrededor de equilibrios para los tres candidatos:

```bash
python3 lure_top3_sphere_robustness.py
```
