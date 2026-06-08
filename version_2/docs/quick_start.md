# Guía de Inicio Rápido — Hidden Attractors

Esta guía proporciona la ruta de entrada recomendada y directa para usuarias nuevas en la versión 2 de la biblioteca.

---

## 1. Instalación Mínima

Para instalar la biblioteca en modo editable junto con las dependencias necesarias:

```bash
# Desde la raíz del repositorio
pip install -e version_2
```

Si deseas ejecutar pruebas unitarias o realizar labores de desarrollo, instala las dependencias de desarrollo:

```bash
pip install -e "version_2[dev]"
```

---

## 2. Ejecutar el Ejemplo Oficial (Ejemplo 1)

El **Ejemplo 1 — Chua fraccionario no suave con función descriptiva sesgada** es el principal punto de partida para comprobar el flujo de negocio del framework.

Para ejecutar la prueba de humo rápida (~1-2 minutos):

```bash
cd version_2/examples/chua_nonsmooth_biased_hidden_attractor
python run_example.py --quick
```

Otras opciones de ejecución para el Ejemplo 1:
- `python run_example.py` (Ejecución estándar con 4 pasos clave, ~10-15 minutos)
- `python run_example.py --all` (Ejecución masiva con búsqueda extendida en paralelo, puede tomar horas)

---

## 3. Ejecutar un Preset de la CLI

La biblioteca incluye configuraciones rápidas (presets) listas para ejecutar mediante el comando unificado `hidden-attractors`. 

Para ejecutar el preset de Chua Fraccionario:

```bash
hidden-attractors run -p chua_fractional
```

Otras opciones de presets disponibles:
* `chua_integer`: Simulación y Lure DF en el sistema Chua de orden entero.
* `chua_arctan`: Análisis en el sistema Chua de orden fraccionario con no-linealidad arcotangente.

---

## 4. ¿Dónde quedan los Resultados?

- **CSV de Trayectorias y Logs de Salida**: Se almacenan en la carpeta `outputs/` en la raíz del repositorio.
- **Gráficos y Metadatos**: Todas las figuras generadas se guardan de forma unificada bajo `version_2/library_figures/` (organizadas por identificador de ejecución o `by_run`).

---

## 5. ⚠️ Qué NO Ejecutar

Para garantizar la estabilidad y reproducibilidad científica del repositorio, sigue estrictamente estas reglas:

* 🚫 **No ejecutar scripts archivados**: Todo código histórico se ha movido a `_archived_figure_scripts/` y no forma parte del flujo de ejecución actual.
* 🚫 **No utilizar `_reference_scripts/`**: Dicha carpeta ha sido archivada en `_archived_figure_scripts/reference_scripts/` y borrada de la raíz.
* 🚫 **No crear scripts nuevos en la raíz del repositorio**: Todo script de prueba temporal o análisis específico debe colocarse dentro de la carpeta de trabajo `version_2/examples/` o en un módulo bajo `version_2/hidden_attractors/`.
* 🚫 **No guardar figuras directamente fuera de `library_figures`**: Queda prohibido llamar directamente a `savefig` en módulos activos. Todo gráfico debe exportarse utilizando la API unificada de ploteo en `export_figure` o `intercept_and_export_path`.
