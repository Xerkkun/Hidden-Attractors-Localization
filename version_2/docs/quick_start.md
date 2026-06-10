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

> [!NOTE]
> **Nota metodológica:** Este ejemplo **no es una reproducción del sistema de Danca (2017)**. El sistema original de Danca **no fue reproducible debido a la falta de información publicada** (condiciones iniciales, detalles espectrales del resolvedor DF y el método de continuación numérica). Este ejemplo realiza una búsqueda sistemática para identificar vecindades compatibles.

Para ejecutar la prueba de humo rápida (~1-2 minutos):

```bash
cd version_2/examples/chua_nonsmooth_biased_hidden_attractor
python run_example.py --quick
```

Otras opciones de ejecución para el Ejemplo 1:
- `python run_example.py` (Ejecución estándar con 4 pasos clave, ~10-15 minutos)
- `python run_example.py --all` (Ejecución masiva con búsqueda extendida en paralelo, puede tomar horas)

---

## 3. Interfaz CLI Unificada (`hidden-attractors`)

La biblioteca incluye una interfaz CLI unificada agrupada por propósito matemático y computacional.

### Comandos Directos:
* `hidden-attractors run -p/--preset <name>` (o con `-c/--config`)
* `hidden-attractors init`: Copia las plantillas de configuración al directorio actual.
* `hidden-attractors inspect-config`: Previsualiza la configuración efectiva normalizada.

### Comandos Agrupados de Análisis Avanzado:
* **Generación de Semillas (`seed`)**:
  * `hidden-attractors seed lure-centered -c <config>` (o `-p <preset>`): Búsqueda de semillas mediante la función descriptiva clásica centrada ($c = 0$, $W_q(j\omega)N(A) = -1$).
  * `hidden-attractors seed lure-biased -c <config>` (o `-p <preset>`): Búsqueda de semillas desplazadas mediante balance de primer armónico y balance DC ($c \neq 0$, $W_q(j\omega)N_1(A, \sigma_0) = -1$).
  * *Nota*: Las familias Machado/FDF (`machado-centered`, `machado-biased`) se encuentran documentadas como planificadas pero no están disponibles para ejecución activa en esta entrega.
  * *Advertencia Científica*: La función descriptiva es únicamente una aproximación armónica para buscar semillas localizadas, **no constituye una prueba matemática de existencia ni de ocultedad**.
* **Continuación Numérica (`continuation`)**:
  * `hidden-attractors continuation run -c <config> -s <seeds.csv>`: Propagación escalar de una semilla deformando la no linealidad mediante $\eta \in [0,1]$.
  * `hidden-attractors continuation multiparameter -c <config>`: Continuación vectorial a lo largo de un camino parametrizado $\Gamma(\tau) = (\eta(\tau), p_1(\tau), \ldots, p_m(\tau))$.
  * *Nota de Caputo*: En sistemas fraccionarios de Caputo, se propaga y registra la historia de memoria/historial. En caso de usar una continuación sin historia (warm-start de último estado), se emitirá una advertencia explícita.
* **Bifurcaciones (`bifurcation`)**:
  * `hidden-attractors bifurcation run -c <config>`: Ejecuta barridos de parámetros para diagramas de bifurcación.
  * `hidden-attractors bifurcation plot -i <csv>`: Genera diagramas de bifurcación estilizados.
  * `hidden-attractors bifurcation inspect -i <json>`: Muestra estadísticas de extremos locales detectados.
* **Exponentes de Lyapunov (`lyapunov`)**:
  * `hidden-attractors lyapunov compute -c <config>`: Estima el espectro de Lyapunov (variación homotópica).
  * `hidden-attractors lyapunov spectrum -t <csv>`: Estima exponentes a partir de trayectorias (mediante `nolds`).
  * `hidden-attractors lyapunov validate -i <json>`: Valida reportes y emite advertencias sobre aproximaciones de tiempo finito.
* **Prueba de Caos (`chaos-test`)**:
  * `hidden-attractors chaos-test zero-one -c <config>` (o `-t <csv>`): Ejecuta la prueba 0-1 de caos.
  * `hidden-attractors chaos-test inspect -i <json>`: Muestra la clasificación (regular, caótico o inconcluyente).

---

## 4. ¿Dónde quedan los Resultados?

- **CSV de Trayectorias y Logs de Salida**: Se almacenan en la carpeta `outputs/` en la raíz del repositorio.
- **Gráficos y Metadatos**: Todas las figuras generadas se guardan de forma unificada bajo `version_2/library_figures/` (organizadas por identificador de ejecución o `by_run`).

---

## 5. ⚠️ Qué NO Ejecutar

Para garantizar la estabilidad y reproducibilidad científica del repositorio, sigue estrictamente estas reglas:

* 🚫 **No ejecutar scripts históricos ni scratch**: Los archivos históricos usados durante la migración no forman parte del repositorio activo ni de la distribución pública. Todo flujo reproducible debe ejecutarse desde `hidden-attractors` o desde los ejemplos oficiales en `version_2/examples/`.
* 🚫 **No crear scripts nuevos en la raíz del repositorio**: Todo script de prueba temporal o análisis específico debe colocarse dentro de la carpeta de trabajo `version_2/examples/` o en un módulo bajo `version_2/hidden_attractors/`.
* 🚫 **No guardar figuras directamente fuera de `library_figures`**: Queda prohibido llamar directamente a `savefig` en módulos activos. Todo gráfico debe exportarse utilizando la API unificada de ploteo en `export_figure` o `intercept_and_export_path`.

---

## 6. Pruebas Unitarias y de Contrato

Para garantizar que los contratos científicos y la lógica de la biblioteca permanezcan invariables, puedes ejecutar la suite de pruebas desde la carpeta `version_2/`:

```bash
# Ejecutar la suite de pruebas rápidas (CI rápida, excluye pruebas lentas)
pytest tests -m "not slow"

# Ejecutar únicamente los contratos científicos rápidos
pytest tests -m "scientific_contract"

# Ejecutar pruebas del CLI público
pytest tests -m "cli"

# Ejecutar la suite completa localmente
pytest tests

# Ejecutar pruebas lentas y reproducciones numéricas pesadas (antes de congelar versión)
pytest tests -m "slow"
```
