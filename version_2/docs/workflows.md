# Flujos de Trabajo (Workflows) — Guía Oficial

Consulta la [Matriz de Afirmaciones de Tesis (Thesis Claims Matrix)](../THESIS_CLAIMS.md) para ver la clasificación actual de resultados y claims defendibles.

Este documento detalla los flujos de trabajo de la biblioteca `hidden-attractors-fo`. 

---

## 1. Ruta Recomendada para Usuarias Nuevas

Si es la primera vez que interactúas con este repositorio, la ruta recomendada de aprendizaje y ejecución es la siguiente:

1. **Instalación**: Instala la librería en modo editable con `pip install -e version_2`.
2. **Ejemplo Oficial**: Ejecuta la prueba rápida del Ejemplo 1 para verificar el pipeline:
   ```bash
   cd version_2/examples/chua_nonsmooth_biased_hidden_attractor
   python run_example.py --quick
   ```
3. **Exploración de Presets**: Ejecuta el comando de CLI unificado para ver un preset estable:
   ```bash
   hidden-attractors run -p chua_fractional
   ```
4. **Lectura de Guías**: Consulta la [Guía de Inicio Rápido](quick_start.md) para comprender dónde se guardan las salidas y qué reglas seguir.

---

## 2. Ejemplo Oficial

El **Ejemplo 1 — Chua fraccionario no suave con función descriptiva sesgada** es el caso de referencia para la búsqueda de candidatos compatibles con ocultedad. Su entrada de ejecución oficial es:
* **Archivo**: [run_example.py](../examples/chua_nonsmooth_biased_hidden_attractor/run_example.py)
* **Lógica interna**: Implementada de forma limpia en el núcleo de la librería (`hidden_attractors/workflows/biased_chua.py`).

> [!NOTE]
> **Advertencia Científica y de Reproducibilidad:** Este ejemplo **no es una reproducción del sistema de Danca (2017)**.
> El sistema original de Danca **no fue reproducible debido a la falta de información publicada** (como las coordenadas exactas de las condiciones iniciales del atractor oculto, parámetros del resolvedor DF, y el método de continuación numérica). Por consiguiente, este ejemplo realiza una búsqueda sistemática de candidatos en un sweep de parámetros para identificar vecindades compatibles con ocultedad.

Este ejemplo ejecuta de forma secuencial las siguientes fases:
1. **Paso 1**: Búsqueda centrada de referencia (DF centrada, $c=0$).
2. **Paso 2**: Búsqueda homotópica afín con función descriptiva sesgada (BDF, $c \neq 0$).
3. **Paso 3**: Verificación de ocultedad estándar mediante barrido local de esferas.
4. **Paso 4** (Opcional): Búsqueda extendida en paralelo de ocultedad volumétrica.
5. **Paso 5**: Resumen y exportación de figuras a la galería centralizada `library_figures/` según la [Política de Exportación de Figuras](figure_export_policy.md).

---

## 3. Presets de CLI y Comandos Unificados

Los presets de CLI son configuraciones empaquetadas listas para ejecutar con el comando `hidden-attractors`. Puedes consultar la lista completa y detalles de cada uno (sistema, estabilidad, propósito, etc.) en el [Índice de Ejemplos y Workflows](examples_index.md).

Ejemplos de uso:
```bash
# Buscar semillas usando la función descriptiva Lur'e centrada
hidden-attractors seed lure-centered -p chua_fractional

# Buscar semillas usando la función descriptiva Lur'e sesgada
hidden-attractors seed lure-biased -p chua_fractional

# Ejecutar continuación escalar
hidden-attractors continuation run -c path/to/config.yaml -s outputs/seeds.csv

# Ejecutar continuación multiparámetro
hidden-attractors continuation multiparameter -c path/to/config.yaml

# Ejecutar preset de Chua Fraccionario estándar
hidden-attractors run -p chua_fractional

# Ejecutar preset de Chua con no-linealidad arcotangente
hidden-attractors run -p chua_arctan

# Ejecutar un barrido de bifurcación
hidden-attractors bifurcation run -p chua_bifurcation

# Ejecutar estimación de exponentes de Lyapunov
hidden-attractors lyapunov compute -c configs/examples/chua_fractional_lyapunov.yaml

# Ejecutar la prueba de caos 0-1
hidden-attractors chaos-test zero-one -c configs/examples/chua_fractional_zero_one.yaml
```

---

## 4. Workflows Especializados

Los workflows especializados no son ejemplos independientes ni metodologías alternativas. Son interfaces de bajo nivel usadas por el pipeline oficial, pruebas de validación o análisis avanzados.

> [!WARNING]
> Los workflows especializados no son ejemplos independientes ni metodologías alternativas. Son interfaces de bajo nivel usadas por el pipeline oficial, pruebas de validación o análisis avanzados.

### Comandos de Workflows Especializados:
* **`hidden-attractors protocol`**: Ejecución secuencial y detallada del protocolo oficial (generación de semillas, continuación, validación, etc.).
* **`hidden-attractors robustness overlay`**: Análisis de robustez numérica variando tamaños de paso y condiciones del resolvedor.
* **`hidden-attractors basin refined`**: Refinamiento fino de las fronteras de cuencas de atracción locales.
* **`hidden-attractors hiddenness sphere-controls`** / **`hidden-attractors published danca-abm-sphere-controls`**: Pruebas de ocultedad en vecindades esféricas alrededor de los puntos de equilibrio.
* **`hidden-attractors basin strict-target-refinement`** / **`hidden-attractors hiddenness strict-target-refinement`**: Refinamiento numérico del atractor localizado.
* **`hidden-attractors report fractional-run`**: Generador automático de reportes científicos unificados.
* **`hidden-attractors validate contract`**: Controladores de validación de la consistencia interna.


---

## 5. API Programática (Python)

Para realizar desarrollos personalizados, la biblioteca expone módulos claros:

```python
from hidden_attractors.workflows.config_loader import load_config
from hidden_attractors.systems import get_system
from hidden_attractors.integrations.selector import integrate

# Cargar configuración desde un archivo YAML
config = load_config("configs/examples/chua_fractional_centered_lure_df.yaml")

# Obtener la definición de un sistema
system = get_system("chua-nonsmooth")

# Integrar numéricamente con el selector de solvers unificado
times, states, status = integrate(
    rhs=system.rhs,
    x0=[0.1, 0.0, 0.0],
    q=0.998,
    h=0.001,
    t_final=100.0,
    integrator="efork3",
    system=system
)
```

---

## 6. Legacy y Archivo Histórico

Historical migration scripts are intentionally excluded from the active repository.
The active implementation lives in `version_2/hidden_attractors/`.

* **Herramientas Legacy**: Conservadas bajo `version_2/tools/legacy/` solo para compatibilidad hacia atrás en resolvedores específicos de C.

