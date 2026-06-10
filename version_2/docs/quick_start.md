# Guía de Inicio Rápido — hidden-attractors-fo

Esta guía proporciona la ruta de entrada recomendada y directa para usuarias nuevas en la versión 2 de la biblioteca.

Los metadatos sincronizados de los manuales se definen en [docs/manual_manifest.yaml](manual_manifest.yaml); las afirmaciones científicas defendibles siguen gobernadas por [THESIS_CLAIMS.md](../THESIS_CLAIMS.md).

Para una descripción completa de instalación, CLI, ejemplos, salidas, etiquetas de evidencia y limitaciones, véase [USER_MANUAL.md](../USER_MANUAL.md).

---

## 1. Alcance Mínimo
Esta biblioteca está diseñada para definir, analizar y ejecutar **workflows reproducibles de candidatos a atractores ocultos** en sistemas dinámicos compatibles con la forma Lur’e (principalmente el circuito de Chua en sus variantes de orden entero y fraccionario).

> [!WARNING]
> **Advertencia Metodológica y Científica:**
> - El análisis de la función descriptiva (DF), Nyquist y los métodos de continuación numérica son herramientas heurísticas que únicamente sirven para **generar semillas o candidatos**. **No constituyen una prueba matemática de existencia ni de ocultedad**.
> - La verificación científica y rigurosa de la ocultedad requiere la comprobación exhaustiva del comportamiento transitorio en vecindades de **todos los puntos de equilibrio** del sistema.
> - Consulta la [Matriz de Afirmaciones de Tesis (Thesis Claims Matrix)](../THESIS_CLAIMS.md) para ver la clasificación actual de resultados y claims defendibles.

---

## 2. Instalación Mínima
Para instalar la biblioteca en modo de desarrollo editable, ejecuta los siguientes comandos desde la **raíz del repositorio**:

```bash
# Instalación estándar editable
pip install -e version_2

# Instalación completa para desarrollo y análisis avanzado
pip install -e "version_2[dev,analysis]"
```

---

## 3. Comando Público Único
La biblioteca se distribuye con un **único comando público y estable**:

```bash
hidden-attractors
```

Este comando expone de manera unificada toda la funcionalidad de la biblioteca. Los comandos independientes antiguos (como `hidden-attractors-protocol`, `hidden-attractors-sphere-controls`, `hidden-attractors-refined-basin`, etc.) ya no se instalan como ejecutables globales, sino que están disponibles a través de subcomandos de `hidden-attractors` o se consideran interfaces de uso interno/desarrollador.

---

## 4. Primer Chequeo de Instalación
Para verificar que el entorno local se ha configurado de manera correcta, puedes realizar las siguientes consultas rápidas al CLI:

```bash
# Mostrar ayuda general y lista de grupos de comandos
hidden-attractors --help

# Inspeccionar los sistemas caóticos registrados en el framework
hidden-attractors inspect systems

# Listar los registros de candidatos a atractores registrados
hidden-attractors inspect candidates
```

---

## 5. Ejecución Mínima Recomendada
La ruta de ejecución más sencilla para comenzar utiliza los presets predefinidos en la biblioteca. Puedes extraer, previsualizar y ejecutar el preset fraccionario por defecto con los siguientes pasos:

```bash
# 1. Inicializar y copiar el preset 'chua_fractional' al directorio de trabajo actual
hidden-attractors init -e chua_fractional

# 2. Previsualizar la configuración efectiva y sus parámetros por defecto
hidden-attractors inspect-config -p chua_fractional

# 3. Ejecutar el pipeline de localización y simulación para el preset
hidden-attractors run -p chua_fractional
```

---

## 6. Ejemplo Oficial 1
El **Ejemplo 1 — Chua fraccionario no suave con función descriptiva sesgada** es el principal flujo completo reproducible para explorar el pipeline BDF (Biased Describing Function) y el integrador fraccionario.

Para ejecutar la prueba de humo rápida (~1-2 minutos):
```bash
cd version_2/examples/chua_nonsmooth_biased_hidden_attractor
python run_example.py --quick
```

> [!NOTE]
> **Aclaraciones sobre el Ejemplo 1:**
> - Es una **prueba de humo** para validar la integración mediante el pipeline BDF y el resolvedor fraccionario.
> - **No constituye una reproducción exacta del sistema de Danca (2017)** debido a la falta de información publicada originalmente (condiciones iniciales del atractor oculto, resolvedor DF y detalles de la continuación).
> - La simulación de este ejemplo **no prueba la ocultedad por sí misma**; las clasificaciones fuertes de ocultedad en la biblioteca dependen estrictamente del contrato de vecindades esféricas alrededor de los equilibrios.

---

## 7. Mapa de Subcomandos Agrupados
Para usuarias avanzadas e investigación, la interfaz CLI unificada agrupa los comandos según su propósito:

```bash
# Generación de semillas iniciales
hidden-attractors seed lure-centered
hidden-attractors seed lure-biased

# Continuación numérica
hidden-attractors continuation run
hidden-attractors continuation multiparameter

# Análisis de bifurcación
hidden-attractors bifurcation run
hidden-attractors bifurcation plot
hidden-attractors bifurcation inspect

# Espectro y exponentes de Lyapunov
hidden-attractors lyapunov compute
hidden-attractors lyapunov spectrum
hidden-attractors lyapunov validate

# Pruebas complementarias de caos
hidden-attractors chaos-test zero-one
hidden-attractors chaos-test inspect

# Pruebas de vecindades y clasificación de ocultedad
hidden-attractors hiddenness sphere-controls
hidden-attractors hiddenness strict-target-refinement
hidden-attractors basin refined
hidden-attractors basin strict-target-refinement

# Validación y contratos de datos
hidden-attractors validate contract
hidden-attractors validate bibliography

# Protocolo oficial Caputo paso a paso
hidden-attractors protocol <substage>
```

> [!NOTE]
> **Nota de Estabilidad:** Los subcomandos avanzados se consideran interfaces de investigación (tier experimental/alfa). La ruta de ejecución estable y recomendada para usuarias nuevas es el comando `hidden-attractors run` configurado mediante presets o archivos YAML.

---

## 8. Qué NO Ejecutar
Para garantizar la estabilidad y reproducibilidad científica del repositorio, sigue estrictamente estas directrices:

* 🚫 **No ejecutar scripts históricos ni scratch**: Todos los scripts de migración o carpetas temporales están excluidos del flujo de trabajo estándar.
* 🚫 **No ejecutar comandos legacy independientes**: No intentes invocar comandos globales antiguos como `hidden-attractors-protocol`. Usa en su lugar la sintaxis de subcomando unificada (`hidden-attractors protocol`).
* 🚫 **No guardar figuras manualmente fuera de `library_figures`**: Todos los gráficos y figuras exportados deben ir a través de la API unificada para centralizar los resultados en `library_figures/`. Para más detalles, consulta la [Política de Exportación de Figuras](figure_export_policy.md).
* 🚫 **No clasificar un candidato como oculto prematuramente**: Nunca declares un candidato como atractor oculto basándote únicamente en simulaciones de un solo punto inicial, Nyquist/DF o continuación. Se requiere la ejecución completa de la validación de vecindades de los equilibrios.

---

## 9. Estados de Evidencia
Las clasificaciones resultantes de los flujos de verificación en la biblioteca se organizan en los siguientes estados oficiales:

| Estado de Evidencia | Descripción |
| :--- | :--- |
| `seed_only` | Solo se ha generado una semilla inicial por análisis armónico (DF). |
| `continuation_survivor` | La semilla sobrevivió a la deformación del parámetro de continuación $\eta \in [0, 1]$. |
| `compatible_with_hiddenness_under_tested_radii` | El atractor no intersecta con vecindades probadas de equilibrios estables. |
| `hiddenness_supported_under_tested_neighborhoods` | El atractor está numéricamente verificado como oculto tras muestrear exhaustivamente esferas de condiciones iniciales alrededor de todos los equilibrios (inestables). |
| `self_excited_contact_detected` | Se ha detectado contacto directo con el flujo originado en un punto de equilibrio; el atractor es autoexcitado. |
| `hiddenness_inconclusive` | Las integraciones numéricas fallaron o los resultados no son concluyentes bajo los parámetros dados. |
| `candidate_rejected` | El candidato divergió o colapsó permanentemente a un punto de equilibrio. |

> [!IMPORTANT]
> El estado `hiddenness_supported_under_tested_neighborhoods` **no representa una demostración matemática global**. Es una clasificación numérica bajo el contrato de integración, paso de tiempo ($h$), orden fraccionario ($q$) y radios de vecindades definidos y probados.

---

## 10. Aspectos Matemáticos y Convenciones
Cuando se trabaja con la función descriptiva fraccionaria (DF), se utiliza la convención estándar para la función de transferencia del sistema lineal $W_q(s)$:

$$W_q(s) = r^T (s^q I - P)^{-1} b$$

Donde la variable compleja de Laplace en el dominio de frecuencia se define por:

$$\lambda = (j \omega)^q$$

*Nota metodológica sobre el signo de BDF:* Si necesitas verificar las ecuaciones detalladas de la función descriptiva sesgada (BDF) con su signo histórico de acoplamiento, consulta la documentación técnica y las fórmulas descritas en los comentarios del código del módulo `hidden_attractors/seed_generation/chua.py`.
