# Quick Start Guide / Guía de Inicio Rápido — hidden-attractors-fo

This guide provides the recommended direct entry point for new users of version 2 of the library.
Esta guía proporciona la ruta de entrada recomendada y directa para usuarias nuevas en la versión 2 de la biblioteca.

Manual metadata synchronization is defined in [docs/manual_manifest.yaml](manual_manifest.yaml); defensible scientific claims remain governed by [THESIS_CLAIMS.md](../THESIS_CLAIMS.md).
Los metadatos sincronizados de los manuales se definen en [docs/manual_manifest.yaml](manual_manifest.yaml); las afirmaciones científicas defendibles siguen gobernadas por [THESIS_CLAIMS.md](../THESIS_CLAIMS.md).

For a complete description of installation, CLI, examples, outputs, evidence labels, and limitations, see [USER_MANUAL.md](../USER_MANUAL.md).
Para una descripción completa de instalación, CLI, ejemplos, salidas, etiquetas de evidencia y limitaciones, véase [USER_MANUAL.md](../USER_MANUAL.md).

---

## 1. Minimum Scope / Alcance Mínimo

This library is designed to define, analyze, and execute **reproducible workflows for hidden attractor candidates** in dynamical systems compatible with the Lur’e form (mainly the Chua circuit in its integer-order and fractional-order variants).
Esta biblioteca está diseñada para definir, analizar y ejecutar **workflows reproducibles de candidatos a atractores ocultos** en sistemas dinámicos compatibles con la forma Lur’e (principalmente el circuito de Chua en sus variantes de orden entero y fraccionario).

> [!WARNING]
> **Methodological and Scientific Warning:**
> - Describing Function (DF) analysis, Nyquist plots, and numerical continuation methods are heuristic tools that only serve to **generate seeds or candidates**. **They do not constitute a mathematical proof of existence or hiddenness**.
> - Rigorous scientific verification of hiddenness requires exhaustive testing of transient behavior in the neighborhoods of **all equilibrium points** of the system.
> - Consult the [Thesis Claims Matrix](../THESIS_CLAIMS.md) to see the current classification of results and defensible claims.
>
> **Advertencia Metodológica y Científica:**
> - El análisis de la función descriptiva (DF), Nyquist y los métodos de continuación numérica son herramientas heurísticas que únicamente sirven para **generar semillas o candidatos**. **No constituyen una prueba matemática de existencia ni de ocultedad**.
> - La verificación científica y rigurosa de la ocultedad requiere la comprobación exhaustiva del comportamiento transitorio en vecindades de **todos los puntos de equilibrio** del sistema.
> - Consulta la [Matriz de Afirmaciones de Tesis (Thesis Claims Matrix)](../THESIS_CLAIMS.md) para ver la clasificación actual de resultados y claims defendibles.

---

## 2. Minimum Installation / Instalación Mínima

To install the library in editable development mode, run the following commands from the **repository root**:
Para instalar la biblioteca en modo de desarrollo editable, ejecuta los siguientes comandos desde la **raíz del repositorio**:

```bash
# Standard editable install
pip install -e version_2

# Full install for development and advanced analysis
pip install -e "version_2[dev,analysis]"
```

---

## 3. Single Public Command / Comando Público Único

The library distributes a **single public and stable command**:
La biblioteca se distribuye con un **único comando público y estable**:

```bash
hidden-attractors
```

This command exposes all library functionality in a unified way. Legacy standalone commands (such as `hidden-attractors-protocol`, `hidden-attractors-sphere-controls`, `hidden-attractors-refined-basin`, etc.) are no longer installed as global executables. They are now available as subcommands of `hidden-attractors` or are considered internal/developer interfaces.

Este comando expone de manera unificada toda la funcionalidad de la biblioteca. Los comandos independientes antiguos (como `hidden-attractors-protocol`, `hidden-attractors-sphere-controls`, `hidden-attractors-refined-basin`, etc.) ya no se instalan como ejecutables globales, sino que están disponibles a través de subcomandos de `hidden-attractors` o se consideran interfaces de uso interno/desarrollador.

---

## 4. First Installation Check / Primer Chequeo de Instalación

To verify that the local environment is configured correctly, run these quick CLI checks:
Para verificar que el entorno local se ha configurado de manera correcta, puedes realizar las siguientes consultas rápidas al CLI:

```bash
# Show general help and command groups
hidden-attractors --help

# Inspect chaotic systems registered in the framework
hidden-attractors inspect systems

# List registered attractor candidate records
hidden-attractors inspect candidates
```

---

## 5. Minimum Recommended Run / Ejecución Mínima Recomendada

The easiest way to start is by using the library's built-in presets. You can extract, preview, and run the default fractional preset using these steps:
La ruta de ejecución más sencilla para comenzar utiliza los presets predefinidos en la biblioteca. Puedes extraer, previsualizar y ejecutar el preset fraccionario por defecto con los siguientes pasos:

```bash
# 1. Initialize and copy the 'chua_fractional' preset to the current working directory
hidden-attractors init -e chua_fractional

# 2. Preview the effective configuration and default parameters
hidden-attractors inspect-config -p chua_fractional

# 3. Run the localization and simulation pipeline for the preset
hidden-attractors run -p chua_fractional
```

---

## 6. Official Example 1 / Ejemplo Oficial 1

**Example 1 — Non-Smooth Fractional Chua with Biased Describing Function** is the primary reproducible workflow to explore the Biased Describing Function (BDF) pipeline and the fractional solver.
El **Ejemplo 1 — Chua fraccionario no suave con función descriptiva sesgada** es el principal flujo completo reproducible para explorar el pipeline BDF (Biased Describing Function) y el integrador fraccionario.

To run the quick smoke check (~1-2 minutes):
Para ejecutar la prueba de humo rápida (~1-2 minutos):
```bash
cd version_2/examples/chua_nonsmooth_biased_hidden_attractor
python run_example.py --quick
```

> [!NOTE]
> **Example 1 Clarifications:**
> - This is a **smoke check** to validate integration using the BDF pipeline and the fractional solver.
> - **It is not an exact reproduction of Danca's (2017) system** due to missing published data (attractor initial conditions, DF solver settings, and continuation details).
> - Simulating this example **does not prove hiddenness on its own**; strong hiddenness classifications in the library depend strictly on the spherical neighborhood contracts around equilibria.
>
> **Aclaraciones sobre el Ejemplo 1:**
> - Es una **prueba de humo** para validar la integración mediante el pipeline BDF y el resolvedor fraccionario.
> - **No constituye una reproducción exacta del sistema de Danca (2017)** debido a la falta de información publicada originalmente (condiciones iniciales del atractor oculto, resolvedor DF y detalles de la continuación).
> - La simulación de este ejemplo **no prueba la ocultedad por sí misma**; las clasificaciones fuertes de ocultedad en la biblioteca dependen estrictamente del contrato de vecindades esféricas alrededor de los equilibrios.

---

## 7. Subcommands Map / Mapa de Subcomandos Agrupados

For advanced users and research, the unified CLI groups commands by purpose:
Para usuarias avanzadas e investigación, la interfaz CLI unificada agrupa los comandos según su propósito:

```bash
# Seed generation (harmonic analysis)
hidden-attractors seed lure-centered
hidden-attractors seed lure-biased

# Numerical continuation
hidden-attractors continuation run
hidden-attractors continuation multiparameter

# Bifurcation sweeps
hidden-attractors bifurcation run
hidden-attractors bifurcation plot
hidden-attractors bifurcation inspect

# Lyapunov exponents and spectra
hidden-attractors lyapunov compute
hidden-attractors lyapunov spectrum
hidden-attractors lyapunov validate

# Complementary chaos diagnostics
hidden-attractors chaos-test zero-one
hidden-attractors chaos-test inspect

# Neighborhood tests and hiddenness classification
hidden-attractors hiddenness sphere-controls
hidden-attractors hiddenness strict-target-refinement
hidden-attractors basin refined
hidden-attractors basin strict-target-refinement

# Validation contracts and checks
hidden-attractors validate contract
hidden-attractors validate bibliography

# Official step-by-step Caputo protocol
hidden-attractors protocol <substage>
```

> [!NOTE]
> **Stability Note:** Advanced subcommands are considered research interfaces (alpha/experimental tier). The recommended stable run path for new users is the `hidden-attractors run` command configured via presets or YAML files.
>
> **Nota de Estabilidad:** Los subcomandos avanzados se consideran interfaces de investigación (tier experimental/alfa). La ruta de ejecución estable y recomendada para usuarias nuevas es el comando `hidden-attractors run` configurado mediante presets o archivos YAML.

---

## 8. What NOT to Run / Qué NO Ejecutar

To guarantee stability and scientific reproducibility of the repository, strictly follow these guidelines:
Para garantizar la estabilidad y reproducibilidad científica del repositorio, sigue estrictamente estas directrices:

* 🚫 **Do not run historical or scratch scripts**: All migration scripts and temporary folders are excluded from the standard workflow.
* 🚫 **Do not run legacy independent commands**: Do not invoke legacy global commands like `hidden-attractors-protocol`. Use the unified subcommand syntax (`hidden-attractors protocol`) instead.
* 🚫 **Do not save figures outside `library_figures`**: All exported plots and figures must go through the unified API to centralize results in `library_figures/`. See [Figure Export Policy](figure_export_policy.md).
* 🚫 **Do not classify a candidate as hidden prematurely**: Never declare a candidate as hidden based solely on single initial point simulations, Nyquist/DF analysis, or continuation. Complete execution of equilibrium-neighborhood validation is required.
* 
* 🚫 **No ejecutar scripts históricos ni scratch**: Todos los scripts de migración o carpetas temporales están excluidos del flujo de trabajo estándar.
* 🚫 **No ejecutar comandos legacy independientes**: No intentes invocar comandos globales antiguos como `hidden-attractors-protocol`. Usa en su lugar la sintaxis de subcomando unificada (`hidden-attractors protocol`).
* 🚫 **No guardar figuras manualmente fuera de `library_figures`**: Todos los gráficos y figuras exportados deben ir a través de la API unificada para centralizar los resultados en `library_figures/`. Para más detalles, consulta la [Política de Exportación de Figuras](figure_export_policy.md).
* 🚫 **No clasificar un candidato como oculto prematuramente**: Nunca declares un candidato como atractor oculto basándote únicamente en simulaciones de un solo punto inicial, Nyquist/DF o continuación. Se requiere la ejecución completa de la validación de vecindades de los equilibrios.

---

## 9. Canonical Attractor Statuses / Estados Canónicos de Atractores

The classifications resulting from the library's verification workflows are organized into the following official canonical statuses:
Las clasificaciones resultantes de los flujos de verificación en la biblioteca se organizan en los siguientes estados oficiales canónicos:

| Status / Estado | Description / Descripción |
| :--- | :--- |
| `candidate` | An initial seed has been generated or survived basic integration. / Solo se ha generado una semilla o sobrevivió a integración básica. |
| `hidden_under_tested_neighborhoods` | Vertically verified as hidden by exhaustively sampling spheres of initial conditions around all (unstable) equilibria. / Verificado numéricamente como oculto tras muestrear exhaustivamente esferas de condiciones iniciales alrededor de todos los equilibrios (inestables). |
| `compatible_with_hiddenness` | The attractor does not intersect tested neighborhoods of stable equilibria. / El atractor no intersecta con vecindades probadas de equilibrios estables. |
| `self_excited` | Direct contact detected with flow originating from an equilibrium point; the attractor is self-excited. / Se ha detectado contacto directo con el flujo originado en un punto de equilibrio; el atractor es autoexcitado. |
| `nonchaotic` | The attractor is regular (limit cycle or quasiperiodic). / El atractor es regular (ciclo límite o cuasiperiódico). |
| `diverged` | The trajectory diverged to infinity. / La trayectoria divergió al infinito. |
| `inconclusive` | Numerical integrations failed or results are inconclusive under given parameters. / Las integraciones numéricas fallaron o los resultados no son concluyentes bajo los parámetros dados. |
| `rejected` | The candidate collapsed to an equilibrium point. / El candidato colapsó permanentemente a un punto de equilibrio. |
| `not_tested` | Verification checks have not yet run on the candidate. / Los chequeos de verificación aún no se han ejecutado sobre el candidato. |

> [!IMPORTANT]
> The status `hidden_under_tested_neighborhoods` **does not represent a global mathematical proof**. It is a numerical classification under the recorded integration step ($h$), fractional order ($q$), and tested neighborhood radii.
>
> El estado `hidden_under_tested_neighborhoods` **no representa una demostración matemática global**. Es una clasificación numérica bajo el contrato de integración, paso de tiempo ($h$), orden fraccionario ($q$) y radios de vecindades definidos y probados.

---

## 10. Mathematical Aspects and Conventions / Aspectos Matemáticos y Convenciones

When working with the fractional describing function (DF), we use the standard convention for the linear system transfer function $W_q(s)$:
Cuando se trabaja con la función descriptiva fraccionaria (DF), se utiliza la convención estándar para la función de transferencia del sistema lineal $W_q(s)$:

$$W_q(s) = r^T (s^q I - P)^{-1} b$$

Where the complex Laplace variable in the frequency domain is defined by:
Donde la variable compleja de Laplace en el dominio de frecuencia se define por:

$$\lambda = (j \omega)^q$$

*Methodological note on the BDF sign:* If you need to verify the detailed equations of the biased describing function (BDF) with its historical coupling sign, consult the technical documentation and the formulas described in the code comments of the module `hidden_attractors/seed_generation/chua.py`.
*Nota metodológica sobre el signo de BDF:* Si necesitas verificar las ecuaciones detalladas de la función descriptiva sesgada (BDF) con su signo histórico de acoplamiento, consulta la documentación técnica y las fórmulas descritas en los comentarios del código del módulo `hidden_attractors/seed_generation/chua.py`.
