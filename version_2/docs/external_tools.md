# External Tools / Herramientas Externas

## Table of Contents / Índice de Contenidos

- [English Version](#english-version)
- [Versión en Español](#versión-en-español)

---

## English Version

### External Tools (EN)

The project should reuse and cite established tools instead of copying mature algorithms into this repository. Local code focuses on adapting outputs from the hidden-attractor workflows to those tools and documenting how they fit.

## PyDSTool (EN)

- Project: [PyDSTool](https://pydstool.github.io/PyDSTool/FrontPage.html)
- Role: dynamical-systems modeling, simulation, phase-plane analysis, continuation, and bifurcation analysis.
- Use here: reference or optional companion tool for continuation and branch tracking when the environment is compatible.
- Proyecto: [PyDSTool](https://pydstool.github.io/PyDSTool/FrontPage.html)

PyDSTool is broader than the local post-processing helpers. The functions in `hidden_attractors.analysis.bifurcation` only extract maxima, minima, or samples from already computed trajectories. They are not a continuation engine.

## pyComplexity Notebook

- Reference: [pyComplexity.ipynb](https://github.com/relopezbriega/relopezbriega.github.io/blob/master/downloads/pyComplexity.ipynb)
- Role: notebook-style exposition of complexity analysis.
- Use here: documentation and notebook style reference for presenting scalar complexity diagnostics clearly.
- Referencia: [pyComplexity.ipynb](https://github.com/relopezbriega/relopezbriega.github.io/blob/master/downloads/pyComplexity.ipynb)

Do not copy notebook code into this repository unless the license and citation requirements have been reviewed. The local adapter delegates scalar measures to installable libraries such as `nolds` and `antropy`.

## nolds (EN)

- Project: [nolds](https://pypi.org/project/nolds/)
- Role: nonlinear time-series measures such as sample entropy, correlation dimension, Lyapunov estimates, Hurst exponent, and DFA.
- Install:
- Proyecto: [nolds](https://pypi.org/project/nolds/)

```bash
python -m pip install nolds
```

Usage:
Uso:

```python
from hidden_attractors.integrations import compute_complexity_measures

metrics = compute_complexity_measures(trajectory[:, 1], backend="nolds")
```

## antropy (EN)

- Project: [antropy](https://pypi.org/project/antropy/)
- Role: entropy and fractal diagnostics such as permutation entropy, spectral entropy, sample entropy, Higuchi fractal dimension, and DFA.
- Install:
- Proyecto: [antropy](https://pypi.org/project/antropy/)

```bash
python -m pip install antropy
```

Usage:
Uso:

```python
from hidden_attractors.integrations import compute_complexity_measures

metrics = compute_complexity_measures(trajectory[:, 1], backend="antropy")
```

## Tool Registry

The current registry is available from Python:

```python
from hidden_attractors.integrations import external_tool_report

for row in external_tool_report():
    print(row["name"], row["available"], row["recommended_use"])
```

This registry is intentionally small. Add new external tools only when they solve a real analysis need and the repository can document how they apply to the fractional Chua/Lur'e study systems.

---

## Versión en Español

### Herramientas Externas (ES)

El proyecto debe reutilizar y citar herramientas establecidas en lugar de copiar algoritmos maduros en este repositorio. El código local se centra en adaptar las salidas de los flujos de trabajo de atractores ocultos a esas herramientas y documentar cómo encajan.

## PyDSTool (ES)

- Project: [PyDSTool](https://pydstool.github.io/PyDSTool/FrontPage.html)
- Proyecto: [PyDSTool](https://pydstool.github.io/PyDSTool/FrontPage.html)
- Rol: modelado de sistemas dinámicos, simulación, análisis del plano de fase, continuación y análisis de bifurcación.
- Uso aquí: herramienta de referencia o acompañante opcional para la continuación y el seguimiento de ramas cuando el entorno sea compatible.

PyDSTool es más amplio que los ayudantes de posprocesamiento locales. Las funciones en `hidden_attractors.analysis.bifurcation` solo extraen máximos, mínimos o muestras de trayectorias ya calculadas. No son un motor de continuación.

## Notebook pyComplexity

- Reference: [pyComplexity.ipynb](https://github.com/relopezbriega/relopezbriega.github.io/blob/master/downloads/pyComplexity.ipynb)
- Referencia: [pyComplexity.ipynb](https://github.com/relopezbriega/relopezbriega.github.io/blob/master/downloads/pyComplexity.ipynb)
- Rol: exposición en estilo notebook del análisis de complejidad.
- Uso aquí: documentación y referencia estilo notebook para presentar claramente los diagnósticos de complejidad escalar.

No copie el código del notebook en este repositorio a menos que se hayan revisado los requisitos de licencia y cita. El adaptador local delega las medidas escalares a bibliotecas instalables como `nolds` y `antropy`.

## nolds (ES)

- Project: [nolds](https://pypi.org/project/nolds/)
- Install:
- Proyecto: [nolds](https://pypi.org/project/nolds/)
- Rol: medidas de series temporales no lineales como entropía muestral, dimensión de correlación, estimaciones de Lyapunov, exponente de Hurst y DFA.
- Instalación:

```bash
python -m pip install nolds
```

Usage:
Uso:

```python
from hidden_attractors.integrations import compute_complexity_measures

metrics = compute_complexity_measures(trajectory[:, 1], backend="nolds")
```

## antropy (ES)

- Project: [antropy](https://pypi.org/project/antropy/)
- Install:
- Proyecto: [antropy](https://pypi.org/project/antropy/)
- Rol: diagnósticos de entropía y fractales como entropía de permutación, entropía espectral, entropía muestral, dimensión fractal de Higuchi y DFA.
- Instalación:

```bash
python -m pip install antropy
```

Usage:
Uso:

```python
from hidden_attractors.integrations import compute_complexity_measures

metrics = compute_complexity_measures(trajectory[:, 1], backend="antropy")
```

## Registro de Herramientas

El registro actual está disponible desde Python:

```python
from hidden_attractors.integrations import external_tool_report

for row in external_tool_report():
    print(row["name"], row["available"], row["recommended_use"])
```

Este registro es intencionadamente pequeño. Agregue nuevas herramientas externas solo cuando resuelvan una necesidad de análisis real y el repositorio pueda documentar cómo se aplican a los sistemas de estudio fraccionarios de Chua/Lur'e.
