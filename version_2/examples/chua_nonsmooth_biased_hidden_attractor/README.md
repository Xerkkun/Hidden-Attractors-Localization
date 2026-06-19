# Hidden Attractor Candidate Search in Non-Smooth Fractional Chua — Official Example / Búsqueda de Candidatos Ocultos en Chua Fraccionario No Suave — Ejemplo Oficial

## Table of Contents / Índice de Contenidos
- [English Version](#english-version)
- [Versión en Español](#versión-en-español)

---

## English Version

> **Library:** `hidden_attractors_fo` · **Version:** 2  
> This example documents the complete methodological process for searching for hidden attractor candidates in the non-smooth fractional Chua system ($q = 0.9998$) using the Biased Describing Function (BDF).
>
> ⚠️ **Scientific and Reproducibility Note:** This example **does not represent a reproduction of the system from Danca (2017)**. The original system in the paper was not reproducible due to the lack of published information (attractor initial conditions, spectral details of DF such as $\omega_0$, and the exact continuation method are not reported). In addition, the sweep parameters, describing function, and numerical continuation used here are different.

### Structure of the Example

The example has been cleaned and modularized. All simulation, homotopy, spectral analysis, and verification logic has been moved to the core of the library. This directory only contains the official entry point:

```text
chua_nonsmooth_biased_hidden_attractor/
├── run_example.py      ← Only official entry point (runs the whole pipeline)
├── README.md           ← (this file)
```

The internal logic for the stages is located in:
- [biased_chua.py](../../hidden_attractors/workflows/biased_chua.py) (Workflows module)
- [biased_chua.py](../../hidden_attractors/plotting/biased_chua.py) (Centralized plotting module)

Historical scripts were used during migration and are intentionally excluded from the active repository. The active workflow is implemented in `version_2/hidden_attractors/`.

### Quick Start / Execution

The execution of all pipeline steps is performed via a single script:

```bash
# Quick smoke check (short simulations and few radii, ~1-2 min)
python run_example.py --quick

# Standard execution (Steps 1, 2, 3, 5 — without the massive extended test, ~10-15 min)
python run_example.py

# Full execution including Step 4 (massive parallelized verification, can take hours)
python run_example.py --all

# Run individual steps (e.g., step 2 and step 5)
python run_example.py --steps 2 5
```

### The System

The non-smooth fractional Chua circuit is described by the fractional Lur'e system of Caputo order $q$:

$$
D^q x = P x + b \cdot f(\sigma), \quad \sigma = r^T x
$$

where the nonlinearity is the bilinear saturation (non-smooth):

$$
f(\sigma) = m_1 \cdot \sigma + \frac{m_0 - m_1}{2}(|\sigma + 1| - |\sigma - 1|)
$$

**Study parameters:**

| Parameter | Value |
|---|---|
| $q$ (Caputo order) | 0.9998 |
| $\alpha$ | 8.4562 |
| $\beta$ | 12.0732 |
| $\gamma$ | 0.0052 |
| $m_1$ (candidate) | −1.1468 |
| $m_0$ (candidate) | −0.1768 |
| Integrator | Caputo ABM, full history |
| $h$ (step size) | 0.01 s |

### Pipeline Steps

1. **Step 1 — Centered Describing Function (Baseline):** Search for standard DF branches with bias $c = 0$. Serves as a baseline comparison. Produces periodic and non-chaotic attractors.
2. **Step 2 — Biased Describing Function (BDF):** Extension to the case with DC bias ($c \neq 0$), root-solving in $(A, c, \omega)$ using the convention $1 + W_q(j\omega)N_1 = 0$, algebraic reconstruction of consistent seeds, affine Caputo ABM continuation from the linearized system ($\eta=0$) to the real system ($\eta=1$), and a final long simulation.
3. **Step 3 — Hiddenness Verification (Standard Protocol):** Generation of 225 initial conditions on spheres of decreasing radii around the 3 stable equilibria to scan for self-excited contacts.
4. **Step 4 — Extended Verification (Multiprocessing):** Massive test of probes in spherical volumes (ball sampling) of large radii (up to $r=2.0$) to evaluate the local penetrability of equilibrium neighborhoods.
5. **Step 5 — Summary and Figure Gallery:** Export of statistical tables, 7-panel attractor report, homotopy plots, and comparative mosaics under the library's unified plotting rules.

### Results and Hiddenness Classification

The pipeline evaluates trajectories under the local spherical neighborhood protocol. The results are summarized below:

| $m_1$ | $m_0$ | bias $c$ | Classification (Step 3) | Note / Contacts detected |
|---|---|---|---|---|
| −1.1468 | −0.1768 | +2.776 | `compatible_with_hiddenness` | Incomplete local protocol (0 contacts in local spheres). No evidence of being locally self-excited. |
| −1.1468 | −0.200  | −2.705 | `compatible_with_hiddenness` | Incomplete local protocol (0 contacts in local spheres). No evidence of being locally self-excited. |
| −1.1468 | −0.240  | −2.581 | `self_excited` | Not compatible with hiddenness (5 contacts in $E+$ and $E-$ neighborhoods). |

#### Relationship with the Official Validation Manifest
It is very important to distinguish this quick example sweep from the rigorous validation of the official manifest (`validation_manifest.json`):
1. **Selected Candidate:** The official manifest evaluates a modified grid candidate `danca2017_nearby_saturation_candidate_q09998` ($m_1 = -1.2$, $m_0 = -0.2$), which was classified as `chaotic_self_excited_candidate_not_hidden_under_tested_equilibrium_neighborhoods` after recording 1305 direct contacts with the attractor.
2. **Robustness and Diagnostics:** The official manifest marks the robustness and diagnostics stages as pending or incomplete, which emphasizes that there is no global certification of a hidden attractor in the suite.
3. **Example 1 (Main Candidate):** The main candidate of this example ($m_1 = -1.1468$, $m_0 = -0.1768$, $c = +2.776$) showed no evidence of being self-excited in the spherical radius tests carried out in Step 3 (0 contacts), remaining provisionally classified as compatible with hiddenness (`compatible_with_hiddenness`) under that limited scope, without this constituting an absolute mathematical or global proof.

All figures corresponding to the runs are automatically exported to the canonical figures folder: `version_2/library_figures/`.

---

## Versión en Español

> **Librería:** `hidden_attractors_fo` · **Versión:** 2  
> Este ejemplo documenta el proceso metodológico completo para buscar candidatos a atractores ocultos en el sistema de Chua fraccionario no suave ($q = 0.9998$) mediante la Función Descriptiva Sesgada (BDF).
>
> ⚠️ **Nota Científica y de Reproducibilidad:** Este ejemplo **no representa una reproducción del sistema de Danca (2017)**. El sistema original del artículo **no fue reproducible debido a la falta de información publicada** (no se reportan las coordenadas de condiciones iniciales del atractor oculto, detalles espectrales de DF como $\omega_0$, ni el método exacto de continuación). Además, los parámetros de sweep, la función descriptiva y la continuación numérica empleadas aquí son diferentes.

### Estructura del Ejemplo

El ejemplo ha sido limpiado y modularizado. Toda la lógica de simulación, homotopía, análisis espectral y verificación se ha movido al core de la librería. Este directorio contiene únicamente el punto de entrada oficial:

```text
chua_nonsmooth_biased_hidden_attractor/
├── run_example.py      ← Único punto de entrada oficial (ejecuta todo el pipeline)
├── README.md           ← (este archivo)
```

La lógica interna de los pasos está alojada en:
- [biased_chua.py](../../hidden_attractors/workflows/biased_chua.py) (Módulo de Workflows)
- [biased_chua.py](../../hidden_attractors/plotting/biased_chua.py) (Módulo de Ploteo centralizado)

Los scripts históricos se utilizaron durante la migración y están excluidos intencionalmente del repositorio activo. El flujo de trabajo activo se implementa en `version_2/hidden_attractors/`.

### Ejecución Rápida

La ejecución de todos los pasos del flujo se realiza a través de un único script:

```bash
# Prueba de humo rápida (simulaciones cortas y pocos radios, ~1-2 min)
python run_example.py --quick

# Ejecución estándar (Pasos 1, 2, 3, 5 — sin el test extendido masivo, ~10-15 min)
python run_example.py

# Ejecución completa incluyendo Paso 4 (verificación extendida en paralelo, puede tomar horas)
python run_example.py --all

# Ejecutar pasos individuales (por ejemplo, paso 2 y paso 5)
python run_example.py --steps 2 5
```

### El Sistema

El circuito de Chua fraccionario no suave se describe mediante el sistema de Lur'e fraccionario de orden Caputo $q$:

$$
D^q x = P x + b \cdot f(\sigma), \quad \sigma = r^T x
$$

donde la no linealidad es la saturación bilineal (no suave):

$$
f(\sigma) = m_1 \sigma + \frac{m_0 - m_1}{2}(|\sigma + 1| - |\sigma - 1|)
$$

**Parámetros del estudio:**

| Parámetro | Valor |
|---|---|
| $q$ (orden Caputo) | 0.9998 |
| $\alpha$ | 8.4562 |
| $\beta$ | 12.0732 |
| $\gamma$ | 0.0052 |
| $m_1$ (candidato) | −1.1468 |
| $m_0$ (candidato) | −0.1768 |
| Integrador | Caputo ABM, memoria completa |
| $h$ (paso) | 0.01 s |

### Pasos del Pipeline

1. **Paso 1 — Función Descriptiva Centrada (Base):** Búsqueda de ramas de la DF estándar con bias $c = 0$. Sirve de línea base de comparación. Produce atractores periódicos y no caóticos.
2. **Paso 2 — Función Descriptiva Sesgada (BDF):** Extensión al caso con bias DC ($c \neq 0$), resolución de raíces en $(A, c, \omega)$ usando la convención $1 + W_q(j\omega)N_1 = 0$, reconstrucción algebraica de semillas consistentes, continuación afín Caputo ABM desde el sistema linealizado ($\eta=0$) al real ($\eta=1$), y simulación final larga.
3. **Paso 3 — Verificación de Ocultedad (Protocolo Estándar):** Generación de 225 condiciones iniciales sobre esferas de radios decrecientes alrededor de los 3 equilibrios estables para buscar contactos autoexcitados.
4. **Paso 4 — Verificación Extendida (Multiprocessing):** Test masivo de sondas en volumen de esferas (ball sampling) de radios grandes (hasta $r=2.0$) para evaluar la penetrabilidad local de las vecindades de los equilibrios.
5. **Paso 5 — Resumen y Galería de Figuras:** Exportación de tablas estadísticas, reporte de atractor de 7 paneles, gráficos de homotopía, y mosaicos comparativos bajo las reglas de ploteo unificado de la librería.

### Resultados y Clasificación de Ocultedad

El pipeline evalúa las trayectorias bajo el protocolo local de vecindades esféricas. Los resultados se resumen a continuación:

| $m_1$ | $m_0$ | bias $c$ | Clasificación (Paso 3) | Nota / Hits detectados |
|---|---|---|---|---|
| −1.1468 | −0.1768 | +2.776 | `compatible_with_hiddenness` | Protocolo local incompleto (0 hits en esferas locales). Sin evidencia de ser autoexcitado localmente. |
| −1.1468 | −0.200  | −2.705 | `compatible_with_hiddenness` | Protocolo local incompleto (0 hits en esferas locales). Sin evidencia de ser autoexcitado localmente. |
| −1.1468 | −0.240  | −2.581 | `self_excited` | No compatible con ocultedad (5 hits en vecindades de $E+$ y $E-$). |

#### Relación con el Manifiesto de Validación Oficial
Es muy importante distinguir este barrido rápido de ejemplo de la validación rigurosa del manifiesto oficial (`validation_manifest.json`):
1. **Candidato Seleccionado:** El manifiesto oficial evalúa un candidato de grilla modificado `danca2017_nearby_saturation_candidate_q09998` ($m_1 = -1.2$, $m_0 = -0.2$), el cual fue clasificado como `chaotic_self_excited_candidate_not_hidden_under_tested_equilibrium_neighborhoods` tras registrar 1305 contactos directos con el atractor.
2. **Robustez y Diagnósticos:** El manifiesto oficial marca las fases de robustez y diagnósticos como pendientes o incompletas, lo cual enfatiza que no hay una certificación global de atractor oculto en la suite.
3. **Ejemplo 1 (Candidato Principal):** El candidato principal de este ejemplo ($m_1 = -1.1468$, $m_0 = -0.1768$, $c = +2.776$) no presentó evidencia de ser autoexcitado en las pruebas de radios esféricos efectuadas en el Paso 3 (0 hits), manteniéndose clasificado provisionalmente como compatible con ocultedad (`compatible_with_hiddenness`) bajo ese alcance limitado, sin que constituya una prueba matemática o global absoluta.

Todas las figuras correspondientes a las ejecuciones se exportan de forma automatizada a la carpeta canónica de figuras: `version_2/library_figures/`.
