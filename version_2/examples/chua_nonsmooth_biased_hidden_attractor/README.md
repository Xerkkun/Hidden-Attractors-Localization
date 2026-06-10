# Búsqueda de Candidatos Ocultos en Chua Fraccionario No Suave — Ejemplo Oficial

> **Librería:** `hidden_attractors_fo` · **Versión:** 2  
> Este ejemplo documenta el proceso metodológico completo para buscar candidatos a atractores ocultos
> en el sistema de Chua fraccionario no suave (q = 0.9998) mediante la Función Descriptiva Sesgada (BDF).
>
> ⚠️ **Nota Científica y de Reproducibilidad:** Este ejemplo **no representa una reproducción del sistema de Danca (2017)**.
> El sistema original del artículo **no fue reproducible debido a la falta de información publicada** (no se reportan
> las coordenadas de condiciones iniciales del atractor oculto, detalles espectrales de DF como omega0, ni el método exacto
> de continuación). Además, los parámetros de sweep, la función descriptiva y la continuación numérica empleadas
> aquí son diferentes.

---

## Estructura del Ejemplo

El ejemplo ha sido limpiado y modularizado. Toda la lógica de simulación, homotopía, análisis espectral y verificación se ha movido al core de la librería. Este directorio contiene únicamente el punto de entrada oficial:

```
chua_nonsmooth_biased_hidden_attractor/
├── run_example.py      ← Único punto de entrada oficial (ejecuta todo el pipeline)
├── README.md           ← (este archivo)
```

La lógica interna de los pasos está alojada en:
- [biased_chua.py](../../hidden_attractors/workflows/biased_chua.py) (Módulo de Workflows)
- [biased_chua.py](../../hidden_attractors/plotting/biased_chua.py) (Módulo de Ploteo centralizado)

Historical scripts were used during migration and are intentionally excluded from the active repository. The active workflow is implemented in `version_2/hidden_attractors/`.

---

## Ejecución Rápida

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

---

## El Sistema

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

---

## Pasos del Pipeline

1. **Paso 1 — Función Descriptiva Centrada (Base):** Búsqueda de ramas de la DF estándar con bias $c = 0$. Sirve de línea base de comparación. Produce atractores periódicos y no caóticos.
2. **Paso 2 — Función Descriptiva Sesgada (BDF):** Extensión al caso con bias DC ($c \neq 0$), resolución de raíces en $(A, c, \omega)$ usando la convención $1 + W_q(j\omega)N_1 = 0$, reconstrucción algebraica de semillas consistentes, continuación afín Caputo ABM desde el sistema linealizado ($\eta=0$) al real ($\eta=1$), y simulación final larga.
3. **Paso 3 — Verificación de Ocultedad (Protocolo Estándar):** Generación de 225 condiciones iniciales sobre esferas de radios decrecientes alrededor de los 3 equilibrios estables para buscar contactos autoexcitados.
4. **Paso 4 — Verificación Extendida (Multiprocessing):** Test masivo de sondas en volumen de esferas (ball sampling) de radios grandes (hasta $r=2.0$) para evaluar la penetrabilidad local de las vecindades de los equilibrios.
5. **Paso 5 — Resumen y Galería de Figuras:** Exportación de tablas estadísticas, reporte de atractor de 7 paneles, gráficos de homotopía, y mosaicos comparativos bajo las reglas de ploteo unificado de la librería.

---

## Resultados y Clasificación de Ocultedad

El pipeline evalúa las trayectorias bajo el protocolo local de vecindades esféricas. Los resultados se resumen a continuación:

| m₁ | m₀ | bias c | Clasificación (Paso 3) | Nota / Hits detectados |
|---|---|---|---|---|
| −1.1468 | −0.1768 | +2.776 | `HIDDEN_COMPATIBLE` | Protocolo local incompleto (0 hits en esferas locales). Sin evidencia de ser autoexcitado localmente. |
| −1.1468 | −0.200  | −2.705 | `HIDDEN_COMPATIBLE` | Protocolo local incompleto (0 hits en esferas locales). Sin evidencia de ser autoexcitado localmente. |
| −1.1468 | −0.240  | −2.581 | `SELF_EXCITED_CONTACT_DETECTED` | No compatible con ocultedad (5 hits en vecindades de $E+$ y $E-$). |

### Relación con el Manifiesto de Validación Oficial
Es muy importante distinguir este barrido rápido de ejemplo de la validación rigurosa del manifiesto oficial (`validation_manifest.json`):
1. **Candidato Seleccionado:** El manifiesto oficial evalúa un candidato de grilla modificado `danca2017_nearby_saturation_candidate_q09998` ($m_1 = -1.2$, $m_0 = -0.2$), el cual fue clasificado como `chaotic_self_excited_candidate_not_hidden_under_tested_equilibrium_neighborhoods` tras registrar 1305 contactos directos con el atractor.
2. **Robustez y Diagnósticos:** El manifiesto oficial marca las fases de robustez y diagnósticos como pendientes o incompletas, lo cual enfatiza que no hay una certificación global de atractor oculto en la suite.
3. **Ejemplo 1 (Candidato Principal):** El candidato principal de este ejemplo ($m_1 = -1.1468$, $m_0 = -0.1768$, $c = +2.776$) no presentó evidencia de ser autoexcitado en las pruebas de radios esféricos efectuadas en el Paso 3 (0 hits), manteniéndose clasificado provisionalmente como compatible con ocultedad (`HIDDEN_COMPATIBLE`) bajo ese alcance limitado, sin que constituya una prueba matemática o global absoluta.

Todas las figuras correspondientes a las ejecuciones se exportan de forma automatizada a la carpeta canónica de figuras: `version_2/library_figures/`.
