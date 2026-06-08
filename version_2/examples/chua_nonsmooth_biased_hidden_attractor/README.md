# Atractor Oculto en Chua Fraccionario No Suave — Ejemplo Oficial

> **Librería:** `hidden_attractors_fo` · **Versión:** 2  
> Este ejemplo documenta el proceso metodológico completo que llevó a la
> confirmación del primer atractor oculto en el sistema de Chua fraccionario
> no suave (q = 0.9998) mediante la Función Descriptiva Sesgada.

---

## Estructura del Ejemplo

El ejemplo ha sido limpiado y modularizado. Toda la lógica de simulación, homotopía, análisis espectral y verificación se ha movido al core de la librería. Este directorio contiene únicamente el punto de entrada oficial:

```
chua_nonsmooth_biased_hidden_attractor/
├── run_example.py      ← Único punto de entrada oficial (ejecuta todo el pipeline)
├── README.md           ← (este archivo)
```

La lógica interna de los pasos está alojada en:
- [biased_chua.py](file:///c:/Users/moren/Desktop/Codes/Hidden%20Attractors%20Fractional%20Order/version_2/hidden_attractors/workflows/biased_chua.py) (Módulo de Workflows)
- [biased_chua.py](file:///c:/Users/moren/Desktop/Codes/Hidden%20Attractors%20Fractional%20Order/version_2/hidden_attractors/plotting/biased_chua.py) (Módulo de Ploteo centralizado)

Los scripts históricos e independientes (`step1_*.py` a `step5_*.py`) han sido archivados en [_archived_figure_scripts/examples/chua_nonsmooth_biased_hidden_attractor/](file:///c:/Users/moren/Desktop/Codes/Hidden%20Attractors%20Fractional%20Order/_archived_figure_scripts/examples/chua_nonsmooth_biased_hidden_attractor/) y están catalogados en el [Índice de Archivos](file:///c:/Users/moren/Desktop/Codes/Hidden%20Attractors%20Fractional%20Order/_archived_figure_scripts/ARCHIVE_INDEX.md).

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
3. **Paso 3 — Verificación de Ocultedad (Protocolo Estándar):** Generación de 225 condiciones iniciales sobre esferas de radios decrecientes alrededor de los 3 equilibrios estables y clasificación de su destino final.
4. **Paso 4 — Verificación Extendida (Multiprocessing):** Test masivo de hasta $28.830$ sondas en volumen de esferas (ball sampling) de radios grandes (hasta $r=2.0$) para confirmar la impenetrabilidad de las cuencas de atracción del atractor oculto.
5. **Paso 5 — Resumen y Galería de Figuras:** Exportación de tablas estadísticas, reporte de atractor de 7 paneles, gráficos de homotopía, y mosaicos comparativos bajo las reglas de ploteo unificado de la librería.

---

## Resultados Confirmados

El pipeline detecta y valida 3 candidatos ocultos con alta reproducibilidad:

| m₁ | m₀ | bias c | Estado |
|---|---|---|---|
| −1.1468 | −0.1768 | +2.776 | `HIDDEN_COMPATIBLE` |
| −1.1468 | −0.200  | −2.705 | `HIDDEN_COMPATIBLE` |
| −1.1468 | −0.240  | −2.581 | `HIDDEN_COMPATIBLE` |

Todas las figuras correspondientes a las ejecuciones se exportan de forma automatizada a la carpeta canónica de figuras: `version_2/library_figures/`.
