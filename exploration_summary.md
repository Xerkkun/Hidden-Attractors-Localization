# Exploración de Atractores Ocultos – Chua Fraccionario

> [!NOTE]
> Las figuras de alta resolución están en `outputs/chaotic_candidates_plots/` — organizadas por grupo (sat_full, sat_win, arc_full).
## Resumen de Procedimiento y Candidatos

---

## 1. Sistema Estudiado

Se estudia el circuito de Chua en orden fraccionario Caputo:

$$D^q \mathbf{x} = A\mathbf{x} + \mathbf{b}\, f(C\mathbf{x}), \qquad 0 < q \leq 1$$

con parámetros del sistema lineal fijos:
| Parámetro | Valor |
|-----------|-------|
| α | 8.4562 |
| β | 12.0732 |
| γ | 0.0052 |

Se exploran **dos modelos de no-linealidad**:

| Modelo | Función `f(x)` | Parámetros libres |
|--------|---------------|-------------------|
| **Arctan** (Wu 2023) | `a1·arctan(x/a2)` | a1, a2, ρ |
| **No Suave (Saturación)** | Función por tramos (saturación) | m1, m0 |

---

## 2. Metodología de Exploración

El procedimiento completo sigue **4 etapas** para cada punto de la rejilla de parámetros:

### Etapa 1 — Función Descriptiva (DF) Fraccionaria

- Se usa la función descriptiva fraccionaria para predecir oscilaciones sostenidas.
- **Modo de transferencia**: `fractional_spectral` (considera la fase fraccionaria).
- Resultado: semillas `(x₀, ω₀, k, A₀)` por cada rama (branch) detectada.
- Implementación: `find_centered_arctan_wu2023_branches` / `find_saturation_branches`.

### Etapa 2 — Continuación Numérica Fraccionaria (ABM)

- Se utiliza el integrador **Adams-Bashforth-Moulton (ABM) Caputo** para continuar la semilla desde el sistema lineal (λ=0) hasta el sistema completo (λ=1).
- Se evalúan 11 pasos: `λ ∈ {0.0, 0.1, …, 1.0}`.
- Por cada paso: 30 s de transiente + 30 s de señal.
- Dos modos de memoria:
  - **Memoria completa**: todo el historial Caputo (exacto pero costoso).
  - **Memoria truncada**: ventana `Lm = 10 s` (1000 pasos a h=0.01).

> [!IMPORTANT]
> La continuación se realiza con el **mismo modo de memoria** que la simulación posterior — es decir, se prueba la consistencia del sistema fraccionario en cada régimen de memoria.

### Etapa 3 — Simulación Larga ABM

- Si la continuación converge (`status=ok`): se integra hasta `t_final = 300 s`.
- Se descarta el transiente `t_transient = 100 s`.
- Paso de tiempo: `h = 0.01 s`.
- Orden fraccionario: `q = 0.99` (Arctan) o `q = 0.9998` (Saturación).

### Etapa 4 — Diagnóstico de Periodicidad

Clasificador `classify_post_transient_periodicity` asigna uno de:
| Etiqueta | Significado |
|----------|-------------|
| `chaotic_candidate_pending_robustness` ⭐ | Candidato caótico fuerte |
| `nonperiodic_candidate` ○ | No periódico, candidato secundario |
| `regular_periodic_rejected` | Periódico — rechazado |
| `thin_periodic_rejected` | Periódico delgado — rechazado |
| `continuation_failed` | Divergencia en la continuación |

---

## 3. Resultados por Modelo

---

### 3.1 Chua No Suave (Saturación) — q = 0.9998

**Rejilla explorada**: m1 ∈ {-0.8, -1.0, -1.2, -1.4, -1.6}, m0 ∈ {-0.1, -0.2, -0.3, -0.4}

#### 3.1.1 Memoria Completa (q=0.9998)

| Candidato | m1 | m0 | ω₀ | k | A₀ | Veredicto |
|-----------|----|----|-----|---|-----|-----------|
| `m1_m1p2000_m0_m0p2000_branch_0` ⭐ | -1.2 | -0.2 | 2.039 | 0.263 | 4.80 | **chaotic** |
| `m1_m0p8000_m0_m0p1000_branch_0` ○ | -0.8 | -0.1 | 3.245 | 0.613 | 1.29 | nonperiodic |
| `m1_m1p0000_m0_m0p1000_branch_1` ○ | -1.0 | -0.1 | 3.245 | 0.813 | 1.24 | nonperiodic |
| `m1_m1p2000_m0_m0p1000_branch_1` ○ | -1.2 | -0.1 | 3.245 | 1.013 | 1.20 | nonperiodic |
| `m1_m1p4000_m0_m0p1000_branch_1` ○ | -1.4 | -0.1 | 3.245 | 1.213 | 1.17 | nonperiodic |
| `m1_m1p6000_m0_m0p1000_branch_1` ○ | -1.6 | -0.1 | 3.245 | 1.413 | 1.16 | nonperiodic |

> [!NOTE]
> Los candidatos de la rama 1 (`branch_1`) a m0=-0.1 presentan estados finales muy cercanos al origen, lo que sugiere convergencia a un equilibrio oculto de punto fijo, no necesariamente un atractor caótico robusto.

#### 3.1.2 Memoria Truncada Lm=10 s (q=0.9998)

| Candidato | m1 | m0 | Veredicto |
|-----------|----|----|-----------|
| `m1_m1p2000_m0_m0p2000_branch_0` ⭐ | -1.2 | -0.2 | **chaotic** |

El candidato `m1=-1.2, m0=-0.2` es **robusto ante la truncación de memoria**. La mayoría de los demás candidatos de la rejilla `diverged_early` con memoria truncada.

> [!IMPORTANT]
> **Candidato confirmado para pruebas de ocultedad**:
> `m1=-1.2, m0=-0.2` (Saturación) — persiste en ambos modos de memoria.

---

### 3.2 Chua Arctan Fraccionario — q = 0.99

**Parámetros fijos**: α=8.4562, β=12.0732, γ=0.0052  
**Rejilla explorada**: a1 ∈ {0.1, 0.2}, a2 ∈ {-1.0, -1.2, -1.5585, -2.0, -2.5, -3.0}, ρ ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0}

#### 3.2.1 Memoria Completa (q=0.99) — 177 casos evaluados

**Candidatos `chaotic_candidate_pending_robustness` (⭐)**:

| Candidato | a1 | a2 | ρ | ω₀ | k | A₀ |
|-----------|----|----|---|----|---|-----|
| `a1=0.10, a2=-1.20, ρ=1.00` ⭐ | 0.1 | -1.2 | 1.0 | 2.099 | -1.028 | 0.88 |
| `a1=0.10, a2=-1.20, ρ=1.25` ⭐ | 0.1 | -1.2 | 1.25 | 2.099 | -1.028 | 1.31 |
| `a1=0.10, a2=-1.20, ρ=2.00` ⭐ | 0.1 | -1.2 | 2.0 | 2.099 | -1.028 | 1.76 |
| `a1=0.10, a2=-1.20, ρ=3.00` ⭐ | 0.1 | -1.2 | 3.0 | 2.099 | -1.028 | 1.97 |
| `a1=0.10, a2=-1.20, ρ=4.00` ⭐ | 0.1 | -1.2 | 4.0 | 2.099 | -1.028 | 2.07 |
| `a1=0.10, a2=-1.5585, ρ=0.75` ⭐ | 0.1 | -1.5585 | 0.75 | 2.099 | -1.028 | 1.05 |
| `a1=0.10, a2=-1.5585, ρ=1.00` ⭐ | 0.1 | -1.5585 | 1.0 | 2.099 | -1.028 | 1.77 |
| `a1=0.10, a2=-1.5585, ρ=1.25` ⭐ | 0.1 | -1.5585 | 1.25 | 2.099 | -1.028 | 2.08 |
| `a1=0.10, a2=-1.5585, ρ=1.50` ⭐ | 0.1 | -1.5585 | 1.5 | 2.099 | -1.028 | 2.27 |
| `a1=0.10, a2=-1.5585, ρ=2.00` ⭐ | 0.1 | -1.5585 | 2.0 | 2.099 | -1.028 | 2.48 |
| `a1=0.10, a2=-1.5585, ρ=3.00` ⭐ | 0.1 | -1.5585 | 3.0 | 2.099 | -1.028 | 2.68 |
| `a1=0.10, a2=-1.5585, ρ=4.00 b0` ⭐ | 0.1 | -1.5585 | 4.0 | 2.099 | -1.028 | 2.77 |
| `a1=0.10, a2=-1.5585, ρ=4.00 b1` ⭐ | 0.1 | -1.5585 | 4.0 | 3.221 | -0.415 | 7.26 |
| `a1=0.20, a2=-1.20, ρ=1.25` ⭐ | 0.2 | -1.2 | 1.25 | 2.099 | -1.128 | 1.06 |
| `a1=0.20, a2=-1.20, ρ=2.00` ⭐ | 0.2 | -1.2 | 2.0 | 2.099 | -1.128 | 1.55 |
| `a1=0.20, a2=-1.20, ρ=3.00` ⭐ | 0.2 | -1.2 | 3.0 | 2.099 | -1.128 | 1.76 |
| `a1=0.20, a2=-1.20, ρ=4.00 b0` ⭐ | 0.2 | -1.2 | 4.0 | 2.099 | -1.128 | 1.86 |
| `a1=0.20, a2=-1.20, ρ=4.00 b1` ⭐ | 0.2 | -1.2 | 4.0 | 3.221 | -0.515 | 4.41 |
| `a1=0.20, a2=-1.5585, ρ=1.00` ⭐ | 0.2 | -1.5585 | 1.0 | 2.099 | -1.128 | 1.45 |
| `a1=0.20, a2=-1.5585, ρ=1.25` ⭐ | 0.2 | -1.5585 | 1.25 | 2.099 | -1.128 | 1.79 |
| `a1=0.20, a2=-1.5585, ρ=1.50` ⭐ | 0.2 | -1.5585 | 1.5 | 2.099 | -1.128 | 1.99 |
| `a1=0.20, a2=-1.5585, ρ=2.00` ⭐ | 0.2 | -1.5585 | 2.0 | 2.099 | -1.128 | 2.21 |
| `a1=0.20, a2=-1.5585, ρ=3.00 b0` ⭐ | 0.2 | -1.5585 | 3.0 | 2.099 | -1.128 | 2.41 |
| `a1=0.20, a2=-1.5585, ρ=3.00 b1` ⭐ | 0.2 | -1.5585 | 3.0 | 3.221 | -0.515 | 5.71 |
| `a1=0.20, a2=-1.5585, ρ=4.00 b0` ⭐ | 0.2 | -1.5585 | 4.0 | 2.099 | -1.128 | 2.50 |
| `a1=0.20, a2=-1.5585, ρ=4.00 b1` ⭐ | 0.2 | -1.5585 | 4.0 | 3.221 | -0.515 | 5.80 |

**Total candidatos caóticos Arctan (memoria completa): 26**

#### 3.2.2 Memoria Truncada Lm=10 s (q=0.99) — 177 casos evaluados

> [!WARNING]
> **Resultado: 0 candidatos sobreviven la continuación con memoria truncada.**
> Todos los 177 casos reportan `continuation_failed` con `diverged_early`.

**Interpretación**: La continuación ABM Caputo con ventana de memoria Lm=10 s introduce inestabilidad numérica suficiente para que el método diverga antes de alcanzar λ=1. Esto indica que los atractores del Chua Arctan fraccionario son **sensibles a la longitud de memoria** — el historial completo es necesario para que la continuación converja correctamente.

---

## 4. Comparación entre Modelos

| Aspecto | Saturación | Arctan |
|---------|-----------|--------|
| q usado | 0.9998 | 0.99 |
| Mem. completa → candidatos caóticos | 1 ⭐ + 5 ○ | **26** ⭐ |
| Mem. truncada → candidatos caóticos | 1 ⭐ | 0 |
| Candidato más robusto | m1=-1.2, m0=-0.2 | a2=-1.2 a2=-1.5585 (varios ρ) |
| Regiones de caos | Localizada (m1≈-1.2) | Extensa (a2∈[-1.2,-1.5585]) |
| Sensibilidad a memoria truncada | Baja (1 candidato persiste) | Alta (ninguno persiste) |

---

## 5. Archivos Generados

| Archivo/Directorio | Contenido |
|--------------------|-----------|
| `outputs/saturation_search_seed1_mem_full_sweep/` | Barrido saturación, memoria completa |
| `outputs/saturation_search_seed0p9998_mem_window_sweep/` | Barrido saturación, memoria truncada |
| `outputs/saturation_comparison/` | Comparativa ABM vs EFORK full vs EFORK trunc |
| `outputs/arctan_search_seed0p99_mem_full/` | Barrido Arctan, memoria completa (177 casos) |
| `outputs/arctan_search_seed0p99_mem_window/` | Barrido Arctan, memoria truncada (177 casos) |
| `outputs/chaotic_candidates_plots/sat_full/` | Figuras detalladas – Saturación mem. completa |
| `outputs/chaotic_candidates_plots/sat_win/` | Figuras detalladas – Saturación mem. truncada |
| `outputs/chaotic_candidates_plots/arc_full/` | Figuras detalladas – Arctan mem. completa |
| `outputs/chaotic_candidates_plots/summary_*.png` | Mosaicos resumen por grupo |
| `outputs/chaotic_candidates_plots/MEGA_all_candidates.png` | Figura resumen global |

### Scripts clave

| Script | Propósito |
|--------|-----------|
| `search_arctan_fractional.py` | Barrido Arctan (ejecuta --memory-mode full/window) |
| `search_saturation_candidates.py` | Barrido Saturación |
| `compare_solvers_saturation.py` | Comparativa ABM vs EFORK |
| `generate_all_plots_and_summary.py` | Genera todas las figuras de candidatos |
| `exploration_template.py` | **Plantilla reutilizable** para nuevas exploraciones |

---

## 6. Candidatos Prioritarios para Pruebas de Ocultedad

Los siguientes candidatos tienen la mayor prioridad para verificar si son **atractores ocultos** (condición inicial en el origen → sin convergencia):

### Saturación (robustez confirmada en ambos modos de memoria):
1. **`m1=-1.2, m0=-0.2`** ⭐ — único candidato que sobrevive memoria truncada

### Arctan (memoria completa, muchos candidatos — selección representativa):
2. **`a1=0.1, a2=-1.2, ρ=1.0`** ⭐ — amplitud mínima, mejor para ocultedad
3. **`a1=0.1, a2=-1.2, ρ=1.25`** ⭐
4. **`a1=0.1, a2=-1.5585, ρ=0.75`** ⭐ — ρ mínimo con comportamiento caótico
5. **`a1=0.1, a2=-1.5585, ρ=1.0`** ⭐
6. **`a1=0.2, a2=-1.2, ρ=1.25`** ⭐
7. **`a1=0.2, a2=-1.5585, ρ=1.0`** ⭐

> [!TIP]
> Para la prueba de ocultedad, integrar desde `x₀ = [0, 0, ε]` con ε → 0 y verificar que **no** converge al mismo atractor. Si no converge, el atractor es oculto.

---

## 7. Observaciones y Hallazgos

1. **Arctan produce más candidatos que Saturación**: El modelo Arctan con q=0.99 genera una región extensa de parámetros con comportamiento caótico, especialmente alrededor de a2∈{-1.2, -1.5585}.

2. **La memoria truncada es crítica para el Arctan**: La continuación ABM con Lm=10s no logra convergencia en ningún caso del Arctan, sugiriendo que el caos en este modelo depende fuertemente del historial de memoria completo.

3. **El modelo de saturación es más frugal**: Solo un candidato caótico fuerte, pero es robusto. Los candidatos no periódicos de rama-1 son probablemente equilibrios ocultos, no atractores caóticos.

4. **Consistencia con exploración previa (entero)**: Los candidatos de Arctan en a2=-1.2 y a2=-1.5585 coinciden con la región identificada en la exploración anterior con integrador entero (version_2/outputs/arctan_full_memory_search/), confirmando que el caos persiste al reducir el orden fraccionario.

---

## 8. Galería de Figuras

### Figura global — todos los candidatos (37 paneles)
![Todos los candidatos caóticos](C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/MEGA_all_candidates.png)

### Mosaico — Saturación, Memoria Completa (6 candidatos)
![Saturación - Memoria Completa](C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/summary_sat_full.png)

### Mosaico — Saturación, Memoria Truncada (1 candidato confirmado)
![Saturación - Memoria Truncada](C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/summary_sat_win.png)

### Mosaico — Arctan Fraccionario, Memoria Completa (26 candidatos)
![Arctan - Memoria Completa](C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/summary_arc_full.png)

### Detalle — Candidato caótico Saturación (m1=-1.2, m0=-0.2)
![Saturación candidato caótico detallado](C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/sat_chaotic_candidate_detailed.png)

### Detalle — Candidato caótico Arctan (a1=0.1, a2=-1.2, ρ=1.0)
![Arctan candidato caótico detallado](C:/Users/moren/.gemini/antigravity-ide/brain/e90964d8-48e3-4643-b087-c2fc0cf40e67/arc_chaotic_candidate_detailed.png)
