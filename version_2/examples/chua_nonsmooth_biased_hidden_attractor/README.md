# Atractor Oculto en Chua Fraccionario No Suave — Primer Ejemplo Exitoso

> **Librería:** `hidden_attractors_fo` · **Versión:** 2  
> Este ejemplo documenta el proceso metodológico completo que llevó a la
> confirmación del primer atractor oculto en el sistema de Chua fraccionario
> no suave (q = 0.9998) mediante la Función Descriptiva Sesgada.

---

## Estructura del Ejemplo

```
chua_nonsmooth_biased_hidden_attractor/
├── run_example.py                   ← Punto de entrada único
├── step1_centered_reference.py      ← Paso 1: DF centrada (línea base)
├── step2_biased_df_search.py        ← Paso 2: DF sesgada + continuación afín
├── step3_hiddenness_verification.py ← Paso 3: Verificación de ocultedad
├── step4_extended_hiddenness.py     ← Paso 4: Test masivo multiprocessing
├── step5_summarize_and_plot.py      ← Paso 5: Resumen y galería
└── README.md                        ← (este archivo)

configs/examples/
└── chua_nonsmooth_biased_df_search.yaml  ← Config YAML central

_reference_scripts/                       ← Scripts temporales originales
└── (scripts de desarrollo/debugging)
```

---

## Ejecución Rápida

```bash
# Prueba de humo (tiempos reducidos, ~2-5 min)
.venv\Scripts\python examples\chua_nonsmooth_biased_hidden_attractor\run_example.py --quick

# Ejecución completa (Pasos 1, 2, 3, 5 — sin el test extendido masivo)
.venv\Scripts\python examples\chua_nonsmooth_biased_hidden_attractor\run_example.py

# Ejecución completa incluyendo Paso 4 (puede tomar 4-8 horas)
.venv\Scripts\python examples\chua_nonsmooth_biased_hidden_attractor\run_example.py --all

# Paso individual
.venv\Scripts\python examples\chua_nonsmooth_biased_hidden_attractor\run_example.py --steps 2
```

---

## El Sistema

El circuito de Chua fraccionario no suave se describe mediante el sistema de
Lur'e fraccionario de orden Caputo $q$:

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

## Proceso Metodológico

### Paso 1 — Función Descriptiva Centrada (línea base)

La Función Descriptiva estándar (DF centrada, $c = 0$) asume que la señal de
entrada a la no linealidad es puramente sinusoidal: $\sigma(t) = A\cos(\omega t)$.

Se buscan cruces de ganancia $k$ tales que exista una oscilación periódica
$1 + W_q(j\omega) \cdot N(A) = 0$, y se traza la continuación para verificar
si la rama produce un atractor caótico.

**Resultado:** Las ramas centradas sobreviven la continuación pero producen
atractores **periódicos**, no caóticos. Ninguna es "oculta" en sentido estricto.

**¿Por qué falló la DF centrada?** El sistema admite un estado de sesgo DC
permanente $\bar{x} \neq 0$ que la DF centrada no puede capturar.

---

### Paso 2 — Función Descriptiva Sesgada (núcleo del método)

#### 2.1 Extensión al caso sesgado

Se generaliza la señal de entrada a:
$$
\sigma(t) = c + A\cos(\omega t) + \text{armónicos superiores}
$$

Los coeficientes de Fourier de la no linealidad se calculan por cuadratura numérica:
$$
\psi_0(A, c) = \frac{1}{2\pi} \int_0^{2\pi} f(c + A\cos\theta)\, d\theta
$$
$$
N_1(A, c) = \frac{1}{A\pi} \int_0^{2\pi} f(c + A\cos\theta)\cos\theta\, d\theta
$$

#### 2.2 Convención de signo — crítica

La condición de oscilación armónica se escribe bajo la convención:
$$
\boxed{1 + W_q(j\omega) \cdot N_1(A, c) = 0}
$$

> **¡Advertencia crítica!** La convención alternativa $1 - W_q \cdot N_1 = 0$
> produce raíces espurias que no corresponden a oscilaciones reales. Se incluye
> una auditoría de signo (`roots_sign_audit.csv`) que verifica cuál convención
> produce residuo nulo.

#### 2.3 Ecuación DC de consistencia

La componente DC impone un equilibrio adicional:
$$
F_0: \quad c - r^T \bar{x} = 0, \quad \bar{x} = -P^{-1} b \psi_0(A, c)
$$

El sistema de ecuaciones completo es:
$$
\mathbf{F}(A, c, \omega) = 
\begin{pmatrix} c - r^T(-P^{-1}b\psi_0) \\ \text{Re}(1 + W_q(j\omega)N_1) \\ \text{Im}(1 + W_q(j\omega)N_1) \end{pmatrix} = \mathbf{0}
$$

#### 2.4 Reconstrucción de la semilla

El vector de estado semilla es:
$$
x_\text{seed} = \bar{x} + \text{Re}(X_1), \quad X_1 = (\lambda I - P)^{-1}b \cdot N_1 A, \quad \lambda = (j\omega)^q
$$

Se verifican dos condiciones de consistencia algebraica:
- **Error DC:** $|r^T \bar{x} - c| < 10^{-5}$
- **Error fasorial:** $\|X_1\| \approx A$

#### 2.5 Homotopía afín Caputo ABM

En lugar de usar la continuación clásica, se usa una **homotopía afín** que
deforma el sistema desde el linealizado (η=0) hasta el original (η=1):

$$
f_\eta(x) = (P + N_1 br^T)x + b(\psi_0 - N_1 c) + \eta b[f(r^T x) - \psi_0 - N_1(r^T x - c)]
$$

**Propiedad clave:** En $\eta = 0$, el sistema es lineal y la semilla $x_\text{seed}$
es exactamente la condición inicial del ciclo límite linealizado. En $\eta = 1$,
el sistema coincide exactamente con el sistema original: $f_1(x) \equiv f(x)$
(verificado en `affine_homotopy_identity.csv`).

---

### Paso 3 — Verificación de Ocultedad

Se aplica el **protocolo de barrido de esferas** alrededor de cada equilibrio
$E_0, E_+, E_-$:

1. Para cada radio $r \in \{10^{-5}, 3\times10^{-5}, 10^{-4}, 3\times10^{-4}, 10^{-3}, 10^{-2}\}$
2. Generar $n$ puntos aleatorios en la superficie de la esfera $\|x - E_i\| = r$
3. Integrar cada punto durante 200 s con Caputo ABM
4. Clasificar el destino de cada sonda

**Clasificaciones posibles:**
- `target_attractor`: el punto alcanzó el atractor candidato → **no es oculto**
- `stable_equilibrium`: el punto convergió al equilibrio
- `other_attractor`: el punto divergió hacia otro atractor
- `divergence`: la solución divergió

**Contrato de ocultedad:** `HIDDEN_COMPATIBLE` si ninguna sonda alcanza el
atractor desde ninguna vecindad del equilibrio en los radios ensayados.

> **Advertencia epistemológica:** La verificación es numérica y finita.
> `HIDDEN_COMPATIBLE` **no** es equivalente a ocultedad matemática probada.
> Es una evidencia computacional bajo los parámetros declarados.

---

### Paso 4 — Verificación Extendida (Multiprocessing)

Para el candidato confirmado ($c \approx +2.776$), se extiende el barrido
hasta $r = 2.0$ usando **ball sampling** (volumen completo, no solo superficie)
con $\sim 9.610$ sondas por equilibrio ejecutadas en paralelo.

**Resultado histórico:** 0 hits TARGET en todas las 28.830 sondas ensayadas
para el candidato con $m_1 = -1.1468$, $m_0 = -0.1768$, $c = +2.776$.

---

### Paso 5 — Resumen y Galería

Genera:
- Galería de figuras (espacio 3D, proyecciones 2D, series de tiempo, FFT)
- Comparación visual del atractor sesgado vs referencia centrada
- MEGA-mosaico de todos los candidatos
- Reporte Markdown con tabla de veredictos

---

## Resultados

Los 3 candidatos confirmados son:

| m₁ | m₀ | bias c | Estado |
|---|---|---|---|
| −1.1468 | −0.1768 | +2.776 | `HIDDEN_COMPATIBLE` |
| −1.1468 | −0.200  | −2.705 | `HIDDEN_COMPATIBLE` |
| −1.1468 | −0.240  | −2.581 | `HIDDEN_COMPATIBLE` |

---

## Cómo Adaptar a Otro Sistema

1. **Modificar el YAML** (`configs/examples/chua_nonsmooth_biased_df_search.yaml`):
   - Cambiar `system.system_id` al ID del nuevo sistema
   - Ajustar `system.parameters`
   - Modificar `parameter_grid` según el espacio de parámetros a explorar

2. **Adaptar `step2_biased_df_search.py`:**
   - Reemplazar `biased_saturation_df` con la DF sesgada del nuevo sistema
   - Ajustar `chua_matrices` con la parametrización Lur'e del nuevo sistema

3. **Los pasos 3, 4 y 5 son agnósticos al sistema** — funcionan directamente
   con cualquier atractor descrito como trayectoria CSV (`t,x,y,z`).

---

## Scripts de Referencia

Los scripts temporales usados durante el desarrollo están preservados en
`_reference_scripts/` (raíz del proyecto):

| Script | Descripción |
|---|---|
| `search_saturation_biased_candidates_corrected.py` | Script monolítico original (núcleo del Paso 2) |
| `hiddenness_cand0_all_extended.py` | Test extendido original (núcleo del Paso 4) |
| `run_hiddenness_biased_candidates.py` | Verificación original (núcleo del Paso 3) |
| `exploration_template.py` | Plantilla genérica DF+continuación |
| `search_saturation_candidates.py` | Búsqueda centrada original |
| `generate_all_plots_and_summary.py` | Generador de figuras global |
| `recover_hiddenness_results.py` | Recuperador de resultados |

---

## Dependencias

```
numpy
scipy
matplotlib
pandas
pyyaml
```

El backend C del integrador ABM se compila automáticamente en la primera ejecución.
