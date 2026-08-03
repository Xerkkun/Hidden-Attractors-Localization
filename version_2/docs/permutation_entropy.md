# Entropía de permutación de Bandt--Pompe

HAFO incorpora una capacidad local de entropía de permutación para series
escalares finitas. El mismo contrato puede analizar una secuencia NumPy o una
componente explícita de un `TrajectoryInput` producido por un sistema de orden
entero o fraccionario. La implementación conserva el histograma ordinal
completo y su procedencia, y ofrece rutas Python, Numba y C/OpenMP sin hacer una
llamada a Julia ni a otra biblioteca por ventana.

Esta cantidad es un diagnóstico empírico de la proyección y la muestra
entregadas. No es, por sí sola, una tasa de entropía, la entropía de
Kolmogorov--Sinai, una prueba de caos, un certificado de atracción ni una prueba
de ocultedad.

## Serie escalar y ventanas forward

Sea una serie real finita

\[
x=(x_0,x_1,\ldots,x_{N-1}),
\]

una dimensión de embedding entera \(m\) y un retardo entero \(\tau\). HAFO
construye ventanas cronológicas hacia delante

\[
y_s^{(m,\tau)}=
\left(x_s,x_{s+\tau},\ldots,x_{s+(m-1)\tau}\right),
\qquad s=0,\ldots,W-1,
\]

con

\[
W=N-(m-1)\tau.
\]

Se exige \(2\leq m\leq 10\), \(\tau\geq1\) y \(W\geq1\). `delay` se mide en
índices de muestra, no automáticamente en unidades de tiempo físico. En una
malla uniforme de paso \(\Delta t\), el retardo físico es
\(\tau\Delta t\); en una malla no uniforme, las separaciones físicas de las
ventanas pueden variar. `TrajectoryInput` permite conservar esa diferencia,
además de la proyección, el solver, el transitorio, la derivada fraccionaria, el
orden, la terminal inferior, la prehistoria y la política de memoria. La rutina
no interpola ni remuestrea silenciosamente.

La operación definida aquí es univariante. Una trayectoria con varias
coordenadas requiere seleccionar exactamente una columna. Aplicar la rutina a
cada columna por separado no equivale a una entropía de permutación
multivariante.

## Patrón ordinal y convenio de empates

Para cada ventana se obtiene la permutación
\(\pi_s=(\pi_{s,0},\ldots,\pi_{s,m-1})\in S_m\) que ordena los índices de la
ventana por amplitud ascendente:

\[
y_{s,\pi_{s,0}}\leq y_{s,\pi_{s,1}}\leq\cdots\leq
y_{s,\pi_{s,m-1}}.
\]

Las igualdades exactas se tratan con una política visible:

| `tie_policy` | Contrato |
|---|---|
| `stable_index` | Ordena por el par `(valor, índice_en_la_ventana)`. Los valores iguales conservan el orden cronológico y se cuentan todas las ventanas. |
| `omit` | Descarta cualquier ventana que contenga al menos una igualdad exacta. |
| `raise` | Detiene el cálculo al detectar una ventana empatada. |

`stable_index` es determinista y materializa el convenio de ordenar igualdades
por orden de aparición. No hace que los empates de una señal cuantizada sean
inocuos: puede introducir orden temporal artificial. `omit` cambia el tamaño
efectivo de muestra y también puede sesgar el conjunto retenido. Por eso el
resultado registra `total_windows`, `valid_windows` y `tied_windows`; una
comparación científica debe declarar la política y, cuando haya empates,
examinar su sensibilidad. La igualdad usada por el núcleo es exacta (`==`), no
una tolerancia oculta; en particular, `-0.0` y `+0.0` empatan.

Una transformación estrictamente monótona conserva los patrones ordinales. Una
transformación que cree empates, una cuantización o un cambio de precisión
puede alterar el resultado.

## Índice denso mediante código de Lehmer

El histograma tiene siempre \(m!\) celdas, incluidas las de conteo cero. HAFO
asigna a cada permutación su rango lexicográfico de Lehmer, empezando en cero.
Para

\[
c_j=\#\{k:j<k<m,\;\pi_{s,k}<\pi_{s,j}\},
\]

el rango es

\[
r(\pi_s)=\sum_{j=0}^{m-2}c_j\,(m-1-j)!,
\qquad 0\leq r(\pi_s)<m!.
\]

Por ejemplo, para \(m=3\), el orden de las celdas es

| Rango | Permutación |
|---:|---|
| 0 | `(0, 1, 2)` |
| 1 | `(0, 2, 1)` |
| 2 | `(1, 0, 2)` |
| 3 | `(1, 2, 0)` |
| 4 | `(2, 0, 1)` |
| 5 | `(2, 1, 0)` |

Si \(n_r\) es el conteo de la celda \(r\), el número aceptado de ventanas es

\[
V=\sum_{r=0}^{m!-1}n_r.
\]

Con `stable_index`, \(V=W\). Con `omit`,
\(V=W-\texttt{tied\_windows}\). El histograma de bajo nivel puede describir
\(V=0\), pero no existe una distribución empírica ni una entropía plugin en
ese caso.

## Probabilidades y entropía plugin

Para \(V>0\), las probabilidades empíricas son

\[
\widehat p_r=\frac{n_r}{V},
\qquad
\sum_{r=0}^{m!-1}\widehat p_r=1.
\]

HAFO calcula la entropía de Shannon plugin en una base finita \(b>1\),
declarada mediante `log_base`:

\[
\widehat H_b(m,\tau)=
-\sum_{r=0}^{m!-1}\widehat p_r\log_b\widehat p_r,
\qquad 0\log_b0:=0.
\]

La versión normalizada es

\[
\widehat H_b^{\mathrm{norm}}(m,\tau)=
\frac{\widehat H_b(m,\tau)}{\log_b(m!)}.
\]

En aritmética exacta,
\(0\leq\widehat H_b^{\mathrm{norm}}\leq1\). La normalización no elimina la
dependencia estadística respecto de \(m\), \(\tau\), el muestreo, la longitud
de la serie o los empates. `entropy` conserva unidades de la base elegida
--bits para \(b=2\)-- y `normalized_entropy` es adimensional.

Éste es el estimador plugin de frecuencias relativas. No aplica
Miller--Madow, estimación bayesiana, extrapolación de cobertura, corrección por
patrones no observados ni intervalos de confianza. Para muestras finitas puede
tener sesgo, especialmente cuando \(V\) no es grande frente a \(m!\). Un valor
cercano a uno sólo indica que los patrones observados están próximos a una
distribución uniforme bajo este contrato; no identifica por sí solo ruido ni
caos.

## Límite \(m\leq10\) y coste

El límite superior es una salvaguarda explícita, no una recomendación de usar
\(m=10\). El número de celdas crece factorialmente: \(10!=3\,628\,800\), por
lo que un histograma `uint64` de orden diez ocupa aproximadamente 27.7 MiB,
antes de contar arreglos auxiliares. El conteo directo cuesta
\(O(Wm^2+m!)\) con el ordenamiento por inserción y el rango de Lehmer usados en
el núcleo; como \(m\) está acotado, el recorrido es lineal en \(W\) para un
\(m\) fijo. La memoria base es \(O(m!)\).

El tamaño \(m!\), la fracción de patrones observados y el cociente \(V/m!\)
deben acompañar la interpretación. El hecho de que una llamada sea
computacionalmente admisible no garantiza que la muestra soporte una
estimación fiable.

## Backends HAFO

| Backend | Función |
|---|---|
| `python` | Referencia legible y determinista para casos pequeños. |
| `numba` | Kernel JIT con las mismas ventanas, empates y celdas Lehmer. |
| `native_c` | Conteo `uint64` en C, compilado de forma diferida y trazado por hash de fuente; usa OpenMP cuando está disponible. |
| `auto` | Política dimensionada del núcleo público; conserva en el resultado el backend solicitado, el efectivo, el umbral aplicable y cualquier fallback. |

La política automática se apoya en el coste conjunto de ventanas y espacio
factorial, no en un único umbral escalar:

| Dimensión \(m\) | Selección automática de C nativo |
|---:|---:|
| 2 | desde 131 072 ventanas |
| 3--7 | desde 32 768 ventanas |
| 8 | desde 131 072 ventanas |
| 9--10 | deshabilitada; `auto` conserva Numba |

`PERMUTATION_ENTROPY_NATIVE_AUTO_WINDOW_THRESHOLDS` expone esta tabla y
`PERMUTATION_ENTROPY_NATIVE_AUTO_MIN_WINDOWS` conserva su mínimo no nulo. La
selección explícita `backend="native_c"` sigue disponible para \(m=9,10\), pero
el despacho automático evita la penalización observada cuando \(m!\) obliga a
usar el histograma atómico de OpenMP.

El kernel C paraleliza ventanas independientes desde 1024 ventanas. Usa un
histograma privado por hilo cuando la suma de esos histogramas no rebasa 64
MiB; para cargas mayores recurre a incrementos atómicos y evita una asignación
no acotada de \(\text{hilos}\times m!\). La normalización y la entropía quedan
en la capa de análisis: el ABI nativo devuelve el histograma denso y los tres
conteos de ventanas. Si no se puede construir o cargar el backend nativo, la
ruta de bajo nivel puede usar un fallback Numba autocontenido; el resultado lo
declara, no lo oculta.

`benchmarks/bench_permutation_entropy.py` separa calentamiento/JIT/compilación,
rota el orden de los tres backends y mide tanto el pipeline público como el
kernel de conteo. El barrido local del 3 de agosto de 2026 cubrió 21 cargas
deterministas, \(m=2,\ldots,10\), 4 096--131 072 ventanas y obtuvo histogramas
idénticos en Python, Numba y C/OpenMP. Esa corrida respalda la tabla en el host
medido, pero conserva como oportunidades conservadoras los cruces variables de
16 384 ventanas; no convierte un benchmark local en superioridad universal.

No se cruza la frontera Python--C dentro de cada ventana. Tampoco se inicia
Julia para una llamada ordinaria. Esta arquitectura permite verificar un solo
contrato matemático en tres implementaciones sin convertir una dependencia
externa en parte del núcleo.

## API pública

`ordinal_pattern_distribution` separa la construcción de la distribución
empírica de su cuantificador de Shannon. Esto permite auditar o reutilizar los
mismos conteos sin reconstruir ventanas:

```python
import numpy as np

from hidden_attractors.analysis import (
    ordinal_pattern_distribution,
    permutation_entropy_from_distribution,
)

x = np.asarray(samples, dtype=np.float64)
distribution = ordinal_pattern_distribution(
    x,
    embedding_dimension=5,
    delay=2,
    tie_policy="stable_index",
    backend="auto",
    fallback=True,
    sampling="uniform dt=0.01 after transient removal",
    projection="measured voltage",
)
result = permutation_entropy_from_distribution(
    distribution,
    log_base=2.0,
)

print(result.entropy)
print(result.normalized_entropy)
print(distribution.counts, distribution.probabilities)
print(
    distribution.valid_windows,
    distribution.tied_windows,
    distribution.backend,
)
```

`permutation_entropy` es la fachada de una sola llamada. Para un
`TrajectoryInput` unidimensional puede omitirse `component`; si la trayectoria
tiene más de una columna, `component` es obligatorio y acepta el índice o la
etiqueta declarada en `projection`:

```python
from hidden_attractors.analysis import permutation_entropy
from hidden_attractors.analysis.contracts import TrajectoryInput

trajectory = TrajectoryInput.from_simulation_result(
    simulation,
    projection=("x", "y", "z"),
    transient_interval=(0.0, 100.0),
)

result = permutation_entropy(
    trajectory,
    component="x",
    embedding_dimension=6,
    delay=8,
    tie_policy="omit",
    log_base=2.0,
    backend="numba",
)

print(result.trajectory_fingerprint)
print(result.distribution.sampling)
```

El conteo nativo también está disponible como operación experimental de bajo
nivel cuando se necesita auditar exactamente el histograma:

```python
from hidden_attractors.analysis.native_permutation_entropy import (
    native_permutation_counts,
)

raw = native_permutation_counts(
    x,
    embedding_dimension=5,
    delay=2,
    tie_policy="stable_index",
    fallback=True,
)
assert raw.counts.dtype == np.uint64
assert raw.counts.size == 120
assert int(raw.counts.sum()) == raw.valid_windows
```

Las tres operaciones de la fachada son:

| Función | Responsabilidad |
|---|---|
| `ordinal_pattern_distribution(data, *, component=None, embedding_dimension=3, delay=1, tie_policy="stable_index", backend="auto", fallback=True, sampling="sample index", projection="supplied scalar signal")` | Valida o extrae la serie escalar, construye las ventanas, cuenta rangos Lehmer y devuelve `OrdinalPatternDistribution`. |
| `permutation_entropy_from_distribution(distribution, *, log_base=2.0)` | Calcula una vez \(\widehat H_b\) y su normalización a partir de conteos ya auditables. |
| `permutation_entropy(data, *, component=None, embedding_dimension=3, delay=1, tie_policy="stable_index", log_base=2.0, backend="auto", fallback=True, sampling="sample index", projection="supplied scalar signal")` | Ejecuta ambas etapas y devuelve `PermutationEntropyResult`. |

`OrdinalPatternDistribution` conserva `counts`, `probabilities`,
`embedding_dimension`, `delay`, `sample_count`, `total_windows`,
`valid_windows`, `tied_windows`, `possible_patterns`, `observed_patterns`,
`tie_policy`, backend solicitado y efectivo, muestreo, proyección, huella de la
trayectoria, advertencias, metadatos, estado y alcance de evidencia.
`PermutationEntropyResult` incorpora esa distribución y añade `entropy`,
`normalized_entropy`, `maximum_entropy`, `log_base`, `estimator` y
`normalization`; sus propiedades `backend` y `trajectory_fingerprint` facilitan
la inspección sin perder el objeto de origen. Ambas clases ofrecen
`as_analysis_result()` para producir el sobre común e inmutable
`AnalysisResult`.

La fachada no elige \(m\), \(\tau\), una política de empates ni una coordenada
a partir de los datos. Esas decisiones son parte del experimento y deben
aparecer en cualquier resultado reproducible.

## Comparadores externos: Julia y AntroPy

HAFO no copia código de estas bibliotecas y no las necesita en ejecución. Se
pueden usar como comparadores externos sólo después de igualar ventana,
retardo, base, normalización y convenio de empates:

- El ecosistema JuliaDynamics expone entropía de Shannon sobre
  `OrdinalPatterns` y la comodidad `entropy_permutation` en
  [ComplexityMeasures.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/tutorial/).
  Esta separación es preferible a introducir una llamada a Julia en el bucle
  de HAFO: Julia queda como implementación independiente para interoperabilidad
  o contraste.
- [AntroPy `perm_entropy`](https://raphaelvallat.com/antropy/generated/antropy.perm_entropy.html)
  ofrece `order`, `delay` y normalización en base dos. Sus optimizaciones y su
  tratamiento de empates dependen de la versión; por ello una coincidencia de
  un único escalar no reemplaza la comparación de ventanas, rangos, conteos y
  probabilidades.

No se debe interpretar una discrepancia como error hasta comprobar todas las
convenciones. De modo simétrico, una coincidencia con Julia o AntroPy no prueba
que \(m\) y \(\tau\) sean físicamente adecuados ni valida una conclusión
dinámica.

## Sistemas enteros y fraccionarios

La entropía de permutación actúa después del integrador y, por ello, puede
recibir muestras de ODE, mapas enteros y cualquiera de las definiciones
fraccionarias soportadas por HAFO. El algoritmo ordinal no cambia con la
derivada; cambia la procedencia científica de la trayectoria.

Para un sistema entero, una coordenada puede ser una proyección de un estado
Markoviano. En un sistema fraccionario, el vector de coordenadas en un instante
normalmente no contiene la historia hereditaria completa. En consecuencia, el
resultado caracteriza sólo la serie escalar seleccionada bajo la terminal
inferior, prehistoria, definición de derivada, orden, discretización y política
de memoria registrados. No es la entropía de un espacio de historia completo.

La comparación entre órdenes entero y fraccionario debe mantener controlados,
como mínimo, el observable, transitorio, duración física, muestreo, \(m\),
\(\tau\), empates y backend numérico. Igualar sólo el número de muestras no
iguala necesariamente el experimento físico.

## Validación y frontera científica

La verificación numérica debe probar por separado:

1. ventanas forward exactas para varios \(m\) y \(\tau\);
2. permutaciones y rangos Lehmer lexicográficos conocidos;
3. histogramas densos y denominador \(V\);
4. las tres políticas de empates;
5. probabilidades, \(H_b\) y normalización;
6. igualdad de Python, Numba, C y un oráculo Wolfram independiente.

Una validación de secuencias finitas demuestra el contrato algebraico y la
concordancia de implementaciones para esos casos. No valida automáticamente
consistencia asintótica, selección de parámetros, robustez frente a ruido,
inferencia causal, tasa entrópica, entropía KS, caos, atracción u ocultedad. La
entropía de permutación tampoco sustituye las pruebas de disipatividad,
Lyapunov, cuencas o continuación que requiere el estudio de un atractor oculto.

## Evidencia localizada con SciSpace

SciSpace se usó como índice de descubrimiento y no como oráculo matemático. Las
fórmulas y decisiones del contrato se contrastaron con los artículos primarios
y después se verifican mediante casos finitos independientes.

Primera pregunta exacta:

> Which peer-reviewed papers define, validate, or critically analyze Bandt-Pompe permutation entropy for finite scalar time series, with explicit treatment of embedding dimension, time delay, ordinal-pattern ties, normalization, and finite-sample bias?

Se solicitó `methods_used` y SciSpace devolvió datos para los cuatro registros
pedidos (`4/4`):

| ID SciSpace | Registro localizado | Resultado `methods_used` |
|---|---|---|
| `3n6afn4kjj` | PENTROPY | disponible |
| `2orrxg30ri` | Zanin et al., revisión, [DOI 10.3390/E14081553](https://doi.org/10.3390/E14081553) | disponible |
| `560v6kub1z` | Rey et al., *The asymptotic distribution of the permutation entropy*, [DOI 10.1063/5.0171508](https://doi.org/10.1063/5.0171508) | disponible |
| `2tz59w5ut1` | Little y Kane, *Permutation entropy with vector embedding delays*, [DOI 10.1103/PhysRevE.96.062205](https://doi.org/10.1103/PhysRevE.96.062205) | disponible |

Segunda pregunta exacta:

> What is the original 2002 Bandt and Pompe paper defining permutation entropy, and what ordinal-pattern convention does it use for equal values in a scalar time series?

Se solicitó `methods_used` y SciSpace devolvió datos para los tres registros
pedidos (`3/3`):

| ID SciSpace | Registro localizado | Resultado `methods_used` |
|---|---|---|
| `54h4wu7fhj` | Traversaro et al., tratamiento de series con empates, [DOI 10.1063/1.5022021](https://doi.org/10.1063/1.5022021) | disponible |
| `13irrb178z` | Unakafova y Keller, codificación eficiente de patrones, [DOI 10.3390/E15104392](https://doi.org/10.3390/E15104392) | disponible |
| `3oohrbcnwc` | Amigó y Keller, [DOI 10.1140/EPJST/E2013-01840-1](https://doi.org/10.1140/EPJST/E2013-01840-1) | disponible |

La segunda consulta no devolvió el artículo original entre esos tres registros;
su identidad se confirmó en la fuente primaria de APS. Los resúmenes
`methods_used` ayudaron a localizar convenciones, eficiencia, empates y
comportamiento finito, pero no se transcribieron como definición normativa.

## Fuentes primarias para este contrato

- Bandt y Pompe, *Permutation Entropy: A Natural Complexity Measure for Time
  Series*, Physical Review Letters 88, 174102 (2002),
  [DOI 10.1103/PhysRevLett.88.174102](https://doi.org/10.1103/PhysRevLett.88.174102).
- Traversaro, Redelico, Risk, Frery y Rosso, *Bandt-Pompe symbolization
  dynamics for time series with tied values: A data-driven approach*, Chaos
  28, 075502 (2018),
  [DOI 10.1063/1.5022021](https://doi.org/10.1063/1.5022021).
- Unakafova y Keller, *Efficiently Measuring Complexity on the Basis of
  Real-World Data*, Entropy 15, 4392--4415 (2013),
  [DOI 10.3390/E15104392](https://doi.org/10.3390/E15104392).
- Rey, Frery, Gambini y Lucini, *The asymptotic distribution of the permutation
  entropy*, Chaos 33, 113108 (2023),
  [DOI 10.1063/5.0171508](https://doi.org/10.1063/5.0171508).
