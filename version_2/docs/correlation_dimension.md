# Dimensión de correlación (q=2)

HAFO ofrece una implementación local de la suma de correlación de
Grassberger--Procaccia para \(q=2\). La misma API consume nubes de puntos o
trayectorias muestreadas procedentes de sistemas de orden entero o
fraccionario, con backends Python, Numba y C nativo. Es un diagnóstico empírico
de una muestra finita: no prueba por sí solo caos, atracción, ocultedad ni la
dimensión del estado hereditario completo de un sistema fraccionario.

## Contrato matemático

Para \(N\) puntos ordenados temporalmente
\(x_0,\ldots,x_{N-1}\in\mathbb{R}^d\) y una ventana de Theiler entera
\(w\geq 0\), HAFO usa exclusivamente pares no ordenados

\[
\mathcal{P}_w=\{(i,j):0\leq i<j<N,\;j-i>w\}.
\]

El rango válido es \(0\leq w\leq N-2\), y el denominador es

\[
M_w=|\mathcal{P}_w|=
\frac{(N-w)(N-w-1)}{2}.
\]

Para cada radio positivo \(r\), el conteo y la suma de correlación son

\[
K(r)=\sum_{(i,j)\in\mathcal{P}_w}
\mathbf{1}\!\left[d(x_i,x_j)<r\right],
\qquad
C_2(r)=\frac{K(r)}{M_w}.
\]

La comparación es estricta, \(d<r\), nunca \(d\leq r\). Esta distinción importa
cuando una distancia coincide exactamente con un radio. Los radios deben ser
finitos, positivos, estrictamente crecientes y sin duplicados. Las métricas
disponibles son `euclidean`, `chebyshev` y `manhattan`.

## Ajuste de una pendiente

`fit_correlation_dimension` ajusta mediante mínimos cuadrados ordinarios

\[
\log C_2(r)=b+D_2\log r
\]

sólo en un intervalo inclusivo `fit_radius_range=(r_min, r_max)` proporcionado
explícitamente por quien llama. Dentro de ese intervalo excluye los puntos
saturados o indefinidos mediante \(0<C_2(r)<1\) y exige al menos tres puntos por
defecto.

HAFO no selecciona automáticamente una región de escala. La pendiente local,
\(R^2\) y el error estándar de la regresión se devuelven como diagnósticos del
ajuste declarado; no son por sí solos un certificado de región de escala ni un
intervalo de incertidumbre completo, porque radios y pares de trayectoria están
correlacionados.

## API pública

```python
import numpy as np

from hidden_attractors.analysis import (
    correlation_sum_curve,
    fit_correlation_dimension,
)

radii = np.geomspace(1.0e-3, 1.0, 80)
curve = correlation_sum_curve(
    points,
    radii,
    theiler_window=25,
    metric="euclidean",
    backend="auto",
    sampling="uniform dt=0.01 after 5000 transient samples",
    projection="delay reconstruction of x, m=3, tau=12 samples",
)
fit = fit_correlation_dimension(
    curve,
    fit_radius_range=(0.02, 0.15),
    minimum_points=5,
)

print(fit.slope, fit.r_squared, curve.backend)
```

`correlation_sum_curve` devuelve radios, conteos enteros, \(C_2(r)\), número de
pares elegibles, métrica, ventana de Theiler, backend solicitado y efectivo, y
metadatos de procedencia. `estimate_correlation_dimension` reúne el conteo y el
ajuste en una llamada, pero también exige `fit_radius_range`; no introduce una
heurística silenciosa.

## Backends y coste

| Backend | Papel | Observaciones |
|---|---|---|
| `python` | Referencia transparente para casos pequeños | Conteo directo con el mismo contrato estricto. |
| `numba` | Ruta compilada de propósito general | Un cálculo de distancia y una búsqueda binaria por par elegible. |
| `native_c` | Kernel C/OpenMP opcional | Usa acumuladores por hilo; puede caer a Numba si `fallback=True`. |
| `auto` | Selección local | Intenta C nativo desde 131 072 pares elegibles; usa Numba por debajo. El resultado registra el backend efectivo y cualquier fallback. |

Con \(R\) radios, dimensión de característica \(d\) y \(M_w\) pares elegibles,
el algoritmo actual cuesta
\(O\!\left(M_w(d+\log R)\right)\) y usa \(O(R)\) memoria de trabajo, sin contar
los acumuladores por hilo del kernel C. Los tres backends comparten conteos,
normalización y comparación estricta; no se cruza Python, Julia o C por cada
paso de un integrador.

El benchmark reproducible `benchmarks/bench_correlation_sum.py` separa
compilación/JIT de las muestras cronometradas y verifica conteos idénticos. En
la ejecución local del 3 de agosto de 2026 (`OMP_NUM_THREADS=4`), para 31 626 pares y 24 radios las
medianas fueron Python `0.428362 s`, Numba `0.001054 s` y C/OpenMP
`0.001227 s`; para 124 750 pares y 48 radios fueron `1.552896 s`, `0.003584 s`
y `0.003183 s`, respectivamente. Son medidas de este host y estas dos cargas,
no un ranking universal. El umbral de `auto` es una política conservadora que
también evita atribuir a la ejecución sostenida el coste de construir el
backend; puede volver a calibrarse con el mismo script en cada plataforma.

## Trayectorias enteras y fraccionarias

La dimensión de correlación pertenece a la capa de análisis de datos y no
modifica el solver que produjo la trayectoria. Para que el resultado sea
auditable se deben registrar, como mínimo, transitorio descartado, muestreo,
coordenadas o reconstrucción delay, métrica, radios, ventana de Theiler e
intervalo de ajuste.

En un ODE de orden entero, las coordenadas completas pueden representar un
estado Markoviano. En una FDE, una proyección finita de \(x(t)\) normalmente no
contiene la historia retenida por la derivada. Por ello, el \(D_2\) calculado
para una trayectoria fraccionaria caracteriza únicamente las coordenadas o la
reconstrucción suministradas, no la dimensión de un espacio de historia
completo.

## Validación finita independiente

El caso `validation/wolfram/cases/correlation_dimension.wl` usa seis puntos 2D
exactos, radios `[1, 1.1, 1.5, 2, 2.1, 2.3]` y \(w=1\). El denominador es 10;
los conteos estrictos esperados son `[0, 4, 6, 6, 8, 10]`, y por tanto
\(C_2=[0,0.4,0.6,0.6,0.8,1]\). En el intervalo explícito `[1.1, 2.1]`, el ajuste
produce pendiente `0.8664716421373693` y \(R^2=0.8241791730183863\). La peor
diferencia Python--Wolfram retenida es `3.552713678800501e-15`, frente a una
tolerancia de `5e-13`.

Este oráculo comprueba únicamente el conteo de pares, la normalización y la
regresión log--log sobre un conjunto finito. No valida que ese intervalo sea una
región de escala física, ni establece consistencia estadística del estimador,
dimensión fractal, caos, atracción u ocultedad.

## Alcance pendiente

No están implementados todavía la selección automática de región de escala,
intervalos por bootstrap, algoritmos boxed o fixed-mass, dimensiones
generalizadas de Rényi, estimadores puntuales/locales ni estructuras de vecinos
para reducir el coste cuadrático en muestras grandes. Esas capacidades no se
simulan mediante una llamada opaca a FractalDimensions.jl o pynamicalsys.

## Fuentes primarias

- Grassberger y Procaccia, *Measuring the Strangeness of Strange Attractors*,
  [DOI 10.1016/0167-2789(83)90298-1](https://doi.org/10.1016/0167-2789(83)90298-1).
- Theiler, *Spurious Dimension from Correlation Algorithms Applied to Limited
  Time-Series Data*,
  [DOI 10.1103/PhysRevA.34.2427](https://doi.org/10.1103/PhysRevA.34.2427).
- Deshmukh, Bradley, Garland y Meiss, *Toward Automated Extraction and
  Characterization of Scaling Regions in Dynamical Systems*,
  [DOI 10.1063/5.0069365](https://doi.org/10.1063/5.0069365).
