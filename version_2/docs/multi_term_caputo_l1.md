# Fachada Caputo multitérmino L1

Estado de implementación: `implemented`. Estabilidad de API: `experimental`.

La API `integrate_multi_term_caputo_l1` resuelve la ecuación finita

\[
 \sum_{j=1}^{R}c_j\,{}_a^CD_t^{\alpha_j}x(t)=f(t,x(t)),
 \qquad 0<\alpha_j\le 1,\quad c_j\ge 0,
 \qquad x(a)=x_0.
\]

Los valores `coefficients` son coeficientes de la ecuación: **no se
normalizan**. Su suma puede ser distinta de uno y puede portar unidades
físicas. La función reutiliza exactamente el kernel L1 combinado del solver de
orden distribuido; no copia sus pesos, su historia ni el corrector.

## Suma finita y medida atómica

La suma anterior puede escribirse como

\[
 \int_{(0,1]}{}_a^CD_t^\alpha x(t)\,\mu(d\alpha),
 \qquad
 \mu=\sum_{j=1}^{R}c_j\delta_{\alpha_j}.
\]

Esta identidad no convierte el problema en una cuadratura aproximada de una
densidad continua. En la fachada:

- `measure_kind="finite_discrete_atomic_order_measure"`;
- `continuous_order_quadrature_used=False`;
- `continuous_order_density_inferred=False`;
- `normalization="none"`.

Una ecuación de orden distribuido genuinamente continuo requiere declarar una
densidad o medida continua y analizar por separado el error de discretización
en el espacio de órdenes. En el modelo multitérmino la suma finita es el modelo
completo, por lo que ese error de cuadratura no existe; persisten los errores
temporales, del corrector y de punto flotante.

## Canonización sin cambiar el modelo

`canonicalize_multi_term_caputo_terms` aplica una sola vez:

1. conversión a vectores `float64` reales, finitos y unidimensionales;
2. validación de todos los órdenes en `(0, 1]`, incluso los asociados a cero;
3. rechazo de coeficientes negativos y del caso todo-cero;
4. política explícita `zero_coefficient_policy="drop"` o `"raise"`;
5. ordenamiento ascendente;
6. agrupación sólo si dos órdenes `float64` son exactamente iguales;
7. suma reproducible de duplicados mediante `math.fsum`.

No se usa tolerancia para agrupar órdenes: reemplazar dos órdenes cercanos por
uno solo modifica el kernel en toda la historia. La salida conserva los
términos originales, grupos de índices, ceros eliminados y términos canónicos.

Por ejemplo,

```python
from hidden_attractors.fractional import (
    canonicalize_multi_term_caputo_terms,
)

terms = canonicalize_multi_term_caputo_terms(
    orders=[0.8, 0.3, 0.3, 0.6],
    coefficients=[0.6, 0.15, 0.25, 0.0],
)

print(terms.orders)        # [0.3, 0.8]
print(terms.coefficients)  # [0.4, 0.6]
print(terms.source_indices)  # ((1, 2), (0,))
```

## Discretización L1 y kernel único

Para `0 < alpha_j < 1`, en `t_n=a+nh`,

\[
 {}_a^CD_t^{\alpha_j}x(t_n)
 \approx
 \frac{h^{-\alpha_j}}{\Gamma(2-\alpha_j)}
 \sum_{k=0}^{n-1}b_k^{(j)}
 \left(x_{n-k}-x_{n-k-1}\right),
\]

\[
 b_k^{(j)}=(k+1)^{1-\alpha_j}-k^{1-\alpha_j}.
\]

Defina

\[
 \rho_j=\frac{c_jh^{-\alpha_j}}{\Gamma(2-\alpha_j)},
 \qquad
 K_k=\sum_{j=1}^{R}\rho_jb_k^{(j)}.
\]

La ecuación discreta completa se reduce a una convolución:

\[
 \sum_{k=0}^{n-1}K_k
 \left(x_{n-k}-x_{n-k-1}\right)=f(t_n,x_n).
\]

Como `K_0=A=sum_j rho_j>0`,

\[
 x_n=B_n+A^{-1}f(t_n,x_n),
\]

\[
 B_n=x_{n-1}-A^{-1}
 \sum_{k=1}^{n-1}K_k
 \left(x_{n-k}-x_{n-k-1}\right).
\]

El predictor evalúa `f(t_n,x_{n-1})` y Picard itera la ecuación implícita. La
salida conserva iteraciones y residuos. HAFO no estima la constante de Lipschitz
ni infiere contractividad.

## Rama exacta de orden entero

Para `alpha_j=1`, HAFO usa el límite exacto backward Euler:

\[
 b_0^{(j)}=1,\qquad b_k^{(j)}=0\;(k\ge1),\qquad
 \rho_j=c_j/h.
\]

Si todos los términos tienen orden uno, su canonización produce un único
término con coeficiente `sum(c_j)`. Ésta es interoperabilidad con el límite
entero, no una afirmación de tasa L1 fraccionaria en `alpha=1`.

## API

```python
import numpy as np

from hidden_attractors.fractional import integrate_multi_term_caputo_l1


def rhs(time, state):
    return -0.35 * state + np.array([0.2])


result = integrate_multi_term_caputo_l1(
    rhs,
    initial_state=[0.8],
    orders=[1/3, 2/3, 1.0],
    coefficients=[0.4, 0.7, 0.75],
    step=0.01,
    n_steps=100,
    initial_regularity="nonsmooth",
    use_acceleration=True,
)

print(result.times, result.states)
print(result.solver_info["coefficient_sum"])  # 1.85, no 1.0
```

`MultiTermCaputoResult` expone la trayectoria y el kernel del resultado
distribuido sin copiarlos. `method="multi_term_caputo_l1"` y
`definition="caputo_multi_term_finite_sum"` mantienen la semántica pública;
`distributed_result` permite auditar el solver delegado.

## Ejemplo ejecutable

```bash
python examples/multi_term_caputo_relaxation.py
```

El ejemplo usa una relajación multiescala forzada con solución afín
manufacturada. Incluye coeficientes cuya suma es `1.85`, un orden repetido, un
término nulo y `alpha=1`. El error finito comprueba consistencia de esa
trayectoria; no prueba una tasa general ni dinámica caótica.

## Complejidad y elección de backend

Después de canonizar `R` términos, el núcleo compartido cuesta

\[
 O(RN+N^2d)
\]

y almacena `O(Nd+N+R)`. La fachada añade sólo ordenamiento y agrupación
`O(R log R)`. `use_acceleration=True` emplea Numba para construir el kernel y
sumar la historia; el RHS y Picard permanecen en Python.

No se creó una segunda implementación C ni Julia para esta fachada:

- duplicaría el solver y aumentaría la superficie de paridad sin acelerar la
  capa semántica pequeña;
- el bucle costoso ya se ejecuta en Numba sin GIL;
- Julia impondría un runtime y serialización adicionales para una operación
  que HAFO ya resuelve localmente;
- C o una ruta SOE/CQ rápida sólo se justifican después de perfilar historiales
  largos y deben atacar el núcleo compartido, no esta fachada.

El benchmark `benchmarks/bench_multi_term_caputo.py` separa el costo de fachada,
la construcción del kernel y la historia. Sus resultados describen el host y
las cargas medidas; no establecen una ventaja universal.

Los resultados medidos dependen del host, la versión de Numba, el calentamiento
y el número de términos coalescibles. Por ello la documentación pública no fija
un factor de aceleración; el benchmark registra por separado fachada,
construcción del kernel e historia ya combinada.

## Base bibliográfica

Las referencias primarias al final de esta página sustentan la formulación
multitérmino, el esquema L1 y la interpretación como medida atómica. La
canonización, el resultado estructurado y la reutilización del kernel son diseño
de HAFO y no se atribuyen a un resumen de búsqueda.

## Verificación Wolfram y pruebas

El caso independiente `validation/wolfram/cases/multi_term_caputo_l1.wl`
construye pesos mediante integración simbólica, incluye la rama `alpha=1`, usa
coeficientes racionales cuya suma es `37/20`, comprueba permutación y
coalescencia, y resuelve una recurrencia afín sin importar HAFO. El comparador
Python llama únicamente a la fachada pública.

El contrato exige residuo simbólico, residuo discreto, paridad del kernel y
paridad de trayectoria dentro de las tolerancias registradas por el caso. Esas
comparaciones son evidencia finita de implementación; no demuestran estabilidad
general, convergencia caótica, atracción ni ocultedad.

Pruebas focales:

```bash
python -m pytest tests/test_multi_term_caputo.py \
  tests/test_multi_term_caputo_example.py \
  tests/test_multi_term_caputo_wolfram.py -q
```

## Límites

- sólo coeficientes escalares no negativos;
- todos los términos comparten terminal inferior, estado inicial e historia;
- sólo `0 < alpha <= 1`;
- un término de orden cero debe modelarse explícitamente en el RHS;
- historia completa y malla física uniforme;
- no hay malla graduada, correcciones de arranque, SOE, adaptatividad ni
  estimador temporal;
- rangos dinámicos extremos pueden absorber contribuciones pequeñas en suma de
  punto flotante aun sin cancelación de signos;
- una trayectoria finita no demuestra estabilidad general, convergencia en un
  sistema caótico, atracción ni ocultedad.

## Referencias principales

- [Diethelm y Ford (2004)](https://doi.org/10.1016/S0096-3003(03)00739-2),
  ecuaciones Caputo multiorden y solución numérica.
- [Ren y Sun (2014)](https://doi.org/10.4208/EAJAM.181113.280514A), L1 para
  ecuaciones Caputo multitérmino.
- [She, Li y Sun (2022)](https://doi.org/10.1016/j.matcom.2021.11.005), L1
  transformado y capa inicial no suave.
- [Zaky y Machado (2020)](https://doi.org/10.1016/j.camwa.2019.07.008),
  distribución atómica mediante deltas de Dirac y reducción multitérmino.
- [Kochubei (2009)](https://doi.org/10.1088/1751-8113/42/31/315203), derivadas
  de orden distribuido respecto a una medida positiva.
