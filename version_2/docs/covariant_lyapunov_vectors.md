# Vectores covariantes de Lyapunov para sistemas enteros

Estado: API pública `experimental` para `q=1`; CLV fraccionario
`research_required`.

## Alcance científico

HAFO implementa los vectores covariantes de Lyapunov (CLV) del cociclo
tangente clásico para flujos y mapas de orden entero, es decir, para
`q = 1`. La implementación es un diagnóstico local de tiempo finito. No
clasifica automáticamente una órbita como caótica, hiperbólica, atractora u
oculta.

Las rutinas que construyen CLV rechazan explícitamente `q != 1`. Esta
restricción no es una limitación sintáctica: una derivada no local requiere
linearizar el operador de memoria y decidir en qué espacio histórico vive el
cociclo. El Jacobiano instantáneo de la parte derecha no contiene por sí solo
esa información.

## Formulación matemática

Para un mapa diferenciable

\[
  x_{n+1}=F_n(x_n),
  \qquad M_n=D F_n(x_n),
\]

la perturbación satisface

\[
  \delta x_{n+1}=M_n\,\delta x_n.
\]

Para un flujo

\[
  \dot{x}=f(t,x),
  \qquad \dot{Y}=J(t,x)Y,
  \qquad J=\frac{\partial f}{\partial x},
\]

el propagador tangente de un segmento se obtiene integrando conjuntamente
`(x,Y)`. HAFO usa DOP853 para esta fachada entera. Si no se suministra un
Jacobiano analítico, calcula diferencias centrales con paso relativo por
componente.

Los CLV \(v_i^{(j)}\) son direcciones proyectivas: bajo el propagador tangente
del segmento cumplen

\[
  M_i v_i^{(j)} \parallel v_{i+1}^{(j)}.
\]

Su norma y signo son una elección de gauge. HAFO normaliza cada columna con la
norma euclidiana, pero toda comparación valida direcciones sin orientación.

## Algoritmo dinámico de Ginelli

En cada segmento forward se factoriza

\[
  M_i Q_i = Q_{i+1}R_{i+1},
\]

donde \(Q_i\in\mathbb{R}^{d\times k}\) tiene columnas ortonormales y
\(R_{i+1}\in\mathbb{R}^{k\times k}\) es triangular superior. HAFO fija la
diagonal de \(R\) positiva cambiando coherentemente los signos de \(Q\) y
\(R\). No se reordenan columnas después de QR porque eso rompería la
correspondencia covariante.

El cálculo se divide en cuatro horizontes distintos:

1. transitorio del estado, descartado antes de inicializar el cociclo;
2. transitorio forward de la base QR;
3. ventana observada, donde se guardan bases y factores triangulares;
4. transitorio backward futuro, que acondiciona la recursión inversa.

Desde una matriz terminal triangular superior \(C_N\), la fase backward
resuelve, sin formar inversas,

\[
  R_{i+1}C_i=C_{i+1},
  \qquad
  C_i\leftarrow\operatorname{normalize\_columns}(C_i).
\]

Finalmente,

\[
  V_i=Q_iC_i,
  \qquad
  V_i\leftarrow\operatorname{normalize\_columns}(V_i).
\]

Una diagonal singular o numéricamente no resoluble de \(R\) se informa como
fallo estructurado. No se sustituye silenciosamente por un valor mínimo: el
recorte que es razonable para almacenar \(\log |R_{jj}|\) no convierte un
sistema triangular singular en uno invertible.

Los exponentes finitos asociados a la ventana observada se calculan mediante

\[
  \lambda_j(T)=\frac{1}{T}
  \sum_i \log |R_{i,jj}|.
\]

Estas tasas no demuestran por sí solas que los CLV hayan convergido.

## Convenios de datos

El resultado público usa los siguientes layouts:

| Campo | Forma | Significado |
|---|---:|---|
| `coordinates` | `(samples,)` | tiempo posterior a los transitorios o iteración observada |
| `sampled_states` | `(samples, dimension)` | estados en las bases almacenadas |
| `vectors` | `(samples, n_vectors, dimension)` | CLV como filas en cada muestra |
| `exponents` | `(n_vectors,)` | tasas QR finitas de la ventana observada |
| `convergence` | `(segments, n_vectors)` | promedios acumulados de tasas, no prueba de convergencia CLV |

El layout de `vectors` coincide con el historial público de SALI/GALI de HAFO.
Dentro del núcleo algebraico, \(Q\), \(R\) y \(C\) conservan vectores por
columnas.

## Ángulos entre CLV y subespacios

Para dos direcciones unitarias se usa el ángulo agudo no orientado

\[
  \theta(u,v)=
  \operatorname{atan2}
  \left(\sqrt{\max(0,1-|u^\top v|^2)},|u^\top v|\right)
  \in[0,\pi/2].
\]

La forma `atan2` evita la pérdida innecesaria de precisión de
`arccos(abs(dot))` cerca de cero. Para dos subespacios se calculan los ángulos
principales con la rutina estable de SciPy y se reporta, de forma explícita, el
mínimo. Un promedio por ventanas es sólo posprocesamiento de ángulos ya
calculados; no sustituye el transitorio backward.

Cuando hay exponentes repetidos o casi degenerados, el subespacio covariante
puede estar definido aunque una base individual dentro de él no sea única. En
ese caso deben compararse proyectores o ángulos de subespacios, no columnas con
signo fijo.

## Decisiones de backend

- NumPy/LAPACK realiza QR, operación ya altamente optimizada y dominante para
  dimensiones moderadas.
- Numba acelera la sustitución triangular backward, la normalización por
  columnas y la reconstrucción por lotes.
- SciPy DOP853 propaga el sistema variacional de flujos con tolerancias
  explícitas.
- Un benchmark histórico del 3 de agosto de 2026 comparó tres cargas con paridad
  previa y siete repeticiones alternadas. En la carga mayor, Numba redujo el
  tiempo completo del mapa en aproximadamente 52.8 %, pero la reconstrucción
  pública Numba ocupó sólo 1.60 % del tiempo end-to-end calentado. Incluso
  reemplazar idealmente toda esa fase por coste cero limitaría la mejora a
  aproximadamente 1.016x. Por ello no se añade ahora un backend C para la
  pasada backward; C se reconsiderará sólo si un perfil de sistemas reales
  muestra un cuello residual material y un candidato C supera pruebas de
  paridad y tiempos end-to-end repetidos.
- ChaosTools.jl no expone actualmente una API pública CLV. Por ello no se paga
  el coste de un puente Julia por paso; Julia permanece como comparación por
  lote para capacidades que sí ofrece.
- pynamicalsys puede servir como comparación externa opcional de `q=1`, pero
  no se importa ni se copia en el núcleo permisivo de HAFO debido a su licencia
  GPL-3.0-only.

Antes de reservar memoria, HAFO estima el workspace de las historias. Con
\(N_o\) segmentos observados, \(N_b\) segmentos futuros, dimensión \(d\) y
\(k\) vectores, la cota base en `float64` es aproximadamente

\[
8\left[(N_o+1)(dk+d)+(N_o+N_b)k^2+k^2\right]\ \text{bytes},
\]

sin contar temporales del integrador. `max_workspace_bytes` permite rechazar
una configuración antes de una asignación grande.

El registro histórico completo, incluido hardware, versiones, calentamiento,
dispersión y frontera probatoria, está en
`validation/outputs/benchmarks/covariant_lyapunov_numpy_numba_20260803.json`
(SHA-256
`15A904AC4F67491015DF4A07CFC4353BBC5F2E19C4BC249121F415CCEE7F61BD`).
Su `script_sha256` corresponde a una revisión anterior y no coincide con el
script del checkout actual. Por ello documenta aquella ejecución y no valida el
rendimiento actual; hace falta repetir el benchmark para una decisión nueva.

## API pública experimental

Las pruebas de layouts, covariancia, paridad NumPy/Numba y manejo de fallos ya
permiten publicar ocho símbolos experimentales en el nivel superior
`hidden_attractors`:

| Símbolo | Contrato |
|---|---|
| `CovariantQRHistoryResult` | CLV y coeficientes reconstruidos desde una historia QR validada |
| `CovariantLyapunovResult` | CLV, exponentes QR finitos, estados, coordenadas, estado de ejecución y metadatos para un flujo o mapa |
| `CovariantAngleResult` | ángulos de pares, ángulos mínimos entre subespacios y promedios móviles opcionales |
| `integer_covariant_vectors_from_qr_history` | fase backward algebraica para `Q.shape == (samples, dimension, n_vectors)` y `R.shape == (samples - 1, n_vectors, n_vectors)` |
| `integer_flow_covariant_lyapunov_vectors` | propagación DOP853 del estado y sistema variacional, seguida de la recursión de Ginelli |
| `integer_map_covariant_lyapunov_vectors` | recurrencia exacta del Jacobiano del mapa o diferencias centrales declaradas |
| `integer_system_covariant_lyapunov_vectors` | dispatcher para un objeto HAFO con `kind`, `evaluate` y Jacobiano opcional |
| `covariant_lyapunov_angles` | posprocesamiento geométrico de una historia `(samples, n_vectors, dimension)` |

Las cuatro rutinas que construyen CLV aceptan `q` sólo para comprobar que todos
sus componentes son numéricamente iguales a uno. No existe un despacho oculto
hacia un integrador fraccionario. `covariant_lyapunov_angles` no acepta `q`
porque sólo procesa vectores suministrados; tampoco infiere que sean CLV
válidos.

Un flujo típico usa la fachada pública de esta forma:

```python
from hidden_attractors import integer_flow_covariant_lyapunov_vectors

result = integer_flow_covariant_lyapunov_vectors(
    rhs,
    jacobian,
    x0,
    t_final=50.0,
    t_burn=10.0,
    forward_transient_time=20.0,
    backward_transient_time=20.0,
    qr_interval=0.25,
    n_vectors=3,
    backend="auto",
    q=1.0,
)

if result.status != "ok":
    raise RuntimeError(f"{result.status}: {result.error_message}")
```

Las fachadas de flujo, mapa y sistema devuelven `CovariantLyapunovResult`; su
`status` debe inspeccionarse antes de usar los arreglos. Los errores de
configuración se elevan inmediatamente, mientras fallos durante la evolución
se registran mediante estados como `invalid_callback`, `singular_cocycle` o
divergencia de la fase correspondiente. La covariancia se verifica con
residuos proyectivos entre muestras; un signo distinto de la misma dirección
no es un error.

Para un mapa se sustituyen los horizontes temporales por `iterations`,
`transient_iterations`, `forward_transient_iterations`,
`backward_transient_iterations` y `qr_interval_iterations`. El ejemplo
ejecutable `examples/covariant_lyapunov_henon_map.py` muestra esa ruta con
Jacobiano analítico y el posprocesador de ángulos.

## Transitorios y trabajo reciente

Ginelli et al. introdujeron el algoritmo dinámico y Kuptsov--Parlitz
desarrollaron su teoría y variantes. Froyland et al. compararon Ginelli,
Wolfe--Samelson, SVD y proyectores de dicotomía. Noethen extendió el análisis a
espacios de Hilbert y relacionó la velocidad de convergencia con la brecha
espectral. El trabajo de du Plessis, Hillebrand y Skokos de 2026 estudia
criterios más eficientes para detener los transitorios y advierte del
deterioro de subespacios centrales tras backward excesivo.

La primera versión HAFO usa horizontes explícitos y registra que no aplica aún
el criterio adaptativo de 2026. Esto mantiene auditable el coste y evita llamar
“convergencia” a un simple horizonte largo. Una extensión posterior podrá
implementar dos estimaciones independientes de subespacios y un umbral de
distancia, con coste y evidencia separados.

## Frontera fraccionaria por operador

| Familia | Qué falta para CLV |
|---|---|
| Caputo y Caputo multitérmino | cociclo del historial, condiciones iniciales y renormalización coherente de memoria |
| Riemann--Liouville y Grünwald--Letnikov | linearización del operador discreto, inicialización fraccionaria y norma del estado histórico |
| Hadamard y Caputo--Hadamard | cociclo en coordenada logarítmica y relación con el tiempo físico no uniforme |
| Caputo/RL templadas | inclusión del parámetro de tempering en el operador tangente y su memoria |
| Caputo--Fabrizio | cociclo del kernel exponencial no singular y su normalización específica |
| Atangana--Baleanu--Caputo | linearización del kernel Mittag--Leffler y de cualquier aproximación de historia rápida |
| orden variable | cociclo no autónomo dependiente del perfil de orden y cachés no invariantes |
| orden distribuido | espacio histórico combinado, cuadratura en orden y errores temporal/de cuadratura separados |
| conformable/local | sólo puede reutilizar CLV entero cuando exista una equivalencia ODE demostrada y registrada |

Los ángulos son una operación geométrica reutilizable si el usuario ya aporta
vectores válidos de otra teoría. Esa reutilización no valida cómo se obtuvieron
los supuestos CLV fraccionarios.

## Referencias primarias

- Ginelli et al. (2007), DOI `10.1103/PhysRevLett.99.130601`.
- Wolfe y Samelson (2007), DOI `10.1111/j.1600-0870.2007.00234.x`.
- Kuptsov y Parlitz (2012), DOI `10.1007/s00332-012-9126-5`.
- Froyland et al. (2013), DOI `10.1016/j.physd.2012.12.005`.
- Ginelli et al. (2013), DOI `10.1088/1751-8113/46/25/254005`.
- Noethen (2021), DOI `10.3934/jcd.2021014`.
- du Plessis, Hillebrand y Skokos (2026), DOI
  `10.1016/j.physd.2026.135237`.
