# SALI, GALI y LDI para flujos y mapas de orden entero

## Alcance científico

HAFO calcula el *Smaller Alignment Index* (SALI) y los *Generalized
Alignment Indices* (GALI) a partir de vectores de desviación de un flujo o un
mapa ordinario, es decir, con orden `q=1` y sin memoria hereditaria. La API
devuelve indicadores geométricos de tiempo finito y su procedencia numérica;
no devuelve una etiqueta automática de caos, atracción u ocultedad.

La extensión a una derivada fraccionaria no consiste en sustituir el
integrador del estado. Para Caputo, Riemann--Liouville, Grünwald--Letnikov,
Caputo--Fabrizio, Atangana--Baleanu y las demás familias no locales hay que
formular primero la ecuación variacional del operador completo, incluida la
historia, y justificar cómo se renormaliza una perturbación en ese espacio.
Por eso el catálogo conserva SALI/GALI fraccionario como
`research_required` y las fachadas ejecutables rechazan `q != 1`.

## Definiciones

Sean vectores de desviación no nulos
\(w_1,\ldots,w_k\in\mathbb R^d\) y

\[
\widehat w_i=\frac{w_i}{\lVert w_i\rVert_2}.
\]

SALI usa dos direcciones:

\[
\operatorname{SALI}
=\min\left(
\lVert\widehat w_1+\widehat w_2\rVert_2,
\lVert\widehat w_1-\widehat w_2\rVert_2
\right).
\]

Si \(W_k=[\widehat w_1\ \cdots\ \widehat w_k]\), GALI es el volumen
del paralelepípedo generado por sus columnas:

\[
\operatorname{GALI}_k
=\left\lVert\widehat w_1\wedge\cdots\wedge\widehat w_k\right\rVert
=\sqrt{\det(W_k^{\mathsf T}W_k)}
=\prod_{j=1}^{k}\sigma_j(W_k).
\]

La última expresión es también el índice de dependencia lineal
\(\operatorname{LDI}_k\) para esas mismas columnas normalizadas:

\[
\operatorname{LDI}_k=\prod_{j=1}^{k}\sigma_j(W_k)
=\operatorname{GALI}_k.
\]

Esta igualdad permite calcular el volumen mediante valores singulares,
sin expandir productos exteriores ni determinantes mal condicionados. Para dos
vectores unitarios, con \(c=\widehat w_1^{\mathsf T}\widehat w_2\),

\[
\operatorname{SALI}^2=2-2|c|,
\qquad
\operatorname{GALI}_2^2=1-c^2,
\]

y por tanto queda la identidad de control

\[
4\operatorname{GALI}_2^2
=\operatorname{SALI}^2\left(4-\operatorname{SALI}^2\right).
\]

Cada vector se normaliza **por separado**. No se reemplazan los vectores
evolucionados por una base ortonormal: un QR recurrente mantendría las columnas
artificialmente ortogonales y destruiría el alineamiento que SALI/GALI debe
medir. QR puede usarse sobre una copia para obtener el volumen o para generar
las desviaciones iniciales, nunca para reiniciar sus direcciones durante la
propagación.

## Propagación variacional

Para un flujo ordinario autónomo,

\[
\dot x=f(x),\qquad \dot V=J_f(x)V,
\]

donde \(V\in\mathbb R^{d\times k}\). HAFO integra conjuntamente el estado y
las columnas tangentes con DOP853. Después de un transitorio que afecta sólo al
estado, cada segmento termina con normalización L2 independiente de las
columnas. El Jacobiano puede ser analítico o una diferencia central registrada
como tal.

Para un mapa ordinario,

\[
x_{n+1}=F(n,x_n),\qquad V_{n+1}=D F(n,x_n)V_n.
\]

El Jacobiano se evalúa en \(x_n\), antes de sustituir el estado por
\(x_{n+1}\). Las ecuaciones en diferencias fraccionarias no entran por esta
ruta: requieren un contrato de estado histórico distinto.

## Método multiparticle

La alternativa `multi_particle` propaga una trayectoria de referencia y
\(k\) trayectorias vecinas. Después de cada intervalo forma las diferencias,
calcula los índices, normaliza cada separación y reinserta el vecino a una
distancia declarada \(d_0\) de la referencia. El estudio de Manda, Hillebrand y
Skokos recomienda, para sus controles Hamiltonianos en doble precisión,
\(d_0\approx\sqrt{\epsilon}\), intervalo de renormalización no mayor que uno y
error relativo de energía no mayor que \(\sqrt{\epsilon}\). Esas cifras son
condiciones estudiadas para esos modelos, no garantías universales para un
sistema disipativo o no suave.

## Núcleo numérico y elección de lenguaje

La ruta de referencia usa SVD de NumPy/LAPACK y conserva tanto el valor directo
como `log_gali`. La ruta Numba calcula por lotes un QR de Householder sobre una
copia de cada matriz normalizada y suma los logaritmos de la diagonal de
\(R\). Así puede conservarse un log-volumen finito aunque el valor directo
subdesborde a cero; una máscara `censored` distingue ese caso de rango
exactamente deficiente.

No se introduce un kernel C específico para estas matrices densas pequeñas.
SVD ya se ejecuta en LAPACK nativo y el bucle por muestras se compila con
Numba; cruzar Python--C por cada muestra añadiría interfaz sin reemplazar el
costo dominante. Un kernel C sólo se promoverá si un benchmark reproducible de
ensambles grandes demuestra una ventaja material. Julia tampoco se invoca por
paso: `ChaosTools.jl` se conserva como comparación por lotes opcional, no como
dependencia de ejecución del núcleo HAFO.

## API

Las funciones públicas son:

- `smaller_alignment_index`: SALI de dos vectores.
- `generalized_alignment_index`: GALI de una matriz instantánea cuyas columnas
  son vectores de desviación.
- `linear_dependence_index`: forma LDI/SVD equivalente del volumen GALI para
  una matriz instantánea.
- `alignment_indices_from_tangent_history`: procesa un historial con forma
  `(n_samples, n_vectors, dimension)`.
- `integer_flow_alignment_indices`: flujo `q=1`, por ecuación variacional o
  partículas vecinas.
- `integer_map_alignment_indices`: mapa entero, por Jacobiano o partículas
  vecinas.
- `integer_system_alignment_indices`: fachada para un `ChaoticSystem` de tipo
  `flow` o `map`.

`AlignmentIndexResult` es el registro público inmutable. Conserva coordenadas
de tiempo/iteración, SALI,
`log_sali`, órdenes GALI, GALI, `log_gali`, máscara de censura, estado final,
desviaciones iniciales/finales, backend, método de propagación, procedencia del
Jacobiano, normalización, referencias y advertencias metodológicas. No duplica
un campo `ldi`: para las mismas columnas normalizadas, `gali` ya almacena la
cantidad algebraicamente equivalente.

## Ejemplo y benchmark reproducibles

```bash
python examples/sali_gali_henon_heiles.py --duration 4
python benchmarks/bench_alignment_indices.py --repeats 7 \
  --output validation/outputs/benchmarks/alignment_indices_numpy_numba_20260803.json
```

En la ejecución local de referencia, el ejemplo Hénon--Heiles conservó una
deriva relativa de energía de `8.42e-16` en ambas rutas; las mayores
diferencias variacional--multiparticle fueron `1.97e-08` para SALI y
`7.34e-09` para GALI. Esto es una comparación numérica finita, no una
etiqueta dinámica.

En Windows 11/AMD64 con Python 3.14.3, NumPy 2.4.5 y Numba 0.65.1, el benchmark
caliente obtuvo razones medianas NumPy/Numba de `3.141x`, `2.122x` y
`2.018x` para 64, 512 y 4096 muestras. La primera llamada Numba, incluida
compilación, tomó `0.532 s`; la peor diferencia GALI fue `1.55e-15` y la
peor diferencia de log-volumen `3.55e-15`. El JSON retenido tiene SHA-256
`FF4AEB083FDDB7594857F70DC841413CDDDBD36533CE31F6936CBD29BF2B13D8`.
Los tiempos son evidencia de ingeniería dependiente del host y la carga.

## Validación independiente

El caso `validation/wolfram/cases/sali_gali_integer.wl` reconstruye las
fórmulas con 80 dígitos y sin importar HAFO. Compara Gram/Cauchy--Binet con
SVD y usa tres controles cerrados:

1. Una rotación ortogonal 3D, que conserva SALI y GALI y detecta un reinicio QR
   accidental.
2. El mapa hiperbólico diagonal
   \(\operatorname{diag}(2,1/2)\), con secuencias exactas
   \(2/\sqrt{16^n+1}\) y \(2\,4^n/(16^n+1)\).
3. El flujo \(A=\operatorname{diag}(1,0,-1)\), contrastado con
   \(V(t)=\exp(At)V(0)\) y fórmulas cerradas para SALI, GALI2 y GALI3.

Estas pruebas demuestran álgebra y propagación lineal finita. No demuestran
convergencia asintótica general, clasificación de una órbita no lineal,
atractividad, ocultedad ni validez fraccionaria.

El resumen promovido `sali_gali_integer_verified` aprobó 12/12 comprobaciones
Wolfram; la comparación de las seis fachadas públicas tuvo discrepancia global
máxima `1.7763568394002505e-15` y máxima de flujo
`8.881784197001252e-16`. Su SHA-256 es
`299CE6A9A0BB7A3C2CF920AE2F8F9C85A1A11D9244729D80302BDA4513ED6787`.
Un resultado fallido conservado en la ruta no verificada no se usa como
oráculo de promoción.

## Referencias primarias

- Skokos (2001), introducción de los índices de alineamiento,
  [DOI 10.1088/0305-4470/34/47/309](https://doi.org/10.1088/0305-4470/34/47/309).
- Skokos, Antonopoulos, Bountis y Vrahatis (2004), comportamiento de SALI,
  [DOI 10.1088/0305-4470/37/24/006](https://doi.org/10.1088/0305-4470/37/24/006).
- Skokos, Bountis y Antonopoulos (2007), definición y leyes de GALI,
  [DOI 10.1016/j.physd.2007.04.004](https://doi.org/10.1016/j.physd.2007.04.004).
- Manda, Hillebrand y Skokos (2025), método multiparticle y análisis de error,
  [DOI 10.1016/j.cnsns.2025.108635](https://doi.org/10.1016/j.cnsns.2025.108635).
- Rolim Sales, Leonel y Antonopoulos (2026), formulación SVD y tasas para mapas
  y flujos,
  [DOI 10.1016/j.chaos.2026.117884](https://doi.org/10.1016/j.chaos.2026.117884).
- Ma, Long y Zhu (2016), cautelas en sistemas disipativos,
  [DOI 10.1142/S0218127416501820](https://doi.org/10.1142/S0218127416501820).
