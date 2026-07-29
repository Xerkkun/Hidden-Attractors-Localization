# Verificacion operacional de ocultedad

Este modulo documenta el contrato numerico usado para evaluar ocultedad en
sistemas dinamicos de orden entero y fraccionario. La conclusion siempre queda
limitada a las vecindades, tiempos, clasificador y tolerancias declarados.

En la literatura cientifica, un atractor \(A\) se clasifica como oculto si su
cuenca de atraccion \(B(A)\) no interseca ninguna vecindad de ningun equilibrio.
Una interseccion detectada dentro de una vecindad local es evidencia contra la
ocultedad bajo el contrato probado.

## Lo que no demuestra ocultedad

El balance armonico, la funcion descriptiva, Nyquist, la continuacion y una
trayectoria acotada pueden localizar o caracterizar una solucion, pero no
demuestran por si solos la no interseccion de cuencas. Del mismo modo, un
exponente de Lyapunov estimado, la prueba 0--1, FFT, PSD o una seccion de
Poincare caracterizan dinamica; no sustituyen el muestreo de vecindades de
todos los equilibrios.

## Geometrias de muestreo

Cada resultado debe identificar sin ambiguedad la geometria usada:

- **Bolas interiores:** muestrean puntos cuya distancia al equilibrio no
  excede el radio declarado. Son la geometria adecuada para una afirmacion
  operacional sobre una vecindad abierta muestreada.
- **Superficies esfericas:** muestrean puntos a distancia fija del equilibrio.
  Solo informan sobre esa frontera y no equivalen a llenar la bola interior.
- **Cascarones esfericos:** muestrean una banda radial declarada entre dos
  radios. Describen esa banda, pero no cubren automaticamente el interior que
  queda fuera del cascaron.

Las tres geometrias son admisibles como evidencia finita si se etiquetan con su
alcance real. Una superficie o un cascaron no se promueven como prueba de una
bola interior. Los radios y conteos concretos pertenecen al registro de
validacion que produjo el resultado, no a esta descripcion publica del metodo.

## Contrato por sonda

Para cada sonda se deben conservar:

- el equilibrio, la geometria y el dominio radial;
- el integrador, paso, horizonte, transitorio y politica de fallos;
- el tiempo inicial declarado;
- la condicion inicial o, para Caputo, la funcion de historia declarada;
- el clasificador del atractor objetivo, su metrica y su umbral;
- el destino numerico y la procedencia del software.

En un problema de Caputo, cada sonda es un problema de valor inicial con memoria.
Una inicializacion nueva debe comenzar en su propio tiempo inicial con la
historia definida por el contrato, por ejemplo una historia constante igual al
estado de la sonda antes de ese tiempo. No se puede heredar silenciosamente la
historia de otra trayectoria, de otro parametro o de otra sonda. Si el contrato
transporta una historia registrada, esa historia y su intervalo deben quedar
identificados de forma explicita.

## Clasificacion fijada antes del barrido

La representacion del atractor de referencia, la metrica de coincidencia, el
umbral, las tolerancias de divergencia y las reglas para estados indeterminados
se fijan antes de iniciar el barrido. No se recalibran despues de observar
contactos. Una calibracion separada puede justificar esos valores, pero no debe
usar las sondas que despues se evaluan como evidencia.

Para cada bloque equilibrio--dominio radial:

- **sin contacto:** ninguna sonda completada alcanza el objetivo bajo el
  clasificador predeclarado;
- **contacto:** al menos una sonda alcanza el objetivo bajo ese clasificador;
- **incompleto:** faltan sondas o hay fallos que el contrato no permite.

## Regla causal opcional de primer contacto

Un barrido radial ordenado puede declarar de antemano la regla
`complete_first_contact_radius`: al aparecer uno o mas contactos, se completan
todas las sondas planeadas para ese mismo radio y para todos los equilibrios.
Solo despues se detiene el barrido y se excluyen los radios mayores.

Esta regla conserva el denominador completo del primer radio con contacto y
evita seleccionar resultados a posteriori. No convierte los radios omitidos en
radios probados, y el informe debe distinguir una terminacion causal de un
protocolo radial completo.

## Vecindades locales versus auditorias extendidas

Local neighborhoods versus extended spherical audits are different numerical
questions. A contact detected on a sphere of large radius around an equilibrium
is not, by itself, evidence that the attractor is self-excited. The operative
hiddenness test concerns sufficiently small neighborhoods of all equilibria.
Large-radius spherical probes are reported as extended basin-geometry audits.

Los dominios locales y macroscopicos se almacenan y se interpretan por separado:

- un contacto local es evidencia contra ocultedad bajo el contrato local;
- un contacto solo en un dominio macro describe geometria de cuenca extendida
  y no invalida por si mismo la afirmacion local;
- cero contactos solo admite una etiqueta condicionada a las vecindades
  efectivamente muestreadas.

## Estados conservadores

- `hidden_under_tested_neighborhoods`: se completo el contrato local declarado
  para todos los equilibrios, sin contactos ni fallos prohibidos.
- `compatible_with_hiddenness`: no se observaron contactos, pero el protocolo
  o su cobertura no permiten la etiqueta anterior.
- `self_excited`: se detecto contacto dentro del dominio local declarado.
- `inconclusive`: faltan datos o existen fallos numericos que bloquean una
  conclusion.

La ausencia de contactos en un muestreo finito no constituye una prueba
matematica global. Es evidencia computacional reproducible bajo el contrato
registrado.
