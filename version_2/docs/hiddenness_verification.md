# Verificacion estricta de ocultedad operacional (Strict Operational Hiddenness Verification)

Este modulo establece un protocolo estricto para validar la condicion operacional de ocultedad en atractores caoticos de orden entero y fraccionario.

En la literatura cientifica, un atractor A se clasifica como oculto si su cuenca de atraccion B(A) no interseca ninguna vecindad de ningun punto de equilibrio del sistema. Si la cuenca interseca alguna vecindad local de equilibrio, el atractor es auto-excitado (self-excited).

---

## Limitacion de metodos analiticos y heuristicos

Metodos como el balance armonico, la funcion descriptiva o DF, el criterio de Nyquist, la continuacion, o la simulacion acotada de un solo seed no son suficientes para certificar un sistema como `hidden_verified`.

* **Funcion descriptiva / Nyquist:** aproximaciones lineales equivalentes locales que sirven para generar candidatos o semillas (`seed_found`). No demuestran la no interseccion global de cuencas.
* **Continuacion:** transporta una estructura oscilatoria al variar un parametro de control, pero no descarta contactos transitorios en la vecindad del equilibrio para el sistema final.
* **Diagnosticos de caos:** Lyapunov positivo, prueba 0-1, FFT, PSD o secciones de Poincare ayudan a caracterizar dinamica, pero no prueban la topologia de cuencas.

Por lo tanto, la biblioteca restringe las etiquetas fuertes de ocultedad a sistemas que aprueban el protocolo operacional de muestreo en vecindades bajo un contrato finito.

---

## Protocolo de muestreo de vecindades (Sphere Probes)

Para verificar numericamente la condicion de ocultedad, se realiza un barrido de trayectorias integradas desde esferas concentricas alrededor de todos los puntos de equilibrio.

### 1. Requisitos de cobertura de radios

El contrato puede exigir radios locales decrecientes, por ejemplo:

```text
1e-2, 1e-3, 1e-4, 1e-5
```

Cualquier omision de radios restringe el veredicto a compatibilidad bajo los radios probados, nunca a una prueba global.

### 2. Condicion de direccion unitaria uniforme

Para cada muestra en el radio epsilon, el punto inicial se genera como:

```text
x0 = X_i^* + epsilon v_j
```

donde `v_j` es una direccion unitaria. La biblioteca audita la norma de la direccion y falla si la muestra no respeta la tolerancia declarada.

### 3. Criterio local de evaluacion

Para cada par equilibrio-radio, se clasifica localmente como:

* **PASS:** cero trayectorias iniciadas en la esfera local alcanzan la vecindad del atractor objetivo.
* **FAIL:** al menos una trayectoria iniciada en el radio local probado interseca el atractor. La interpretacion depende del contrato radial: un contacto local es evidencia contra ocultedad bajo ese contrato; un contacto solo en radio extendido se reporta como auditoria de geometria de cuenca.
* **INCOMPLETE:** fallo numerico o muestreo incompleto.

---

## Vecindades locales versus auditorias esfericas extendidas

A contact detected on a sphere of large radius around an equilibrium is not, by itself, evidence that the attractor is self-excited. The operative hiddenness test concerns sufficiently small neighborhoods of all equilibria. Large-radius spherical probes are reported as extended basin-geometry audits.

La interpretacion uniforme por escala radial es:

* `local_neighborhood_contact_detected` o local self-excited contact: evidencia contra ocultedad bajo el contrato local probado.
* `extended_radius_contact_detected` o `macro_radius_contact_detected`: contacto macroscopico de auditoria extendida; no rechazo automatico de la ocultedad local.
* `hiddenness_supported_under_tested_local_neighborhoods` o `compatible_with_hiddenness_under_tested_radii`: cero contactos en las vecindades locales probadas, con alcance finito.
* `candidate_rejected_under_local_contract`: rechazo por contacto local o por otro bloqueo del contrato probado, no por contacto extendido aislado.

---

## Estados del contrato de ocultedad (HiddennessVerificationStatus)

1. **HIDDEN_VERIFIED:** protocolo de vecindades completado para todos los equilibrios y radios exigidos, con cero contactos y cero fallos numericos bajo el contrato.
2. **HIDDEN_COMPATIBLE:** no se detectaron contactos, pero el protocolo esta incompleto o limitado a un subconjunto de radios, muestras o equilibrios.
3. **SELF_EXCITED_CONTACT_DETECTED:** se detecto al menos un contacto entre vecindades locales declaradas de los equilibrios y el atractor. El atractor no es oculto bajo ese contrato local.
4. **NUMERICAL_FAILURE:** ocurrieron fallos de integracion durante el barrido y el contrato no los permite.
5. **CANDIDATE_NOT_AVAILABLE / SEED_NOT_AVAILABLE:** la simulacion del seed o la continuacion fallaron en localizar el atractor de referencia.

---

## Nota metodologica importante

La declaracion de ocultedad es estrictamente operacional. La ausencia de contactos numericos bajo tolerancias y radios especificados no constituye una prueba matematica global. Representa evidencia computacional bajo un contrato de discretizacion finito. Las dinamicas fraccionarias de memoria larga pueden requerir tiempos de integracion superiores a los limites computacionales del protocolo.
