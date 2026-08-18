# Cierre de funciones actuales

Este documento fija el criterio ejecutable de cierre de las funciones que ya
existían en HAFO. No incorpora métodos científicos nuevos.

Una implementación actual se considera cerrada cuando tiene contrato
matemático y numérico explícito, pruebas automatizadas, referencia documental,
una ruta pública experimental y un estado reproducible. La estabilidad de la
API continúa siendo `experimental`: `implemented` describe la madurez de la
ruta numérica, no una promesa de compatibilidad perpetua ni una prueba global
de caos u ocultamiento.

La matriz canónica es
`validation/software_audit/current_function_closure.json`. La prueba
`tests/test_current_function_closure.py` exige que:

- todos los métodos de `FRACTIONAL_METHODS` aparezcan exactamente una vez;
- toda ruta ejecutable esté marcada `implemented` y apunte a pruebas y
  documentación reales;
- las rutas planeadas y teóricas sigan sin ejecutarse;
- las capacidades de retardo, recurrencia, incertidumbre de cuencas y
  adaptadores de complejidad estén en la API pública experimental;
- cualquier reproducción Lyapunov con discrepancia publicada permanezca en
  cuarentena y requiera aceptación explícita para ejecutarse por el despachador
  común;
- RK4 sea la ruta entera canónica y el esquema retirado no reaparezca en las
  fuentes activas.

## Límites que no se promocionan

`fractional_difference_equations`, `cf_predictor_corrector`,
`abc_fast_soe_predictor_corrector`, `tempered_symbol_shift_cq`,
`variable_order_pece`, `distributed_order_quadrature`, órbitas periódicas y
pruebas por sustitutos siguen siendo trabajo futuro. Caputo--Fabrizio conserva
la definición FDE en `research_required`; sólo su operador muestral recurrente
está implementado.

La ruta DK2018 reproduce cuantitativamente Lorenz, pero no el tercer exponente
publicado de Rabinovich--Fabrikant. La ruta de dinámica clonada también conserva
discrepancias publicadas. Ambas se cierran como reproducciones negativas
auditables, no como validaciones positivas.

## Reproducción

Desde `version_2`:

```text
python -m pytest -q
python -m pytest -q -m wolfram
python -m pytest -q -m native
python -m mkdocs build --strict
```

Los diagnósticos se interpretan siempre bajo su contrato finito. Una
trayectoria, un espectro de Lyapunov, una FFT, una matriz de recurrencia o un
muestreo de cuenca no demuestran por sí solos ocultamiento ni caos global.
