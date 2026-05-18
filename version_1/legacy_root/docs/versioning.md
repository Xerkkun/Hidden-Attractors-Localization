# Versioning policy

Este proyecto queda separado en dos superficies de trabajo.

## Version 1

`version_1/` describe la etapa historica del repositorio: scripts raiz largos,
salidas ya usadas en reportes y corridas que dependen de nombres o rutas
existentes. No es el lugar para agregar ejemplos o analisis nuevos.

Se permite tocar V1 solo para:

- reproducir una corrida anterior;
- corregir un bug que impide ejecutar un resultado historico;
- dejar un wrapper de compatibilidad hacia una implementacion de V2.

## Version 2

`version_2/` es la version activa. A partir de esta separacion:

- ejemplos nuevos van en `version_2/examples/`;
- notas e indices de analisis nuevos van en `version_2/analysis/`;
- codigo reusable va en `hidden_attractors/analysis/`,
  `hidden_attractors/workflows/`, `hidden_attractors/plotting/`,
  `hidden_attractors/models/`, `hidden_attractors/native/` o
  `hidden_attractors/basins/`, segun corresponda;
- scripts raiz nuevos deben ser wrappers delgados y solo cuando haga falta
  conservar un nombre de comando.

## Migration rule

Cuando un script raiz empiece a duplicar logica:

1. extraer la funcion reusable a `hidden_attractors/`;
2. dejar el script raiz como CLI de compatibilidad;
3. agregar un ejemplo minimo en `version_2/examples/`;
4. registrar el analisis o contrato numerico en `version_2/analysis/`;
5. mantener salidas nuevas bajo `outputs/` con carpeta propia o timestamp.
