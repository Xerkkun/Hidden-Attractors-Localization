# Version 1: legacy research scripts

Esta carpeta fija la referencia de la version historica del proyecto.
`legacy_root/` es una copia completa de la raiz antigua, con scripts, configs,
docs, outputs y pruebas. Se conserva para reproducir corridas previas sin
depender de los archivos que queden fuera de `version_1/` y `version_2/`.

Uso recomendado:

- reproducir corridas antiguas desde `version_1/legacy_root/`;
- consultar `MANIFEST.md` para saber que pertenece a la etapa historica;
- evitar agregar aqui ejemplos o analisis nuevos.

Si una modificacion nueva necesita reutilizar logica de un script historico,
esa logica debe extraerse a `hidden_attractors/` y documentarse en
`version_2/`.
