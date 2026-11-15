# Auditoría bibliográfica actual

El corpus tiene **141 entradas y 141 claves únicas**, sin claves citadas ausentes ni DOI duplicados en el BibTeX. Los AUX compilados coinciden exactamente con el recorrido de las dos ediciones: **97 citas distintas en la monografía y 114 en la guía**, con 97 comunes y 17 exclusivas de la guía.

Zotero contiene las **141 referencias locales**. Se importaron 92 fichas faltantes, sin fusionar ni eliminar registros previos. La biblioteca tiene 254 fichas regulares y la colección Tesis 176; 131 entradas del corpus pertenecen a esa colección y otras 10 ya existían en la biblioteca. Las notas hijas no se cuentan como referencias. Los duplicados previamente existentes en Zotero se conservan.

La comprobación estricta de identidad, año y autoría de las 92 importadas pasa **91/92**. En `scipy2020`, los 111 autores personales son correctos, pero falta el crédito corporativo **SciPy 1.0 Contributors** en Zotero. La referencia del PDF es válida: `and others` es una convención de BibTeX. Esta discrepancia no implica una ficha ausente ni un error del documento. No se repitió la importación.

La entrada previa `kuznetsov2017localization` presenta además una diferencia de manifestación: Zotero contiene un DOI de arXiv y la entrada local describe el artículo de IFAC sin DOI. Se conserva esa diferencia sin asignar el DOI del preprint al artículo.

## Archivos de consulta

- `bibliografia_completa_auditoria.csv`: las 141 claves, citas por edición, correspondencia con Zotero, diferencias y alcance real de lectura.
- `bibliografia_completa_resumen.json`: conteos, duplicados, hashes y reconciliación exacta con AUX.
- `mapa_claves_latex_zotero.csv`: clave LaTeX original y clave interna Zotero para cada referencia. Las claves del manuscrito se conservan; `note` fue enviado como nota bibliográfica, no como campo Extra.
- `verificacion_importacion_zotero.json`: cotejo de las 92 fichas nuevas y discrepancia de crédito corporativo.
- `discrepancias_actuales_zotero.csv`: las dos cautelas de metadatos vigentes.
- `errores_metadatos_corregidos.csv`: errores detectados y correcciones comprobadas en cinco entradas de la bibliografía local.
- `bibliografia_auditoria.csv` y `seleccion_bibliografica_verificada.json`: selección focalizada de 15 trabajos con fuentes y límites de lectura; 14 pertenecen al corpus. El preprint de Fazzino queda como candidato externo, sin uso como sustento.
- `recomendaciones_coasesor.csv`: prioridades científicas y criterios de cierre.

No se afirma lectura integral de las 141 obras. En este corpus se documenta lectura primaria selectiva de seis fuentes, lectura de resumen o metadatos primarios de ocho y cotejo de metadatos con Zotero de las otras 127. SciSpace ayuda a encontrar y comparar fuentes; sus resúmenes no sustituyen demostraciones.

## Repetir las comprobaciones

Con Zotero abierto y su API local habilitada, ejecutar desde esta carpeta:

```powershell
python auditar_bibliografia_completa.py
python verificar_importacion_zotero.py
```

Ambos scripts consultan Zotero en modo lectura y actualizan archivos de auditoría dentro del paquete donde residen. Las rutas se derivan de `__file__`; no dependen de una carpeta temporal ni escriben en la copia canónica al ejecutarse desde otra copia. Los metadatos de investigación necesarios se distribuyen aquí. Para reconciliar citas se requieren los AUX de `../../build/`; si faltan, el auditor declara esa limitación.

Los archivos `importar_*.bib` y `referencias_locales_faltantes_importar.bib` contienen fichas que **ya están importadas**. No deben volver a importarse: producirían duplicados. No se consultaron adjuntos ni texto completo de Zotero durante este cotejo.
