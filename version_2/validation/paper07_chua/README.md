# Evidencia cerrada de validación para paper07

Este directorio conserva únicamente dos registros numéricos cerrados. Cada
resultado está limitado al contrato declarado y no constituye una prueba
global de cuenca ni una certificación de caos.

## Chua fraccionario no suave corregido

El contrato local usa bolas interiores alrededor de los tres equilibrios con
radios `r <= 0.01`. En ese dominio se registraron:

- `7,200` sondas;
- `0` contactos con el conjunto objetivo;
- `0` fallos numéricos.

Los radios `0.03`, `0.1` y `0.3` forman una auditoría macroscópica separada:
`10,200` sondas adicionales y `37` contactos, todos en `r = 0.3`. La etiqueta
dinámica registrada es regular/periódica; por tanto, este caso no sostiene una
afirmación de caos.

Archivos canónicos:

- `evidence/nonsmooth_corrected/extended_first_contact_clean/extended_result.json`;
- `evidence/nonsmooth_corrected/extended_first_contact_clean/extended_probe_summary.csv`;
- `evidence/nonsmooth_corrected/extended_first_contact_clean/extended_numerical_contract.json`.

## Chua fraccionario arctan c590

El registro c590 conserva evidencia finita y limitada por radio:

- radios locales `r <= 0.3`: `8,400` sondas finitas y `0` contactos;
- radio macroscópico `r = 1.0`: `22 / 2,400` contactos;
- radio macroscópico `r = 2.0`: `588 / 2,700` contactos.

La conclusión pública se limita a compatibilidad con ocultedad bajo las
vecindades locales probadas. Los contactos macroscópicos describen geometría
extendida de cuenca y no cambian retroactivamente el contrato local. Este
registro tampoco establece caos.

Archivos canónicos:

- `../chua_fractional_arctan_c590/validation_summary.json`;
- `../chua_fractional_arctan_c590/summary_by_radius.csv`;
- `../chua_fractional_arctan_c590/summary_by_radius_equilibrium_status_contact.csv`.

## Integridad del paquete

El manifiesto SHA-256 contiene `43` artefactos cerrados en tres grupos:
resultados no suaves corregidos, filas de ocultedad c590 y trayectorias
representativas finitas.

Desde `version_2`, la integridad se verifica sin depender de archivos locales:

```bash
python validation/paper07_chua/scripts/sync_paper07_evidence.py --verify
```

El mapa compacto de alcance y rutas canónicas está en `manifest.json`.
