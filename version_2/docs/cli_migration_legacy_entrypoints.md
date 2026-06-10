# Migración de Comandos CLI Legacy a Subcomandos Unificados

Este documento describe la migración de los antiguos comandos independientes a la nueva interfaz CLI unificada `hidden-attractors`.

Los comandos antiguos independientes han dejado de instalarse como ejecutables públicos en `pyproject.toml` para limpiar la interfaz de la biblioteca, pero toda su funcionalidad se mantiene accesible.

## Tabla de Migración

| Comando Antiguo (Legacy) | Reemplazo con Subcomando Unificado | Estado |
| :--- | :--- | :--- |
| `hidden-attractors-list-candidates` | `hidden-attractors inspect candidates` | Legacy / no público |
| `hidden-attractors-systems` | `hidden-attractors inspect systems` | Legacy / no público |
| `hidden-attractors-workflow-requirements` | `hidden-attractors inspect workflow-requirements` | Legacy / no público |
| `hidden-attractors-check-validation` | `hidden-attractors validate contract` | Legacy / no público |
| `hidden-attractors-protocol` | `hidden-attractors protocol ...` | Legacy / no público |
| `hidden-attractors-robustness-overlay` | `hidden-attractors robustness overlay` | Legacy / no público |
| `hidden-attractors-sphere-controls` | `hidden-attractors hiddenness sphere-controls` | Legacy / no público |
| `hidden-attractors-refined-basin` | `hidden-attractors basin refined` | Legacy / no público |
| `hidden-attractors-strict-target-refinement` | `hidden-attractors basin strict-target-refinement` / `hidden-attractors hiddenness strict-target-refinement` | Legacy / no público |
| `hidden-attractors-danca-abm-sphere-controls` | `hidden-attractors published danca-abm-sphere-controls` | Legacy / no público |
| `hidden-attractors-fractional-report-run` | `hidden-attractors report fractional-run` | Legacy / no público |
