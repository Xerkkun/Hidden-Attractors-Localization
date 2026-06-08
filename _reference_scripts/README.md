# _reference_scripts/ — Scripts de Desarrollo

Esta carpeta contiene los scripts temporales originales que se usaron durante
el proceso de investigación. Se preservan aquí como referencia histórica.

**NO son parte de la librería** — el proceso reproducible y documentado
está en `version_2/examples/chua_nonsmooth_biased_hidden_attractor/`.

---

## Inventario

### Scripts de `version_2/` (raíz del paquete)

| Script | Descripción | Equivalente en el ejemplo |
|---|---|---|
| `search_saturation_biased_candidates_corrected.py` | **Script núcleo** — DF sesgada corregida, homotopía afín, clasificación | `step2_biased_df_search.py` |
| `search_saturation_biased_candidates.py` | Versión anterior sin corrección de convención de signo | Obsoleto |
| `run_hiddenness_biased_candidates.py` | Verificación de ocultedad con contrato | `step3_hiddenness_verification.py` |
| `hiddenness_cand0_all_extended.py` | **Test extendido** multiprocessing para cand0 (todos los equilibrios, hasta r=2.0) | `step4_extended_hiddenness.py` |
| `hiddenness_cand0_e0_extended.py` | Test focalizado en E₀ | Redundante con `all_extended` |
| `hiddenness_cand0_ep_em_extended.py` | Test focalizado en E₊/E₋ | Redundante con `all_extended` |
| `hiddenness_cand0_intensive.py` | Versión intensiva (misma lógica) | Redundante con `all_extended` |
| `hiddenness_single_candidate.py` | Prueba para un candidato individual | Fusionado en `step3_hiddenness_verification.py` |
| `run_hiddenness_parallel.py` | Lanzador paralelo (versión temprana) | Fusionado en `step4_extended_hiddenness.py` |
| `recover_hiddenness_results.py` | Recuperador de resultados guardados | `step5_summarize_and_plot.py` |
| `00_make_chua_fractional_memory_experiment_matrix.py` | Matriz de experimentos de memoria | Experimento auxiliar independiente |

### Scripts de la raíz del proyecto (`root_*`)

| Script | Descripción | Equivalente |
|---|---|---|
| `root_exploration_template.py` | Plantilla genérica DF+continuación (arctan y saturación) | Referencia de estructura |
| `root_search_saturation_candidates.py` | Búsqueda centrada original | `step1_centered_reference.py` |
| `root_generate_all_plots_and_summary.py` | Generador de figuras global para todos los candidatos | `step5_summarize_and_plot.py` |
| `root_plot_chaotic_candidates.py` | Plots de candidatos caóticos | Fusionado en `step5_summarize_and_plot.py` |

---

## Cronología del Proceso

1. `root_search_saturation_candidates.py` — Primera exploración centrada; motiva la necesidad de la DF sesgada
2. `search_saturation_biased_candidates.py` — Primera implementación sesgada (convención de signo incorrecta)
3. `search_saturation_biased_candidates_corrected.py` — **Versión corregida** con auditoría de signo
4. `hiddenness_single_candidate.py` → `run_hiddenness_biased_candidates.py` — Verificación iterativa
5. `hiddenness_cand0_all_extended.py` — Test masivo para el candidato confirmado
