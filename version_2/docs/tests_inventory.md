# Tests Inventory

This file lists all test files in the test suite, their classification, and planned actions.

| archivo | objetivo real | categoría propuesta | velocidad estimada | tipo | dependencias externas | escribe en outputs reales | usa simulaciones largas | acción | justificación |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tests/cli/test_continuation_order_cli.py | Verificar continuation order cli | cli | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/cli/test_seed_order_cli.py | Verificar seed order cli | cli | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/hygiene/test_tests_do_not_write_real_artifacts.py | Verificar tests do not write real artifacts | plotting | slow | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/scientific_contract/test_continuation_order_contract.py | Verificar continuation order contract | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/scientific_contract/test_seed_transfer_contract.py | Verificar seed transfer contract | scientific_contract | fast | contract | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_abm_fractional_diagnostics.py | Verificar abm fractional diagnostics | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_adm_wu2023.py | Verificar adm wu2023 | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_archive_independence.py | Verificar archive independence | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_attractor_only.py | Verificar attractor only | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_basin_smoke.py | Verificar basin smoke | integration | medium | integration | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_biased_chua_example_single_entrypoint.py | Verificar biased chua example single entrypoint | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_biased_figure_manifest.py | Verificar biased figure manifest | plotting | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_bibliographic_validation.py | Verificar bibliographic validation | literature_traceability | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_bifurcation_cli.py | Verificar bifurcation cli | cli | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_bifurcation_smoke.py | Verificar bifurcation smoke | integration | fast | integration | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_boundedness_diagnostics.py | Verificar boundedness diagnostics | unit | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_candidate_gate.py | Verificar candidate gate | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_centered_lure_workflow.py | Verificar centered lure workflow | integration | medium | integration | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_chua_arctan_wu2023_algebra.py | Verificar chua arctan wu2023 algebra | scientific_contract | fast | contract | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_chua_arctan_wu2023_seed_generation.py | Verificar chua arctan wu2023 seed generation | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_chua_arctan_wu2023_validation_contract.py | Verificar chua arctan wu2023 validation contract | validation_contract | fast | contract | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_classical_route_scope.py | Verificar classical route scope | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_cli_no_redundant_public_scripts.py | Verificar cli no redundant public scripts | cli | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_cli_overrides.py | Verificar cli overrides | cli | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_cli_smoke.py | Verificar cli smoke | cli | medium | unit | ninguna | no | sí | fusionar | Fusionar en test parametrizado de CLI. |
| tests/test_cloned_dynamics_discrepancy_diagnostics.py | Verificar cloned dynamics discrepancy diagnostics | unit | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_cloned_dynamics_fischer_published.py | Verificar cloned dynamics fischer published | slow | slow | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_cloned_dynamics_fischer_published_all_rows.py | Verificar cloned dynamics fischer published all rows | slow | slow | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_cloned_dynamics_published_structure.py | Verificar cloned dynamics published structure | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_cloned_dynamics_reproduction_limitations.py | Verificar cloned dynamics reproduction limitations | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_cloned_dynamics_synthetic.py | Verificar cloned dynamics synthetic | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_config_loader.py | Verificar config loader | unit | medium | unit | ninguna | sí | sí | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_continuation_cli.py | Verificar continuation cli | cli | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_continuation_memory_policy.py | Verificar continuation memory policy | unit | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_continuation_memory_validation.py | Verificar continuation memory validation | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_current_candidate_selection.py | Verificar current candidate selection | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_current_protocol_evidence_scope.py | Verificar current protocol evidence scope | unit | medium | unit | ninguna | sí | sí | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_danca_candidate_traceability.py | Verificar danca candidate traceability | unit | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_deferred_robustness_scope.py | Verificar deferred robustness scope | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_diagnostics_partial_scope.py | Verificar diagnostics partial scope | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_documentation_references.py | Verificar documentation references | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_dynamical_analysis.py | Verificar dynamical analysis | unit | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_efork_published_validation.py | Verificar efork published validation | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_f4_internal_lyapunov_validation.py | Verificar f4 internal lyapunov validation | unit | medium | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_f4_no_false_method_validation.py | Verificar f4 no false method validation | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_f5_diagnostics_contract.py | Verificar f5 diagnostics contract | unit | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_f5_global_runner.py | Verificar f5 global runner | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_f6_integrated_chaos_validator.py | Verificar f6 integrated chaos validator | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_f7_method_comparison.py | Verificar f7 method comparison | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_figure_export_contract.py | Verificar figure export contract | plotting | medium | unit | matplotlib | sí | sí | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_figure_manifest.py | Verificar figure manifest | plotting | medium | unit | matplotlib | sí | sí | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_final_freeze_audit_artifact.py | Verificar final freeze audit artifact | unit | medium | unit | ninguna | sí | sí | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_fractional_memory_validation.py | Verificar fractional memory validation | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_fractional_nonsmooth_algebra.py | Verificar fractional nonsmooth algebra | scientific_contract | fast | contract | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_fractional_variational_benchmarks.py | Verificar fractional variational benchmarks | slow | slow | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_freeze_audit_json_outputs.py | Verificar freeze audit json outputs | unit | medium | unit | ninguna | sí | sí | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_general_solver.py | Verificar general solver | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_hiddenness_contract.py | Verificar hiddenness contract | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_homogenization_rules.py | Verificar homogenization rules | unit | fast | unit | wolframscript | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_integer_lure_workflow.py | Verificar integer lure workflow | integration | medium | integration | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_integrator_crosscheck.py | Verificar integrator crosscheck | unit | medium | unit | ninguna | sí | sí | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_integrator_method_validation.py | Verificar integrator method validation | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_integrator_selector.py | Verificar integrator selector | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_integrator_workflow_compatibility.py | Verificar integrator workflow compatibility | integration | fast | integration | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_library_seed_and_workflow_api.py | Verificar library seed and workflow api | integration | fast | integration | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_lure_compatibility.py | Verificar lure compatibility | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_lure_seed_families.py | Verificar lure seed families | unit | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_lure_seed_filtering.py | Verificar lure seed filtering | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_lure_seeds_modal.py | Verificar lure seeds modal | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_lyapunov_api.py | Verificar lyapunov api | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_lyapunov_cli.py | Verificar lyapunov cli | cli | medium | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_lyapunov_fractional_variational.py | Verificar lyapunov fractional variational | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_lyapunov_integer_audit.py | Verificar lyapunov integer audit | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_lyapunov_promotion.py | Verificar lyapunov promotion | unit | medium | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_lyapunov_registry_f3.py | Verificar lyapunov registry f3 | unit | medium | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_memory_normalization.py | Verificar memory normalization | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_native_fractional_variational_backend.py | Verificar native fractional variational backend | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_no_absolute_local_paths.py | Verificar no absolute local paths | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_no_direct_savefig_outside_export.py | Verificar no direct savefig outside export | plotting | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_no_false_certification.py | Verificar no false certification | unit | medium | unit | wolframscript | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_no_false_chaos_certification_f3.py | Verificar no false chaos certification f3 | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_no_loose_figure_scripts.py | Verificar no loose figure scripts | plotting | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_no_plot_titles.py | Verificar no plot titles | plotting | fast | unit | matplotlib | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_nonsmooth_nonlinearities.py | Verificar nonsmooth nonlinearities | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_official_protocol.py | Verificar official protocol | unit | medium | unit | ninguna | sí | sí | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_official_wolfram_artifacts.py | Verificar official wolfram artifacts | unit | fast | unit | wolframscript | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_package_smoke.py | Verificar package smoke | integration | medium | integration | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_packaged_yaml_examples.py | Verificar packaged yaml examples | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_phase_F_closure.py | Verificar phase F closure | unit | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_plotting_style_contract.py | Verificar plotting style contract | plotting | fast | unit | matplotlib | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_poincare_diagnostics.py | Verificar poincare diagnostics | unit | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_poincare_method_validation.py | Verificar poincare method validation | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_published_case_reproduction.py | Verificar published case reproduction | unit | medium | unit | ninguna | sí | sí | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_published_continuation_comparison.py | Verificar published continuation comparison | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_published_validation_coverage.py | Verificar published validation coverage | unit | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_reference_data.py | Verificar reference data | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_regression_reference_values.py | Verificar regression reference values | unit | medium | regression | ninguna | sí | sí | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_report_figure_paths.py | Verificar report figure paths | plotting | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_reproducibility.py | Verificar reproducibility | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_reproducibility_metadata.py | Verificar reproducibility metadata | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_scientific_scope_docs.py | Verificar scientific scope docs | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_scientific_software_algebra.py | Verificar scientific software algebra | scientific_contract | fast | contract | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_scientific_software_hiddenness.py | Verificar scientific software hiddenness | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_scientific_software_integrators.py | Verificar scientific software integrators | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |
| tests/test_seed_cli.py | Verificar seed cli | cli | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_spectral_diagnostics.py | Verificar spectral diagnostics | unit | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_stage_flags.py | Verificar stage flags | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_symmetry_validator.py | Verificar symmetry validator | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_systems_and_legacy_facade.py | Verificar systems and legacy facade | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_test_suite_classification.py | Verificar suite classification | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_validation_contract.py | Verificar validation contract | validation_contract | fast | contract | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_validation_manifest.py | Verificar validation manifest | unit | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_validation_states.py | Verificar validation states | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_weyl_caputo_justification.py | Verificar weyl caputo justification | scientific_contract | fast | contract | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_wolfram_python_consistency.py | Verificar wolfram python consistency | unit | fast | unit | wolframscript | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_wolfram_validations.py | Verificar wolfram validations | unit | fast | unit | wolframscript | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_workflow_smoke.py | Verificar workflow smoke | integration | fast | integration | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_yaml_loading.py | Verificar yaml loading | unit | fast | unit | ninguna | no | no | conservar | Verifica contrato o comportamiento único. |
| tests/test_zero_one_diagnostics.py | Verificar zero one diagnostics | unit | fast | unit | ninguna | sí | no | refactorizar | Escribe en outputs reales; cambiar a tmp_path. |
| tests/test_zero_one_test.py | Verificar zero one test | unit | medium | unit | ninguna | no | sí | conservar | Verifica contrato o comportamiento único. |