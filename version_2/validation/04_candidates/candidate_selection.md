# Criba de candidatos fraccionarios actuales

Se ejecutó la corrida `chua_fractional_nonsmooth_q09998_efork3_20260523_173051` para el Chua no suave con \(q=0.9998\). La
etapa DF utiliza cuadratura Python de corta duración (3.598 s);
la continuación y la clasificación dinámica que determinan cada preselección se
ejecutaron con `chua_frac_backend_lib.c` y el estadio corregido `K3 = a31*K1 + a32*K2`.

La criba inicial produjo ternas no triviales en EFORK con historial completo y
con ventana finita \(L_m=8\). El posdiagnóstico de retorno al periodo
dominante rechazó todos los regímenes acotados: ninguno satisface
`dominant_period_return_ratio >= 0.05`. Las tablas
`preselected_regimes_rejected.csv` y
`preselected_regimes_rejected_finite_window.csv` conservan los valores
rechazados; `selected_candidates*.json` no contiene candidatos promovidos.
No se utilizaron salidas históricas para estos valores.
