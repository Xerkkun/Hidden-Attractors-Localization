# Diagnóstico de vecindades sobre referencias rechazadas

Para `chua_fractional_nonsmooth_q09998_efork3_20260523_173051` se sondearon esferas discretas alrededor de cada equilibrio
mediante integración C `chua_frac_backend_lib.c`. El ensayo se realizó por
separado bajo EFORK con historial completo y bajo EFORK con ventana
\(L_m=8\). Las referencias usadas inicialmente fueron después rechazadas por
el filtro de retorno casi periódico, por lo que estos sondeos no se promueven
como pruebas de ocultedad de un atractor caótico.

No hubo impactos que refinar; `strict_refinement_summary.csv` registra
explícitamente este caso con conteo cero. La búsqueda de nuevas referencias
caóticas, las cuencas completas y los diagramas de bifurcación permanecen
pendientes.
