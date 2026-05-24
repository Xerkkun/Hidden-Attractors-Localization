# Análisis dinámico fraccionario actual y descarte

Las trayectorias de ambas preselecciones fueron calculadas dentro del tramo de
observación de su continuación causal EFORK-3 C para `chua_fractional_nonsmooth_q09998_efork3_20260523_173051`. El espectro
se obtiene como posprocesamiento ligero de esas trayectorias C.

Las gráficas son trazas delgadas. Para formalizar este diagnóstico se midió el
error normalizado de retorno tras un periodo de la frecuencia dominante. En
historial completo los 12 regímenes no triviales dan valores entre 0.00525 y
0.01206; con ventana finita, entre 0.00533 y 0.01079. Como todos quedan por
debajo del umbral de exclusión 0.05, se rechazan como candidatos caóticos.
Los resultados están en `periodicity_screen.csv` y
`finite_window_periodicity_screen.csv`.

Los exponentes de Lyapunov permanecen pendientes: el backend existente
reinicia por bloques y no se promueve como evidencia de una trayectoria que
transporta historia de continuación.
