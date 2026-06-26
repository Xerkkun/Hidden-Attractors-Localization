# Published validation data extraction v1

Este archivo concentra datos extraídos de los artículos adjuntos para configurar benchmarks y validaciones de la librería. Los valores marcados como faltantes no deben inventarse.


## kuznetsov2017_chua_integer_df

**Referencia:** N. V. Kuznetsov, O. A. Kuznetsova, G. A. Leonov, T. N. Mokaev, N. V. Stankevich (2017), *Hidden attractors localization in Chua circuit via the describing function method*.


### kuznetsov2017_case_18_hidden_chaotic

- Parámetros: `{'alpha': 8.4562, 'beta': 12.0732, 'gamma': 0.0052, 'm0': -0.1768, 'm1': -1.1468}`

- DF/semilla: `{'omega0': 2.0392, 'k': 0.2098, 'a0': 5.8576}`; seed+: `[5.8576, 0.3694, -8.3686]`

- Estado reportado: two symmetric hidden chaotic attractors localized by direct use of DF initial data; authors state they skip multistep small-parameter continuation for this case.


### kuznetsov2017_case_21_hidden_chaotic_branch

- Parámetros: `{'alpha': 8.4, 'beta': 12.0, 'gamma': -0.005, 'm0': -1.2, 'm1': -0.05}`

- DF/semilla: `{'omega0': 2.026, 'k': -0.889, 'a0': 1.5187}`; seed+: `[1.5187, 0.0926, -2.1682]`

- Estado reportado: two symmetric hidden chaotic attractors


### kuznetsov2017_case_21_hidden_periodic_branch

- Parámetros: `{'alpha': 8.4, 'beta': 12.0, 'gamma': -0.005, 'm0': -1.2, 'm1': -0.05}`

- DF/semilla: `{'omega0': 3.2396, 'k': -0.1244, 'a0': 11.7546}`; seed+: `[11.7546, 9.7044, -16.7367]`

- Estado reportado: hidden periodic attractor / stable limit cycle coexisting with hidden chaotic attractors


## danca2017_fractional_hidden_attractors

**Referencia:** Marius-F. Danca (2017), *Hidden chaotic attractors in fractional-order systems*.


### generalized_lorenz_fractional

- Parámetros: `{'r': 6.8, 'a': -0.5, 'sigma': 3.4, 'q': 0.995}`

- Equilibrios: `{'X0': [0, 0, 0], 'X12': '±(3.476,1.807,6.280)'}`

- Eigenspectra: `{'X0': [2.5576, -1.0, -7.5576], 'X1': ['-5.9570', '-0.0215 ± 3.6026i']}`

- Test vecindades: `{'unstable_equilibrium': 'X0', 'radius_delta': 0.3, 'reported_trajectories': 50}`


### rabinovich_fabrikant_fractional

- Parámetros: `{'a': 0.1, 'b': 0.2876, 'q': 0.998}`

- Equilibrios: `{'X0': [0, 0, 0], 'X12': '(∓1.1600, ±0.2479, 0.1223)', 'X34': '(∓0.0850, ±3.3827, 0.9953)'}`

- Eigenspectra: `{'X0': ['-0.5752', '0.1 ± i'], 'X1': ['-0.2562', '-0.0595 ± 1.4731i'], 'X3': ['0.1981', '-0.2866 ± 4.7743i']}`

- Test vecindades: `{'unstable_equilibria': ['X0', 'X3', 'X4'], 'radius_delta': 0.1}`


### chua_fractional_saturation

- Parámetros: `{'m0': -0.1768, 'm1': -1.1468, 'alpha': 8.4562, 'beta': 12.0732, 'gamma': 0.0052, 'q': 0.9998}`

- Equilibrios: `{'X0': [0, 0, 0], 'X12': '(±6.5883, ±0.0029, ∓6.5855)'}`

- Eigenspectra: `{'X0': ['-7.9587', '-0.0038 ± 3.2494i'], 'X12': ['2.2193', '-0.9916 ± 2.4068i']}`

- Test vecindades: `{'unstable_equilibria': ['X1', 'X2'], 'radius_delta': 0.01, 'reported_trajectories': 200}`


## wu2023_chua_fractional_arctan

**Referencia:** Xianming Wu, Longxiang Fu, Shaobo He, Zhao Yao, Huihai Wang, Jiayu Han (2023), *Hidden attractors in a new fractional-order Chua system with arctan nonlinearity and its DSP implementation*.

- Sistema: `['D^q x = alpha*(y-x) - alpha*f(x)', 'D^q y = x-y+z', 'D^q z = -beta*y - gamma*z', 'f(x)=m*x+(n-m)*arctan(x)']`

- Parámetros: `{'alpha': 8.4562, 'beta': 12.0732, 'gamma_or_r': 0.0052, 'm': 0.4, 'n': -1.1585}`

- Equilibrios: `{'P1': [-0.573, -0.0002468, 0.573], 'P2': [0.573, 0.0002468, -0.573]}`

- Simulación: `{'q': 0.99, 'N': 10000, 'h': 0.01, 'method_reported': 'ADM for numerical solution; DSP uses ADM and Romberg integration for arctan', 'initial_conditions_hidden': {'x0_plus': [13.8, 0.7093, -19.8768], 'x0_minus': [-13.8, -0.7093, 19.8768]}, 'phase_figure': 'Fig. 9', 'basin_figure': 'Fig. 10'}`

- Barridos: `{'q_sweep': {'range': [0.8, 1.0], 'step': 0.0008, 'minimum_order_for_chaos': 0.9028, 'ICs': [[13, 0.7, -19], [-13, -0.7, 19]]}, 'alpha_sweep': {'beta': 12.0732, 'gamma': 0.0052, 'm': 0.4, 'n': -1.1585, 'q': 0.99, 'range': [9.2, 8.7], 'step': 0.002}, 'beta_sweep': {'alpha': 8.4562, 'gamma': 0.0052, 'm': 0.4, 'n': -1.1585, 'q': 0.99, 'range': [11.5, 12.5], 'step': 0.004}, 'complexity_map': {'alpha_range': [8.2, 8.7], 'alpha_step': 0.005, 'beta_range': [11.5, 12.5], 'beta_step': 0.01, 'q': 0.99}}`


## danca_kuznetsov2018_lyapunov_fo

**Referencia:** Marius-F. Danca, Nikolay Kuznetsov (2018), *Matlab Code for Lyapunov Exponents of Fractional-Order Systems*.


### DK2018_RF_q0999

- Sistema: Rabinovich-Fabrikant; parámetros: `{'a': 0.1, 'p_or_bifurcation_parameter': 0.98, 'q': 0.999}`

- Comando LE: `{'ne': 3, 't_start': 0, 'h_norm': 0.02, 't_end': 300, 'x_start': [0.1, 0.1, 0.1], 'h': 0.005, 'q': 0.999, 'out': 1000}`

- LE esperado: `[0.0749, 0.0018, -2.085]`; nivel: `quantitative`


### DK2018_Lorenz_q0985

- Sistema: Lorenz; parámetros: `{'sigma': 10, 'beta': 2.6666666666666665, 'p': 200, 'q': 0.985}`

- Comando LE: `{'ne': 3, 't_start': 0, 'h_norm': 5, 't_end': 500, 'x_start': [0.1, 0.1, 0.1], 'h': 0.001, 'q': 0.985, 'out': 10}`

- LE esperado: `[-0.0026, -0.087, -1.6225]`; nivel: `quantitative`


### DK2018_4D_nonsmooth_q098

- Sistema: nonsmooth_4D; parámetros: `{'a': 1, 'b': 0.5, 'q': 0.98}`

- Comando LE: `{'ne': 4, 't_start': 0, 'h_norm': 0.02, 't_end': 300, 'x_start': [0.1, 0.1, 0.1, 0.1], 'h': 0.005, 'q': 0.98, 'out': 1000}`

- LE esperado: `[0.1262, 0.0846, 0.0778, -1.5244]`; nivel: `qualitative_or_experimental_only`


## fischer2020_cloned_dynamics

**Referencia:** C. Fischer, K. L. A. Zourmba, A. Mohamadou (2020), *Lyapunov exponents spectrum estimation of fractional order nonlinear systems using Cloned Dynamics*.


### jerk_system

- Ecuaciones: `['D^alpha1 x = y', 'D^alpha2 y = z', 'D^alpha3 z = -a*z - f(y) - x', 'f(y)=Ic*(exp(y/(n*VT))-1)']`

- Parámetros: `{'a': 0.5, 'Ic': '10*10^-9 A', 'VT': 0.026, 'n': 2}`; CI: `[0.1, 0.1, 0.1]`

- Tabla LCE/0-1 Jerk: `[{'type': 'Comm', 'orders': [1.0, 1.0, 1.0], 'LE': [0.1899, 0.0413, -0.4246], 'K01': 0.9866}, {'type': 'Comm', 'orders': [0.9, 0.9, 0.9], 'LE': [-0.0042, -0.0267, -0.4656], 'K01': 0.0758}, {'type': 'Comm', 'orders': [0.8, 0.8, 0.8], 'LE': [-0.0047, -0.2864, -0.3634], 'K01': 0.0151}, {'type': 'Comm', 'orders': [0.7, 0.7, 0.7], 'LE': [-0.0527, -0.0528, -0.2763], 'K01': -0.1425}, {'type': 'Incomm', 'orders': [0.9, 1.0, 1.0], 'LE': [0.1744, 0.0246, -0.46], 'K01': 0.9928}, {'type': 'Incomm', 'orders': [0.8, 1.0, 1.0], 'LE': [0.081, -0.0162, -0.4017], 'K01': 0.3786}, {'type': 'Incomm', 'orders': [0.7, 1.0, 1.0], 'LE': [-0.0395, -0.0888, -0.2442], 'K01': 0.1543}, {'type': 'Incomm', 'orders': [0.6, 1.0, 1.0], 'LE': [0.0179, -0.0971, -0.2816], 'K01': 0.1329}]`


### financial_system

- Ecuaciones: `['D^alpha1 x = z+(y-a)*x', 'D^alpha2 y = 1-b*y-|x|', 'D^alpha3 z = -x-c*z']`

- Parámetros: `{'a': 1.0, 'b': 0.15, 'c': 1.0}`; CI: `[1.0, 1.0, 1.0]`

- Tabla LCE/0-1 Financial: `[{'type': 'Comm', 'orders': [1.0, 1.0, 1.0], 'LE': [0.0891, 0.0058, -0.4348], 'K01': 0.9984}, {'type': 'Comm', 'orders': [0.9, 0.9, 0.9], 'LE': [0.1133, -0.003, -0.337], 'K01': 0.9974}, {'type': 'Comm', 'orders': [0.8, 0.8, 0.8], 'LE': [-0.0407, -0.0406, -0.2648], 'K01': 0.1325}, {'type': 'Comm', 'orders': [0.7, 0.7, 0.7], 'LE': [-0.2125, -0.4539, -0.4539], 'K01': 0.2491}, {'type': 'Incomm', 'orders': [0.9, 1.0, 1.0], 'LE': [0.1445, 0.0012, -0.4419], 'K01': 0.998}, {'type': 'Incomm', 'orders': [0.8, 1.0, 1.0], 'LE': [0.1235, -0.0026, -0.3412], 'K01': 0.9975}, {'type': 'Incomm', 'orders': [0.7, 1.0, 1.0], 'LE': [0.0935, -0.0476, -0.29], 'K01': 0.9986}, {'type': 'Incomm', 'orders': [0.6, 1.0, 1.0], 'LE': [-0.0678, -0.068, -0.3308], 'K01': 0.1037}]`


### four_wing_system

- Ecuaciones: `['D^alpha1 x=-x+y*z', 'D^alpha2 y=y-x*z', 'D^alpha3 z=b-c*z+x*y']`

- Parámetros: `{'b': 0.53, 'c': 3.0}`; CI: `[0.1, 0.1, 0.1]`

- Tabla LCE/0-1 Four Wing: `[{'type': 'Comm', 'orders': [1.0, 1.0, 1.0], 'LE': [0.3358, 0.0133, -1.2608], 'K01': 0.998}, {'type': 'Comm', 'orders': [0.9, 0.9, 0.9], 'LE': [0.2684, -0.0259, -0.9699], 'K01': 0.9982}, {'type': 'Comm', 'orders': [0.8, 0.8, 0.8], 'LE': [-0.1543, -0.1531, -0.801], 'K01': 0.1352}, {'type': 'Comm', 'orders': [0.7, 0.7, 0.7], 'LE': [-0.4825, -0.4818, -0.6942], 'K01': 0.1752}, {'type': 'Incomm', 'orders': [0.9, 1.0, 1.0], 'LE': [0.3623, -0.0501, -1.0984], 'K01': 0.9983}, {'type': 'Incomm', 'orders': [0.8, 1.0, 1.0], 'LE': [0.346, -0.1048, -0.9196], 'K01': 0.9978}, {'type': 'Incomm', 'orders': [0.7, 1.0, 1.0], 'LE': [0.3598, -0.0345, -0.916], 'K01': 0.9976}, {'type': 'Incomm', 'orders': [0.6, 1.0, 1.0], 'LE': [0.3371, -0.0628, -0.9076], 'K01': 0.9969}]`

