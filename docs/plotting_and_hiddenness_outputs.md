# Plotting and Hiddenness Output Visualizations

This document explains the premium scientific visualizations, data exports, and configuration rules implemented in the `Hidden-Attractors-Localization` library.

---

## 1. Homogeneous Visual and Structural Rules

To maintain high academic readability, all figures are rendered at high resolution (`300 DPI`) and saved as separate projection files. 

### Equilibria Suppression Rule
- **Phase Space Trajectory Plots (`attractor` type)**: Do **not** render equilibrium points. This includes final attractor plots and candidate seed plots.
- **Time Series Plots (`timeseries` type)**: Do **not** render equilibrium points.
- **Verification Plots**: Equilibria points are **mandatory** and are only displayed in:
  1. **Control Sphere Plots** (Pruebas de esferas / neighborhood control)
  2. **Basin of Attraction Slices** (Cuencas de atracción)
  3. **Matignon Plots** (Plano complejo de estabilidad fraccionaria)

---

## 2. Generated Plot Artifacts

When `plot_enabled: true`, the workflow automatically generates the following files inside the run's timestamped `outputs/<system_id>/<timestamp>/figures/` folder:

### A. Phase Space Attractors (Separate Projections)
For the final trajectory (`final_attractor`) and each candidate seed (`seed_candidate_00`):
- `figures/final_attractor_3d.png` / `figures/seed_candidate_00_attractor_3d.png`: A premium 3D phase space representation.
- `figures/final_attractor_xy.png` / `figures/seed_candidate_00_xy.png`: 2D projection on the $x-y$ plane.
- `figures/final_attractor_xz.png` / `figures/seed_candidate_00_xz.png`: 2D projection on the $x-z$ plane.
- `figures/final_attractor_yz.png` / `figures/seed_candidate_00_yz.png`: 2D projection on the $y-z$ plane.

*Configurable properties:*
```yaml
attractor_plots:
  enabled: true
  include_equilibria: false     # Default is false, suppressing markers in attractors
  use_tail_after_burn: true    # Excludes initial transient states
  line_width: 0.7              # Linewidth of attractor trace
  point_size: 0.0              # Scatter points size (0.0 for lines only)
```

### B. State Variable Time Series
For the final trajectory and candidates, individual state components and combined variables are saved as plots and CSV databases:
- `figures/final_timeseries_x.png` / `figures/seed_candidate_00_timeseries_x.png`
- `figures/final_timeseries_y.png` / `figures/seed_candidate_00_timeseries_y.png`
- `figures/final_timeseries_z.png` / `figures/seed_candidate_00_timeseries_z.png`
- `figures/final_timeseries_xyz.png` / `figures/seed_candidate_00_timeseries_xyz.png`
- `final_timeseries.csv` / `seed_candidate_00_timeseries.csv`

*Configurable properties:*
```yaml
plot_timeseries: true
timeseries_use_tail_after_burn: false
timeseries_max_points: 20000
```

### C. Matignon Fractional Stability Plane
A dedicated complex plane representation of eigenvalues and fractional stability boundaries:
- `figures/matignon_equilibria.png`

**Features shown:**
1. The complex plane containing eigenvalue markers for each equilibrium ($E_0$, $E_+$, $E_-$).
2. Transparent shading separating the complex plane: **Light Green** background for stable regions ($|\arg(\lambda)| > q\pi/2$) and **Light Red** sector for unstable regions ($|\arg(\lambda)| \le q\pi/2$).
3. Red dashed border rays representing the exact fractional order boundary: $|\arg(\lambda)| = q\pi/2$.
4. A detail textbox summarizing stability metrics:
   $$\alpha_{min} = \min_{i} |\arg(\lambda_i)|$$
   $$\text{instability\_measure} = q - \frac{2 \alpha_{min}}{\pi}$$
   $$\text{stable} = \text{True / False}$$

*Configurable properties:*
```yaml
plot_matignon: true
```

---

## 3. Early Stopping Configurations

Early stopping cuts off simulations as soon as a trajectory is classified, saving up to 90% of integration runtime.

### A. Divergence Early Stopping
If a trajectory is unbounded, it will stop before integrating up to `t_final`.
```yaml
early_stop:
  enabled: true
  divergence_enabled: true
  divergence_norm: 80.0
  divergence_consecutive_steps: 5
  divergence_growth_factor: 1.25
```
- **Rule 1**: If $\|X(t)\| > 80.0$ for 5 consecutive steps, abort and report `status = "diverged_early"`.
- **Rule 2**: If $\|X_k\| > 1.25 \times \|X_{k-1}\|$ for 5 consecutive steps, abort early.

### B. Equilibrium Convergence Early Stopping
If a trajectory settles on an equilibrium point, it will stop.
```yaml
early_stop:
  equilibrium_enabled: true
  equilibrium_tol: 1e-3
  equilibrium_derivative_tol: 1e-4
  equilibrium_consecutive_steps: 200
  equilibrium_min_time: 5.0
```
- **Rule**: If $\|X(t) - E_i\| < 1\times 10^{-3}$ and $\|f(X(t))\| < 1\times 10^{-4}$ for 200 consecutive steps after $t \ge 5.0$, abort and report `status = "converged_equilibrium_early"`.

---

## 4. Decoupled Simulation Limits

Simulation times are configured separately for different phases, keeping quick check sweeps short while allowing long attractor simulations:
```yaml
# Decoupled times configuration
final_simulation:
  t_final: 500.0
  t_burn: 120.0

sphere_tests:
  enabled: false
  t_final: 80.0
  t_burn: 20.0
  early_stop_enabled: true

basin:
  enabled: false
  t_final: 80.0
  t_burn: 20.0
  early_stop_enabled: true
```
