# Getting Started

## 1. Import the model

```python
from hidden_attractors import chua_piecewise_parameters
from hidden_attractors.models import equilibria_piecewise, rhs_piecewise
```

## 2. Inspect equilibria

```python
import numpy as np

params = chua_piecewise_parameters()
for name, point in equilibria_piecewise(params).items():
    print(name, point, np.linalg.norm(rhs_piecewise(point, params)))
```

Equivalent command:

```bash
python examples/quickstart_equilibria.py
```

## 3. Load reference candidates

```python
from hidden_attractors import load_final_candidate_records

records = load_final_candidate_records()
for record in records:
    print(record.candidate_id, record.route, record.q)
```

Equivalent command:

```bash
python examples/list_final_candidates.py
```

## 4. Run workflow help

Long workflows expose command-line help without launching simulations:

```bash
python tools/cli/robustness_overlay_c_trajectories.py --help
python tools/cli/lure_top3_sphere_robustness.py --help
python tools/cli/refine_project_basin_classification.py --help
```

After installation, console entry points are also available:

```bash
hidden-attractors-robustness-overlay --help
hidden-attractors-sphere-controls --help
hidden-attractors-refined-basin --help
```
