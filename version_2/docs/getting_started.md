# Getting Started

## 1. Import the model

```python
from hidden_attractors import chua_nonsmooth_parameters
from hidden_attractors.models import equilibria_nonsmooth, rhs_nonsmooth
```

## 2. Inspect equilibria

```python
import numpy as np

params = chua_nonsmooth_parameters()
for name, point in equilibria_nonsmooth(params).items():
    print(name, point, np.linalg.norm(rhs_nonsmooth(point, params)))
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

## 4. Run Workflow Help

The official protocol exposes one command family. These commands validate and
write stage summaries without changing the methodology:

```bash
hidden-attractors-protocol generate-seeds --help
hidden-attractors-protocol soft-precheck --help
hidden-attractors-protocol continue --help
hidden-attractors-protocol filter-survivors --help
hidden-attractors-protocol build-reference --help
hidden-attractors-protocol robustness --help
hidden-attractors-protocol hiddenness --help
hidden-attractors-protocol diagnostics --help
```

Compatibility adapters remain available while their numerical engines are
migrated behind the protocol interface:

```bash
hidden-attractors-robustness-overlay --help
hidden-attractors-sphere-controls --help
hidden-attractors-refined-basin --help
```

`hidden-attractors-sphere-controls` now samples inside equilibrium-centered
balls. Its installed name is retained for compatibility, not as a
surface-only hiddenness method.
