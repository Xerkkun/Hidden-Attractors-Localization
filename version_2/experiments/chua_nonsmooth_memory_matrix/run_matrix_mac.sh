#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION2_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MATRIX_ROOT="${MATRIX_ROOT:-$VERSION2_ROOT/outputs/chua_nonsmooth_fractional_memory_matrix}"
WORKERS="${WORKERS:-1}"

cd "$VERSION2_ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python3 experiments/chua_nonsmooth_memory_matrix/run_shared_cache_tasks.py --tasks "$MATRIX_ROOT/tasks/shared_cache_tasks.csv" --workers "$WORKERS"
python3 experiments/chua_nonsmooth_memory_matrix/run_continuation_tasks.py --tasks "$MATRIX_ROOT/tasks/continuation_tasks.csv" --workers "$WORKERS"
python3 experiments/chua_nonsmooth_memory_matrix/run_hiddenness_tasks.py --tasks "$MATRIX_ROOT/tasks/hiddenness_tasks.csv" --workers "$WORKERS"
python3 figure_scripts/chua_nonsmooth_memory_matrix_run_figure_tasks.py --tasks "$MATRIX_ROOT/tasks/figure_tasks.csv" --workers "$WORKERS"
python3 experiments/chua_nonsmooth_memory_matrix/aggregate_results.py --root "$MATRIX_ROOT"
