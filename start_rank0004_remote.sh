#!/bin/bash
set -euo pipefail
cd "/Users/Xerk/Desktop/Proyectos/Hidden Attractors Fractional Order/codigo_mac"
mkdir -p outputs/lure_biased_multiparam_q09998
nohup python3 lure_biased_multiparam_continuation.py \
  --config configs/lure_biased_multiparam_q09998.yaml \
  --post-continuation-only \
  --resume \
  --execute-early-filter \
  --execute-robustness \
  --survivor-id lure_biased_q_0p99980_rank_0004 \
  > outputs/lure_biased_multiparam_q09998/rank0004_realrepo_stdout.log \
  2> outputs/lure_biased_multiparam_q09998/rank0004_realrepo_stderr.log \
  < /dev/null &
echo $! > outputs/lure_biased_multiparam_q09998/rank0004_realrepo.pid