#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH=src
for checkpoint in experiments/results/checkpoints/*.pt; do
    [ -f "$checkpoint" ] || continue
    "${PYTHON:-python}" experiments/tem_spatial_probe.py --checkpoint "$checkpoint" \
        --out "${checkpoint%.pt}-spatial"
done
