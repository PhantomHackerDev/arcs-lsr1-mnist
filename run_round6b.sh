#!/bin/bash
# wait for the initial modern_fmnist sweep (pid $1), then adaptive lr extension + seeds 1,2
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
.venv/bin/python src/sweep_round6.py --workers 14 --threads 1 --epochs 10
echo "round 6 finished"
