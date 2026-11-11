#!/bin/bash
# round 6: modern baselines. Waits for the F-MNIST grid (pid in $1), then
# phase 1 = lr sweep with seed 0, phase 2 = seeds 1,2 at the lr with the lowest training loss.
cd "$(dirname "$0")"
while kill -0 "$1" 2>/dev/null; do sleep 30; done
G=".venv/bin/python src/grid.py --workers 14 --threads 1 --epochs 10"
$G --suite modern_mnist --bs 256 --seeds 0
$G --suite modern_fmnist --bs 64,256 --seeds 0
$G --suite modern_mnist --bs 256 --seeds 1,2 --only "$(.venv/bin/python src/select_lr.py modern_mnist 256)"
$G --suite modern_fmnist --bs 64 --seeds 1,2 --only "$(.venv/bin/python src/select_lr.py modern_fmnist 64)"
$G --suite modern_fmnist --bs 256 --seeds 1,2 --only "$(.venv/bin/python src/select_lr.py modern_fmnist 256)"
echo "round 6 finished"
