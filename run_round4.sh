#!/bin/bash
# round 4: flipped Hessian-momentum weights + rank cap (3 seeds), then secant-residual probe (1 seed, 3 epochs)
cd "$(dirname "$0")"
.venv/bin/python src/grid.py --suite flip --bs 256 --workers 12 --threads 1 --epochs 10 &
.venv/bin/python src/grid.py --suite secant --bs 256 --seeds 0 --workers 4 --threads 1 --epochs 3 &
wait
