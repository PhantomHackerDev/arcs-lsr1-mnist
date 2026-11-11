#!/bin/bash
# Follow-up grid: deep net, per-layer mu (Q1) then memory / Hessian momentum (Q2)
cd "$(dirname "$0")"
.venv/bin/python src/grid.py --suite layer --bs 64,256 --workers 8 --threads 2 --epochs 10
.venv/bin/python src/grid.py --suite memory --bs 256 --workers 8 --threads 2 --epochs 10
