### Round 5: F-MNIST, the paper's CNN (10 epochs, mean ± sd over seeds)

| bs | y | method | test acc % (final) | best-epoch acc % | train loss | accept % | iterations with a μ at cap % |
|---|---|---|---|---|---|---|---|
| 64 |  | SGD+momentum (lr 0.1) | 83.06 ± 0.48 | 84.03 | 0.432 |  |  |
| 64 |  | Adam (lr 1e-3) | 90.03 ± 0.29 | 90.11 | 0.205 |  |  |
| 64 | same | ARCs-LSR1, full network, m=10 | 73.40 ± 1.62 | 78.12 | 0.713 | 92.8 | 0.0 |
| 64 | same | per-layer models, shared μ, m=10 | 76.39 ± 1.88 | 78.58 | 0.600 | 50.7 | 44.6 |
| 64 | same | per-layer μ_l + floor, m=10 | 75.67 ± 1.52 | 78.02 | 0.606 | 100.0 | 0.0 |
| 64 | same | per-layer models, shared μ, m=2 | 76.25 ± 2.36 | 79.75 | 0.633 | 93.1 | 0.0 |
| 64 | same | per-layer μ_l + floor, m=2 | 76.78 ± 3.48 | 81.02 | 0.690 | 100.0 | 0.0 |
| 64 | cross | full network, m=10 | 54.84 ± 6.96 | 61.26 | 1.208 | 53.1 | 14.7 |
| 64 | cross | per-layer models, shared μ, m=10 | 76.28 ± 5.05 | 76.28 | 0.603 | 9.1 | 84.5 |
| 64 | cross | per-layer μ_l + floor, m=10 | 70.72 ± 2.98 | 70.72 | 0.779 | 2.7 | 98.1 |
| 256 |  | SGD+momentum (lr 0.1) | 87.89 ± 0.73 | 88.24 | 0.259 |  |  |
| 256 |  | Adam (lr 1e-3) | 88.60 ± 0.12 | 88.70 | 0.269 |  |  |
| 256 | same | ARCs-LSR1, full network, m=10 | 81.54 ± 2.89 | 84.25 | 0.459 | 91.2 | 0.0 |
| 256 | same | per-layer models, shared μ, m=10 | 81.49 ± 1.45 | 83.75 | 0.485 | 63.7 | 28.7 |
| 256 | same | per-layer μ_l + floor, m=10 | 83.93 ± 0.79 | 85.19 | 0.390 | 100.0 | 0.0 |
| 256 | same | per-layer models, shared μ, m=2 | 83.34 ± 1.69 | 85.69 | 0.415 | 86.1 | 1.8 |
| 256 | same | per-layer μ_l + floor, m=2 | 85.58 ± 0.34 | 85.81 | 0.344 | 100.0 | 0.0 |
| 256 | cross | full network, m=10 | 67.60 ± 3.46 | 81.86 | 0.842 | 76.5 | 0.0 |
| 256 | cross | per-layer models, shared μ, m=10 | 68.26 ± 8.27 | 68.26 | 0.937 | 3.1 | 93.6 |
| 256 | cross | per-layer μ_l + floor, m=10 | 70.36 ± 2.55 | 70.36 | 0.803 | 5.8 | 96.2 |
