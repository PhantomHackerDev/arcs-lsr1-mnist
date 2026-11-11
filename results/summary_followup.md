### Q1: one μ vs. μ_l per layer (deep net, 10 epochs, mean ± sd over seeds)

| bs | y | method | test acc % | train loss | accept % | fallback % | final μ_l (layers 1/2/3/4, median) | max μ | weight moved ‖Δw‖/‖w₀‖ (layers 1/2/3/4) | time s |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 |  | SGD+momentum (lr 0.1) | 67.72 ± 49.91 | 0.832 |  |  |  |  | 7.4 / 8.1 / 14 / 41 | 275 |
| 64 |  | Adam (lr 1e-3) | 98.11 ± 0.06 | 0.013 |  |  |  |  | 3.4 / 2.8 / 1.8 / 1.2 | 376 |
| 64 | same | full network, one μ | 98.02 ± 0.08 | 0.012 | 87.2 | 0.0 | 2 | 1.3e+02 | 2.2 / 1.8 / 1.4 / 3.4 | 3021 |
| 64 | same | per-layer models, shared μ | 98.20 ± 0.25 | 0.011 | 76.5 | 0.0 | 1e+12 / 1e+12 / 1e+12 / 1e+12 | 1e+12 | – | 3130 |
| 64 | same | per-layer models, μ_l per layer | 97.88 ± 0.14 | 0.020 | 99.6 | 10.7 | 2.6e+02 / 8 / 1e+12 / 1e+12 | 1e+12 | – | 3410 |
| 64 | same | μ_l per layer + floor μ_l ≥ μ_g (round 3) | 98.15 ± 0.16 | 0.009 | 99.9 | 7.5 | 16 / 16 / 16 / 16 | 8.4e+06 | 1.8 / 1.9 / 1.7 / 3.9 | 4627 |
| 64 | same | μ_l per layer, ρ_l on next batch (round 3) | 76.82 ± 1.76 | 0.812 | 80.0 | 2.2 | 1e+12 / 1e+12 / 1e+12 / 1e+12 | 1e+12 | 0.18 / 0.23 / 0.21 / 0.51 | 3938 |
| 64 | cross | full network, one μ | 77.56 ± 12.59 | 0.820 | 71.4 | 0.0 | 64 | 1.7e+10 | 6.5 / 6.2 / 6.5 / 11 | 3308 |
| 64 | cross | per-layer models, shared μ | 94.11 ± 2.19 | 0.227 | 20.0 | 0.0 | 1e+12 / 1e+12 / 1e+12 / 1e+12 | 1e+12 | 1.8 / 1.9 / 2 / 4 | 1514 |
| 64 | cross | per-layer models, μ_l per layer | 41.32 ± 12.16 | 1.745 | 42.2 | 0.8 | 1e+12 / 1e+12 / 1e+12 / 1e+12 | 1e+12 | 64 / 5.7 / 2.2 / 1.8 | 1732 |
| 64 | cross | μ_l per layer + floor μ_l ≥ μ_g (round 3) | 94.08 ± 0.88 | 0.219 | 30.1 | 0.4 | 1e+12 / 1e+12 / 1e+12 / 1e+12 | 1e+12 | 2.3 / 2.1 / 2.5 / 2.9 | 2927 |
| 64 | cross | μ_l per layer, ρ_l on next batch (round 3) | 81.52 ± 2.72 | 0.633 | 5.9 | 0.0 | 1e+12 / 1e+12 / 1e+12 / 1e+12 | 1e+12 | 0.43 / 0.45 / 0.35 / 0.47 | 2368 |
| 256 |  | SGD+momentum (lr 0.1) | 98.05 ± 0.14 | 0.011 |  |  |  |  | 1.4 / 1.3 / 1.2 / 4 | 152 |
| 256 |  | Adam (lr 1e-3) | 98.14 ± 0.06 | 0.011 |  |  |  |  | 1.7 / 1.4 / 0.8 / 0.63 | 116 |
| 256 | same | full network, one μ | 98.31 ± 0.08 | 0.003 | 94.9 | 0.0 | 4 | 16 | 1.3 / 1 / 0.88 / 2.8 | 2262 |
| 256 | same | per-layer models, shared μ | 98.34 ± 0.05 | 0.003 | 94.3 | 0.0 | 8 / 8 / 8 / 8 | 32 | 1.2 / 1.2 / 1.2 / 3.4 | 1763 |
| 256 | same | per-layer models, μ_l per layer | 98.35 ± 0.02 | 0.003 | 100.0 | 6.4 | 64 / 8 / 2 / 8 | 2.6e+02 | 1.1 / 1.2 / 1.3 / 3.2 | 853 |
| 256 | same | μ_l per layer + floor μ_l ≥ μ_g (round 3) | 98.29 ± 0.05 | 0.003 | 100.0 | 5.4 | 16 / 16 / 16 / 16 | 64 | 1.1 / 1 / 1.1 / 3.1 | 1367 |
| 256 | same | μ_l per layer, ρ_l on next batch (round 3) | 84.93 ± 0.81 | 0.541 | 79.8 | 3.4 | 1e+12 / 1e+12 / 1e+12 / 1e+12 | 1e+12 | 0.21 / 0.23 / 0.31 / 0.56 | 1147 |
| 256 | cross | full network, one μ | 90.38 ± 1.88 | 0.323 | 81.0 | 0.0 | 1 | 4.1e+03 | 5.7 / 5.9 / 5.6 / 7.7 | 666 |
| 256 | cross | per-layer models, shared μ | 90.34 ± 11.58 | 0.310 | 42.8 | 0.0 | 4 / 4 / 4 / 4 | 2.1e+09 | 1.8 / 2.1 / 2.1 / 3.6 | 511 |
| 256 | cross | per-layer models, μ_l per layer | 58.22 ± 9.90 | 1.381 | 92.8 | 3.8 | 1e+12 / 1e+12 / 8.2e+03 / 1e+12 | 1e+12 | 44 / 42 / 1.5 / 0.86 | 1233 |
| 256 | cross | μ_l per layer + floor μ_l ≥ μ_g (round 3) | 89.50 ± 2.85 | 0.400 | 37.2 | 1.0 | 1e+12 / 1e+12 / 1e+12 / 1e+12 | 1e+12 | 3 / 3.3 / 3.5 / 0.79 | 677 |
| 256 | cross | μ_l per layer, ρ_l on next batch (round 3) | 89.19 ± 1.51 | 0.402 | 42.5 | 0.4 | 1e+12 / 1e+12 / 1e+12 / 1e+12 | 1e+12 | 0.51 / 0.45 / 0.55 / 0.32 | 858 |

### Q2: history size vs. Hessian momentum (deep net, bs 256, per-layer μ_l, same-batch y)

B̃ = w_old·B̃ + w_new·B_new. Rounds 2 used w_new = 0.5 / 0.1; round 4 flips it to w_new = 0.7 / 0.9 (Aditya's suggestion) and also tries rank cap m.

| history m | weight on B_new | rank cap | n-vectors stored per layer | test acc % | train loss | time s |
|---|---|---|---|---|---|---|
| 1 | – (plain L-SR1) | – | 2 | 98.19 ± 0.09 | 0.003 | 275 |
| 1 | 0.9 | 1 | 3 | 98.18 ± 0.16 | 0.006 | 838 |
| 1 | 0.7 | 1 | 3 | 98.19 ± 0.04 | 0.005 | 850 |
| 2 | – (plain L-SR1) | – | 4 | 98.39 ± 0.13 | 0.004 | 499 |
| 1 | 0.9 | 2 | 4 | 98.05 ± 0.04 | 0.005 | 864 |
| 1 | 0.7 | 2 | 4 | 98.14 ± 0.08 | 0.003 | 908 |
| 1 | 0.5 | 2 | 4 | 98.11 ± 0.08 | 0.006 | 402 |
| 1 | 0.1 | 2 | 4 | 98.23 ± 0.13 | 0.005 | 556 |
| 2 | 0.9 | 2 | 6 | 98.35 ± 0.07 | 0.004 | 1222 |
| 2 | 0.7 | 2 | 6 | 98.32 ± 0.15 | 0.004 | 1224 |
| 2 | 0.9 | 4 | 8 | 98.34 ± 0.09 | 0.003 | 1288 |
| 2 | 0.7 | 4 | 8 | 98.38 ± 0.14 | 0.003 | 1282 |
| 2 | 0.5 | 4 | 8 | 98.31 ± 0.09 | 0.003 | 1066 |
| 2 | 0.1 | 4 | 8 | 98.25 ± 0.14 | 0.002 | 1313 |
| 5 | – (plain L-SR1) | – | 10 | 98.43 ± 0.01 | 0.003 | 1253 |
| 10 | – (plain L-SR1) | – | 20 | 98.35 ± 0.02 | 0.003 | 853 |
| Adam (lr 1e-3) | | | 0–2 | 98.14 ± 0.06 | 0.011 | 116 |
| SGD+momentum (lr 0.1) | | | 0–2 | 98.05 ± 0.14 | 0.011 | 152 |

Wall-clock times (time s) are not comparable between rows: up to 8 runs shared 16 cores with varying load. Only order-of-magnitude differences (first-order vs. ARCs) are meaningful.

### Secant residual vs. weight on B_new (m = 2, rank cap 4, seed 0, 3 epochs, batch 256)

‖B̃ s_k − y_k‖ / ‖y_k‖ for the newest pair, averaged over layers. With weight w on B_new, B̃ s_k − y_k = (1 − w)(B̃_old s_k − y_k), so the residual should shrink linearly as w → 1.

| weight on B_new | median residual | mean residual | test acc % after 3 epochs |
|---|---|---|---|
| 0.1 | 0.607 | 0.61 | 96.59 |
| 0.3 | 0.44 | 0.444 | 96.32 |
| 0.5 | 0.304 | 0.306 | 97.60 |
| 0.7 | 0.184 | 0.184 | 96.84 |
| 0.9 | 0.0625 | 0.0624 | 96.46 |
| 1.0 (plain L-SR1) | 3.59e-14 | 4.07e-14 | 96.24 |
