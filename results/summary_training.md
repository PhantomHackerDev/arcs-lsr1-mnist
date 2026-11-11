### Training response vs. the weight w on B_new (deep net, batch 256, per-layer μ_l, same-batch, mean over 3 seeds)

Training loss is on the full 60k training set. The per-batch decrease is f_B(x_k) − f_B(x_k+1) on the batch the step was computed on, as % of f_B(x_k), averaged over the iterations of that epoch (rejected steps count as 0). Time per epoch uses the idle-machine benchmark (results/overhead.md).

| method | train loss ep 1 | ep 3 | ep 10 | epochs to 0.05 | to 0.02 | to 0.01 | per-batch decrease ep 1 / 5 / 10 (%) | accept % | est. s / epoch |
|---|---|---|---|---|---|---|---|---|---|
| m=2, w = 0.1 | 0.2748 | 0.0744 | 0.0023 | 4.0 | 6.0 | 7.2 | 22.2 / 75.2 / 79.3 | 100 | 10.3 |
| m=2, w = 0.5 | 0.1901 | 0.1051 | 0.0027 | 4.6 | 5.9 | 7.6 | 25.5 / 75.9 / 81.7 | 100 | 10.3 |
| m=2, w = 0.7 | 0.2307 | 0.0772 | 0.0026 | 4.3 | 5.7 | 8.2 | 23.8 / 75.0 / 81.0 | 100 | 10.3 |
| m=2, w = 0.9 | 0.2769 | 0.0963 | 0.0031 | 4.6 | 6.3 | 8.0 | 20.6 / 73.4 / 82.3 | 100 | 10.3 |
| m=2, w = 1 (plain L-SR1) | 0.2334 | 0.1162 | 0.0043 | 4.4 | 6.6 | 7.6 | 20.6 / 72.9 / 81.7 | 100 | 7.5 |
| m=10, plain L-SR1 | 0.1903 | 0.0703 | 0.0032 | 4.2 | 6.6 | 8.0 | 36.6 / 76.0 / 81.9 | 100 | 13.1 |
| SGD+momentum (lr 0.1) | 0.1059 | 0.0474 | 0.0106 | 2.9 | 6.2 | 8.9 | – | – | 0.8 |
| Adam (lr 1e-3) | 0.1291 | 0.0475 | 0.0111 | 2.9 | 7.9 | – | – | – | 1.0 |
