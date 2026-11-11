### Round 6: modern baselines vs. ARCs-LSR1 (10 epochs, mean ± sd over 3 seeds)

Modern baselines: learning rate chosen by the lowest final training loss on seed 0 over a sweep that was extended until the best value was inside the range (results/round6_selection.json); decoupled weight decay 0.01. SGD and Adam use the paper's settings.

**MNIST, deep MLP, batch 256**

| method | lr | test acc % | final train loss | train loss epoch 1 | epoch 3 |
|---|---|---|---|---|---|
| ARCs-LSR1, full network, m=10 | – | 98.31 ± 0.08 | 0.0026 ± 0.0012 | 0.1571 | 0.0666 |
| ARCs-LSR1, per-layer μ_l (+ floor on F-MNIST), m=2 | – | 98.39 ± 0.13 | 0.0043 ± 0.0007 | 0.2334 | 0.1162 |
| AdamW (tuned lr) | 0.0003 | 97.92 ± 0.06 | 0.0163 ± 0.0017 | 0.2135 | 0.0922 |
| Muon (tuned lr) | 0.0017 | 98.45 ± 0.03 | 0.0001 ± 0.0001 | 0.2527 | 0.0300 |
| Shampoo, Adam-grafted (tuned lr) | 0.0003 | 98.47 ± 0.03 | 0.0005 ± 0.0000 | 0.1149 | 0.0274 |
| SophiaG (tuned lr) | 0.0001 | 97.88 ± 0.14 | 0.0139 ± 0.0007 | 0.2557 | 0.0789 |
| SGD+momentum (paper: lr 0.1) | 0.1 | 98.05 ± 0.14 | 0.0106 ± 0.0022 | 0.1059 | 0.0474 |
| Adam (paper: lr 1e-3) | 1e-3 | 98.14 ± 0.06 | 0.0111 ± 0.0017 | 0.1291 | 0.0475 |

**F-MNIST, CNN, batch 64**

| method | lr | test acc % | final train loss | train loss epoch 1 | epoch 3 |
|---|---|---|---|---|---|
| ARCs-LSR1, full network, m=10 | – | 73.40 ± 1.62 | 0.7128 ± 0.0502 | 0.6955 | 0.6903 |
| ARCs-LSR1, per-layer μ_l (+ floor on F-MNIST), m=2 | – | 76.78 ± 3.48 | 0.6901 ± 0.1180 | 0.5854 | 0.5359 |
| AdamW (tuned lr) | 0.003 | 89.55 ± 0.23 | 0.1977 ± 0.0089 | 0.3422 | 0.2676 |
| Muon (tuned lr) | 0.005 | 90.25 ± 0.14 | 0.1281 ± 0.0014 | 0.2964 | 0.2079 |
| Shampoo, Adam-grafted (tuned lr) | 0.001 | 90.06 ± 0.18 | 0.1370 ± 0.0112 | 0.3211 | 0.2249 |
| SophiaG (tuned lr) | 0.0003 | 89.97 ± 0.14 | 0.1827 ± 0.0037 | 0.3719 | 0.2737 |
| SGD+momentum (paper: lr 0.1) | 0.1 | 83.06 ± 0.48 | 0.4315 ± 0.0208 | 0.5389 | 0.4438 |
| Adam (paper: lr 1e-3) | 1e-3 | 90.03 ± 0.29 | 0.2049 ± 0.0134 | 0.4043 | 0.2942 |

**F-MNIST, CNN, batch 256**

| method | lr | test acc % | final train loss | train loss epoch 1 | epoch 3 |
|---|---|---|---|---|---|
| ARCs-LSR1, full network, m=10 | – | 81.54 ± 2.89 | 0.4589 ± 0.0775 | 0.4837 | 0.5005 |
| ARCs-LSR1, per-layer μ_l (+ floor on F-MNIST), m=2 | – | 85.58 ± 0.34 | 0.3440 ± 0.0132 | 0.5911 | 0.4270 |
| AdamW (tuned lr) | 0.003 | 89.79 ± 0.26 | 0.2102 ± 0.0141 | 0.4053 | 0.2997 |
| Muon (tuned lr) | 0.005 | 90.42 ± 0.12 | 0.1394 ± 0.0074 | 0.4126 | 0.2486 |
| Shampoo, Adam-grafted (tuned lr) | 0.003 | 89.82 ± 0.13 | 0.1458 ± 0.0054 | 0.3310 | 0.2344 |
| SophiaG (tuned lr) | 0.001 | 89.76 ± 0.33 | 0.1956 ± 0.0093 | 0.3987 | 0.2870 |
| SGD+momentum (paper: lr 0.1) | 0.1 | 87.89 ± 0.73 | 0.2587 ± 0.0278 | 0.4196 | 0.3327 |
| Adam (paper: lr 1e-3) | 1e-3 | 88.60 ± 0.12 | 0.2687 ± 0.0036 | 0.4957 | 0.3719 |

