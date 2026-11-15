### Round 9: larger batches, F-MNIST CNN

ARCs-LSR1: full network, same-batch y, m = 10, η₁ = 0.1, 10 epochs (each step costs two gradients, so 10 epochs ≈ 20 full-gradient evaluations). Baselines: 20 epochs, so they can be compared both at the same epoch (10) and at the same compute (20 full-gradient evaluations). All selection is by the seed-0 final training loss; the selected configurations have seeds 0-2.

#### Batch 1024 (58 steps per epoch)

ARCs-LSR1 grid, seed 0, final training loss after 10 epochs:

| μ_min | η₂ = 0.75, pairs | η₂ = 0.75, B = I | η₂ = 0.5, pairs | η₂ = 0.5, B = I | η₂ = 0.25, pairs | η₂ = 0.25, B = I |
|---|---|---|---|---|---|---|
| none | 0.3730 | 0.6105 | 0.3448 | 0.6028 | 0.4003 | 0.6220 |
| 1 | 0.3214 | 0.6071 | 0.2881 | 0.6204 | 0.3255 | 0.6434 |
| 4 | 0.3050 | 0.6147 | 0.2973 | 0.6089 | 0.2951 | 0.6220 |
| 16 | 0.3264 | 0.6430 | 0.3131 | 0.6460 | 0.3264 | 0.6835 |
| 64 | 0.3413 | 0.6773 | 0.3431 | 0.6875 | 0.3461 | 0.7387 |
| 256 | 0.3948 | 0.7008 | 0.3948 | 0.7035 | 0.3948 | 0.7394 |

Selected configurations (mean ± sd over seeds; compute in full-gradient evaluations):

| method | setting | train loss, epoch 10 | test acc %, epoch 10 | train loss at 20 grads | test acc % at 20 grads | accepted % | μ above floor % |
|---|---|---|---|---|---|---|---|
| ARCs-LSR1 | η₂ = 0.5, μ_min = 1 | 0.328 ± 0.047 (n=3) | 86.84 | 0.328 ± 0.047 (n=3) | 86.84 | 76 | 39 |
| ARCs, no pairs (B = I) | η₂ = 0.5, μ_min = none | 0.588 ± 0.046 (n=3) | 77.19 | 0.588 ± 0.046 (n=3) | 77.19 | 83 | 100 |
| SGD+momentum | lr 0.1 | 0.276 ± 0.028 (n=3) | 88.18 | 0.230 ± 0.025 (n=3) | 88.71 | – | – |
| AdamW | lr 0.01 | 0.224 ± 0.011 (n=3) | 89.26 | 0.173 ± 0.007 (n=3) | 89.72 | – | – |
| Muon | lr 0.015 | 0.138 ± 0.004 (n=3) | 90.36 | 0.072 ± 0.003 (n=3) | 89.70 | – | – |
| Shampoo | lr 0.01 / 0.003 | 0.164 ± 0.016 (n=3) | 89.61 | 0.126 ± 0.014 (n=3) | 89.69 | – | – |
| SophiaG | lr 0.001 | 0.233 ± 0.011 (n=3) | 88.98 | 0.156 ± 0.007 (n=3) | 90.06 | – | – |
| Adam (paper, lr 1e-3) | lr 1e-3 | 0.349 ± 0.003 (n=3) | 86.51 | 0.285 ± 0.008 (n=3) | 88.23 | – | – |

Baseline lr: chosen at epoch 10 / at epoch 20 when they differ; '(edge)' = best at the edge of the sweep after the extensions.

#### Batch 2048 (29 steps per epoch)

ARCs-LSR1 grid, seed 0, final training loss after 10 epochs:

| μ_min | η₂ = 0.75, pairs | η₂ = 0.75, B = I | η₂ = 0.5, pairs | η₂ = 0.5, B = I | η₂ = 0.25, pairs | η₂ = 0.25, B = I |
|---|---|---|---|---|---|---|
| none | 0.3932 | 0.6782 | 0.3952 | 0.6884 | 0.3817 | 0.7160 |
| 1 | 0.3512 | 0.6867 | 0.3748 | 0.6935 | 0.3857 | 0.7222 |
| 4 | 0.4141 | 0.6836 | 0.3731 | 0.6982 | 0.3699 | 0.7209 |
| 16 | 0.3793 | 0.7445 | 0.3985 | 0.7577 | 0.3683 | 0.7800 |
| 64 | 0.4186 | 0.7597 | 0.4403 | 0.7720 | 0.4205 | 0.8032 |
| 256 | 0.4798 | 0.7840 | 0.4689 | 0.7931 | 0.4698 | 0.8268 |

Selected configurations (mean ± sd over seeds; compute in full-gradient evaluations):

| method | setting | train loss, epoch 10 | test acc %, epoch 10 | train loss at 20 grads | test acc % at 20 grads | accepted % | μ above floor % |
|---|---|---|---|---|---|---|---|
| ARCs-LSR1 | η₂ = 0.75, μ_min = 1 | 0.344 ± 0.008 (n=3) | 86.57 | 0.344 ± 0.008 (n=3) | 86.57 | 77 | 96 |
| ARCs, no pairs (B = I) | η₂ = 0.75, μ_min = none | 0.695 ± 0.079 (n=3) | 73.17 | 0.695 ± 0.079 (n=3) | 73.17 | 91 | 100 |
| SGD+momentum | lr 0.1 | 0.337 ± 0.048 (n=3) | 86.42 | 0.273 ± 0.032 (n=3) | 88.28 | – | – |
| AdamW | lr 0.01 | 0.259 ± 0.018 (n=3) | 88.71 | 0.200 ± 0.026 (n=3) | 89.43 | – | – |
| Muon | lr 0.045 | 0.136 ± 0.010 (n=3) | 89.60 | 0.078 ± 0.011 (n=3) | 89.05 | – | – |
| Shampoo | lr 0.01 | 0.189 ± 0.024 (n=3) | 89.81 | 0.128 ± 0.022 (n=3) | 89.47 | – | – |
| SophiaG | lr 0.003 | 0.261 ± 0.004 (n=3) | 88.52 | 0.193 ± 0.004 (n=3) | 89.21 | – | – |
| Adam (paper, lr 1e-3) | lr 1e-3 | 0.405 ± 0.006 (n=3) | 84.59 | 0.331 ± 0.009 (n=3) | 87.13 | – | – |

Baseline lr: chosen at epoch 10 / at epoch 20 when they differ; '(edge)' = best at the edge of the sweep after the extensions.

#### Batch 4096 (14 steps per epoch)

ARCs-LSR1 grid, seed 0, final training loss after 10 epochs:

| μ_min | η₂ = 0.75, pairs | η₂ = 0.75, B = I | η₂ = 0.5, pairs | η₂ = 0.5, B = I | η₂ = 0.25, pairs | η₂ = 0.25, B = I |
|---|---|---|---|---|---|---|
| none | 0.4291 | 0.7999 | 0.4157 | 0.8054 | 0.4246 | 0.8469 |
| 1 | 0.4064 | 0.7848 | 0.4061 | 0.8101 | 0.4284 | 0.8270 |
| 4 | 0.4482 | 0.7802 | 0.4532 | 0.7922 | 0.4476 | 0.8021 |
| 16 | 0.4637 | 0.8228 | 0.4788 | 0.8218 | 0.4635 | 0.8734 |
| 64 | 0.5000 | 0.8442 | 0.5060 | 0.8517 | 0.5028 | 0.9023 |
| 256 | 0.5747 | 0.8696 | 0.5694 | 0.8730 | 0.5714 | 0.9128 |

Selected configurations (mean ± sd over seeds; compute in full-gradient evaluations):

| method | setting | train loss, epoch 10 | test acc %, epoch 10 | train loss at 20 grads | test acc % at 20 grads | accepted % | μ above floor % |
|---|---|---|---|---|---|---|---|
| ARCs-LSR1 | η₂ = 0.5, μ_min = 1 | 0.405 ± 0.009 (n=3) | 84.33 | 0.405 ± 0.009 (n=3) | 84.33 | 66 | 87 |
| ARCs, no pairs (B = I) | η₂ = 0.75, μ_min = 4 | 0.775 ± 0.073 (n=3) | 70.78 | 0.775 ± 0.073 (n=3) | 70.78 | 89 | 96 |
| SGD+momentum | lr 0.1 | 0.427 ± 0.015 (n=3) | 83.35 | 0.324 ± 0.011 (n=3) | 86.91 | – | – |
| AdamW | lr 0.01 | 0.316 ± 0.003 (n=3) | 87.26 | 0.248 ± 0.007 (n=3) | 88.86 | – | – |
| Muon | lr 0.045 | 0.189 ± 0.004 (n=3) | 89.52 | 0.085 ± 0.008 (n=3) | 89.30 | – | – |
| Shampoo | lr 0.01 | 0.248 ± 0.006 (n=3) | 89.10 | 0.166 ± 0.013 (n=3) | 89.82 | – | – |
| SophiaG | lr 0.003 | 0.338 ± 0.002 (n=3) | 86.49 | 0.230 ± 0.005 (n=3) | 88.87 | – | – |
| Adam (paper, lr 1e-3) | lr 1e-3 | 0.484 ± 0.003 (n=3) | 81.28 | 0.394 ± 0.007 (n=3) | 85.07 | – | – |

Baseline lr: chosen at epoch 10 / at epoch 20 when they differ; '(edge)' = best at the edge of the sweep after the extensions.

#### L-SR1 memory m (selected η₂ and μ_min per batch size)

'safe' is the paper's γ rule (γ = λ̂_min / 2, or γ = 1 when λ̂_min ≤ 0); 'safe_keep' keeps the previous γ when λ̂_min ≤ 0. Training loss after 10 epochs (≈ 20 full-gradient evaluations); the optimizer's own arithmetic is given relative to the model's forward/backward FLOPs.

| batch | m | γ rule | seeds | train loss | test acc % | accepted % | median γ | λ̂_min ≤ 0 (% of steps) | optimizer / model FLOPs |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 5 | safe | 1 | 0.419 | 84.26 | 62 | 1.41 | 0.0 | 0.05% |
| 1024 | 10 | safe | 3 | 0.328 ± 0.047 (n=3) | 86.84 | 76 | 0.202 | 0.0 | 0.10% |
| 1024 | 20 | safe | 1 | 0.260 | 88.80 | 97 | 0.116 | 0.0 | 0.20% |
| 1024 | 50 | safe | 1 | 0.228 | 89.52 | 99 | 0.0799 | 0.0 | 0.49% |
| 1024 | 10 | safe_keep | 3 | 0.328 ± 0.047 (n=3) | 86.84 | 76 | 0.202 | 0.0 | 0.10% |
| 1024 | 20 | safe_keep | 3 | 0.271 ± 0.017 (n=3) | 88.42 | 95 | 0.112 | 0.0 | 0.20% |
| 1024 | 50 | safe_keep | 3 | 0.238 ± 0.009 (n=3) | 89.32 | 99 | 0.0734 | 0.0 | 0.49% |
| 1024 | 100 | safe_keep | 3 | 0.225 ± 0.010 (n=3) | 89.84 | 99 | 0.0478 | 0.0 | 0.99% |
| 2048 | 5 | safe | 1 | 0.553 | 78.21 | 66 | 4.19 | 2.4 | 0.02% |
| 2048 | 10 | safe | 3 | 0.344 ± 0.008 (n=3) | 86.57 | 77 | 0.297 | 1.4 | 0.05% |
| 2048 | 20 | safe | 1 | 0.328 | 86.95 | 83 | 0.0932 | 17.6 | 0.10% |
| 2048 | 50 | safe | 1 | 0.408 | 84.42 | 78 | 1 | 99.7 | 0.25% |
| 2048 | 10 | safe_keep | 3 | 0.352 ± 0.021 (n=3) | 86.23 | 77 | 0.295 | 1.5 | 0.05% |
| 2048 | 20 | safe_keep | 3 | 0.322 ± 0.023 (n=3) | 87.26 | 87 | 0.0877 | 2.7 | 0.10% |
| 2048 | 50 | safe_keep | 3 | 0.259 ± 0.007 (n=3) | 88.93 | 97 | 0.0319 | 6.1 | 0.25% |
| 2048 | 100 | safe_keep | 3 | 0.305 ± 0.102 (n=3) | 87.33 | 92 | 0.0196 | 33.2 | 0.49% |
| 4096 | 5 | safe | 1 | 0.543 | 78.35 | 61 | 1.77 | 0.0 | 0.01% |
| 4096 | 10 | safe | 3 | 0.405 ± 0.009 (n=3) | 84.33 | 66 | 0.167 | 0.0 | 0.02% |
| 4096 | 20 | safe | 1 | 0.321 | 87.02 | 77 | 0.0244 | 0.0 | 0.05% |
| 4096 | 50 | safe | 1 | 0.285 | 88.04 | 94 | 0.00576 | 0.0 | 0.12% |
| 4096 | 10 | safe_keep | 3 | 0.405 ± 0.009 (n=3) | 84.33 | 66 | 0.167 | 0.0 | 0.02% |
| 4096 | 20 | safe_keep | 3 | 0.334 ± 0.012 (n=3) | 86.83 | 79 | 0.0287 | 0.0 | 0.05% |
| 4096 | 50 | safe_keep | 3 | 0.294 ± 0.019 (n=3) | 88.07 | 93 | 0.00566 | 0.0 | 0.12% |
| 4096 | 100 | safe_keep | 3 | 0.284 ± 0.017 (n=3) | 88.33 | 92 | 0.00203 | 0.0 | 0.25% |

