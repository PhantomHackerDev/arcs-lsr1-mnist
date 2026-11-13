### Round 7: F-MNIST CNN, batch 64, full-network ARCs-LSR1 (m=10, same-batch y), 10 epochs

Seed 0 for screening; the selected configurations also have seeds 1-2 (seeds column). Compute counts every forward/backward sample, so passes on the larger batch are charged at their size; 1.0 = one full-training-set gradient.

**R: ratio test on a larger batch (gradient and pairs on the current 64)**

| run | seeds | final train loss | test acc % | train loss ep 1 / 3 | accepted % | median μ, last 10% | median ‖s‖ (accepted), last 10% | compute / epoch (full-gradient units) |
|---|---|---|---|---|---|---|---|---|
| ratio on 256 samples, (η₁, η₂) = (0.1, 0.75) | 1 | 0.7321 | 72.44 | 0.734 / 0.733 | 92 | 1e+12 | 2.2e-06 | 5.0 |
| ratio on 256 samples, (η₁, η₂) = (0.01, 0.25) | 3 | 0.3657 ± 0.0198 | 84.98 ± 0.65 | 0.495 / 0.424 | 82 | 1 | 0.697 | 5.0 |
| ratio on 256 samples, (η₁, η₂) = (0, 0.1) | 1 | 0.6645 | 74.53 | 0.519 / 0.545 | 65 | 0.25 | 1.23 | 5.0 |
| ratio on 1024 samples, (η₁, η₂) = (0.1, 0.75) | 1 | 0.9181 | 65.90 | 0.919 / 0.919 | 83 | 1e+12 | 1.79e-06 | 13.8 |
| ratio on 1024 samples, (η₁, η₂) = (0.01, 0.25) | 1 | 0.7134 | 72.20 | 0.729 / 0.724 | 84 | 2.5e+11 | 4.42e-06 | 13.8 |
| ratio on 1024 samples, (η₁, η₂) = (0, 0.1) | 1 | 0.4722 | 81.27 | 0.613 / 0.542 | 62 | 1.31e+05 | 0.00647 | 13.8 |

**Y: curvature pairs and ratio test on the larger batch**

| run | seeds | final train loss | test acc % | train loss ep 1 / 3 | accepted % | median μ, last 10% | median ‖s‖ (accepted), last 10% | compute / epoch (full-gradient units) |
|---|---|---|---|---|---|---|---|---|
| pairs + ratio on 256 samples, (η₁, η₂) = (0.01, 0.25) | 3 | 0.3713 ± 0.0180 | 84.91 ± 0.25 | 0.440 / 0.398 | 84 | 2 | 0.612 | 9.0 |
| pairs + ratio on 256 samples, (η₁, η₂) = (0, 0.1) | 1 | 0.5780 | 79.35 | 0.506 / 0.497 | 65 | 0.5 | 1.35 | 9.0 |
| pairs + ratio on 1024 samples, (η₁, η₂) = (0.01, 0.25) | 1 | 0.7579 | 71.50 | 0.775 / 0.774 | 81 | 1.25e+11 | 6e-06 | 33.0 |
| pairs + ratio on 1024 samples, (η₁, η₂) = (0, 0.1) | 1 | 0.4541 | 82.02 | 0.584 / 0.528 | 63 | 1.31e+05 | 0.00578 | 33.0 |

**H: hyperparameters (everything on the current batch)**

| run | seeds | final train loss | test acc % | train loss ep 1 / 3 | accepted % | median μ, last 10% | median ‖s‖ (accepted), last 10% | compute / epoch (full-gradient units) |
|---|---|---|---|---|---|---|---|---|
| defaults (round 5) | 3 | 0.7128 ± 0.0502 | 73.40 ± 1.62 | 0.695 / 0.690 | 93 | 0.25 | 1.23 | 2.0 |
| η₁ = 0 | 1 | 1.1492 | 55.02 | 0.682 / 1.121 | 84 | 0.125 | 1.56 | 2.0 |
| η₁ = 0.01 | 1 | 2.3399 | 10.00 | 0.633 / 0.748 | 80 | 0.0156 | 1.59 | 2.0 |
| η₁ = 0.25 | 1 | 0.4679 | 80.84 | 0.507 / 0.420 | 88 | 0.5 | 0.872 | 2.0 |
| η₁ = 0.5 | 1 | 0.2855 | 87.15 | 0.456 / 0.343 | 71 | 2 | 0.448 | 2.0 |
| η₂ = 0.25 | 1 | 1.0918 | 59.20 | 0.802 / 1.346 | 20 | 1e+12 | 0 | 2.0 |
| η₂ = 0.5 | 1 | 1.0807 | 54.78 | 0.785 / 1.025 | 65 | 1 | 0.603 | 2.0 |
| η₂ = 0.9 | 1 | 0.4758 | 82.08 | 0.515 / 0.557 | 97 | 0.5 | 0.991 | 2.0 |
| (η₁, η₂) = (0.01, 0.25) | 1 | 2.3313 | 10.00 | 1.507 / 2.331 | 15 | 1e+12 | 0 | 2.0 |
| (η₁, η₂) = (0, 0.1) | 1 | 2.3146 | 10.00 | 2.381 / 2.315 | 14 | 1e+12 | 0 | 2.0 |
| μ₀ = 0.01 | 1 | 0.6279 | 76.37 | 0.757 / 0.709 | 92 | 0.32 | 1.26 | 2.0 |
| μ₀ = 100 | 1 | 0.7715 | 71.37 | 0.635 / 0.746 | 94 | 0.391 | 1.24 | 2.0 |
| μ_min = 1 | 1 | 0.3971 | 83.93 | 0.550 / 0.415 | 99 | 1 | 0.727 | 2.0 |
| μ_min = 4 | 1 | 0.2641 | 87.58 | 0.441 / 0.328 | 100 | 4 | 0.403 | 2.0 |
| μ_min = 16 | 3 | 0.2195 ± 0.0104 | 88.72 ± 0.30 | 0.391 / 0.290 | 100 | 16 | 0.223 | 2.0 |
| μ_min = 64 | 3 | 0.2029 ± 0.0124 | 89.54 ± 0.29 | 0.398 / 0.291 | 100 | 64 | 0.127 | 2.0 |
| μ_min = 256 | 1 | 0.2259 | 89.15 | 0.456 / 0.301 | 100 | 256 | 0.0752 | 2.0 |
| η₁ = 0.7 | 1 | 0.2703 | 88.01 | 0.426 / 0.372 | 54 | 8 | 0.292 | 2.0 |
| γ rule: BB | 1 | 0.4151 | 84.03 | 0.660 / 0.495 | 23 | 1e+12 | 0 | 2.0 |

**A: ablation, same μ_min without curvature pairs (B = I)**

| run | seeds | final train loss | test acc % | train loss ep 1 / 3 | accepted % | median μ, last 10% | median ‖s‖ (accepted), last 10% | compute / epoch (full-gradient units) |
|---|---|---|---|---|---|---|---|---|
| no pairs (B = I), μ_min = 4 | 1 | 0.2631 | 86.89 | 0.475 / 0.295 | 100 | 4 | 0.389 | 2.0 |
| no pairs (B = I), μ_min = 16 | 3 | 0.2134 ± 0.0148 | 89.00 ± 0.42 | 0.402 / 0.286 | 100 | 16 | 0.238 | 2.0 |
| no pairs (B = I), μ_min = 64 | 3 | 0.2113 ± 0.0066 | 89.34 ± 0.21 | 0.427 / 0.299 | 100 | 64 | 0.137 | 2.0 |

**C: combinations with μ_min**

| run | seeds | final train loss | test acc % | train loss ep 1 / 3 | accepted % | median μ, last 10% | median ‖s‖ (accepted), last 10% | compute / epoch (full-gradient units) |
|---|---|---|---|---|---|---|---|---|
| μ_min = 16, η₁ = 0.5 | 1 | 0.2111 | 88.90 | 0.401 / 0.288 | 98 | 16 | 0.218 | 2.0 |
| μ_min = 16, ratio on 256 samples, (η₁, η₂) = (0.01, 0.25) | 1 | 0.2084 | 89.06 | 0.362 / 0.292 | 94 | 16 | 0.22 | 5.0 |
| μ_min = 16, pairs + ratio on 256 samples, (η₁, η₂) = (0.01, 0.25) | 1 | 0.2270 | 88.23 | 0.373 / 0.287 | 91 | 16 | 0.255 | 9.0 |

**Baselines (round 6)**

| run | seeds | final train loss | test acc % | train loss ep 1 / 3 | accepted % | median μ, last 10% | median ‖s‖ (accepted), last 10% | compute / epoch (full-gradient units) |
|---|---|---|---|---|---|---|---|---|
| SGD+momentum (paper: lr 0.1) | 3 | 0.4315 ± 0.0208 | 83.06 ± 0.48 | 0.539 / 0.444 | – | – | – | 1.0 |
| Adam (paper: lr 1e-3) | 3 | 0.2049 ± 0.0134 | 90.03 ± 0.29 | 0.404 / 0.294 | – | – | – | 1.0 |
| AdamW (tuned lr 0.003) | 3 | 0.1977 ± 0.0089 | 89.55 ± 0.23 | 0.342 / 0.268 | – | – | – | 1.0 |
| Muon (tuned lr 0.005) | 3 | 0.1281 ± 0.0014 | 90.25 ± 0.14 | 0.296 / 0.208 | – | – | – | 1.0 |
| Shampoo (tuned lr 0.001) | 3 | 0.1370 ± 0.0112 | 90.06 ± 0.18 | 0.321 / 0.225 | – | – | – | 1.0 |
| SophiaG (tuned lr 0.0003) | 3 | 0.1827 ± 0.0037 | 89.97 ± 0.14 | 0.372 / 0.274 | – | – | – | 1.1 |

Best by seed-0 training loss: R: r7_rho256_eta0.01-0.25 (0.3788), Y: r7_yrho256_eta0.01-0.25 (0.3597), H: r7_mumin_64 (0.2044), A: r7_nopairs_mumin_16 (0.1998), C: r7_mumin16_rho256_eta0.01-0.25 (0.2084)

