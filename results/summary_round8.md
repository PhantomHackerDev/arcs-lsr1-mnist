### Round 8: value of the L-SR1 curvature vs. the floor μ_min (10 epochs)

Each cell: final training loss on the full training set (test accuracy). Seed 0 unless n is shown. 'no pairs' stores no curvature pairs, so B = I and the step is a cubic-regularized gradient step (it still pays for the second gradient, which only its ratio test would need as a forward pass).

**F-MNIST CNN, batch 64**

| μ_min | with L-SR1 pairs | no pairs (B = I) |
|---|---|---|
| none (default 1e-8) | 0.713 ± 0.050 (73.40%, n=3) | 0.395 (83.18%) |
| 1 | 0.397 (83.93%) | 0.320 (85.73%) |
| 4 | 0.264 (87.58%) | 0.263 (86.89%) |
| 16 | 0.220 ± 0.010 (88.72%, n=3) | 0.213 ± 0.015 (89.00%, n=3) |
| 64 | 0.203 ± 0.012 (89.54%, n=3) | 0.211 ± 0.007 (89.34%, n=3) |
| 256 | 0.226 (89.15%) | 0.232 (89.06%) |

Baselines, 3 seeds: Muon 0.1281, AdamW 0.1977, Adam (paper) 0.2049

Selected μ_min, pairs (μ_min = 64) vs. no pairs (μ_min = 16), 3 / 3 seeds: geometric mean of the training loss over epochs 1-10 0.2593 ± 0.0083 vs. 0.2614 ± 0.0029; ratio (pairs / no pairs) per epoch 0.95-1.04.

**F-MNIST CNN, batch 256**

| μ_min | with L-SR1 pairs | no pairs (B = I) |
|---|---|---|
| none (default 1e-8) | 0.459 ± 0.078 (81.54%, n=3) | 0.374 (84.60%) |
| 1 | 0.271 (86.72%) | 0.345 ± 0.026 (85.73%, n=3) |
| 4 | 0.227 ± 0.003 (89.09%, n=3) | 0.333 ± 0.031 (85.93%, n=3) |
| 16 | 0.237 (89.14%) | 0.311 (86.89%) |
| 64 | 0.249 (89.25%) | 0.309 (87.14%) |
| 256 | 0.286 (88.35%) | 0.368 (85.10%) |

Baselines, 3 seeds: Muon 0.1394, AdamW 0.2102, Adam (paper) 0.2687

Selected μ_min, pairs (μ_min = 4) vs. no pairs (μ_min = 4), 3 / 3 seeds: geometric mean of the training loss over epochs 1-10 0.284 ± 0.0029 vs. 0.4331 ± 0.032; ratio (pairs / no pairs) per epoch 0.61-0.70.

**F-MNIST CNN, batch 1024**

| μ_min | with L-SR1 pairs | no pairs (B = I) |
|---|---|---|
| none (default 1e-8) | 0.373 (85.18%) | 0.610 (75.95%) |
| 1 | 0.321 (87.15%) | 0.608 ± 0.056 (76.49%, n=3) |
| 4 | 0.296 ± 0.009 (88.21%, n=3) | 0.628 ± 0.083 (75.74%, n=3) |
| 16 | 0.326 (87.10%) | 0.643 (74.97%) |
| 64 | 0.341 (86.89%) | 0.677 (73.80%) |
| 256 | 0.395 (84.73%) | 0.701 (73.02%) |

Selected μ_min, pairs (μ_min = 4) vs. no pairs (μ_min = 1), 3 / 3 seeds: geometric mean of the training loss over epochs 1-10 0.3956 ± 0.014 vs. 0.6862 ± 0.066; ratio (pairs / no pairs) per epoch 0.49-0.72.

**MNIST deep MLP, batch 256**

| μ_min | with L-SR1 pairs | no pairs (B = I) |
|---|---|---|
| none (default 1e-8) | 0.0026 ± 0.0012 (98.31%, n=3) | 0.0044 (98.17%) |
| 1 | 0.0023 (98.39%) | 0.0068 ± 0.0032 (98.19%, n=3) |
| 4 | 0.0068 ± 0.0079 (98.22%, n=3) | 0.0042 (98.21%) |
| 16 | 0.0042 (98.30%) | 0.0043 (98.13%) |
| 64 | 0.0130 (97.90%) | 0.0187 (97.88%) |
| 256 | 0.0375 (97.73%) | 0.0459 (97.61%) |

Baselines, 3 seeds: Muon 0.000135, AdamW 0.0163, Adam (paper) 0.01113

Selected μ_min, pairs (μ_min = 4) vs. no pairs (μ_min = 1), 3 / 3 seeds: geometric mean of the training loss over epochs 1-10 0.02095 ± 0.0027 vs. 0.03532 ± 0.0044; ratio (pairs / no pairs) per epoch 0.48-0.69.

